#!/usr/bin/env python3
"""Strict, importable R5J-default-off replay comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple


R5J_DIAGNOSTIC_FIELDS = (
    "ics_residual_preemption_trigger",
    "ics_residual_preemption_range_rate_available",
)
ALLOWED_TOP_LEVEL_KEYS = {"result_path"}
EXPECTED_CHECKPOINT_SHA256 = "9b0ab9df5dda083b1121d722cd79ba4fd59fdbd10610a4db2467444ba2c44ac2"
SOURCE_VARIANT = "r5g_downatten_z010"


def _load(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def exactly_equal(left: Any, right: Any) -> bool:
    """Compare JSON-shaped values without tolerances or coercion."""
    if isinstance(left, float) and isinstance(right, float):
        return left == right or (math.isnan(left) and math.isnan(right))
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            exactly_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            exactly_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def _is_diagnostic_key(key: str) -> bool:
    return key in R5J_DIAGNOSTIC_FIELDS or any(
        key.endswith(f"diagnostics.{field}") for field in R5J_DIAGNOSTIC_FIELDS
    )


def _path_label(path: Tuple[str, ...]) -> str:
    return ".".join(path)


def collect_r5j_summaries(value: Any, *, path: Tuple[str, ...] = ()) -> Dict[str, Any]:
    """Collect every present direct or flattened R5J summary before filtering."""
    found: Dict[str, Any] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = path + (key,)
            if _is_diagnostic_key(key):
                found[_path_label(child_path)] = child
                continue
            found.update(collect_r5j_summaries(child, path=child_path))
        return found
    if isinstance(value, list):
        for index, child in enumerate(value):
            found.update(collect_r5j_summaries(child, path=path + (str(index),)))
    return found


def remove_allowed_fields(
    value: Any,
    *,
    allowed_diagnostic_paths: Iterable[str],
    path: Tuple[str, ...] = (),
) -> Any:
    """Remove only explicitly allowed top-level or short-suite diagnostic deltas."""
    allowed = set(allowed_diagnostic_paths)
    if isinstance(value, dict):
        cleaned = {}
        for key, child in value.items():
            child_path = path + (key,)
            label = _path_label(child_path)
            if not path and key in ALLOWED_TOP_LEVEL_KEYS:
                continue
            if label in allowed:
                continue
            cleaned[key] = remove_allowed_fields(
                child,
                allowed_diagnostic_paths=allowed,
                path=child_path,
            )
        return cleaned
    if isinstance(value, list):
        return [
            remove_allowed_fields(
                child,
                allowed_diagnostic_paths=allowed,
                path=path + (str(index),),
            )
            for index, child in enumerate(value)
        ]
    return value


def validate_disabled_summary(summary: Any) -> Tuple[bool, str]:
    """Disabled R5J summaries must be populated, finite, and identically zero."""
    if not isinstance(summary, dict):
        return False, "summary is not an object"
    for key in ("count", "finite_count", "mean", "min", "max"):
        if key not in summary:
            return False, f"missing {key}"
    for key in ("count", "finite_count"):
        value = summary[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            return False, f"{key} must be finite and positive"
    for key, value in summary.items():
        if key in {"count", "finite_count"}:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False, f"{key} is not a numeric summary statistic"
        if not math.isfinite(value) or value != 0.0:
            return False, f"{key} must be finite and exactly zero"
    return True, "ok"


def expected_short_diagnostic_keys() -> Tuple[str, ...]:
    flattened = tuple(
        f"eval/{pass_name}/diagnostics.{field}"
        for pass_name in ("station", "tracking")
        for field in R5J_DIAGNOSTIC_FIELDS
    )
    per_pass = tuple(
        f"passes.{pass_name}.eval/diagnostics.{field}"
        for pass_name in ("station_static_mid360", "tracking_static_mid360")
        for field in R5J_DIAGNOSTIC_FIELDS
    )
    return flattened + per_pass


def compare_documents(
    baseline: Dict[str, Any],
    replay: Optional[Dict[str, Any]],
    *,
    checkpoint_sha256: str,
    checkpoint_path_matches: bool,
    resolved_seed_is_zero: bool,
    legacy_overrides_unchanged: bool,
    gate_evaluator: Callable[[Dict[str, Any]], Dict[str, Any]],
    expected_diagnostic_keys: Iterable[str] = (),
    wrapper_failure: Optional[str] = None,
) -> Dict[str, Any]:
    """Return a serialisable exact-equivalence report without filesystem I/O."""
    common_checks = {
        "checkpoint_sha256": checkpoint_sha256 == EXPECTED_CHECKPOINT_SHA256,
        "resolved_seed": bool(resolved_seed_is_zero),
        "stored_overrides_unchanged": bool(legacy_overrides_unchanged),
    }
    if replay is None:
        checks = {
            **common_checks,
            "replay_exists": False,
            "wrapper_recorded_failure": bool(wrapper_failure),
        }
        return {
            "status": "HOLD",
            "allowed_differences": sorted(ALLOWED_TOP_LEVEL_KEYS | set(R5J_DIAGNOSTIC_FIELDS)),
            "checks": checks,
            "hold_reason": wrapper_failure or "Replay JSON is missing and no wrapper failure was recorded.",
        }

    # Validate every present diagnostic before removing only the approved paths.
    baseline_diagnostics = collect_r5j_summaries(baseline)
    replay_diagnostics = collect_r5j_summaries(replay)
    expected = tuple(expected_diagnostic_keys)
    diagnostic_checks = {
        f"baseline:{key}": validate_disabled_summary(value)[0]
        for key, value in baseline_diagnostics.items()
    }
    diagnostic_checks.update({
        f"replay:{key}": validate_disabled_summary(value)[0]
        for key, value in replay_diagnostics.items()
    })
    missing_expected = [key for key in expected if key not in replay_diagnostics]
    for key in missing_expected:
        diagnostic_checks[f"replay:{key}"] = False
    allowed_paths = set(expected) | set(R5J_DIAGNOSTIC_FIELDS)
    baseline_clean = remove_allowed_fields(baseline, allowed_diagnostic_paths=allowed_paths)
    replay_clean = remove_allowed_fields(replay, allowed_diagnostic_paths=allowed_paths)
    baseline_gates = gate_evaluator(baseline_clean)
    replay_gates = gate_evaluator(replay_clean)
    checks = {
        **common_checks,
        "replay_exists": True,
        "checkpoint_path": checkpoint_path_matches,
        "existing_json_fields_exact": exactly_equal(baseline_clean, replay_clean),
        "disabled_r5j_diagnostics_exact_zero": bool(diagnostic_checks) and all(diagnostic_checks.values()),
        "expected_disabled_r5j_diagnostics_present": not missing_expected,
        "gate_report_exact": exactly_equal(baseline_gates, replay_gates),
    }
    return {
        "status": "GO (design only)" if all(checks.values()) else "HOLD",
        "allowed_differences": sorted(ALLOWED_TOP_LEVEL_KEYS | set(R5J_DIAGNOSTIC_FIELDS)),
        "checks": checks,
        "diagnostic_checks": diagnostic_checks,
        "baseline_r5j_diagnostics": baseline_diagnostics,
        "replay_r5j_diagnostics": replay_diagnostics,
        "baseline_gate_report": baseline_gates,
        "replay_gate_report": replay_gates,
    }


def stored_and_replay_argv(root: Path, replay_path: Path):
    summary = _load(root / "docs/instinctRL_devlog/tests/artifacts/sweeps/20260714_234801/summary.json")
    source = next(job for job in summary["jobs"] if job["variant"] == SOURCE_VARIANT)
    stored = list(source["eval_command"])
    result_indexes = [index for index, item in enumerate(stored) if item.startswith("result_path=")]
    if len(result_indexes) != 1:
        raise ValueError("stored R5G argv must contain exactly one result_path override")
    replay_argv = list(stored)
    replay_argv[result_indexes[0]] = f"result_path={replay_path}"
    replay_argv.append("instinctRL.ics.residual_preemption_enabled=false")
    unchanged_legacy_overrides = (
        replay_argv[:result_indexes[0]] + replay_argv[result_indexes[0] + 1:-1]
        == stored[:result_indexes[0]] + stored[result_indexes[0] + 1:]
    )
    return stored, replay_argv, unchanged_legacy_overrides, source["checkpoint_path"]


def resolved_eval_seed(root: Path) -> Optional[int]:
    text = (root / "isaac-training/training/cfg/eval.yaml").read_text(encoding="utf-8")
    match = re.search(r"^seed:\s*(\d+)\s*$", text, flags=re.MULTILINE)
    return int(match.group(1)) if match else None


def _gate_evaluator(root: Path) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    import sys

    sys.path.insert(0, str(root / "isaac-training" / "training" / "scripts"))
    from instinctRL.gates import evaluate_gates  # pylint: disable=import-outside-toplevel

    return lambda value: evaluate_gates(value).to_dict()


def compare_files(
    root: Path,
    baseline_path: Path,
    replay_path: Path,
    checkpoint_path: Path,
    *,
    wrapper_failure: Optional[str] = None,
) -> Dict[str, Any]:
    stored, replay_argv, overrides_unchanged, stored_checkpoint = stored_and_replay_argv(root, replay_path)
    sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    replay = _load(replay_path) if replay_path.exists() else None
    report = compare_documents(
        _load(baseline_path),
        replay,
        checkpoint_sha256=sha256,
        checkpoint_path_matches=replay is not None and replay.get("checkpoint_path") == stored_checkpoint,
        resolved_seed_is_zero=resolved_eval_seed(root) == 0,
        legacy_overrides_unchanged=overrides_unchanged,
        gate_evaluator=_gate_evaluator(root),
        expected_diagnostic_keys=expected_short_diagnostic_keys(),
        wrapper_failure=wrapper_failure,
    )
    report.update({
        "baseline": str(baseline_path),
        "replay": str(replay_path),
        "checkpoint_sha256": sha256,
        "stored_command": stored,
        "replay_command": replay_argv,
    })
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("replay", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wrapper-record", type=Path)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[6]
    wrapper_failure = None
    if args.wrapper_record and args.wrapper_record.exists():
        wrapper_failure = _load(args.wrapper_record).get("failure")
    report = compare_files(root, args.baseline, args.replay, args.checkpoint, wrapper_failure=wrapper_failure)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(report["status"])
    return 0 if report["status"] == "GO (design only)" else 1


if __name__ == "__main__":
    raise SystemExit(main())
