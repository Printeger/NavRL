"""Hard gates for instinctRL diagnostic evaluation summaries.

The gates are intentionally independent from Isaac Sim. They consume the JSON
files produced by ``training/scripts/eval.py`` and turn handbook metrics into a
single pass/fail report plus a sortable score for short diagnostic sweeps.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class GateSpec:
    name: str
    keys: Tuple[str, ...]
    category: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    required: bool = True


@dataclass(frozen=True)
class GateResult:
    name: str
    category: str
    key: Optional[str]
    value: Optional[float]
    min_value: Optional[float]
    max_value: Optional[float]
    passed: bool
    normalized_violation: float
    reason: str


@dataclass(frozen=True)
class GateReport:
    passed: bool
    safety_passed: bool
    score: float
    passed_count: int
    total_count: int
    results: Tuple[GateResult, ...]

    def to_dict(self):
        return {
            "passed": self.passed,
            "safety_passed": self.safety_passed,
            "score": self.score,
            "passed_count": self.passed_count,
            "total_count": self.total_count,
            "results": [asdict(result) for result in self.results],
        }


DEFAULT_GATE_SPECS: Tuple[GateSpec, ...] = (
    GateSpec(
        name="station_drift_mean",
        keys=("eval/station/handbook.station_keeping_drift_mean",),
        category="station",
        max_value=1.3,
    ),
    GateSpec(
        name="station_drift_p95",
        keys=("eval/station/handbook.station_keeping_drift_p95",),
        category="station",
        max_value=2.6,
    ),
    GateSpec(
        name="station_null_speed_mean",
        keys=("eval/station/handbook.null_command_speed_mean",),
        category="station",
        max_value=0.08,
    ),
    GateSpec(
        name="station_anchor_error_mean",
        keys=("eval/station/handbook.anchor_error_mean",),
        category="station",
        max_value=2.0,
    ),
    GateSpec(
        name="tracking_rmse_actual",
        keys=(
            "eval/tracking/handbook.tracking_rmse_actual_body_vs_v_cmd",
            "eval/handbook.tracking_rmse_actual_body_vs_v_cmd",
        ),
        category="tracking",
        max_value=0.45,
    ),
    GateSpec(
        name="tracking_preservation_ratio",
        keys=(
            "eval/tracking/handbook.command_preservation_ratio",
            "eval/handbook.command_preservation_ratio",
        ),
        category="tracking",
        min_value=0.75,
        max_value=1.05,
    ),
    GateSpec(
        name="tracking_command_amplification_mean",
        keys=(
            "eval/tracking/handbook.command_amplification_mean",
            "eval/handbook.command_amplification_mean",
        ),
        category="tracking",
        max_value=0.05,
    ),
    GateSpec(
        name="tracking_command_amplification_rate",
        keys=(
            "eval/tracking/handbook.command_amplification_rate",
            "eval/handbook.command_amplification_rate",
        ),
        category="tracking",
        max_value=0.15,
    ),
    GateSpec(
        name="safety_collision_rate",
        keys=(
            "eval/handbook.safety_collision_rate",
            "eval/tracking/handbook.safety_collision_rate",
            "eval/station/handbook.safety_collision_rate",
        ),
        category="safety",
        max_value=0.0,
    ),
    GateSpec(
        name="safety_min_clearance_p05",
        keys=(
            "eval/handbook.safety_min_clearance_p05",
            "eval/tracking/handbook.safety_min_clearance_p05",
            "eval/station/handbook.safety_min_clearance_p05",
        ),
        category="safety",
        min_value=1.0,
    ),
    GateSpec(
        name="ics_violation_rate",
        keys=(
            "eval/handbook.ics_violation_rate",
            "eval/tracking/handbook.ics_violation_rate",
            "eval/station/handbook.ics_violation_rate",
        ),
        category="safety",
        max_value=0.005,
    ),
    GateSpec(
        name="termination_collision",
        keys=("eval/handbook.termination_collision",),
        category="safety",
        max_value=0.0,
    ),
    GateSpec(
        name="termination_below_bound",
        keys=("eval/handbook.termination_below_bound",),
        category="termination",
        max_value=0.0,
    ),
    GateSpec(
        name="termination_above_bound",
        keys=("eval/handbook.termination_above_bound",),
        category="termination",
        max_value=0.0,
    ),
)


def load_eval_summary(path: str | Path) -> Mapping[str, object]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_gates(
    summary: Mapping[str, object],
    specs: Sequence[GateSpec] = DEFAULT_GATE_SPECS,
) -> GateReport:
    results = tuple(_evaluate_one(summary, spec) for spec in specs)
    passed_count = sum(1 for result in results if result.passed)
    violation = sum(result.normalized_violation for result in results)
    passed = all(result.passed for result in results)
    safety_results = [result for result in results if result.category == "safety"]
    safety_passed = all(result.passed for result in safety_results)
    score = float(passed_count - violation)
    return GateReport(
        passed=passed,
        safety_passed=safety_passed,
        score=score,
        passed_count=passed_count,
        total_count=len(results),
        results=results,
    )


def format_gate_report(report: GateReport) -> str:
    lines = [
        f"passed={report.passed} safety_passed={report.safety_passed} "
        f"score={report.score:.3f} gates={report.passed_count}/{report.total_count}",
    ]
    for result in report.results:
        status = "PASS" if result.passed else "FAIL"
        value = "missing" if result.value is None else f"{result.value:.6g}"
        bounds = []
        if result.min_value is not None:
            bounds.append(f">={result.min_value:g}")
        if result.max_value is not None:
            bounds.append(f"<={result.max_value:g}")
        bound_text = " ".join(bounds)
        lines.append(
            f"{status} {result.category}.{result.name}: {value} {bound_text} "
            f"({result.reason})"
        )
    return "\n".join(lines)


def _evaluate_one(summary: Mapping[str, object], spec: GateSpec) -> GateResult:
    key, value = _first_finite(summary, spec.keys)
    if key is None or value is None:
        return GateResult(
            name=spec.name,
            category=spec.category,
            key=None,
            value=None,
            min_value=spec.min_value,
            max_value=spec.max_value,
            passed=not spec.required,
            normalized_violation=100.0 if spec.required else 0.0,
            reason="missing required metric" if spec.required else "missing optional metric",
        )

    violation = 0.0
    reasons = []
    if spec.min_value is not None and value < spec.min_value:
        delta = spec.min_value - value
        violation += delta / max(abs(spec.min_value), 1.0)
        reasons.append(f"below min by {delta:.6g}")
    if spec.max_value is not None and value > spec.max_value:
        delta = value - spec.max_value
        violation += delta / max(abs(spec.max_value), 1.0)
        reasons.append(f"above max by {delta:.6g}")
    passed = violation <= 1e-12
    return GateResult(
        name=spec.name,
        category=spec.category,
        key=key,
        value=value,
        min_value=spec.min_value,
        max_value=spec.max_value,
        passed=passed,
        normalized_violation=float(violation),
        reason="within bounds" if passed else "; ".join(reasons),
    )


def _first_finite(
    summary: Mapping[str, object],
    keys: Iterable[str],
) -> tuple[Optional[str], Optional[float]]:
    for key in keys:
        if key not in summary:
            continue
        try:
            value = float(summary[key])
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return key, value
    return None, None


def _main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate instinctRL hard gates.")
    parser.add_argument("summary_json", help="Path to eval.py summary JSON")
    parser.add_argument("--json", action="store_true", help="Print machine-readable report")
    args = parser.parse_args()

    report = evaluate_gates(load_eval_summary(args.summary_json))
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(format_gate_report(report))
    return 0 if report.passed else 2


if __name__ == "__main__":
    raise SystemExit(_main())
