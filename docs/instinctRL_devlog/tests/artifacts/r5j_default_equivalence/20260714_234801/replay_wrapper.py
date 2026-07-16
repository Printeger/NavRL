#!/usr/bin/env python3
"""Run exactly one disabled R5J replay with fail-closed provenance."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from compare_disabled_replay import (
    EXPECTED_BRANCH,
    EXPECTED_CHECKPOINT_SHA256,
    compare_files,
    git_value,
    resolved_eval_seed,
    stored_and_replay_argv,
)


ARTIFACT_DIR = Path(__file__).resolve().parent
ROOT = ARTIFACT_DIR.parents[5]
BASELINE = ROOT / "docs/instinctRL_devlog/tests/artifacts/r5e3_braking_residual/20260714_234801/r5e3_r5g_downatten_z010_eval.json"
ATTEMPTS_DIR = ARTIFACT_DIR / "attempts"


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _attempt_id(root: Path) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    commit = git_value(root, "rev-parse", "--short", "HEAD") or "unknown"
    return f"{stamp}-{commit}"


def _run(command):
    try:
        completed = subprocess.run(command, shell=False, capture_output=True, text=True, check=False)
        return {
            "argv": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except OSError as error:
        return {"argv": command, "returncode": None, "stdout": "", "stderr": str(error)}


def cuda_preflight(eval_python: str):
    nvidia = _run(["nvidia-smi"])
    torch = _run([eval_python, "-c", "import torch; print(torch.cuda.is_available())"])
    torch_available = torch["returncode"] == 0 and torch["stdout"].strip().lower() == "true"
    ready = nvidia["returncode"] == 0 and torch_available
    if ready:
        failure = None
    elif nvidia["returncode"] != 0:
        failure = "nvidia-smi failed (exit {}): {}".format(
            nvidia["returncode"], (nvidia["stderr"] or nvidia["stdout"]).strip()
        )
    else:
        failure = "torch CUDA preflight failed: {}".format(
            (torch["stderr"] or torch["stdout"]).strip()
        )
    return {"ready": ready, "nvidia_smi": nvidia, "torch": torch, "failure": failure}


def _write_json(path: Path, value: Dict[str, Any]):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _freshness(result_path: Path, started_ns: int) -> Dict[str, Any]:
    freshness = {
        "result_exists": result_path.is_file(),
        "mtime_after_attempt_start": False,
        "nonempty": False,
        "json_result_path_matches": False,
        "ready": False,
    }
    if not freshness["result_exists"]:
        return freshness
    stat = result_path.stat()
    freshness.update({"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    freshness["mtime_after_attempt_start"] = stat.st_mtime_ns >= started_ns
    freshness["nonempty"] = stat.st_size > 0
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        freshness["json_result_path_matches"] = payload.get("result_path") == str(result_path)
        freshness["result_sha256"] = hashlib.sha256(result_path.read_bytes()).hexdigest()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        freshness["json_result_path_matches"] = False
    freshness["ready"] = all(
        freshness[key]
        for key in ("result_exists", "mtime_after_attempt_start", "nonempty", "json_result_path_matches")
    )
    return freshness


def _write_logs(record: Dict[str, Any], stdout: str, stderr: str):
    Path(record["stdout_path"]).write_text(stdout, encoding="utf-8")
    Path(record["stderr_path"]).write_text(stderr, encoding="utf-8")


def run_attempt(
    *,
    root: Path = ROOT,
    artifact_dir: Path = ARTIFACT_DIR,
    attempt_id: Optional[str] = None,
    preflight_fn: Callable[[str], Dict[str, Any]] = cuda_preflight,
    eval_runner: Callable[..., Any] = subprocess.run,
    compare_fn: Callable[..., Dict[str, Any]] = compare_files,
) -> Dict[str, Any]:
    """Execute one attempt; return the final wrapper record for unit tests and CLI."""
    attempt_id = attempt_id or _attempt_id(root)
    attempt_dir = artifact_dir / "attempts" / attempt_id
    collision_index = 0
    while attempt_dir.exists():
        collision_index += 1
        attempt_dir = artifact_dir / "attempts" / f"{attempt_id}-collision{collision_index}"
    attempt_id = attempt_dir.name
    attempt_dir.mkdir(parents=True, exist_ok=False)
    result_path = attempt_dir / "r5j_r5g_downatten_z010_eval.json"
    record_path = attempt_dir / "wrapper_record.json"
    comparison_path = attempt_dir / "comparison.json"
    started_at = _timestamp()
    started_ns = time.time_ns()
    record: Dict[str, Any] = {
        "attempt_id": attempt_id,
        "attempt_id_collision_avoided": collision_index > 0,
        "status": "RUNNING",
        "started_at": started_at,
        "branch": git_value(root, "branch", "--show-current"),
        "commit": git_value(root, "rev-parse", "HEAD"),
        "worktree": str(root),
        "worktree_status": git_value(root, "status", "--short"),
        "result_path": str(result_path),
        "stdout_path": str(attempt_dir / "replay.stdout.log"),
        "stderr_path": str(attempt_dir / "replay.stderr.log"),
        "record_path": str(record_path),
        "comparison_path": str(comparison_path),
        "exit_code": None,
        "failure": None,
        "freshness": {"ready": False},
    }
    try:
        stored_argv, replay_argv, overrides_unchanged, checkpoint_name = stored_and_replay_argv(root, result_path)
        checkpoint = Path(checkpoint_name)
        record.update({
            "stored_argv": stored_argv,
            "argv": replay_argv,
            "cwd": str(root / "isaac-training"),
            "checkpoint_path": str(checkpoint),
            "resolved_seed": resolved_eval_seed(root),
            "seed_verified": resolved_eval_seed(root) == 0,
            "legacy_overrides_unchanged": overrides_unchanged,
            "result_path_unused": not result_path.exists(),
        })
        record["checkpoint_sha256"] = (
            hashlib.sha256(checkpoint.read_bytes()).hexdigest() if checkpoint.is_file() else None
        )
        record["checkpoint_verified"] = record["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256
        record["branch_verified"] = record["branch"] == EXPECTED_BRANCH
        record["commit_verified"] = bool(record["commit"])
        record["cwd_verified"] = Path(record["cwd"]).is_dir()
    except (OSError, ValueError, StopIteration) as error:
        record["failure"] = f"attempt setup failed: {error}"

    preconditions = (
        "checkpoint_verified",
        "seed_verified",
        "legacy_overrides_unchanged",
        "result_path_unused",
        "branch_verified",
        "commit_verified",
        "cwd_verified",
    )
    if not record.get("failure"):
        failed = [key for key in preconditions if record.get(key) is not True]
        if failed:
            record["failure"] = f"pre-eval provenance failed: {', '.join(failed)}"

    if not record.get("failure"):
        record["cuda_preflight"] = preflight_fn(record["argv"][0])
        if not record["cuda_preflight"].get("ready"):
            record["failure"] = record["cuda_preflight"].get("failure") or "CUDA preflight failed"
        _write_logs(
            record,
            record["cuda_preflight"]["nvidia_smi"].get("stdout", "") + record["cuda_preflight"]["torch"].get("stdout", ""),
            record["cuda_preflight"]["nvidia_smi"].get("stderr", "") + record["cuda_preflight"]["torch"].get("stderr", ""),
        )
    else:
        record["cuda_preflight"] = {"ready": False, "failure": "not run because pre-eval provenance failed"}
        _write_logs(record, "", record["failure"])

    if not record.get("failure"):
        completed = eval_runner(
            record["argv"],
            cwd=record["cwd"],
            shell=False,
            capture_output=True,
            text=True,
            check=False,
        )
        _write_logs(record, completed.stdout, completed.stderr)
        record["exit_code"] = completed.returncode
        if completed.returncode != 0:
            record["failure"] = f"eval subprocess failed with exit {completed.returncode}"
        else:
            record["freshness"] = _freshness(result_path, started_ns)
            if not record["freshness"]["ready"]:
                record["failure"] = "eval exited 0 but did not produce a fresh replay result"

    checkpoint_for_compare = Path(record.get("checkpoint_path", root / "missing-checkpoint"))
    report = compare_fn(
        root,
        root / "docs/instinctRL_devlog/tests/artifacts/r5e3_braking_residual/20260714_234801/r5e3_r5g_downatten_z010_eval.json",
        result_path,
        checkpoint_for_compare,
        wrapper_failure=record.get("failure"),
        attempt_record=record,
    )
    if report["status"] == "HOLD" and not record.get("failure"):
        record["failure"] = "comparator returned HOLD"
    record["comparator_outcome"] = report["status"]
    record["status"] = report["status"]
    record["finished_at"] = _timestamp()
    _write_json(comparison_path, report)
    _write_json(record_path, record)
    return record


def main() -> int:
    record = run_attempt()
    print(record["status"])
    return 0 if record["status"] == "GO (design only)" else 1


if __name__ == "__main__":
    raise SystemExit(main())
