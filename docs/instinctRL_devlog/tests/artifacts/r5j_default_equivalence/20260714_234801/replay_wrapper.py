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
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    freshness["json_result_path_matches"] = payload.get("result_path") == str(result_path)
    freshness["result_sha256"] = hashlib.sha256(result_path.read_bytes()).hexdigest()
    freshness["ready"] = all(
        freshness[key]
        for key in ("result_exists", "mtime_after_attempt_start", "nonempty", "json_result_path_matches")
    )
    return freshness


def _write_logs(record: Dict[str, Any], stdout: str, stderr: str):
    Path(record["stdout_path"]).write_text(stdout, encoding="utf-8")
    Path(record["stderr_path"]).write_text(stderr, encoding="utf-8")


def _exception_failure(stage: str, error: Exception) -> str:
    return f"{stage} exception {type(error).__name__}: {error}"


def _hold_report(failure: str) -> Dict[str, Any]:
    """Produce parseable fail-closed output even when comparison cannot run."""
    return {
        "status": "HOLD",
        "checks": {"wrapper_failure": True},
        "hold_reason": failure,
    }


def _append_failure(record: Dict[str, Any], failure: str) -> None:
    existing = record.get("failure")
    record["failure"] = f"{existing}; {failure}" if existing else failure
    record["status"] = "HOLD"
    record["comparator_outcome"] = "HOLD"


def _persist_artifacts(
    record: Dict[str, Any], report: Dict[str, Any], stdout: str, stderr: str
) -> Dict[str, Any]:
    """Best-effort durable output; failures can never leave a GO decision."""
    try:
        _write_logs(record, stdout, stderr)
    except Exception as error:
        _append_failure(record, _exception_failure("artifact logs", error))
        report = _hold_report(record["failure"])
    try:
        _write_json(Path(record["comparison_path"]), report)
    except Exception as error:
        _append_failure(record, _exception_failure("artifact comparison", error))
    try:
        _write_json(Path(record["record_path"]), record)
    except Exception as error:
        _append_failure(record, _exception_failure("artifact record", error))
        # The record path itself failed, but preserve a parseable HOLD comparison
        # where possible and return the failure to the caller.
        try:
            _write_json(Path(record["comparison_path"]), _hold_report(record["failure"]))
        except Exception:
            pass
    return report


def _new_record(
    *,
    root: Path,
    attempt_id: str,
    record_path: Path,
    comparison_path: Path,
    stdout_path: Path,
    stderr_path: Path,
    result_path: Path,
    source_commit: Optional[str],
    commit: Optional[str],
    pre_attempt_worktree_status: Optional[str],
) -> Dict[str, Any]:
    worktree_clean = pre_attempt_worktree_status == ""
    return {
        "attempt_id": attempt_id,
        "status": "RUNNING",
        "started_at": _timestamp(),
        "branch": git_value(root, "branch", "--show-current"),
        "commit": commit,
        "source_commit": source_commit,
        "pre_attempt_worktree_status": pre_attempt_worktree_status,
        "worktree_clean": worktree_clean,
        "worktree": str(root),
        "worktree_status": pre_attempt_worktree_status,
        "result_path": str(result_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "record_path": str(record_path),
        "comparison_path": str(comparison_path),
        "exit_code": None,
        "failure": None,
        "freshness": {"ready": False},
    }


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
    # This gate intentionally runs before attempts/<id> is created: a dirty tree
    # cannot be represented as a CUDA-capable replay attempt.
    pre_attempt_worktree_status = git_value(
        root, "status", "--porcelain=v1", "--untracked-files=all"
    )
    source_commit = git_value(root, "rev-parse", "--verify", "HEAD")
    # Compatibility `commit` is the exact source value; a second read detects a
    # HEAD race before any CUDA-capable attempt is allowed.
    commit = source_commit
    current_head = git_value(root, "rev-parse", "HEAD")
    pre_attempt_failures = []
    if pre_attempt_worktree_status != "":
        pre_attempt_failures.append("worktree_clean")
    if not source_commit:
        pre_attempt_failures.append("source_commit")
    if not current_head:
        pre_attempt_failures.append("commit")
    if source_commit != current_head:
        pre_attempt_failures.append("source_commit_matches_commit")
    if pre_attempt_failures:
        hold_dir = artifact_dir / "pre_attempt_holds"
        try:
            hold_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            # The record below still carries paths and is returned fail-closed.
            pass
        prefix = hold_dir / attempt_id
        record = _new_record(
            root=root,
            attempt_id=attempt_id,
            record_path=Path(f"{prefix}.wrapper_record.json"),
            comparison_path=Path(f"{prefix}.comparison.json"),
            stdout_path=Path(f"{prefix}.stdout.log"),
            stderr_path=Path(f"{prefix}.stderr.log"),
            result_path=Path(f"{prefix}.no_eval.json"),
            source_commit=source_commit,
            commit=commit,
            pre_attempt_worktree_status=pre_attempt_worktree_status,
        )
        record["failure"] = "pre-attempt provenance failed: " + ", ".join(pre_attempt_failures)
        record["cuda_preflight"] = {"ready": False, "failure": "not run because pre-attempt provenance failed"}
        record["status"] = "HOLD"
        record["comparator_outcome"] = "HOLD"
        record["finished_at"] = _timestamp()
        _persist_artifacts(record, _hold_report(record["failure"]), "", record["failure"])
        return record

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
    started_ns = time.time_ns()
    record = _new_record(
        root=root,
        attempt_id=attempt_id,
        record_path=record_path,
        comparison_path=comparison_path,
        stdout_path=attempt_dir / "replay.stdout.log",
        stderr_path=attempt_dir / "replay.stderr.log",
        result_path=result_path,
        source_commit=source_commit,
        commit=commit,
        pre_attempt_worktree_status=pre_attempt_worktree_status,
    )
    record["attempt_id_collision_avoided"] = collision_index > 0
    stdout = ""
    stderr = ""
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
    except Exception as error:
        record["failure"] = _exception_failure("attempt setup", error)

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
        try:
            record["cuda_preflight"] = preflight_fn(record["argv"][0])
            if not record["cuda_preflight"].get("ready"):
                record["failure"] = record["cuda_preflight"].get("failure") or "CUDA preflight failed"
            stdout = (
                record["cuda_preflight"]["nvidia_smi"].get("stdout", "")
                + record["cuda_preflight"]["torch"].get("stdout", "")
            )
            stderr = (
                record["cuda_preflight"]["nvidia_smi"].get("stderr", "")
                + record["cuda_preflight"]["torch"].get("stderr", "")
            )
        except Exception as error:
            record["failure"] = _exception_failure("preflight", error)
            record["cuda_preflight"] = {"ready": False, "failure": record["failure"]}
    else:
        record["cuda_preflight"] = {"ready": False, "failure": "not run because pre-eval provenance failed"}
        stderr = record["failure"]

    if not record.get("failure"):
        try:
            completed = eval_runner(
                record["argv"],
                cwd=record["cwd"],
                shell=False,
                capture_output=True,
                text=True,
                check=False,
            )
            stdout, stderr = completed.stdout, completed.stderr
            record["exit_code"] = completed.returncode
            if completed.returncode != 0:
                record["failure"] = f"eval subprocess failed with exit {completed.returncode}"
        except Exception as error:
            record["failure"] = _exception_failure("eval", error)

    if not record.get("failure") and record.get("exit_code") == 0:
        try:
            record["freshness"] = _freshness(result_path, started_ns)
            if not record["freshness"]["ready"]:
                record["failure"] = "eval exited 0 but did not produce a fresh replay result"
        except Exception as error:
            record["failure"] = _exception_failure("freshness", error)

    checkpoint_for_compare = Path(record.get("checkpoint_path", root / "missing-checkpoint"))
    try:
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
    except Exception as error:
        record["failure"] = _exception_failure("comparator", error)
        report = _hold_report(record["failure"])
    record["comparator_outcome"] = report.get("status", "HOLD")
    record["status"] = "HOLD" if record.get("failure") else report.get("status", "HOLD")
    record["finished_at"] = _timestamp()
    _persist_artifacts(record, report, stdout, stderr)
    return record


def main() -> int:
    record = run_attempt()
    print(record["status"])
    return 0 if record["status"] == "GO (design only)" else 1


if __name__ == "__main__":
    raise SystemExit(main())
