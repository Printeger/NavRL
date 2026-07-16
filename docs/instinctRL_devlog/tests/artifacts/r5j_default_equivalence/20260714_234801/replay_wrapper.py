#!/usr/bin/env python3
"""Run the one permitted disabled R5J replay, with complete local provenance."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from compare_disabled_replay import compare_files, resolved_eval_seed, stored_and_replay_argv


ARTIFACT_DIR = Path(__file__).resolve().parent
ROOT = ARTIFACT_DIR.parents[5]
BASELINE = ROOT / "docs/instinctRL_devlog/tests/artifacts/r5e3_braking_residual/20260714_234801/r5e3_r5g_downatten_z010_eval.json"
REPLAY = ARTIFACT_DIR / "r5j_r5g_downatten_z010_eval.json"
RECORD = ARTIFACT_DIR / "wrapper_record.json"
COMPARISON = ARTIFACT_DIR / "comparison.json"
STDOUT = ARTIFACT_DIR / "replay.stdout.log"
STDERR = ARTIFACT_DIR / "replay.stderr.log"


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run(command):
    try:
        completed = subprocess.run(command, shell=False, capture_output=True, text=True, check=False)
        return {"argv": command, "returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}
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
        detail = (nvidia["stderr"] or nvidia["stdout"]).strip()
        failure = f"nvidia-smi failed (exit {nvidia['returncode']}): {detail}"
    else:
        detail = (torch["stderr"] or torch["stdout"]).strip()
        failure = f"torch CUDA preflight failed: {detail}"
    return {"ready": ready, "nvidia_smi": nvidia, "torch": torch, "failure": failure}


def main() -> int:
    started_at = _timestamp()
    stored_argv, replay_argv, legacy_overrides_unchanged, checkpoint_name = stored_and_replay_argv(ROOT, REPLAY)
    checkpoint = Path(checkpoint_name)
    resolved_seed = resolved_eval_seed(ROOT)
    preflight = cuda_preflight(replay_argv[0])
    record = {
        "status": "HOLD",
        "started_at": started_at,
        "argv": replay_argv,
        "stored_argv": stored_argv,
        "cwd": str(ROOT / "isaac-training"),
        "result_path": str(REPLAY),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "resolved_seed": resolved_seed,
        "seed_verified": resolved_seed == 0,
        "legacy_overrides_unchanged": legacy_overrides_unchanged,
        "stdout_path": str(STDOUT),
        "stderr_path": str(STDERR),
        "cuda_preflight": preflight,
        "exit_code": None,
    }
    if not record["seed_verified"]:
        record["failure"] = f"eval.yaml seed must resolve to 0, got {resolved_seed!r}"
    elif not legacy_overrides_unchanged:
        record["failure"] = "stored legacy eval overrides changed while constructing replay argv"
    elif preflight["ready"]:
        completed = subprocess.run(
            replay_argv,
            cwd=record["cwd"],
            shell=False,
            capture_output=True,
            text=True,
            check=False,
        )
        STDOUT.write_text(completed.stdout, encoding="utf-8")
        STDERR.write_text(completed.stderr, encoding="utf-8")
        record["exit_code"] = completed.returncode
        if completed.returncode != 0:
            record["failure"] = f"eval subprocess failed with exit {completed.returncode}"
    else:
        STDOUT.write_text(preflight["nvidia_smi"]["stdout"] + preflight["torch"]["stdout"], encoding="utf-8")
        STDERR.write_text(preflight["nvidia_smi"]["stderr"] + preflight["torch"]["stderr"], encoding="utf-8")
        record["failure"] = preflight["failure"]
    report = compare_files(ROOT, BASELINE, REPLAY, checkpoint, wrapper_failure=record.get("failure"))
    COMPARISON.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    record["comparator_outcome"] = report["status"]
    record["comparison_path"] = str(COMPARISON)
    record["finished_at"] = _timestamp()
    RECORD.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(record["status"] if not preflight["ready"] else report["status"])
    return 0 if report["status"] == "GO (design only)" else 1


if __name__ == "__main__":
    raise SystemExit(main())
