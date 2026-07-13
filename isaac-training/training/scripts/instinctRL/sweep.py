"""Small-scale corrective sweep runner for instinctRL.

Default behavior is dry-run. Pass ``--execute`` to launch train/eval jobs.
The generated candidates are short 128k/256k-style diagnostics intended to
select top configurations before any 1M/2M run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import datetime as _dt
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Iterable, List, Optional, Sequence


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from instinctRL.gates import evaluate_gates, load_eval_summary  # noqa: E402


CHECKPOINT_RE = re.compile(r"Final model saved to (?P<path>.+checkpoint_final\.pt)")


@dataclass(frozen=True)
class SweepVariant:
    name: str
    overrides: tuple[str, ...]


@dataclass
class SweepJob:
    variant: str
    seed: int
    train_command: List[str]
    eval_command: List[str]
    result_path: str
    checkpoint_path: Optional[str] = None
    gate_report: Optional[dict] = None
    error: Optional[str] = None

    def to_dict(self):
        data = asdict(self)
        data["train_command_text"] = shlex.join(self.train_command)
        data["eval_command_text"] = shlex.join(self.eval_command)
        return data


def default_safety_preservation_variants() -> tuple[SweepVariant, ...]:
    r5_base = (
        "algo.instinctRL.governor.v_corr_limit=0.35",
        "instinctRL.reward.preservation_high_weight=2.0",
        "instinctRL.reward.command_amplification_weight=2.0",
        "instinctRL.reward.proxy_tracking_weight=0.5",
        "instinctRL.reward.safety_weight=1.2",
        "instinctRL.reward.clearance_margin=0.4",
        "instinctRL.ics.active_horizon_margin=1.0",
        "instinctRL.ics.clearance_margin=0.15",
    )
    null_speed = (
        "instinctRL.reward.null_command_speed_weight=4.0",
    )
    amp3 = (
        "instinctRL.reward.command_amplification_weight=3.0",
    )
    height16 = (
        "instinctRL.reward.height_floor_weight=16.0",
    )
    safety_margin = (
        "instinctRL.reward.safety_weight=1.5",
        "instinctRL.reward.clearance_margin=0.5",
        "instinctRL.ics.active_horizon_margin=1.2",
        "instinctRL.ics.clearance_margin=0.2",
    )
    null_amp_height = null_speed + amp3 + height16
    return (
        SweepVariant(
            "r5_null_speed4",
            r5_base + null_speed,
        ),
        SweepVariant(
            "r5_amp3",
            r5_base + amp3,
        ),
        SweepVariant(
            "r5_height16",
            r5_base + height16,
        ),
        SweepVariant(
            "r5_safety_margin",
            r5_base + safety_margin,
        ),
        SweepVariant(
            "r5_null_amp_height",
            r5_base + null_amp_height,
        ),
        SweepVariant(
            "r5_null_amp_height_safety",
            r5_base + null_amp_height + safety_margin,
        ),
    )


def build_train_command(
    *,
    python_exe: str,
    variant: SweepVariant,
    seed: int,
    frames: int,
    run_name: str,
) -> List[str]:
    base = [
        python_exe,
        "training/scripts/train.py",
        "instinctRL.mode=train",
        "instinctRL.task=command_governor",
        "instinctRL.command.source=curriculum_generator",
        "instinctRL.command.curriculum_profile=station_first",
        "instinctRL.reward.enabled=true",
        "instinctRL.reward.use_privileged_velocity_for_reward=true",
        "instinctRL.reward.anchor_weight=4.0",
        "instinctRL.reward.null_command_output_weight=0.1",
        "instinctRL.ics.enabled=true",
        "algo.instinctRL.governor.null_vcorr_gate_enabled=true",
        "algo.instinctRL.governor.null_vcorr_gate_eps=0.25",
        "algo.instinctRL.governor.null_vcorr_gate_min=0.25",
        "env.num_envs=32",
        "env.num_obstacles=350",
        "env_dyn.num_obstacles=0",
        "algo.training_frame_num=16",
        "algo.num_minibatches=8",
        "algo.training_epoch_num=4",
        f"max_frame_num={int(frames)}",
        "eval_interval=250",
        "save_interval=250",
        "wandb.mode=offline",
        f"wandb.name={run_name}",
        f"seed={int(seed)}",
        "headless=true",
    ]
    return base + list(variant.overrides)


def build_eval_command(
    *,
    python_exe: str,
    checkpoint_path: str,
    result_path: str,
) -> List[str]:
    return [
        python_exe,
        "training/scripts/eval.py",
        f"checkpoint_path={checkpoint_path}",
        f"result_path={result_path}",
        "env.num_envs=32",
        "env.max_episode_length=1000",
        "env.num_obstacles=350",
        "env_dyn.num_obstacles=0",
        "instinctRL.eval.suite=short_diagnostic",
        "instinctRL.observability.enabled=true",
        "instinctRL.observability.mode=proxy",
        "wandb.mode=offline",
        "headless=true",
    ]


def build_jobs(
    *,
    python_exe: str,
    variants: Sequence[SweepVariant],
    seeds: Iterable[int],
    frames: int,
    artifacts_dir: Path,
    tag: str,
) -> list[SweepJob]:
    jobs = []
    for variant in variants:
        for seed in seeds:
            run_name = f"instinctrl_{tag}_{variant.name}_{frames}_seed{seed}"
            result_path = artifacts_dir / f"{tag}_{variant.name}_{frames}_seed{seed}_eval.json"
            train_command = build_train_command(
                python_exe=python_exe,
                variant=variant,
                seed=seed,
                frames=frames,
                run_name=run_name,
            )
            eval_command = build_eval_command(
                python_exe=python_exe,
                checkpoint_path="<filled-after-training>",
                result_path=str(result_path),
            )
            jobs.append(
                SweepJob(
                    variant=variant.name,
                    seed=int(seed),
                    train_command=train_command,
                    eval_command=eval_command,
                    result_path=str(result_path),
                )
            )
    return jobs


def execute_jobs(jobs: Sequence[SweepJob], *, cwd: Path) -> list[SweepJob]:
    for job in jobs:
        result_path = Path(job.result_path)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        train = subprocess.run(
            job.train_command,
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        (result_path.with_suffix(".train.log")).write_text(train.stdout, encoding="utf-8")
        if train.returncode != 0:
            job.error = f"train failed with exit code {train.returncode}"
            continue
        checkpoint = _parse_checkpoint(train.stdout)
        if checkpoint is None:
            job.error = "could not parse checkpoint_final.pt from training log"
            continue
        job.checkpoint_path = checkpoint
        job.eval_command = build_eval_command(
            python_exe=job.eval_command[0],
            checkpoint_path=checkpoint,
            result_path=job.result_path,
        )
        eval_run = subprocess.run(
            job.eval_command,
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        result_path.with_suffix(".eval.log").write_text(eval_run.stdout, encoding="utf-8")
        if eval_run.returncode != 0:
            job.error = f"eval failed with exit code {eval_run.returncode}"
            continue
        report = evaluate_gates(load_eval_summary(job.result_path))
        job.gate_report = report.to_dict()
    return list(jobs)


def rank_jobs(jobs: Sequence[SweepJob]) -> list[SweepJob]:
    def key(job: SweepJob):
        report = job.gate_report or {}
        return (
            bool(report.get("passed", False)),
            bool(report.get("safety_passed", False)),
            float(report.get("score", -1e9)),
        )

    return sorted(jobs, key=key, reverse=True)


def _parse_checkpoint(output: str) -> Optional[str]:
    matches = CHECKPOINT_RE.findall(output)
    if not matches:
        return None
    return matches[-1].strip()


def _main() -> int:
    parser = argparse.ArgumentParser(description="Run instinctRL corrective sweep.")
    parser.add_argument("--execute", action="store_true", help="Launch train/eval jobs")
    parser.add_argument("--frames", type=int, default=131072)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--limit", type=int, default=0, help="Limit number of generated jobs")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--tag", default="a2r5_sweep")
    parser.add_argument(
        "--artifacts-dir",
        default="../docs/instinctRL_devlog/tests/artifacts/sweeps",
    )
    args = parser.parse_args()

    isaac_training_root = Path(__file__).resolve().parents[3]
    artifacts_dir = Path(args.artifacts_dir)
    if not artifacts_dir.is_absolute():
        artifacts_dir = isaac_training_root / artifacts_dir
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    artifacts_dir = artifacts_dir / timestamp

    jobs = build_jobs(
        python_exe=args.python,
        variants=default_safety_preservation_variants(),
        seeds=args.seeds,
        frames=args.frames,
        artifacts_dir=artifacts_dir,
        tag=args.tag,
    )
    if args.limit > 0:
        jobs = jobs[: args.limit]

    if args.execute:
        jobs = execute_jobs(jobs, cwd=isaac_training_root)
        jobs = rank_jobs(jobs)

    summary = {
        "execute": bool(args.execute),
        "frames": int(args.frames),
        "jobs": [job.to_dict() for job in jobs],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.execute:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        (artifacts_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
