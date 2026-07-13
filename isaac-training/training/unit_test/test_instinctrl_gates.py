import os
import sys
from pathlib import Path


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPTS = os.path.join(ROOT, "training", "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

from instinctRL.gates import evaluate_gates  # noqa: E402
from instinctRL.sweep import (  # noqa: E402
    build_jobs,
    default_safety_preservation_variants,
)

SWEEP_PATH = os.path.join(SCRIPTS, "instinctRL", "sweep.py")


def _passing_summary():
    return {
        "eval/station/handbook.station_keeping_drift_mean": 0.5,
        "eval/station/handbook.station_keeping_drift_p95": 1.5,
        "eval/station/handbook.null_command_speed_mean": 0.02,
        "eval/station/handbook.null_command_output_speed_mean": 0.02,
        "eval/station/handbook.anchor_error_mean": 0.5,
        "eval/tracking/handbook.tracking_rmse_actual_body_vs_v_cmd": 0.35,
        "eval/tracking/handbook.command_preservation_ratio": 0.9,
        "eval/tracking/handbook.command_amplification_mean": 0.02,
        "eval/tracking/handbook.command_amplification_rate": 0.04,
        "eval/handbook.safety_collision_rate": 0.0,
        "eval/handbook.safety_min_clearance_p05": 1.2,
        "eval/handbook.ics_violation_rate": 0.0,
        "eval/handbook.termination_collision": 0.0,
        "eval/handbook.termination_below_bound": 0.0,
        "eval/handbook.termination_above_bound": 0.0,
    }


def test_hard_gate_passes_only_when_all_handbook_bounds_pass():
    report = evaluate_gates(_passing_summary())

    assert report.passed is True
    assert report.safety_passed is True
    assert report.passed_count == report.total_count


def test_hard_gate_fails_safety_preservation_and_station_regressions():
    summary = _passing_summary()
    summary.update({
        "eval/station/handbook.station_keeping_drift_mean": 2.0,
        "eval/station/handbook.null_command_output_speed_mean": 0.25,
        "eval/station/handbook.anchor_error_mean": 2.5,
        "eval/tracking/handbook.command_preservation_ratio": 1.2,
        "eval/tracking/handbook.command_amplification_rate": 0.25,
        "eval/handbook.safety_min_clearance_p05": 0.6,
        "eval/handbook.ics_violation_rate": 0.12,
    })

    report = evaluate_gates(summary)
    failed = {result.name for result in report.results if not result.passed}

    assert report.passed is False
    assert report.safety_passed is False
    assert "station_drift_mean" in failed
    assert "station_anchor_error_mean" in failed
    assert "tracking_preservation_ratio" in failed
    assert "tracking_command_amplification_rate" in failed
    assert "safety_min_clearance_p05" in failed
    assert "ics_violation_rate" in failed


def test_hard_gate_treats_null_output_speed_as_diagnostic_only():
    summary = _passing_summary()
    summary["eval/station/handbook.null_command_output_speed_mean"] = 10.0

    report = evaluate_gates(summary)
    failed = {result.name for result in report.results if not result.passed}

    assert report.passed is True
    assert "station_null_output_speed_mean" not in failed


def test_hard_gate_missing_required_metric_fails():
    summary = _passing_summary()
    del summary["eval/handbook.safety_min_clearance_p05"]

    report = evaluate_gates(summary)

    assert report.passed is False
    assert any(
        result.name == "safety_min_clearance_p05" and "missing" in result.reason
        for result in report.results
    )


def test_sweep_dry_run_jobs_include_a2r5_config_only_overrides():
    variants = default_safety_preservation_variants()
    jobs = build_jobs(
        python_exe="python",
        variants=variants,
        seeds=[0],
        frames=131072,
        artifacts_dir=Path("/tmp/instinctrl_sweep"),
        tag="unit",
    )

    expected_names = [
        "r5_null_speed4",
        "r5_amp3",
        "r5_height16",
        "r5_safety_margin",
        "r5_null_amp_height",
        "r5_null_amp_height_safety",
    ]

    assert [variant.name for variant in variants] == expected_names
    assert len(jobs) == len(expected_names)
    command_text = " ".join(jobs[0].train_command)
    assert "algo.instinctRL.governor.null_vcorr_gate_enabled=true" in command_text
    assert "algo.instinctRL.governor.null_vcorr_gate_eps=0.25" in command_text
    assert "algo.instinctRL.governor.null_vcorr_gate_min=0.25" in command_text
    assert "algo.instinctRL.governor.v_corr_limit=0.35" in command_text
    assert "instinctRL.reward.anchor_weight=4.0" in command_text
    assert "instinctRL.reward.null_command_output_weight=0.1" in command_text
    assert "instinctRL.reward.preservation_high_weight=2.0" in command_text
    assert "instinctRL.reward.command_amplification_weight=2.0" in command_text
    assert "instinctRL.reward.proxy_tracking_weight=0.5" in command_text
    assert "instinctRL.reward.safety_weight=1.2" in command_text
    assert "instinctRL.reward.clearance_margin=0.4" in command_text
    assert "instinctRL.ics.active_horizon_margin=1.0" in command_text
    assert "instinctRL.ics.clearance_margin=0.15" in command_text
    assert "instinctRL.eval.suite=short_diagnostic" in " ".join(jobs[0].eval_command)

    null_speed = next(job for job in jobs if job.variant == "r5_null_speed4")
    assert (
        "instinctRL.reward.null_command_speed_weight=4.0"
        in " ".join(null_speed.train_command)
    )

    amp = next(job for job in jobs if job.variant == "r5_amp3")
    assert "instinctRL.reward.command_amplification_weight=3.0" in " ".join(
        amp.train_command
    )

    height = next(job for job in jobs if job.variant == "r5_height16")
    assert "instinctRL.reward.height_floor_weight=16.0" in " ".join(
        height.train_command
    )

    safety = next(job for job in jobs if job.variant == "r5_safety_margin")
    safety_text = " ".join(safety.train_command)
    assert "instinctRL.reward.safety_weight=1.5" in safety_text
    assert "instinctRL.reward.clearance_margin=0.5" in safety_text
    assert "instinctRL.ics.active_horizon_margin=1.2" in safety_text
    assert "instinctRL.ics.clearance_margin=0.2" in safety_text

    combined = next(job for job in jobs if job.variant == "r5_null_amp_height")
    combined_text = " ".join(combined.train_command)
    assert "instinctRL.reward.null_command_speed_weight=4.0" in combined_text
    assert "instinctRL.reward.command_amplification_weight=3.0" in combined_text
    assert "instinctRL.reward.height_floor_weight=16.0" in combined_text

    combined_safety = next(
        job for job in jobs if job.variant == "r5_null_amp_height_safety"
    )
    combined_safety_text = " ".join(combined_safety.train_command)
    assert "instinctRL.reward.null_command_speed_weight=4.0" in combined_safety_text
    assert "instinctRL.reward.command_amplification_weight=3.0" in combined_safety_text
    assert "instinctRL.reward.height_floor_weight=16.0" in combined_safety_text
    assert "instinctRL.reward.safety_weight=1.5" in combined_safety_text
    assert "instinctRL.reward.clearance_margin=0.5" in combined_safety_text
    assert "instinctRL.ics.active_horizon_margin=1.2" in combined_safety_text
    assert "instinctRL.ics.clearance_margin=0.2" in combined_safety_text

    for job in jobs:
        assert "instinctRL.reward.anchor_weight=5.0" not in job.train_command

    sweep_source = open(SWEEP_PATH, encoding="utf-8").read()
    assert 'parser.add_argument("--tag", default="a2r5_sweep")' in sweep_source
