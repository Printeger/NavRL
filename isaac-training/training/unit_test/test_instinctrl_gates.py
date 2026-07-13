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


def _last_override(command, key):
    prefix = f"{key}="
    values = [arg.split("=", 1)[1] for arg in command if arg.startswith(prefix)]
    assert values, f"missing override for {key}"
    return values[-1]


def test_sweep_dry_run_jobs_include_a2r5d_vertical_mechanism_overrides():
    variants = default_safety_preservation_variants()
    jobs = build_jobs(
        python_exe="python",
        variants=variants,
        seeds=[0],
        frames=131072,
        artifacts_dir=Path("/tmp/instinctrl_sweep"),
        tag="a2r5d_sweep",
    )

    expected_names = [
        "r5d_zlimit020",
        "r5d_zlimit012",
        "r5d_trackzgain050",
        "r5d_trackzgain000",
        "r5d_zlimit020_trackzgain050",
        "r5d_zlimit012_trackzgain000",
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
    assert "instinctRL.reward.command_amplification_weight=2.5" in command_text
    assert "instinctRL.reward.proxy_tracking_weight=0.5" in command_text
    assert "instinctRL.reward.safety_weight=1.2" in command_text
    assert "instinctRL.reward.clearance_margin=0.4" in command_text
    assert "instinctRL.ics.active_horizon_margin=1.0" in command_text
    assert "instinctRL.ics.clearance_margin=0.15" in command_text
    assert "instinctRL.reward.null_command_speed_weight=4.0" in command_text
    assert "instinctRL.reward.height_floor=0.5" in command_text
    assert "instinctRL.reward.height_floor_weight=8.0" in command_text
    assert "instinctRL.reward.height_ceiling=4.0" in command_text
    assert "instinctRL.reward.height_ceiling_weight=8.0" in command_text
    assert "instinctRL.eval.suite=short_diagnostic" in " ".join(jobs[0].eval_command)

    base_overrides = {
        "algo.instinctRL.governor.v_corr_limit": "0.35",
        "instinctRL.reward.preservation_high_weight": "2.0",
        "instinctRL.reward.command_amplification_weight": "2.5",
        "instinctRL.reward.proxy_tracking_weight": "0.5",
        "instinctRL.reward.safety_weight": "1.2",
        "instinctRL.reward.clearance_margin": "0.4",
        "instinctRL.ics.active_horizon_margin": "1.0",
        "instinctRL.ics.clearance_margin": "0.15",
        "instinctRL.reward.null_command_speed_weight": "4.0",
        "instinctRL.reward.height_floor": "0.5",
        "instinctRL.reward.height_floor_weight": "8.0",
        "instinctRL.reward.height_ceiling": "4.0",
        "instinctRL.reward.height_ceiling_weight": "8.0",
    }
    expected_effective_overrides = {
        "r5d_zlimit020": {
            "algo.instinctRL.governor.v_corr_z_limit": "0.20",
        },
        "r5d_zlimit012": {
            "algo.instinctRL.governor.v_corr_z_limit": "0.12",
        },
        "r5d_trackzgain050": {
            "algo.instinctRL.governor.tracking_vcorr_z_gate_enabled": "true",
            "algo.instinctRL.governor.tracking_vcorr_z_gate_eps": "0.001",
            "algo.instinctRL.governor.tracking_vcorr_z_gain": "0.50",
        },
        "r5d_trackzgain000": {
            "algo.instinctRL.governor.tracking_vcorr_z_gate_enabled": "true",
            "algo.instinctRL.governor.tracking_vcorr_z_gate_eps": "0.001",
            "algo.instinctRL.governor.tracking_vcorr_z_gain": "0.0",
        },
        "r5d_zlimit020_trackzgain050": {
            "algo.instinctRL.governor.v_corr_z_limit": "0.20",
            "algo.instinctRL.governor.tracking_vcorr_z_gate_enabled": "true",
            "algo.instinctRL.governor.tracking_vcorr_z_gate_eps": "0.001",
            "algo.instinctRL.governor.tracking_vcorr_z_gain": "0.50",
        },
        "r5d_zlimit012_trackzgain000": {
            "algo.instinctRL.governor.v_corr_z_limit": "0.12",
            "algo.instinctRL.governor.tracking_vcorr_z_gate_enabled": "true",
            "algo.instinctRL.governor.tracking_vcorr_z_gate_eps": "0.001",
            "algo.instinctRL.governor.tracking_vcorr_z_gain": "0.0",
        },
    }
    jobs_by_variant = {job.variant: job for job in jobs}
    for variant_name, expected_overrides in expected_effective_overrides.items():
        job = jobs_by_variant[variant_name]
        assert job.eval_overrides == variants[expected_names.index(variant_name)].overrides
        for command in (job.train_command, job.eval_command):
            for key, expected_value in base_overrides.items():
                assert _last_override(command, key) == expected_value
            for key, expected_value in expected_overrides.items():
                assert _last_override(command, key) == expected_value

    for job in jobs:
        command = job.train_command + job.eval_command
        assert "instinctRL.reward.anchor_weight=5.0" not in command
        assert "instinctRL.reward.safety_weight=1.5" not in command
        assert "instinctRL.reward.clearance_margin=0.5" not in command
        assert "instinctRL.ics.active_horizon_margin=1.2" not in command
        assert "instinctRL.ics.clearance_margin=0.2" not in command
        assert "instinctRL.reward.command_amplification_weight=3.0" not in command
        assert "instinctRL.reward.height_floor_weight=12.0" not in command
        assert "algo.instinctRL.governor.v_corr_limit=0.30" not in command
        assert not any("height_clamp" in arg for arg in command)
        assert not any("height_safety_clamp" in arg for arg in command)
        assert any(
            arg.startswith("wandb.name=instinctrl_a2r5d_sweep_")
            for arg in job.train_command
        )

    sweep_source = open(SWEEP_PATH, encoding="utf-8").read()
    assert 'parser.add_argument("--tag", default="a2r5d_sweep")' in sweep_source
