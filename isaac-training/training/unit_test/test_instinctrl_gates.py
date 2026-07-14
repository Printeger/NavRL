import os
import sys
from pathlib import Path


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPTS = os.path.join(ROOT, "training", "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

from instinctRL.gates import DEFAULT_GATE_SPECS, evaluate_gates  # noqa: E402
from instinctRL.sweep import (  # noqa: E402
    build_jobs,
    default_r5f_mechanism_variants,
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


def test_hard_gate_spec_snapshot_is_unchanged():
    snapshot = [
        (
            spec.name,
            spec.keys,
            spec.category,
            spec.min_value,
            spec.max_value,
            spec.required,
        )
        for spec in DEFAULT_GATE_SPECS
    ]

    assert snapshot == [
        (
            "station_drift_mean",
            ("eval/station/handbook.station_keeping_drift_mean",),
            "station",
            None,
            1.3,
            True,
        ),
        (
            "station_drift_p95",
            ("eval/station/handbook.station_keeping_drift_p95",),
            "station",
            None,
            2.6,
            True,
        ),
        (
            "station_null_speed_mean",
            ("eval/station/handbook.null_command_speed_mean",),
            "station",
            None,
            0.08,
            True,
        ),
        (
            "station_anchor_error_mean",
            ("eval/station/handbook.anchor_error_mean",),
            "station",
            None,
            2.0,
            True,
        ),
        (
            "tracking_rmse_actual",
            (
                "eval/tracking/handbook.tracking_rmse_actual_body_vs_v_cmd",
                "eval/handbook.tracking_rmse_actual_body_vs_v_cmd",
            ),
            "tracking",
            None,
            0.45,
            True,
        ),
        (
            "tracking_preservation_ratio",
            (
                "eval/tracking/handbook.command_preservation_ratio",
                "eval/handbook.command_preservation_ratio",
            ),
            "tracking",
            0.75,
            1.05,
            True,
        ),
        (
            "tracking_command_amplification_mean",
            (
                "eval/tracking/handbook.command_amplification_mean",
                "eval/handbook.command_amplification_mean",
            ),
            "tracking",
            None,
            0.05,
            True,
        ),
        (
            "tracking_command_amplification_rate",
            (
                "eval/tracking/handbook.command_amplification_rate",
                "eval/handbook.command_amplification_rate",
            ),
            "tracking",
            None,
            0.15,
            True,
        ),
        (
            "safety_collision_rate",
            (
                "eval/handbook.safety_collision_rate",
                "eval/tracking/handbook.safety_collision_rate",
                "eval/station/handbook.safety_collision_rate",
            ),
            "safety",
            None,
            0.0,
            True,
        ),
        (
            "safety_min_clearance_p05",
            (
                "eval/handbook.safety_min_clearance_p05",
                "eval/tracking/handbook.safety_min_clearance_p05",
                "eval/station/handbook.safety_min_clearance_p05",
            ),
            "safety",
            1.0,
            None,
            True,
        ),
        (
            "ics_violation_rate",
            (
                "eval/handbook.ics_violation_rate",
                "eval/tracking/handbook.ics_violation_rate",
                "eval/station/handbook.ics_violation_rate",
            ),
            "safety",
            None,
            0.005,
            True,
        ),
        (
            "termination_collision",
            ("eval/handbook.termination_collision",),
            "safety",
            None,
            0.0,
            True,
        ),
        (
            "termination_below_bound",
            ("eval/handbook.termination_below_bound",),
            "termination",
            None,
            0.0,
            True,
        ),
        (
            "termination_above_bound",
            ("eval/handbook.termination_above_bound",),
            "termination",
            None,
            0.0,
            True,
        ),
    ]


def _last_override(command, key):
    prefix = f"{key}="
    values = [arg.split("=", 1)[1] for arg in command if arg.startswith(prefix)]
    assert values, f"missing override for {key}"
    return values[-1]


def test_sweep_dry_run_jobs_include_a2r5f_mechanism_overrides():
    variants = default_safety_preservation_variants()
    assert variants == default_r5f_mechanism_variants()
    jobs = build_jobs(
        python_exe="python",
        variants=variants,
        seeds=[0],
        frames=131072,
        artifacts_dir=Path("/tmp/instinctrl_sweep"),
        tag="a2r5f_sweep",
    )

    expected_names = [
        "r5f_null_axis_xy050_z000",
        "r5f_null_axis_xy075_z000",
        "r5f_zsign_opp050_reinf100",
        "r5f_zsign_opp100_reinf050",
        "r5f_downatten",
        "r5f_preserve_h050_v100",
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
    assert "algo.instinctRL.governor.v_corr_z_limit=0.12" in command_text
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
        "algo.instinctRL.governor.v_corr_z_limit": "0.12",
    }
    expected_variant_overrides = {
        "r5f_null_axis_xy050_z000": (
            "algo.instinctRL.governor.null_vcorr_axis_split_enabled=true",
            "algo.instinctRL.governor.null_vcorr_xy_gate_min=0.50",
            "algo.instinctRL.governor.null_vcorr_z_gate_min=0.0",
        ),
        "r5f_null_axis_xy075_z000": (
            "algo.instinctRL.governor.null_vcorr_axis_split_enabled=true",
            "algo.instinctRL.governor.null_vcorr_xy_gate_min=0.75",
            "algo.instinctRL.governor.null_vcorr_z_gate_min=0.0",
        ),
        "r5f_zsign_opp050_reinf100": (
            "algo.instinctRL.governor.tracking_vcorr_z_sign_gate_enabled=true",
            "algo.instinctRL.governor.tracking_vcorr_z_gate_eps=0.001",
            "algo.instinctRL.governor.tracking_vcorr_z_opposing_gain=0.50",
            "algo.instinctRL.governor.tracking_vcorr_z_reinforcing_gain=1.0",
        ),
        "r5f_zsign_opp100_reinf050": (
            "algo.instinctRL.governor.tracking_vcorr_z_sign_gate_enabled=true",
            "algo.instinctRL.governor.tracking_vcorr_z_gate_eps=0.001",
            "algo.instinctRL.governor.tracking_vcorr_z_opposing_gain=1.0",
            "algo.instinctRL.governor.tracking_vcorr_z_reinforcing_gain=0.50",
        ),
        "r5f_downatten": (
            "instinctRL.ics.downward_attenuation_enabled=true",
            "instinctRL.ics.downward_ray_min_z=0.25",
            "instinctRL.ics.downward_clearance_margin=0.0",
        ),
        "r5f_preserve_h050_v100": (
            "instinctRL.reward.horizontal_preservation_weight=0.5",
            "instinctRL.reward.vertical_preservation_weight=1.0",
        ),
    }
    jobs_by_variant = {job.variant: job for job in jobs}
    for variant_name, expected_overrides in expected_variant_overrides.items():
        job = jobs_by_variant[variant_name]
        assert job.eval_overrides == variants[expected_names.index(variant_name)].overrides
        assert job.eval_overrides[-len(expected_overrides):] == expected_overrides
        for command in (job.train_command, job.eval_command):
            for key, expected_value in base_overrides.items():
                assert _last_override(command, key) == expected_value
            for override in expected_overrides:
                key, expected_value = override.split("=", 1)
                assert _last_override(command, key) == expected_value
                assert override in command

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
        assert "algo.instinctRL.governor.v_corr_z_limit=0.20" not in command
        assert "algo.instinctRL.governor.tracking_vcorr_z_gain=0.0" not in command
        assert "algo.instinctRL.governor.tracking_vcorr_z_gain=0.50" not in command
        assert "instinctRL.safety_filter.privileged_height_floor_enabled=true" not in command
        assert not any("height_clamp" in arg for arg in command)
        assert not any("height_safety_clamp" in arg for arg in command)
        assert not any(
            arg.startswith("instinctRL.safety_filter.")
            for arg in command
        )
        assert any(
            arg.startswith("wandb.name=instinctrl_a2r5f_sweep_")
            for arg in job.train_command
        )

    sweep_source = open(SWEEP_PATH, encoding="utf-8").read()
    assert 'parser.add_argument("--tag", default="a2r5f_sweep")' in sweep_source
    assert "a2r5d_sweep" not in sweep_source
