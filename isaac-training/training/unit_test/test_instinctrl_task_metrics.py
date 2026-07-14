import importlib.util
import os
import sys

import torch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPTS = os.path.join(ROOT, "training", "scripts")
TASK_PATH = os.path.join(SCRIPTS, "instinctRL", "task_metrics.py")


def _load_task_metrics():
    spec = importlib.util.spec_from_file_location("instinctrl_task_metrics_test", TASK_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_world_to_body_velocity_inverse_quaternion_rotation():
    metrics = _load_task_metrics()
    yaw_90_body_to_world = torch.tensor([[0.70710678, 0.0, 0.0, 0.70710678]])
    world_velocity = torch.tensor([[0.0, 1.0, 0.0]])

    body_velocity = metrics.world_to_body_velocity(world_velocity, yaw_90_body_to_world)

    assert torch.allclose(body_velocity, torch.tensor([[1.0, 0.0, 0.0]]), atol=1e-5)


def test_termination_reason_codes_are_exclusive_and_diagnostic():
    metrics = _load_task_metrics()
    below = torch.tensor([[False], [True], [False], [False], [True]])
    above = torch.tensor([[False], [False], [True], [False], [False]])
    collision = torch.tensor([[False], [False], [False], [True], [True]])
    truncated = torch.tensor([[False], [False], [False], [False], [True]])

    stats = metrics.compute_termination_stats(
        below_bound=below,
        above_bound=above,
        collision=collision,
        truncated=truncated,
    )

    assert stats["termination_reason_code"].reshape(-1).tolist() == [
        metrics.TERMINATION_NONE,
        metrics.TERMINATION_BELOW_BOUND,
        metrics.TERMINATION_ABOVE_BOUND,
        metrics.TERMINATION_COLLISION,
        metrics.TERMINATION_COLLISION,
    ]
    assert stats["terminated_below_bound"].sum().item() == 2.0
    assert stats["terminated_above_bound"].sum().item() == 1.0
    assert stats["terminated_collision"].sum().item() == 2.0
    assert stats["truncated_timeout"].sum().item() == 1.0


def test_command_curriculum_starts_conservative_then_adds_adversarial_modes():
    metrics = _load_task_metrics()

    early = metrics.command_curriculum_probabilities(0)
    middle = metrics.command_curriculum_probabilities(600_000)
    late = metrics.command_curriculum_probabilities(2_500_000)

    for probs in (early, middle, late):
        assert len(probs) == 5
        assert abs(sum(probs) - 1.0) < 1e-6

    assert early[metrics.COMMAND_MODE_RECOVERY] > late[metrics.COMMAND_MODE_RECOVERY]
    assert early[metrics.COMMAND_MODE_ADVERSARIAL] == 0.0
    assert middle[metrics.COMMAND_MODE_ADVERSARIAL] > 0.0
    assert late[metrics.COMMAND_MODE_ADVERSARIAL] > middle[metrics.COMMAND_MODE_ADVERSARIAL]


def test_station_first_curriculum_prioritizes_recovery_before_mixed_tracking():
    metrics = _load_task_metrics()

    early = metrics.command_curriculum_probabilities(0, profile="station_first")
    middle = metrics.command_curriculum_probabilities(1_500_000, profile="station_first")
    late = metrics.command_curriculum_probabilities(3_500_000, profile="station_first")
    diagnostic = metrics.command_curriculum_probabilities(0, profile="diagnostic_mixed")

    assert early[metrics.COMMAND_MODE_RECOVERY] == 0.70
    assert early[metrics.COMMAND_MODE_ADVERSARIAL] == 0.0
    assert middle[metrics.COMMAND_MODE_RECOVERY] == 0.25
    assert abs(late[metrics.COMMAND_MODE_ADVERSARIAL] - 0.05) < 1e-6
    assert abs(diagnostic[metrics.COMMAND_MODE_RECOVERY] - 0.10) < 1e-6
    for probs in (early, middle, late, diagnostic):
        assert abs(sum(probs) - 1.0) < 1e-6


def test_handbook_metric_summary_uses_actual_tracking_and_command_proxy():
    metrics = _load_task_metrics()
    v_cmd = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    actual = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    v_final = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    min_clearance = torch.tensor([[0.5], [2.0]])
    beta = torch.tensor([[0.5], [1.0]])

    summary = metrics.compute_handbook_step_metrics(
        v_cmd_b=v_cmd,
        actual_velocity_b=actual,
        v_final_b=v_final,
        min_clearance=min_clearance,
        ics_beta=beta,
        ics_emergency=torch.zeros(2, 1),
        anchor_active=torch.tensor([[1.0], [0.0]]),
        anchor_error_mean=torch.tensor([[0.2], [0.0]]),
        anchor_error_max=torch.tensor([[0.4], [0.0]]),
        anchor_loss=torch.tensor([[0.1], [0.0]]),
        collision=torch.zeros(2, 1),
    )

    assert summary["tracking_actual_error_sq"].shape == (2, 1)
    assert summary["tracking_proxy_error_sq"].shape == (2, 1)
    assert summary["command_preservation_ratio"].shape == (2, 1)
    assert summary["tracking_actual_error_sq"][0].item() > 0.0
    assert summary["tracking_proxy_error_sq"][0].item() == 0.0
    assert summary["ics_intervention"].reshape(-1).tolist() == [1.0, 0.0]


def test_handbook_step_metrics_report_null_command_and_amplification():
    metrics = _load_task_metrics()
    v_cmd = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    actual = torch.tensor([
        [0.3, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    v_final = torch.tensor([
        [0.4, 0.0, 0.0],
        [1.2, 0.0, 0.0],
        [0.3, 0.0, 0.0],
    ])

    summary = metrics.compute_handbook_step_metrics(
        v_cmd_b=v_cmd,
        actual_velocity_b=actual,
        v_final_b=v_final,
        min_clearance=torch.ones(3, 1) * 2.0,
        ics_beta=torch.tensor([[1.0], [1.0], [0.3]]),
        ics_emergency=torch.zeros(3, 1),
        collision=torch.zeros(3, 1),
        command_eps=0.05,
    )

    assert abs(summary["null_command_speed"][0].item() - 0.3) < 1e-6
    assert abs(summary["null_command_output_speed"][0].item() - 0.4) < 1e-6
    assert summary["null_command_speed"][1].item() == 0.0
    assert summary["command_amplification"][0].item() == 0.0
    assert abs(summary["command_amplification"][1].item() - 0.2) < 1e-6
    assert summary["command_amplification"][2].item() == 0.0
    assert summary["command_amplification_active"][1].item() == 1.0
    assert summary["command_amplification_active"][2].item() == 0.0


def test_handbook_step_metrics_report_height_and_split_command_amplification():
    metrics = _load_task_metrics()
    v_cmd = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
    ])
    actual = v_cmd.clone()
    v_final = torch.tensor([
        [1.5, 0.0, 0.0],
        [0.0, 0.0, 1.5],
        [0.0, 0.0, 1.5],
        [0.0, 0.0, 2.0],
    ])

    summary = metrics.compute_handbook_step_metrics(
        v_cmd_b=v_cmd,
        actual_velocity_b=actual,
        v_final_b=v_final,
        min_clearance=torch.ones(4, 1) * 2.0,
        height_w=torch.tensor([[0.25], [4.25], [3.0], [4.0]]),
        ics_beta=torch.tensor([[1.0], [1.0], [0.3], [1.0]]),
        ics_emergency=torch.zeros(4, 1),
        collision=torch.zeros(4, 1),
        height_floor=0.5,
        height_ceiling=4.0,
        command_eps=0.05,
    )

    assert abs(summary["command_amplification_horizontal"][0].item() - 0.5) < 1e-6
    assert summary["command_amplification_horizontal_active"].reshape(-1).tolist() == [
        1.0,
        0.0,
        0.0,
        0.0,
    ]
    assert abs(summary["command_amplification_vertical"][1].item() - 0.5) < 1e-6
    assert summary["command_amplification_vertical"][2].item() == 0.0
    assert summary["command_amplification_vertical_active"].reshape(-1).tolist() == [
        0.0,
        1.0,
        0.0,
        0.0,
    ]
    assert summary["height_world_z"].reshape(-1).tolist() == [0.25, 4.25, 3.0, 4.0]
    assert abs(summary["height_floor_violation"][0].item() - 0.25) < 1e-6
    assert abs(summary["height_ceiling_violation"][1].item() - 0.25) < 1e-6
    assert abs(summary["height_ceiling_margin"][1].item() + 0.25) < 1e-6
    assert summary["v_cmd_z"].reshape(-1).tolist() == [0.0, 1.0, 1.0, 0.0]
    assert summary["v_final_b_z"].reshape(-1).tolist() == [0.0, 1.5, 1.5, 2.0]


def test_vertical_channel_metrics_report_sign_masks_saturation_and_conditionals():
    metrics = _load_task_metrics()

    summary = metrics.compute_vertical_channel_step_metrics(
        v_cmd_z=torch.tensor([[0.0], [1.0], [-1.0], [1.0]]),
        v_corr_z=torch.tensor([[0.2], [0.5], [0.5], [-0.1]]),
        v_gov_z=torch.tensor([[0.2], [1.5], [-0.5], [0.9]]),
        v_final_z=torch.tensor([[0.1], [1.3], [-0.4], [0.7]]),
        station_drift=torch.tensor([[2.0], [3.0], [4.0], [5.0]]),
        command_preservation_ratio=torch.tensor([[1.0], [0.8], [0.9], [1.1]]),
        command_amplification_vertical=torch.tensor([[0.0], [0.1], [0.2], [0.3]]),
        ics_beta=torch.tensor([[1.0], [0.5], [1.0], [0.2]]),
        ics_emergency=torch.tensor([[0.0], [1.0], [0.0], [0.0]]),
        v_corr_limit=0.5,
        command_eps=0.05,
        saturation_tol=1e-4,
    )

    assert summary["vertical_command_active"].reshape(-1).tolist() == [0.0, 1.0, 1.0, 1.0]
    assert summary["vertical_command_null"].reshape(-1).tolist() == [1.0, 0.0, 0.0, 0.0]
    assert summary["vertical_corr_z_positive"].reshape(-1).tolist() == [1.0, 1.0, 1.0, 0.0]
    assert summary["vertical_corr_z_negative"].reshape(-1).tolist() == [0.0, 0.0, 0.0, 1.0]
    assert summary["vertical_corr_z_saturated"].reshape(-1).tolist() == [0.0, 1.0, 1.0, 0.0]
    assert torch.allclose(
        summary["vertical_gov_minus_cmd_z"].reshape(-1),
        torch.tensor([0.2, 0.5, 0.5, -0.1]),
    )
    assert torch.allclose(
        summary["vertical_final_minus_cmd_z"].reshape(-1),
        torch.tensor([0.1, 0.3, 0.6, -0.3]),
    )
    assert torch.allclose(
        summary["vertical_ics_delta_z"].reshape(-1),
        torch.tensor([-0.1, -0.2, 0.1, -0.2]),
    )
    assert summary["vertical_corr_reinforces_command"].reshape(-1).tolist() == [
        0.0,
        1.0,
        0.0,
        0.0,
    ]
    assert summary["vertical_corr_opposes_command"].reshape(-1).tolist() == [
        0.0,
        0.0,
        1.0,
        1.0,
    ]
    assert summary["vertical_null_corr_active"].reshape(-1).tolist() == [1.0, 0.0, 0.0, 0.0]
    assert torch.allclose(
        summary["vertical_null_corr_abs"].reshape(-1),
        torch.tensor([0.2, 0.0, 0.0, 0.0]),
    )
    assert torch.allclose(
        summary["vertical_null_station_drift_when_corr_active"].reshape(-1),
        torch.tensor([2.0, 0.0, 0.0, 0.0]),
    )
    assert summary["vertical_tracking_corr_active"].reshape(-1).tolist() == [
        0.0,
        1.0,
        1.0,
        1.0,
    ]
    assert torch.allclose(
        summary["vertical_tracking_amplification_when_corr_active"].reshape(-1),
        torch.tensor([0.0, 0.1, 0.2, 0.3]),
    )
    assert torch.allclose(
        summary["vertical_tracking_preservation_when_corr_active"].reshape(-1),
        torch.tensor([0.0, 0.8, 0.9, 1.1]),
    )


def test_r5e_metrics_split_null_actual_and_output_speed_axes():
    metrics = _load_task_metrics()

    summary = metrics.compute_r5e_mechanism_step_metrics(
        v_cmd_b=torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]),
        actual_velocity_b=torch.tensor([
            [3.0, 4.0, 2.0],
            [10.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
        ]),
        v_gov_b=torch.zeros(3, 3),
        v_final_b=torch.tensor([
            [0.0, 2.0, -0.5],
            [9.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]),
        height_world_z=torch.ones(3, 1),
        min_clearance=torch.ones(3, 1),
        command_eps=0.05,
    )

    null_count = summary["r5e_null_command"].sum().item()
    assert null_count == 2.0
    assert abs(summary["r5e_null_actual_speed_xy"].sum().item() / null_count - 2.5) < 1e-6
    assert abs(summary["r5e_null_actual_speed_z_abs"].sum().item() / null_count - 1.5) < 1e-6
    assert abs(summary["r5e_null_output_speed_xy"].sum().item() / null_count - 1.0) < 1e-6
    assert abs(summary["r5e_null_output_speed_z_abs"].sum().item() / null_count - 0.25) < 1e-6


def test_r5e_metrics_split_pre_post_ics_and_axis_preservation():
    metrics = _load_task_metrics()

    summary = metrics.compute_r5e_mechanism_step_metrics(
        v_cmd_b=torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ]),
        actual_velocity_b=torch.zeros(3, 3),
        v_gov_b=torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 2.0],
            [5.0, 0.0, 0.0],
        ]),
        v_final_b=torch.tensor([
            [0.5, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [5.0, 0.0, 0.0],
        ]),
        height_world_z=torch.ones(3, 1),
        min_clearance=torch.ones(3, 1),
        command_eps=0.05,
    )

    command_count = summary["r5e_command_active"].sum().item()
    assert command_count == 2.0
    assert abs(summary["r5e_command_preservation_pre_ics"].sum().item() / command_count - 1.5) < 1e-6
    assert abs(summary["r5e_command_preservation_post_ics"].sum().item() / command_count - 0.75) < 1e-6
    assert abs(summary["r5e_command_preservation_ics_loss"].sum().item() / command_count - 0.75) < 1e-6
    assert summary["r5e_command_horizontal_active"].sum().item() == 1.0
    assert abs(summary["r5e_command_preservation_horizontal"].sum().item() - 0.5) < 1e-6
    assert summary["r5e_command_vertical_active"].sum().item() == 1.0
    assert abs(summary["r5e_command_preservation_vertical_abs"].sum().item() - 1.0) < 1e-6


def test_r5e_metrics_mask_near_floor_and_report_near_floor_ics_violation():
    metrics = _load_task_metrics()

    summary = metrics.compute_r5e_mechanism_step_metrics(
        v_cmd_b=torch.tensor([
            [0.0, 0.0, -0.2],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 0.4],
        ]),
        actual_velocity_b=torch.zeros(3, 3),
        v_gov_b=torch.tensor([
            [0.0, 0.0, -0.1],
            [0.0, 0.0, -2.0],
            [0.0, 0.0, 0.3],
        ]),
        v_final_b=torch.tensor([
            [0.0, 0.0, -0.05],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 0.2],
        ]),
        height_world_z=torch.tensor([[0.60], [0.61], [0.40]]),
        min_clearance=torch.tensor([[0.70], [0.10], [0.90]]),
        ics_beta=torch.tensor([[0.5], [0.1], [1.0]]),
        ics_emergency=torch.tensor([[0.0], [0.0], [0.0]]),
        height_floor=0.5,
        d_safe=0.8,
        command_eps=0.05,
    )

    near_floor = summary["r5e_near_floor"]
    near_count = near_floor.sum().item()
    assert near_floor.reshape(-1).tolist() == [1.0, 0.0, 1.0]
    assert near_count == 2.0
    assert abs(summary["r5e_near_floor_v_cmd_z"].sum().item() / near_count - 0.1) < 1e-6
    assert abs(summary["r5e_near_floor_v_gov_z"].sum().item() / near_count - 0.1) < 1e-6
    assert abs(summary["r5e_near_floor_v_final_z"].sum().item() / near_count - 0.075) < 1e-6
    assert abs(summary["r5e_near_floor_ics_beta"].sum().item() / near_count - 0.75) < 1e-6
    assert torch.isfinite(summary["r5e_near_floor_clearance"].reshape(-1)).tolist() == [
        True,
        False,
        True,
    ]
    assert abs(summary["r5e_ics_violation_near_floor"].sum().item() / near_count - 0.5) < 1e-6


def test_r5g_station_anchor_metrics_split_command_motion_and_anchor_conditions():
    metrics = _load_task_metrics()

    summary = metrics.compute_r5g_station_anchor_step_metrics(
        v_cmd_b=torch.tensor([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]),
        actual_velocity_b=torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [9.0, 0.0, 0.0],
        ]),
        v_final_b=torch.tensor([
            [0.25, 0.0, 0.0],
            [0.0, 0.0, 0.2],
            [9.0, 0.0, 0.0],
        ]),
        station_drift=torch.tensor([[2.0], [3.0], [4.0]]),
        anchor_active=torch.tensor([[1.0], [1.0], [0.0]]),
        anchor_valid_fraction=torch.tensor([[0.2], [0.05], [0.0]]),
        anchor_error_mean=torch.tensor([[1.0], [2.0], [0.0]]),
        anchor_loss=torch.tensor([[0.1], [0.0], [0.0]]),
        observability_valid_fraction=torch.tensor([[0.02], [0.0], [0.0]]),
        command_eps=0.05,
        min_anchor_valid_fraction=0.1,
        anchor_loss_high_threshold=0.05,
        observability_min_valid_fraction=0.01,
    )

    null_count = summary["r5g_station_null_command"].sum().item()
    assert null_count == 2.0
    assert abs(summary["r5g_station_null_mismatch_xy"].sum().item() / null_count - 0.375) < 1e-6
    assert abs(summary["r5g_station_null_mismatch_z_abs"].sum().item() / null_count - 0.4) < 1e-6
    assert summary["r5g_station_null_alignment_xy_active"].sum().item() == 1.0
    assert abs(summary["r5g_station_null_alignment_xy"].sum().item() - 1.0) < 1e-6
    assert summary["r5g_station_null_output_xy_active"].sum().item() == 1.0
    assert abs(summary["r5g_station_null_actual_output_xy_ratio"].sum().item() - 4.0) < 1e-6
    assert summary["r5g_anchor_active"].sum().item() == 2.0
    assert summary["r5g_anchor_valid"].sum().item() == 1.0
    assert summary["r5g_anchor_invalid"].sum().item() == 1.0
    assert summary["r5g_anchor_high_loss"].sum().item() == 1.0
    assert summary["r5g_anchor_obs_valid"].sum().item() == 1.0
    assert summary["r5g_anchor_obs_poor"].sum().item() == 1.0
    assert abs(summary["r5g_anchor_station_drift_when_valid"].sum().item() - 2.0) < 1e-6
    assert abs(summary["r5g_anchor_error_when_invalid"].sum().item() - 2.0) < 1e-6


def test_r5g_downward_metrics_mask_active_effectiveness():
    metrics = _load_task_metrics()

    summary = metrics.compute_r5g_downward_step_metrics(
        downward_active=torch.tensor([[1.0], [0.0], [1.0]]),
        downward_has_ray=torch.tensor([[1.0], [1.0], [0.0]]),
        downward_beta=torch.tensor([[0.5], [1.0], [0.2]]),
        downward_min_clearance=torch.tensor([[0.3], [0.4], [0.5]]),
        downward_pre_z=torch.tensor([[-1.0], [0.0], [-2.0]]),
        downward_post_z=torch.tensor([[-0.5], [0.0], [-0.4]]),
        downward_z_delta_abs=torch.tensor([[0.5], [0.0], [1.6]]),
        downward_attenuation_ratio=torch.tensor([[0.5], [0.0], [0.8]]),
    )

    active = summary["r5g_downward_active"].sum().item()
    assert active == 2.0
    assert summary["r5g_downward_has_ray"].sum().item() == 2.0
    assert abs(summary["r5g_downward_beta_when_active"].sum().item() / active - 0.35) < 1e-6
    assert torch.isfinite(
        summary["r5g_downward_min_clearance_when_active"].reshape(-1)
    ).tolist() == [True, False, True]
    assert abs(summary["r5g_downward_z_delta_abs_when_active"].sum().item() / active - 1.05) < 1e-6
    assert abs(
        summary["r5g_downward_attenuation_ratio_when_active"].sum().item() / active - 0.65
    ) < 1e-6
