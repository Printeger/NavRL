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
