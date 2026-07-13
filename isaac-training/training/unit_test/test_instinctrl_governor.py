import os
import sys

import pytest
import torch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPTS = os.path.join(ROOT, "training", "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

from instinctRL.governor import (  # noqa: E402
    MinimalGovernor,
    TrainableGovernorDecoder,
    extract_latest_prev_action_b,
    extract_latest_v_cmd_b,
)


def _state_vec(v_cmd, prev_action=None, history_len=4):
    n = v_cmd.shape[0]
    if prev_action is None:
        prev_action = torch.zeros_like(v_cmd)
    state = torch.zeros(n, history_len * 13)
    state[:, -7:-4] = v_cmd
    state[:, -4:-1] = prev_action
    return state


def test_trainable_governor_bounds_formula_and_clip():
    decoder = TrainableGovernorDecoder(v_corr_limit=0.5, velocity_limit=1.0)
    v_cmd = torch.tensor([[2.0, 0.0, 0.0], [0.0, 0.5, 0.0]])
    action = torch.tensor([
        [0.5, 1.0, 0.5, 0.5],
        [0.25, 0.5, 1.0, 0.5],
    ])

    out = decoder(action, _state_vec(v_cmd))

    assert torch.all((out.alpha >= 0.0) & (out.alpha <= 1.0))
    assert torch.all(out.v_corr <= 0.5 + 1e-6)
    assert torch.all(out.v_corr >= -0.5 - 1e-6)
    raw = out.alpha * v_cmd + out.v_corr
    expected = raw / torch.linalg.norm(raw, dim=-1, keepdim=True).clamp_min(1.0)
    assert torch.allclose(out.v_gov, expected, atol=1e-6)
    assert torch.all(torch.linalg.norm(out.v_gov, dim=-1) <= 1.0 + 1e-6)


def test_trainable_governor_default_z_limit_inherits_v_corr_limit():
    decoder = TrainableGovernorDecoder(v_corr_limit=0.4, velocity_limit=10.0)
    v_cmd = torch.tensor([[0.2, -0.1, 0.3], [-0.3, 0.2, -0.1]])
    action = torch.tensor([
        [0.25, 1.0, 0.0, 1.0],
        [1.0, 0.75, 0.25, 0.0],
    ])

    out = decoder(action, _state_vec(v_cmd))

    expected_v_corr = (2.0 * action[:, 1:4] - 1.0) * 0.4
    assert torch.allclose(out.v_corr, expected_v_corr)
    assert torch.allclose(out.v_gov, out.alpha * v_cmd + expected_v_corr)


def test_trainable_governor_v_corr_z_limit_changes_only_z_correction():
    baseline = TrainableGovernorDecoder(v_corr_limit=0.5, velocity_limit=10.0)
    split_z = TrainableGovernorDecoder(
        v_corr_limit=0.5,
        v_corr_z_limit=0.2,
        velocity_limit=10.0,
    )
    v_cmd = torch.zeros(1, 3)
    action = torch.tensor([[0.0, 1.0, 0.0, 1.0]])

    baseline_out = baseline(action, _state_vec(v_cmd))
    split_out = split_z(action, _state_vec(v_cmd))

    assert torch.allclose(split_out.v_corr[..., :2], baseline_out.v_corr[..., :2])
    assert torch.allclose(baseline_out.v_corr[..., 2:3], torch.tensor([[0.5]]))
    assert torch.allclose(split_out.v_corr[..., 2:3], torch.tensor([[0.2]]))


def test_trainable_governor_tracking_z_attenuation_uses_abs_v_cmd_z_only():
    decoder = TrainableGovernorDecoder(
        v_corr_limit=0.5,
        v_corr_z_limit=0.5,
        velocity_limit=10.0,
        tracking_vcorr_z_gate_enabled=True,
        tracking_vcorr_z_gate_eps=0.1,
        tracking_vcorr_z_gain=0.25,
    )
    action = torch.tensor([
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
    ])
    v_cmd = torch.tensor([
        [0.0, 0.0, 0.2],
        [0.0, 0.0, -0.2],
        [0.0, 0.0, 0.1],
        [0.3, 0.0, 0.0],
    ])

    out = decoder(action, _state_vec(v_cmd))

    assert torch.allclose(out.v_corr[..., :2], torch.tensor([
        [0.5, -0.5],
        [0.5, -0.5],
        [0.5, -0.5],
        [0.5, -0.5],
    ]))
    assert torch.allclose(out.v_corr[..., 2:3], torch.tensor([
        [0.125],
        [0.125],
        [0.5],
        [0.5],
    ]))


def test_trainable_governor_tracking_z_gain_zero_preserves_null_station_correction():
    decoder = TrainableGovernorDecoder(
        v_corr_limit=0.5,
        velocity_limit=10.0,
        null_vcorr_gate_enabled=True,
        null_vcorr_gate_eps=0.2,
        null_vcorr_gate_min=0.25,
        tracking_vcorr_z_gate_enabled=True,
        tracking_vcorr_z_gate_eps=0.1,
        tracking_vcorr_z_gain=0.0,
    )
    action = torch.tensor([
        [0.0, 0.5, 0.5, 1.0],
        [0.0, 0.5, 0.5, 1.0],
    ])
    v_cmd = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.2],
    ])

    out = decoder(action, _state_vec(v_cmd))

    assert torch.allclose(out.v_corr[..., :2], torch.zeros(2, 2))
    assert torch.allclose(out.v_corr[..., 2:3], torch.tensor([[0.125], [0.0]]))


def test_trainable_governor_smoothing_uses_actor_clean_prev_action():
    decoder = TrainableGovernorDecoder(
        v_corr_limit=0.0,
        velocity_limit=2.0,
        smoothing_tau=0.25,
    )
    v_cmd = torch.tensor([[1.0, 0.0, 0.0]])
    prev = torch.tensor([[0.0, 1.0, 0.0]])
    action = torch.tensor([[1.0, 0.5, 0.5, 0.5]])

    out = decoder(action, _state_vec(v_cmd, prev))

    assert torch.allclose(out.v_gov, torch.tensor([[0.75, 0.25, 0.0]]))


def test_trainable_governor_null_command_gate_allows_soft_station_correction():
    decoder = TrainableGovernorDecoder(
        v_corr_limit=0.5,
        velocity_limit=2.0,
        null_vcorr_gate_enabled=True,
        null_vcorr_gate_eps=0.2,
        null_vcorr_gate_min=0.25,
    )
    action = torch.tensor([[1.0, 1.0, 0.5, 0.5]])

    null_out = decoder(action, _state_vec(torch.zeros(1, 3)))
    ramp_out = decoder(action, _state_vec(torch.tensor([[0.1, 0.0, 0.0]])))
    active_out = decoder(action, _state_vec(torch.tensor([[0.3, 0.0, 0.0]])))

    assert torch.allclose(null_out.v_corr, torch.tensor([[0.125, 0.0, 0.0]]))
    assert torch.allclose(null_out.v_gov, torch.tensor([[0.125, 0.0, 0.0]]))
    assert torch.allclose(ramp_out.v_corr, torch.tensor([[0.25, 0.0, 0.0]]))
    assert torch.allclose(active_out.v_corr, torch.tensor([[0.5, 0.0, 0.0]]))


def test_trainable_governor_null_command_gate_validation():
    with pytest.raises(ValueError, match="null_vcorr_gate_eps"):
        TrainableGovernorDecoder(null_vcorr_gate_eps=0.0)
    with pytest.raises(ValueError, match="null_vcorr_gate_min"):
        TrainableGovernorDecoder(null_vcorr_gate_min=1.5)


def test_trainable_governor_z_limit_and_tracking_gate_validation():
    with pytest.raises(ValueError, match="v_corr_z_limit"):
        TrainableGovernorDecoder(v_corr_z_limit=-0.1)
    with pytest.raises(ValueError, match="v_corr_z_limit"):
        TrainableGovernorDecoder(v_corr_z_limit=float("nan"))
    with pytest.raises(ValueError, match="tracking_vcorr_z_gate_eps"):
        TrainableGovernorDecoder(tracking_vcorr_z_gate_eps=0.0)
    with pytest.raises(ValueError, match="tracking_vcorr_z_gain"):
        TrainableGovernorDecoder(tracking_vcorr_z_gain=-0.1)
    with pytest.raises(ValueError, match="tracking_vcorr_z_gain"):
        TrainableGovernorDecoder(tracking_vcorr_z_gain=1.1)


def test_trainable_governor_extractors_and_shape_validation():
    v_cmd = torch.tensor([[0.1, 0.2, 0.3]])
    prev = torch.tensor([[0.4, 0.5, 0.6]])
    state = _state_vec(v_cmd, prev)

    assert torch.allclose(extract_latest_v_cmd_b(state), v_cmd)
    assert torch.allclose(extract_latest_prev_action_b(state), prev)

    decoder = TrainableGovernorDecoder()
    with pytest.raises(ValueError, match="dim 4"):
        decoder(torch.zeros(1, 3), state)
    with pytest.raises(ValueError, match="state_vec"):
        decoder(torch.zeros(1, 4), torch.zeros(1, 12))


def test_minimal_governor_preserves_direction_with_norm_clip():
    governor = MinimalGovernor(velocity_limit=1.0)
    out = governor(torch.tensor([[2.0, 0.0, 0.0]]))

    assert torch.allclose(out.alpha, torch.ones(1, 1))
    assert torch.allclose(out.v_corr, torch.zeros(1, 3))
    assert torch.allclose(out.v_gov, torch.tensor([[1.0, 0.0, 0.0]]))
