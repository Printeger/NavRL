import os
import sys
import importlib.util
from types import SimpleNamespace

import pytest
import torch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPTS = os.path.join(ROOT, "training", "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

try:
    if not (importlib.util.find_spec("tensordict") and importlib.util.find_spec("torchrl")):
        raise ImportError("tensordict/torchrl not installed")
    from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec
    from ppo import PPO  # noqa: E402
    from utils import BetaActor, ValueNorm  # noqa: E402
    from instinctRL.ppo_stability import (  # noqa: E402
        safe_normalize_advantage,
        save_diagnostic_snapshot,
        tensor_stats,
    )
except Exception as exc:
    CompositeSpec = None
    UnboundedContinuousTensorSpec = None
    PPO = None
    PPO_IMPORT_ERROR = exc
else:
    PPO_IMPORT_ERROR = None


def _skip_if_unavailable():
    if PPO is None:
        pytest.skip(
            "PPO stability test dependencies unavailable: "
            f"{type(PPO_IMPORT_ERROR).__name__}: {PPO_IMPORT_ERROR}"
        )


def _cfg(tmp_path=None, target_kl=0.0):
    return SimpleNamespace(
        feature_extractor=SimpleNamespace(learning_rate=1e-4),
        actor=SimpleNamespace(
            learning_rate=1e-4,
            action_limit=2.0,
            clip_ratio=0.2,
            beta_alpha_min=1.0,
            beta_alpha_max=30.0,
            beta_beta_min=1.0,
            beta_beta_max=30.0,
            action_eps=1e-6,
        ),
        critic=SimpleNamespace(learning_rate=1e-4, clip_ratio=0.2),
        training_epoch_num=1,
        num_minibatches=1,
        entropy_loss_coefficient=0.0,
        max_grad_norm=0.5,
        target_kl=target_kl,
        finite_audit=True,
        diagnostic_dir=str(tmp_path or "ppo_diagnostics_test"),
        value_norm=SimpleNamespace(max_abs_bootstrap=1000.0, max_abs_return=1000.0),
        instinctRL=SimpleNamespace(
            governor=SimpleNamespace(
                alpha_mode="learned",
                v_corr_limit=0.5,
                velocity_limit=2.0,
                smoothing_tau=0.0,
            )
        ),
    )


class _ActionSpec:
    shape = (1, 3)


def _obs_spec(batch_size=4):
    return CompositeSpec({
        "agents": CompositeSpec({
            "observation": CompositeSpec({
                "lidar_grid": UnboundedContinuousTensorSpec((12, 16, 8)),
                "state_vec": UnboundedContinuousTensorSpec((52,)),
            }),
        }).expand(batch_size),
        "info": CompositeSpec({
            "drone_state": UnboundedContinuousTensorSpec((1, 13)),
            "v_cmd": UnboundedContinuousTensorSpec((1, 3)),
            "actual_velocity_b": UnboundedContinuousTensorSpec((1, 3)),
            "min_clearance": UnboundedContinuousTensorSpec((1, 1)),
        }).expand(batch_size),
    }, shape=[batch_size])


def _policy(tmp_path=None, batch_size=4, target_kl=0.0):
    _skip_if_unavailable()
    return PPO(_cfg(tmp_path, target_kl=target_kl), _obs_spec(batch_size), _ActionSpec(), "cpu")


def _minibatch(policy, batch_size=4):
    td = _obs_spec(batch_size).zero().clone()
    with torch.no_grad():
        policy(td)
    td["sample_log_prob"] = td["sample_log_prob"].detach()
    td["adv"] = torch.linspace(-1.0, 1.0, batch_size).unsqueeze(-1)
    td["ret"] = torch.zeros(batch_size, 1)
    td["state_value"] = torch.zeros(batch_size, 1)
    return td


def _find_beta_actor(policy):
    for module in policy.actor.modules():
        if isinstance(module, BetaActor):
            return module
    raise AssertionError("BetaActor not found")


def test_bounded_beta_params_stay_finite_for_extreme_raw_outputs():
    _skip_if_unavailable()
    actor = BetaActor(4, alpha_min=1.0, alpha_max=30.0, beta_min=1.0, beta_max=30.0)
    features = torch.ones(2, 3)
    actor(features)
    with torch.no_grad():
        actor.alpha_layer.weight.fill_(1000.0)
        actor.alpha_layer.bias.fill_(1000.0)
        actor.beta_layer.weight.fill_(-1000.0)
        actor.beta_layer.bias.fill_(-1000.0)

    alpha, beta = actor(features)

    assert torch.isfinite(alpha).all()
    assert torch.isfinite(beta).all()
    assert torch.all((alpha >= 1.0) & (alpha <= 30.0))
    assert torch.all((beta >= 1.0) & (beta <= 30.0))


def test_policy_action_normalized_finite_for_random_finite_observations(tmp_path):
    policy = _policy(tmp_path)
    td = _obs_spec().zero().clone()
    td["agents", "observation", "lidar_grid"] = torch.rand_like(
        td["agents", "observation", "lidar_grid"]
    )
    td["agents", "observation", "state_vec"] = torch.randn_like(
        td["agents", "observation", "state_vec"]
    )

    out = policy(td)

    assert torch.isfinite(out["agents", "action_normalized"]).all()
    assert torch.all(out["agents", "action_normalized"] >= policy.action_eps)
    assert torch.all(out["agents", "action_normalized"] <= 1.0 - policy.action_eps)


def test_nan_raw_actor_output_is_caught_before_governor_decoder(tmp_path):
    policy = _policy(tmp_path)
    beta_actor = _find_beta_actor(policy)
    with torch.no_grad():
        beta_actor.alpha_layer.bias[0] = float("nan")

    with pytest.raises(ValueError, match="raw_alpha.*diagnostic_snapshot"):
        policy(_obs_spec().zero().clone())
    assert list(tmp_path.glob("ppo_stability_*actor_forward_failure.pt"))


def test_nonfinite_gradients_are_caught_before_optimizer_step(tmp_path):
    policy = _policy(tmp_path)
    td = _minibatch(policy)
    parameter = next(policy.actor.parameters())
    parameter.grad = torch.zeros_like(parameter)
    parameter.grad.flatten()[0] = float("nan")

    with pytest.raises(ValueError, match="non-finite gradients.*diagnostic_snapshot"):
        policy._audit_module_gradients("actor", policy.actor, policy._context("unit"), td)
    assert list(tmp_path.glob("ppo_stability_*nonfinite_gradients_actor.pt"))


def test_nonfinite_parameters_are_caught_after_optimizer_step(tmp_path):
    policy = _policy(tmp_path)
    td = _minibatch(policy)
    parameter = next(policy.actor.parameters())
    with torch.no_grad():
        parameter.flatten()[0] = float("nan")

    with pytest.raises(ValueError, match="non-finite parameters.*diagnostic_snapshot"):
        policy._audit_module_parameters("actor", policy.actor, policy._context("unit"), td)
    assert list(tmp_path.glob("ppo_stability_*nonfinite_parameters_actor.pt"))


def test_advantage_normalization_with_zero_std_is_finite():
    adv = torch.ones(4, 1)
    normalized = safe_normalize_advantage(adv)
    assert torch.isfinite(normalized).all()
    assert torch.allclose(normalized, torch.zeros_like(normalized))


def test_tensor_stats_samples_large_tensors_without_full_finite_copy():
    tensor = torch.ones(2_000_001)
    tensor[-1] = float("inf")

    stats = tensor_stats(tensor)

    assert stats["present"]
    assert stats["numel"] == 2_000_001
    assert stats["count_exact"] is False
    assert stats["finite_count"] > 0


def test_diagnostic_snapshot_handles_large_tensors(tmp_path):
    tensor = torch.ones(2_000_001)
    tensor[-1] = float("nan")

    path = save_diagnostic_snapshot(
        str(tmp_path),
        "large_tensor_unit",
        {"phase": "unit"},
        tensors={"large": tensor},
    )
    payload = torch.load(path)

    assert payload["reason"] == "large_tensor_unit"
    assert payload["tensors"]["large"]["numel"] == 2_000_001
    assert payload["tensors"]["large"]["count_exact"] is False


def test_value_norm_update_keeps_second_moment_finite_for_large_returns():
    value_norm = ValueNorm(1)
    returns = torch.full((8, 1), 1.0e20)

    value_norm.update(returns)

    assert value_norm.running_mean.dtype == torch.float64
    assert value_norm.running_mean_sq.dtype == torch.float64
    assert torch.isfinite(value_norm.running_mean).all()
    assert torch.isfinite(value_norm.running_mean_sq).all()


def test_value_norm_update_rejects_nonfinite_returns_without_polluting_state():
    value_norm = ValueNorm(1)
    returns = torch.tensor([[float("inf")]])

    with pytest.raises(ValueError, match="non-finite returns"):
        value_norm.update(returns)

    assert torch.isfinite(value_norm.running_mean).all()
    assert torch.isfinite(value_norm.running_mean_sq).all()
    assert torch.isfinite(value_norm.debiasing_term).all()


def test_value_norm_denormalize_clips_before_float32_overflow():
    value_norm = ValueNorm(1)
    value_norm.running_mean.fill_(1.0e40)
    value_norm.running_mean_sq.fill_(1.0e80)
    value_norm.debiasing_term.fill_(1.0)
    normalized_value = torch.ones(4, 1, dtype=torch.float32)

    denormalized = value_norm.denormalize(normalized_value, max_abs=1000.0)

    assert denormalized.dtype == torch.float32
    assert torch.isfinite(denormalized).all()
    assert torch.all(denormalized.abs() <= 1000.0)


def test_grad_clipping_is_applied_to_all_module_groups(tmp_path, monkeypatch):
    policy = _policy(tmp_path)
    td = _minibatch(policy)
    original_clip = torch.nn.utils.clip_grad.clip_grad_norm_
    calls = []

    def recording_clip(parameters, max_norm, *args, **kwargs):
        params = list(parameters)
        calls.append((len(params), float(max_norm)))
        return original_clip(params, max_norm, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils.clip_grad, "clip_grad_norm_", recording_clip)

    stats = policy._update(td)

    assert torch.isfinite(stats["actor_loss"])
    assert len(calls) == 4
    assert all(max_norm == pytest.approx(0.5) for _, max_norm in calls)


def test_target_kl_early_stop_triggers(tmp_path):
    policy = _policy(tmp_path, target_kl=0.001)
    td = _minibatch(policy)
    td["sample_log_prob"] = torch.full_like(td["sample_log_prob"], 10.0)

    stats = policy._update(td)

    assert stats["kl_early_stop"].item() == 1.0
    assert stats["approx_kl"].item() > 0.001


def test_no_training_path_silently_replaces_nan_action_with_zero():
    root = os.path.join(ROOT, "training", "scripts")
    for relpath in ["ppo.py", "utils.py", os.path.join("instinctRL", "governor.py")]:
        with open(os.path.join(root, relpath), "r", encoding="utf-8") as handle:
            source = handle.read()
        assert "nan_to_num" not in source
