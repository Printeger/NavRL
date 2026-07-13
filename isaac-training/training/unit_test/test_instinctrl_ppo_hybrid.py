import os
import sys
import importlib.util
from types import SimpleNamespace

import torch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPTS = os.path.join(ROOT, "training", "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

try:
    if not (importlib.util.find_spec("tensordict") and importlib.util.find_spec("torchrl")):
        raise ImportError("tensordict/torchrl not installed")
    from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec
    from torchrl.envs.utils import ExplorationType, set_exploration_type
    from ppo import PPO  # noqa: E402
except Exception as exc:
    CompositeSpec = None
    UnboundedContinuousTensorSpec = None
    PPO = None
    PPO_IMPORT_ERROR = exc
else:
    PPO_IMPORT_ERROR = None


def _cfg():
    return SimpleNamespace(
        feature_extractor=SimpleNamespace(learning_rate=1e-4),
        actor=SimpleNamespace(learning_rate=1e-4, action_limit=2.0, clip_ratio=0.2),
        critic=SimpleNamespace(clip_ratio=0.2),
        training_epoch_num=1,
        num_minibatches=1,
        entropy_loss_coefficient=0.0,
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


def _obs_spec(batch_size=2):
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


def test_ppo_learned_governor_forward_and_actor_critic_separation():
    if PPO is None:
        message = (
            "PPO hybrid test dependencies unavailable: "
            f"{type(PPO_IMPORT_ERROR).__name__}: {PPO_IMPORT_ERROR}"
        )
        try:
            import pytest
        except Exception:
            print(f"SKIP {message}")
            return
        pytest.skip(message)
        return
    policy = PPO(_cfg(), _obs_spec(), _ActionSpec(), "cpu")
    td = _obs_spec().zero()
    out = policy(td.clone())
    assert ("agents", "action") in out.keys(True)
    assert out["agents", "action"].shape == torch.Size([2, 3])
    assert out["agents", "action_normalized"].shape == torch.Size([2, 4])
    assert out["governor_alpha"].shape == torch.Size([2, 1])
    assert out["governor_v_corr"].shape == torch.Size([2, 3])
    assert out["governor_v_corr_z"].shape == torch.Size([2, 1])
    assert out["governor_v_cmd_b_z"].shape == torch.Size([2, 1])
    assert out["governor_v_gov_b"].shape == torch.Size([2, 3])
    assert out["governor_v_gov_b_z"].shape == torch.Size([2, 1])
    assert torch.all((out["governor_alpha"] >= 0.0) & (out["governor_alpha"] <= 1.0))
    assert torch.all(torch.linalg.norm(out["governor_v_gov_b"], dim=-1) <= 2.0 + 1e-6)
    assert policy.verify_actor_critic_separation(td.clone())


def test_ppo_update_recomputes_critic_features_for_minibatch():
    if PPO is None:
        message = (
            "PPO hybrid test dependencies unavailable: "
            f"{type(PPO_IMPORT_ERROR).__name__}: {PPO_IMPORT_ERROR}"
        )
        try:
            import pytest
        except Exception:
            print(f"SKIP {message}")
            return
        pytest.skip(message)
        return

    batch_size = 4
    policy = PPO(_cfg(), _obs_spec(batch_size), _ActionSpec(), "cpu")
    td = _obs_spec(batch_size).zero().clone()
    td["agents", "action_normalized"] = torch.full((batch_size, 4), 0.5)
    td["sample_log_prob"] = torch.zeros(batch_size)
    td["adv"] = torch.linspace(-1.0, 1.0, batch_size).unsqueeze(-1)
    td["ret"] = torch.zeros(batch_size, 1)
    td["state_value"] = torch.zeros(batch_size, 1)
    td.exclude("_critic_feature", inplace=True)

    stats = policy._update(td)

    assert "_critic_feature" in td.keys(True)
    assert torch.isfinite(stats["critic_loss"])


def test_ppo_deterministic_mean_action_and_checkpoint_roundtrip(tmp_path):
    if PPO is None:
        message = (
            "PPO hybrid test dependencies unavailable: "
            f"{type(PPO_IMPORT_ERROR).__name__}: {PPO_IMPORT_ERROR}"
        )
        try:
            import pytest
        except Exception:
            print(f"SKIP {message}")
            return
        pytest.skip(message)
        return

    cfg = _cfg()
    policy = PPO(cfg, _obs_spec(), _ActionSpec(), "cpu")
    td = _obs_spec().zero()
    td["agents", "observation", "state_vec"][:, -7:-4] = torch.tensor([[1.0, 0.0, 0.0]])

    with set_exploration_type(ExplorationType.MEAN), torch.no_grad():
        out_a = policy(td.clone())
        out_b = policy(td.clone())
    assert torch.allclose(out_a["agents", "action_normalized"], out_b["agents", "action_normalized"])
    assert torch.allclose(out_a["governor_v_gov_b"], out_b["governor_v_gov_b"])

    ckpt = tmp_path / "policy.pt"
    torch.save(policy.state_dict(), ckpt)
    reloaded = PPO(cfg, _obs_spec(), _ActionSpec(), "cpu")
    reloaded.load_state_dict(torch.load(ckpt, map_location="cpu"))

    with set_exploration_type(ExplorationType.MEAN), torch.no_grad():
        out_c = reloaded(td.clone())
    assert torch.allclose(out_a["agents", "action_normalized"], out_c["agents", "action_normalized"])
    assert torch.allclose(out_a["governor_v_gov_b"], out_c["governor_v_gov_b"])


def test_ppo_direct_velocity_baseline_still_supported():
    if PPO is None:
        message = (
            "PPO hybrid test dependencies unavailable: "
            f"{type(PPO_IMPORT_ERROR).__name__}: {PPO_IMPORT_ERROR}"
        )
        try:
            import pytest
        except Exception:
            print(f"SKIP {message}")
            return
        pytest.skip(message)
        return

    cfg = _cfg()
    cfg.instinctRL.governor.alpha_mode = "fixed"
    policy = PPO(cfg, _obs_spec(), _ActionSpec(), "cpu")
    out = policy(_obs_spec().zero().clone())

    assert out["agents", "action_normalized"].shape == torch.Size([2, 3])
    assert out["agents", "action"].shape == torch.Size([2, 3])
    assert "governor_v_gov_b" not in out.keys(True)
