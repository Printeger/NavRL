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
            "target_rpos": UnboundedContinuousTensorSpec((1, 3)),
            "target_distance": UnboundedContinuousTensorSpec((1, 1)),
        }).expand(batch_size),
    }, shape=[batch_size])


def test_ppo_hybrid_forward_and_actor_critic_separation():
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
    td["agents", "action_normalized"] = torch.full((batch_size, 3), 0.5)
    td["sample_log_prob"] = torch.zeros(batch_size)
    td["adv"] = torch.linspace(-1.0, 1.0, batch_size).unsqueeze(-1)
    td["ret"] = torch.zeros(batch_size, 1)
    td["state_value"] = torch.zeros(batch_size, 1)
    td.exclude("_critic_feature", inplace=True)

    stats = policy._update(td)

    assert "_critic_feature" in td.keys(True)
    assert torch.isfinite(stats["critic_loss"])
