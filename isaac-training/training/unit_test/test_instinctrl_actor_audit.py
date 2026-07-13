import importlib.util
import os
import sys
from types import SimpleNamespace

import pytest
import torch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPTS = os.path.join(ROOT, "training", "scripts")
MODULE_PATH = os.path.join(SCRIPTS, "instinctRL", "audit.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("instinctrl_audit_test", MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _td(obs):
    return {"agents": {"observation": obs}}


def test_actor_schema_accepts_only_lidar_grid_and_state_vec():
    audit = _load_module()
    td = _td({
        "lidar_grid": torch.zeros(2, 12, 360, 59),
        "state_vec": torch.zeros(2, 52),
    })
    assert audit.check_actor_input(td)[0]
    assert audit.check_actor_schema(td, history_len=4)[0]


def test_actor_audit_rejects_forbidden_key_and_extra_schema_key():
    audit = _load_module()
    bad_key = _td({
        "lidar_grid": torch.zeros(2, 12, 360, 59),
        "state_vec": torch.zeros(2, 52),
        "position": torch.zeros(2, 3),
    })
    assert not audit.check_actor_input(bad_key)[0]
    assert not audit.check_actor_schema(bad_key, history_len=4)[0]


def test_rollout_audit_requires_learned_governor_action_dim_and_finite_reward():
    audit = _load_module()
    cfg = SimpleNamespace(
        instinctRL=SimpleNamespace(observation=SimpleNamespace(history_len=4)),
        algo=SimpleNamespace(
            instinctRL=SimpleNamespace(governor=SimpleNamespace(alpha_mode="learned"))
        ),
    )
    td = {
        "agents": {
            "observation": {
                "lidar_grid": torch.zeros(2, 12, 360, 59),
                "state_vec": torch.zeros(2, 52),
            },
            "action_normalized": torch.full((2, 4), 0.5),
        },
        "sample_log_prob": torch.zeros(2),
        "state_value": torch.zeros(2, 1),
        "next": {
            "agents": {
                "observation": {
                    "lidar_grid": torch.zeros(2, 12, 360, 59),
                    "state_vec": torch.zeros(2, 52),
                },
                "reward": torch.zeros(2, 1),
            }
        },
    }

    assert audit.audit_rollout_batch(td, cfg)["rollout_batch"]
    td["agents"]["action_normalized"] = torch.full((2, 3), 0.5)
    with pytest.raises(RuntimeError, match="expected 4"):
        audit.audit_rollout_batch(td, cfg)


def test_checkpoint_audit_requires_loadable_file(tmp_path):
    audit = _load_module()
    path = tmp_path / "checkpoint.pt"
    torch.save({"ok": torch.ones(1)}, path)
    assert audit.audit_checkpoint_file(str(path))["checkpoint_file"]

    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"")
    with pytest.raises(RuntimeError, match="empty file"):
        audit.audit_checkpoint_file(str(bad))


def test_learned_governor_source_does_not_read_privileged_v_cmd():
    ppo_source = os.path.join(SCRIPTS, "ppo.py")
    gov_source = os.path.join(SCRIPTS, "instinctRL", "governor.py")
    ppo_text = open(ppo_source, encoding="utf-8").read()
    gov_text = open(gov_source, encoding="utf-8").read()
    decode_block = ppo_text.split("def decode_action", 1)[1].split(
        "def verify_actor_critic_separation", 1
    )[0]
    actor_block = ppo_text.split("self.actor_feature_extractor = TensorDictSequential", 1)[1].split(
        "self.critic_feature_extractor = TensorDictSequential", 1
    )[0]
    text = gov_text + decode_block + actor_block

    assert '("info", "v_cmd")' not in text
    assert '["info", "v_cmd"]' not in text
    assert "info\", \"v_cmd" not in text


def test_r5b_height_diagnostics_do_not_enter_actor_observation_source():
    env_source = open(os.path.join(SCRIPTS, "env.py"), encoding="utf-8").read()
    ppo_source = open(os.path.join(SCRIPTS, "ppo.py"), encoding="utf-8").read()
    env_actor_block = env_source.split("# -----------------Network Input Final--------------", 1)[1]
    env_actor_block = env_actor_block.split("# ============================================", 1)[0]
    ppo_actor_block = ppo_source.split("self.actor_feature_extractor = TensorDictSequential", 1)[1]
    ppo_actor_block = ppo_actor_block.split("self.critic_feature_extractor = TensorDictSequential", 1)[0]

    for forbidden in [
        "height_world_z",
        "height_floor_violation",
        "height_ceiling_violation",
        "height_ceiling_margin",
        "v_final_b_z",
        "governor_v_corr_z",
        "governor_v_final_b_z",
        "drone_state",
        "root_state",
    ]:
        assert forbidden not in env_actor_block
        assert forbidden not in ppo_actor_block
