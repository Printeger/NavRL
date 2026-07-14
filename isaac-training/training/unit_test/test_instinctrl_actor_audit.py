import importlib.util
import os
import re
import sys
from types import SimpleNamespace

import pytest
import torch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPTS = os.path.join(ROOT, "training", "scripts")
MODULE_PATH = os.path.join(SCRIPTS, "instinctRL", "audit.py")
SAFETY_FILTER_PATH = os.path.join(SCRIPTS, "instinctRL", "safety_filter.py")


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


def test_actor_audit_rejects_velocity_map_slam_and_privileged_state_keys():
    audit = _load_module()

    for forbidden_key in [
        "vertical_velocity",
        "height_root_state",
        "local_map",
        "slam_pose",
        "privileged_sim_state",
        "r5e_near_floor_rate",
        "r5g_station_null_mismatch_xy_mean",
        "r5g_anchor_error_mean_when_valid",
        "r5g_any_future_diagnostic",
        "r5f_axis_split_gate",
        "privileged_root_height",
        "world_z",
        "root_height",
        "governor_v_gov_b",
        "min_clearance",
        "ics_beta",
        "ics_downward_active",
        "ics_downward_attenuation_ratio",
        "ics_downward_has_ray_rate",
        "safety_filter_height_scale",
        "safety_filter_height_active",
        "safety_filter_privileged_height",
        "safety_filter_diagnostic_height",
    ]:
        td = _td({
            "lidar_grid": torch.zeros(2, 12, 360, 59),
            "state_vec": torch.zeros(2, 52),
            forbidden_key: torch.zeros(2, 1),
        })
        assert not audit.check_actor_input(td)[0]
        assert not audit.check_actor_schema(td, history_len=4)[0]


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


def test_ppo_actor_feature_extractor_source_uses_exact_actor_observation_keys():
    ppo_source = os.path.join(SCRIPTS, "ppo.py")
    ppo_text = open(ppo_source, encoding="utf-8").read()
    actor_block = ppo_text.split("self.actor_feature_extractor = TensorDictSequential", 1)[1]
    actor_block = actor_block.split("self.critic_feature_extractor = TensorDictSequential", 1)[0]

    observation_keys = re.findall(
        r'\("agents", "observation", "([^"]+)"\)',
        actor_block,
    )
    assert sorted(observation_keys) == ["lidar_grid", "state_vec"]

    for forbidden_key in [
        "height_world_z",
        "world_z",
        "height_root_state",
        "root_state",
        "root_height",
        "privileged_root_height",
        "pose",
        "odometry",
        "actual_velocity_b",
        "velocity_body",
        "governor_v_gov_b",
        "r5e_near_floor_rate",
        "r5g_station_null_mismatch_xy_mean",
        "r5g_anchor_error_mean_when_valid",
        "r5f_axis_split_gate",
        "r5e_command_preservation_pre_ics_ratio",
        "min_clearance",
        "safety_min_clearance",
        "safety_filter_height_scale",
        "ics_beta",
        "ics_emergency",
        "ics_downward_active",
        "ics_downward_attenuation_ratio",
        "ics_downward_has_ray_rate",
        "local_map",
        "slam_pose",
        "privileged_sim_state",
        "drone_state",
    ]:
        assert f'("agents", "observation", "{forbidden_key}")' not in actor_block


def test_r5b_height_diagnostics_do_not_enter_actor_observation_source():
    env_source = open(os.path.join(SCRIPTS, "env.py"), encoding="utf-8").read()
    ppo_source = open(os.path.join(SCRIPTS, "ppo.py"), encoding="utf-8").read()
    env_actor_block = env_source.split("# -----------------Network Input Final--------------", 1)[1]
    env_actor_block = env_actor_block.split("# ============================================", 1)[0]
    ppo_actor_block = ppo_source.split("self.actor_feature_extractor = TensorDictSequential", 1)[1]
    ppo_actor_block = ppo_actor_block.split("self.critic_feature_extractor = TensorDictSequential", 1)[0]

    for forbidden in [
        "height_world_z",
        "world_z",
        "height_floor_violation",
        "height_ceiling_violation",
        "height_ceiling_margin",
        "privileged_root_height",
        "root_height",
        "vertical_",
        "v_final_b_z",
        "governor_v_corr_z",
        "governor_v_final_b_z",
        "governor_v_cmd_b",
        "governor_v_gov_b",
        "r5e_",
        "r5g_",
        "r5f_",
        "actual_velocity_b",
        "min_clearance",
        "safety_min_clearance",
        "safety_filter",
        "ics_beta",
        "ics_emergency",
        "ics_downward_",
        "safety_filter_diagnostic",
        "local_map",
        "slam_pose",
        "privileged_sim_state",
        "drone_state",
        "root_state",
    ]:
        assert forbidden not in env_actor_block
        assert forbidden not in ppo_actor_block


def test_privileged_height_filter_source_is_sim_eval_only_not_actor_method():
    safety_source = open(SAFETY_FILTER_PATH, encoding="utf-8").read()
    train_source = open(os.path.join(SCRIPTS, "train.py"), encoding="utf-8").read()
    eval_source = open(os.path.join(SCRIPTS, "eval.py"), encoding="utf-8").read()
    ppo_source = open(os.path.join(SCRIPTS, "ppo.py"), encoding="utf-8").read()
    normalized_safety_source = " ".join(safety_source.split())

    assert "sim/eval-only" in safety_source
    assert "must never enter actor observation" in normalized_safety_source
    assert "Paper-1 deployable actor-method claim" in safety_source
    assert "(sim/eval-only)" in train_source
    assert "(sim/eval-only)" in eval_source

    actor_block = ppo_source.split("self.actor_feature_extractor = TensorDictSequential", 1)[1]
    actor_block = actor_block.split("self.critic_feature_extractor = TensorDictSequential", 1)[0]
    assert "PrivilegedHeightFloorSafetyFilter" not in actor_block
    assert "root_height_w" not in actor_block
    assert "safety_filter" not in actor_block
