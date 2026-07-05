import importlib.util
import math
import os
import sys

import torch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPTS = os.path.join(ROOT, "training", "scripts")
REWARD_PATH = os.path.join(SCRIPTS, "instinctRL", "rewards.py")
ENV_PATH = os.path.join(SCRIPTS, "env.py")
CFG_PATH = os.path.join(ROOT, "training", "cfg", "train.yaml")


def _load_rewards():
    spec = importlib.util.spec_from_file_location("instinctrl_rewards_test", REWARD_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _computer(**kwargs):
    mod = _load_rewards()
    params = {
        "tracking_weight": 1.0,
        "anchor_weight": 0.5,
        "safety_weight": 1.0,
        "ics_compliance_weight": 1.0,
        "intervention_weight": 0.1,
        "smoothness_weight": 0.1,
        "collision_weight": 10.0,
        "clearance_safe": 0.8,
        "clearance_margin": 0.2,
        "max_reward_abs": 20.0,
        "min_anchor_valid_fraction": 0.1,
        "tracking_beta_gate": True,
    }
    params.update(kwargs)
    cfg = mod.RewardConfig(**params)
    return mod, mod.InstinctRLRewardComputer(cfg, device="cpu")


def _base_inputs(**overrides):
    inputs = {
        "v_cmd_b": torch.tensor([[1.0, 0.0, 0.0]]),
        "v_final_b": torch.tensor([[1.0, 0.0, 0.0]]),
        "prev_v_final_b": torch.tensor([[1.0, 0.0, 0.0]]),
        "anchor_loss": torch.zeros(1, 1),
        "anchor_active": torch.zeros(1, 1),
        "anchor_valid_fraction": torch.ones(1, 1),
        "ics_beta": torch.ones(1, 1),
        "ics_emergency": torch.zeros(1, 1),
        "ics_active_beam_count": torch.zeros(1, 1),
        "min_clearance": torch.ones(1, 1) * 2.0,
        "collision": torch.zeros(1, 1, dtype=torch.bool),
    }
    inputs.update(overrides)
    return inputs


def test_config_validation():
    mod = _load_rewards()
    mod.RewardConfig()
    bad_kwargs = [
        {"tracking_weight": float("nan")},
        {"max_reward_abs": 0.0},
        {"clearance_safe": 0.0},
        {"clearance_margin": -0.1},
        {"min_anchor_valid_fraction": -0.1},
        {"min_anchor_valid_fraction": 1.1},
        {"command_eps": 0.0},
    ]
    for kwargs in bad_kwargs:
        try:
            mod.RewardConfig(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {kwargs}")


def test_tracking_reward_prefers_command_match():
    _, computer = _computer()
    match = computer.compute(**_base_inputs())
    mismatch = computer.compute(**_base_inputs(v_final_b=torch.tensor([[0.0, 0.0, 0.0]])))
    assert match.components["reward_tracking"].item() == 0.0
    assert mismatch.components["reward_tracking"].item() < match.components["reward_tracking"].item()
    assert mismatch.total.item() < match.total.item()


def test_beta_and_emergency_gate_tracking_penalty():
    _, computer = _computer(intervention_weight=0.0)
    mismatch = torch.tensor([[0.0, 0.0, 0.0]])
    full = computer.compute(**_base_inputs(v_final_b=mismatch, ics_beta=torch.ones(1, 1)))
    low_beta = computer.compute(**_base_inputs(v_final_b=mismatch, ics_beta=torch.full((1, 1), 0.25)))
    emergency = computer.compute(**_base_inputs(
        v_final_b=mismatch,
        prev_v_final_b=mismatch,
        ics_beta=torch.zeros(1, 1),
        ics_emergency=torch.ones(1, 1),
    ))
    assert low_beta.components["reward_tracking"].item() > full.components["reward_tracking"].item()
    assert low_beta.components["reward_ics_compliance"].item() > 0.0
    assert emergency.components["reward_tracking"].item() == 0.0
    assert emergency.components["reward_ics_compliance"].item() > 0.0


def test_anchor_reward_active_loss_and_valid_mask():
    _, computer = _computer(anchor_weight=2.0, min_anchor_valid_fraction=0.5)
    inactive = computer.compute(**_base_inputs(anchor_loss=torch.ones(1, 1), anchor_active=torch.zeros(1, 1)))
    active = computer.compute(**_base_inputs(anchor_loss=torch.ones(1, 1), anchor_active=torch.ones(1, 1)))
    low_valid = computer.compute(**_base_inputs(
        anchor_loss=torch.ones(1, 1),
        anchor_active=torch.ones(1, 1),
        anchor_valid_fraction=torch.full((1, 1), 0.25),
    ))
    assert inactive.components["reward_anchor"].item() == 0.0
    assert active.components["reward_anchor"].item() == -2.0
    assert low_valid.components["reward_anchor"].item() == 0.0


def test_safety_lower_clearance_is_worse_and_invalid_is_finite():
    _, computer = _computer(safety_weight=1.0)
    far = computer.compute(**_base_inputs(min_clearance=torch.tensor([[2.0]])))
    near = computer.compute(**_base_inputs(min_clearance=torch.tensor([[0.4]])))
    invalid = computer.compute(**_base_inputs(min_clearance=torch.tensor([[float("nan")]])))
    missing = computer.compute(**_base_inputs(min_clearance=None))
    assert near.components["reward_safety"].item() < far.components["reward_safety"].item()
    assert torch.isfinite(torch.cat([v.reshape(-1).float() for v in invalid.components.values()])).all()
    assert torch.isfinite(missing.total).all()


def test_intervention_smoothness_and_collision_terms():
    _, computer = _computer()
    beta_one = computer.compute(**_base_inputs(ics_beta=torch.ones(1, 1)))
    beta_low = computer.compute(**_base_inputs(ics_beta=torch.full((1, 1), 0.2)))
    smooth = computer.compute(**_base_inputs(
        v_final_b=torch.tensor([[1.0, 0.0, 0.0]]),
        prev_v_final_b=torch.tensor([[1.0, 0.0, 0.0]]),
    ))
    jump = computer.compute(**_base_inputs(
        v_final_b=torch.tensor([[1.0, 0.0, 0.0]]),
        prev_v_final_b=torch.tensor([[-1.0, 0.0, 0.0]]),
    ))
    collision = computer.compute(**_base_inputs(collision=torch.ones(1, 1, dtype=torch.bool)))
    assert beta_low.components["reward_intervention"].item() < beta_one.components["reward_intervention"].item()
    assert jump.components["reward_smoothness"].item() < smooth.components["reward_smoothness"].item()
    assert collision.components["reward_collision"].item() == -10.0


def test_total_is_component_sum_and_clipped_with_scaled_components():
    _, computer = _computer(max_reward_abs=1.0, collision_weight=10.0)
    out = computer.compute(**_base_inputs(collision=torch.ones(1, 1, dtype=torch.bool)))
    component_sum = sum(
        value for key, value in out.components.items() if key != "reward_total"
    )
    assert torch.allclose(out.total, component_sum)
    assert torch.allclose(out.total, out.components["reward_total"])
    assert out.total.abs().item() <= 1.0


def test_disabled_modules_degrade_gracefully_and_optional_privileged_velocity():
    _, computer = _computer(use_privileged_velocity_for_reward=False)
    defaulted = computer.compute(
        v_cmd_b=torch.tensor([[1.0, 0.0, 0.0]]),
        v_final_b=torch.tensor([[1.0, 0.0, 0.0]]),
        prev_v_final_b=torch.tensor([[1.0, 0.0, 0.0]]),
        min_clearance=None,
        collision=torch.zeros(1, 1, dtype=torch.bool),
    )
    assert defaulted.components["reward_anchor"].item() == 0.0
    assert defaulted.components["reward_intervention"].item() == 0.0

    _, privileged = _computer(use_privileged_velocity_for_reward=True)
    out = privileged.compute(**_base_inputs(actual_velocity_b=torch.zeros(1, 3)))
    assert out.components["reward_tracking"].item() < 0.0
    assert privileged.cfg.use_privileged_velocity_for_reward is True


def test_public_component_shapes_and_cache_boundary():
    mod, computer = _computer()
    out = computer.compute(**_base_inputs(v_cmd_b=torch.zeros(2, 3), v_final_b=torch.zeros(2, 3), prev_v_final_b=torch.zeros(2, 3), collision=torch.zeros(2, 1, dtype=torch.bool)))
    assert isinstance(out, mod.RewardTerms)
    expected = set(mod.REWARD_COMPONENT_KEYS)
    assert set(out.components) == expected
    for key, value in out.components.items():
        assert value.shape == (2, 1), key
    assert "tracking_error_norm" in out.cache
    assert "tracking_error_norm" not in out.components


def test_source_level_actor_privileged_and_env_integration_contracts():
    env_source = open(ENV_PATH, encoding="utf-8").read()
    actor_block = env_source.split("# -----------------Network Input Final--------------", 1)[1]
    actor_block = actor_block.split("# ============================================", 1)[0]
    assert '"lidar_grid": obs_hist["lidar_grid"]' in actor_block
    assert '"state_vec": obs_hist["state_vec"]' in actor_block
    for token in ['"reward_', '"anchor_', '"ics_', '"map', '"odom', '"root_state']:
        assert token not in actor_block

    assert "InstinctRLRewardComputer" in env_source
    assert "cfg.instinctRL.reward.enabled" in env_source
    assert "for key, value in reward_terms.components.items()" in env_source
    assert "self.stats[key] += value" in env_source
    assert "reward_safety_static" in env_source
    assert "else:" in env_source

    reward_source = open(REWARD_PATH, encoding="utf-8").read()
    assert "use_privileged_velocity_for_reward: bool = False" in reward_source
    assert "actual_velocity_b" in reward_source

    cfg_source = open(CFG_PATH, encoding="utf-8").read()
    assert "reward:" in cfg_source
    assert "enabled: true" in cfg_source
