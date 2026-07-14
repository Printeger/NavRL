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
        "null_command_speed_weight": 0.0,
        "null_command_output_weight": 0.0,
        "null_output_anchor_loss_threshold": 0.05,
        "proxy_tracking_weight": 0.0,
        "preservation_low_weight": 0.0,
        "preservation_high_weight": 0.0,
        "horizontal_preservation_weight": 0.0,
        "vertical_preservation_weight": 0.0,
        "preservation_lower": 0.75,
        "preservation_upper": 1.05,
        "command_amplification_weight": 0.0,
        "height_floor": 0.5,
        "height_floor_weight": 8.0,
        "height_ceiling": 4.0,
        "height_ceiling_weight": 0.0,
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
        {"preservation_low_weight": -0.1},
        {"horizontal_preservation_weight": -0.1},
        {"vertical_preservation_weight": -0.1},
        {"preservation_lower": -0.1},
        {"preservation_lower": 1.1, "preservation_upper": 1.0},
        {"command_eps": 0.0},
        {"height_floor": -0.1},
        {"height_floor_weight": -0.1},
        {"height_ceiling": 0.4},
        {"height_ceiling_weight": -0.1},
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


def test_height_floor_penalty_is_quadratic_below_floor_only():
    _, computer = _computer(height_floor=0.5, height_floor_weight=8.0)
    at_floor = computer.compute(**_base_inputs(height_w=torch.tensor([[0.5]])))
    above_floor = computer.compute(**_base_inputs(height_w=torch.tensor([[1.0]])))
    below_floor = computer.compute(**_base_inputs(height_w=torch.tensor([[0.25]])))

    assert at_floor.components["reward_height_floor"].item() == 0.0
    assert above_floor.components["reward_height_floor"].item() == 0.0
    assert abs(below_floor.components["reward_height_floor"].item() + 0.5) < 1e-6
    assert abs(below_floor.cache["height_floor_violation"].item() - 0.25) < 1e-6
    assert below_floor.total.item() < at_floor.total.item()


def test_height_ceiling_penalty_is_dormant_by_default_and_quadratic_when_enabled():
    _, dormant = _computer(height_ceiling=4.0, height_ceiling_weight=0.0)
    above_ceiling = dormant.compute(**_base_inputs(height_w=torch.tensor([[4.25]])))
    assert above_ceiling.components["reward_height_ceiling"].item() == 0.0
    assert abs(above_ceiling.cache["height_ceiling_violation"].item() - 0.25) < 1e-6
    assert abs(above_ceiling.cache["height_ceiling_margin"].item() + 0.25) < 1e-6

    _, active = _computer(height_ceiling=4.0, height_ceiling_weight=8.0)
    at_ceiling = active.compute(**_base_inputs(height_w=torch.tensor([[4.0]])))
    below_ceiling = active.compute(**_base_inputs(height_w=torch.tensor([[3.5]])))
    above_ceiling = active.compute(**_base_inputs(height_w=torch.tensor([[4.25]])))

    assert at_ceiling.components["reward_height_ceiling"].item() == 0.0
    assert below_ceiling.components["reward_height_ceiling"].item() == 0.0
    assert abs(above_ceiling.components["reward_height_ceiling"].item() + 0.5) < 1e-6
    assert above_ceiling.total.item() < at_ceiling.total.item()


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


def test_null_command_penalizes_motion_and_output_bias():
    _, computer = _computer(
        null_command_speed_weight=2.0,
        null_command_output_weight=0.5,
        command_eps=0.05,
    )

    stable = computer.compute(**_base_inputs(
        v_cmd_b=torch.zeros(1, 3),
        v_final_b=torch.zeros(1, 3),
        prev_v_final_b=torch.zeros(1, 3),
        actual_velocity_b=torch.zeros(1, 3),
    ))
    moving = computer.compute(**_base_inputs(
        v_cmd_b=torch.zeros(1, 3),
        v_final_b=torch.tensor([[0.4, 0.0, 0.0]]),
        prev_v_final_b=torch.zeros(1, 3),
        actual_velocity_b=torch.tensor([[0.3, 0.0, 0.0]]),
    ))
    nonzero_command = computer.compute(**_base_inputs(
        v_cmd_b=torch.tensor([[1.0, 0.0, 0.0]]),
        v_final_b=torch.tensor([[0.4, 0.0, 0.0]]),
        prev_v_final_b=torch.zeros(1, 3),
        actual_velocity_b=torch.tensor([[0.3, 0.0, 0.0]]),
    ))

    assert stable.components["reward_null_command_speed"].item() == 0.0
    assert stable.components["reward_null_command_output"].item() == 0.0
    assert abs(moving.components["reward_null_command_speed"].item() + 0.6) < 1e-6
    assert abs(moving.components["reward_null_command_output"].item() + 0.2) < 1e-6
    assert nonzero_command.components["reward_null_command_speed"].item() == 0.0
    assert nonzero_command.components["reward_null_command_output"].item() == 0.0
    assert moving.total.item() < stable.total.item()


def test_null_command_output_penalty_is_anchor_aware():
    _, computer = _computer(
        null_command_speed_weight=0.0,
        null_command_output_weight=0.5,
        null_output_anchor_loss_threshold=0.05,
        command_eps=0.05,
    )

    base = dict(
        v_cmd_b=torch.zeros(1, 3),
        v_final_b=torch.tensor([[0.4, 0.0, 0.0]]),
        prev_v_final_b=torch.zeros(1, 3),
        actual_velocity_b=torch.zeros(1, 3),
        anchor_active=torch.ones(1, 1),
        anchor_valid_fraction=torch.ones(1, 1),
    )
    low_loss = computer.compute(**_base_inputs(**base, anchor_loss=torch.tensor([[0.01]])))
    high_loss = computer.compute(**_base_inputs(**base, anchor_loss=torch.tensor([[0.2]])))
    inactive_base = dict(base)
    inactive_base["anchor_active"] = torch.zeros(1, 1)
    inactive_anchor = computer.compute(**_base_inputs(
        **inactive_base,
        anchor_loss=torch.tensor([[0.2]]),
    ))

    assert abs(low_loss.components["reward_null_command_output"].item() + 0.2) < 1e-6
    assert high_loss.components["reward_null_command_output"].item() == 0.0
    assert abs(inactive_anchor.components["reward_null_command_output"].item() + 0.2) < 1e-6
    assert low_loss.cache["null_command_output_bias_gate"].item() == 1.0
    assert high_loss.cache["null_command_output_bias_gate"].item() == 0.0


def test_proxy_tracking_and_command_amplification_penalize_unsafe_governor_bias():
    _, computer = _computer(
        proxy_tracking_weight=0.25,
        command_amplification_weight=0.5,
    )
    match = computer.compute(**_base_inputs(
        v_cmd_b=torch.tensor([[1.0, 0.0, 0.0]]),
        v_final_b=torch.tensor([[1.0, 0.0, 0.0]]),
        actual_velocity_b=torch.tensor([[1.0, 0.0, 0.0]]),
    ))
    amplified = computer.compute(**_base_inputs(
        v_cmd_b=torch.tensor([[1.0, 0.0, 0.0]]),
        v_final_b=torch.tensor([[1.5, 0.0, 0.0]]),
        actual_velocity_b=torch.tensor([[1.0, 0.0, 0.0]]),
    ))
    attenuated_by_ics = computer.compute(**_base_inputs(
        v_cmd_b=torch.tensor([[1.0, 0.0, 0.0]]),
        v_final_b=torch.tensor([[0.2, 0.0, 0.0]]),
        actual_velocity_b=torch.tensor([[1.0, 0.0, 0.0]]),
        ics_beta=torch.full((1, 1), 0.2),
    ))

    assert match.components["reward_proxy_tracking"].item() == 0.0
    assert match.components["reward_command_amplification"].item() == 0.0
    assert abs(amplified.components["reward_proxy_tracking"].item() + 0.125) < 1e-6
    assert abs(amplified.components["reward_command_amplification"].item() + 0.25) < 1e-6
    assert attenuated_by_ics.components["reward_proxy_tracking"].item() == 0.0
    assert attenuated_by_ics.components["reward_command_amplification"].item() == 0.0
    assert amplified.total.item() < match.total.item()


def test_preservation_band_penalizes_command_loss_and_gain_only_when_ics_clear():
    _, computer = _computer(
        preservation_low_weight=2.0,
        preservation_high_weight=3.0,
        preservation_lower=0.75,
        preservation_upper=1.05,
    )

    inside = computer.compute(**_base_inputs(
        v_cmd_b=torch.tensor([[1.0, 0.0, 0.0]]),
        v_final_b=torch.tensor([[0.9, 0.0, 0.0]]),
    ))
    too_slow = computer.compute(**_base_inputs(
        v_cmd_b=torch.tensor([[1.0, 0.0, 0.0]]),
        v_final_b=torch.tensor([[0.5, 0.0, 0.0]]),
    ))
    too_fast = computer.compute(**_base_inputs(
        v_cmd_b=torch.tensor([[1.0, 0.0, 0.0]]),
        v_final_b=torch.tensor([[1.2, 0.0, 0.0]]),
    ))
    attenuated_by_ics = computer.compute(**_base_inputs(
        v_cmd_b=torch.tensor([[1.0, 0.0, 0.0]]),
        v_final_b=torch.tensor([[0.5, 0.0, 0.0]]),
        ics_beta=torch.full((1, 1), 0.2),
    ))

    assert inside.components["reward_preservation_low"].item() == 0.0
    assert inside.components["reward_preservation_high"].item() == 0.0
    assert abs(too_slow.components["reward_preservation_low"].item() + 0.5) < 1e-6
    assert abs(too_fast.components["reward_preservation_high"].item() + 0.45) < 1e-6
    assert attenuated_by_ics.components["reward_preservation_low"].item() == 0.0
    assert too_slow.total.item() < inside.total.item()


def test_axis_preservation_terms_default_to_zero_and_do_not_change_reward():
    _, computer = _computer()
    out = computer.compute(**_base_inputs(
        v_cmd_b=torch.tensor([[1.0, 0.0, 1.0]]),
        v_final_b=torch.tensor([[0.2, 0.0, 0.2]]),
    ))

    assert out.components["reward_horizontal_preservation"].item() == 0.0
    assert out.components["reward_vertical_preservation"].item() == 0.0
    assert out.cache["horizontal_preservation_violation"].item() > 0.0
    assert out.cache["vertical_preservation_violation"].item() > 0.0


def test_axis_preservation_terms_are_command_and_ics_safety_gated():
    _, computer = _computer(
        horizontal_preservation_weight=2.0,
        vertical_preservation_weight=3.0,
        preservation_lower=0.75,
        preservation_upper=1.05,
    )

    active = computer.compute(**_base_inputs(
        v_cmd_b=torch.tensor([[1.0, 0.0, 1.0]]),
        v_final_b=torch.tensor([[0.5, 0.0, 0.5]]),
    ))
    attenuated_by_ics = computer.compute(**_base_inputs(
        v_cmd_b=torch.tensor([[1.0, 0.0, 1.0]]),
        v_final_b=torch.tensor([[0.5, 0.0, 0.5]]),
        ics_beta=torch.full((1, 1), 0.2),
    ))
    horizontal_only = computer.compute(**_base_inputs(
        v_cmd_b=torch.tensor([[1.0, 0.0, 0.0]]),
        v_final_b=torch.tensor([[0.5, 0.0, 0.0]]),
    ))

    assert abs(active.components["reward_horizontal_preservation"].item() + 0.5) < 1e-6
    assert abs(active.components["reward_vertical_preservation"].item() + 0.75) < 1e-6
    assert attenuated_by_ics.components["reward_horizontal_preservation"].item() == 0.0
    assert attenuated_by_ics.components["reward_vertical_preservation"].item() == 0.0
    assert abs(horizontal_only.components["reward_horizontal_preservation"].item() + 0.5) < 1e-6
    assert horizontal_only.components["reward_vertical_preservation"].item() == 0.0


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
    assert "height_w=self.root_state[..., 2].reshape(self.num_envs, 1)" in env_source
    assert "reward_safety_static" in env_source
    assert "else:" in env_source

    reward_source = open(REWARD_PATH, encoding="utf-8").read()
    assert "use_privileged_velocity_for_reward: bool = False" in reward_source
    assert "actual_velocity_b" in reward_source
    assert '"reward_height_floor"' in reward_source

    cfg_source = open(CFG_PATH, encoding="utf-8").read()
    assert "reward:" in cfg_source
    assert "enabled: true" in cfg_source
    assert "height_floor: 0.5" in cfg_source
    assert "height_floor_weight: 8.0" in cfg_source
    assert "height_ceiling: 4.0" in cfg_source
    assert "height_ceiling_weight: 0.0" in cfg_source
    assert "horizontal_preservation_weight: 0.0" in cfg_source
    assert "vertical_preservation_weight: 0.0" in cfg_source


def test_train_eval_semantics_are_handbook_aligned_by_default():
    env_source = open(ENV_PATH, encoding="utf-8").read()
    cfg_source = open(CFG_PATH, encoding="utf-8").read()

    assert 'task: "command_governor"' in cfg_source
    assert 'source: "curriculum_generator"' in cfg_source
    assert "use_privileged_velocity_for_reward: true" in cfg_source
    assert "curriculum_profile: \"station_first\"" in cfg_source
    assert "anchor_weight: 4.0" in cfg_source
    assert "null_command_speed_weight: 2.0" in cfg_source
    assert "null_command_output_weight: 0.1" in cfg_source
    assert "preservation_lower: 0.75" in cfg_source
    assert "preservation_upper: 1.05" in cfg_source
    assert "command_amplification_weight: 0.5" in cfg_source
    assert "height_floor: 0.5" in cfg_source
    assert "height_floor_weight: 8.0" in cfg_source
    assert "height_ceiling: 4.0" in cfg_source
    assert "height_ceiling_weight: 0.0" in cfg_source
    assert "horizontal_preservation_weight: 0.0" in cfg_source
    assert "vertical_preservation_weight: 0.0" in cfg_source
    assert "actual_velocity_b=actual_velocity_b" in env_source
    assert "height_w=self.root_state[..., 2].reshape(self.num_envs, 1)" in env_source
    assert "compute_termination_stats(" in env_source
    assert "terminated_below_bound" in env_source
    assert "truncated_timeout" in env_source

    ics_block = cfg_source.split("  ics:", 1)[1].split("  # instinctRL-F", 1)[0]
    assert "enabled: true" in ics_block


def test_eval_summary_reports_handbook_metrics_not_reach_goal_as_success():
    utils_path = os.path.join(SCRIPTS, "utils.py")
    utils_source = open(utils_path, encoding="utf-8").read()

    for key in [
        "eval/handbook.tracking_rmse_actual_body_vs_v_cmd",
        "eval/handbook.tracking_rmse_v_final_body_vs_v_cmd",
        "eval/handbook.command_preservation_ratio",
        "eval/handbook.anchor_error_mean",
        "eval/handbook.safety_min_clearance_p05",
        "eval/handbook.ics_intervention_frequency",
        "eval/handbook.termination_below_bound",
        "eval/handbook.null_command_speed_mean",
        "eval/handbook.null_command_output_speed_mean",
        "eval/handbook.command_amplification_mean",
        "eval/handbook.command_amplification_rate",
        "eval/handbook.command_amplification_horizontal_mean",
        "eval/handbook.command_amplification_vertical_mean",
        "eval/handbook.height_world_z_mean",
        "eval/handbook.height_ceiling_violation_mean",
    ]:
        assert key in utils_source

    assert "eval/stats.reach_goal: success rate" not in utils_source
