import importlib.util
import math
import os
import sys

import torch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPTS = os.path.join(ROOT, "training", "scripts")
ICS_PATH = os.path.join(SCRIPTS, "instinctRL", "ics.py")
SAFETY_FILTER_PATH = os.path.join(SCRIPTS, "instinctRL", "safety_filter.py")
OBS_PATH = os.path.join(SCRIPTS, "instinctRL", "observation.py")
ENV_PATH = os.path.join(SCRIPTS, "env.py")
TRAIN_PATH = os.path.join(SCRIPTS, "train.py")


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_ics():
    return _load_module(ICS_PATH, "instinctrl_ics_test")


def _load_safety_filter():
    return _load_module(SAFETY_FILTER_PATH, "instinctrl_safety_filter_test")


def _load_observation():
    return _load_module(OBS_PATH, "instinctrl_observation_ics_test")


def _attenuator(**kwargs):
    mod = _load_ics()
    params = {
        "d_safe": 0.5,
        "emergency_clearance": 0.2,
        "a_max": 1.0,
        "velocity_limit": 3.0,
        "active_horizon_margin": 0.5,
        "latency_sec": 0.0,
        "clearance_margin": 0.0,
        "approach_eps": 0.0,
        "range_rate_eps": 0.0,
    }
    params.update(kwargs)
    cfg = mod.ICSConfig(**params)
    return mod, mod.RangeHistoryICSAttenuator(cfg, device="cpu")


def _history(ranges, masks=None, weights=None):
    ranges = torch.as_tensor(ranges, dtype=torch.float32)
    if ranges.dim() == 2:
        ranges = ranges.unsqueeze(0)
    if masks is None:
        masks = torch.ones_like(ranges)
    else:
        masks = torch.as_tensor(masks, dtype=torch.float32)
        if masks.dim() == 2:
            masks = masks.unsqueeze(0)
    if weights is None:
        weights = torch.ones_like(ranges)
    else:
        weights = torch.as_tensor(weights, dtype=torch.float32)
        if weights.dim() == 2:
            weights = weights.unsqueeze(0)
    return ranges, masks, weights


def _rays(count):
    base = torch.tensor([
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ])
    return base[:count]


def test_config_validation():
    mod = _load_ics()
    mod.ICSConfig()
    bad_kwargs = [
        {"a_max": 0.0},
        {"velocity_limit": 0.0},
        {"d_safe": 0.0},
        {"emergency_clearance": 2.0, "d_safe": 1.0},
        {"min_reliability": 0.0},
        {"min_reliability": 1.1},
        {"brake_mode": "reverse"},
        {"range_rate_mode": "other"},
        {"downward_ray_min_z": -0.1},
        {"downward_clearance_margin": -0.1},
    ]
    for kwargs in bad_kwargs:
        try:
            mod.ICSConfig(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {kwargs}")


def test_shape_validation_for_histories_rays_and_commands():
    _, ics = _attenuator()
    ranges, masks, weights = _history([[[1.0, 1.0], [0.9, 0.9]]])
    rays = _rays(2)
    out = ics(ranges, masks, weights, rays, torch.tensor([[0.1, 0.0, 0.0]]))
    assert out.v_final_b.shape == (1, 3)

    ranges4 = ranges.reshape(1, 2, 1, 2)
    masks4 = masks.reshape(1, 2, 1, 2)
    weights4 = weights.reshape(1, 2, 1, 2)
    out = ics(ranges4, masks4, weights4, rays, torch.tensor([[[0.1, 0.0, 0.0]]]))
    assert out.v_final_b.shape == (1, 1, 3)

    bad_inputs = [
        (ranges[:, :, :1], masks, weights, rays, torch.zeros(1, 3)),
        (ranges, masks, weights, rays[:1], torch.zeros(1, 3)),
        (ranges, masks, weights, rays, torch.zeros(1, 2)),
        (ranges, masks, weights, rays, torch.full((1, 3), float("nan"))),
    ]
    for args in bad_inputs:
        try:
            ics(*args)
        except ValueError:
            pass
        else:
            raise AssertionError("malformed ICS input must fail")


def test_empty_active_set_preserves_command_unless_clipped():
    _, ics = _attenuator(velocity_limit=3.0)
    ranges, masks, weights = _history([[[2.0, 2.0], [2.0, 2.0]]])
    out = ics(ranges, masks, weights, _rays(2), torch.tensor([[1.0, 0.0, 0.0]]))
    assert out.metrics["ics_beta"].item() == 1.0
    assert out.metrics["ics_active_beam_count"].item() == 0.0
    assert torch.allclose(out.v_final_b, torch.tensor([[1.0, 0.0, 0.0]]))

    _, clipped = _attenuator(velocity_limit=0.5)
    out = clipped(ranges, masks, weights, _rays(2), torch.tensor([[2.0, 0.0, 0.0]]))
    assert out.metrics["ics_beta"].item() == 1.0
    assert torch.allclose(out.v_final_b, torch.tensor([[0.5, 0.0, 0.0]]))


def test_emergency_bypass_forces_zero_command():
    _, ics = _attenuator()
    ranges, masks, weights = _history([[[0.15, 2.0], [0.15, 2.0]]])
    out = ics(ranges, masks, weights, _rays(2), torch.tensor([[1.0, 0.0, 0.0]]))
    assert out.metrics["ics_emergency"].item() == 1.0
    assert out.metrics["ics_beta"].item() == 0.0
    assert torch.allclose(out.v_final_b, torch.zeros(1, 3))


def test_beta_is_monotonic_with_clearance_and_speed():
    _, ics = _attenuator()
    rays = _rays(1)
    near = _history([[[0.7], [0.7]]])
    far = _history([[[0.9], [0.9]]])
    slow = torch.tensor([[0.5, 0.0, 0.0]])
    fast = torch.tensor([[1.0, 0.0, 0.0]])

    beta_near = ics(*near, rays, slow).metrics["ics_beta"].item()
    beta_far = ics(*far, rays, slow).metrics["ics_beta"].item()
    beta_slow = ics(*far, rays, slow).metrics["ics_beta"].item()
    beta_fast = ics(*far, rays, fast).metrics["ics_beta"].item()

    assert beta_near <= beta_far
    assert beta_fast <= beta_slow


def test_active_set_rules_and_ratio_clamp():
    _, ics = _attenuator()
    ranges, masks, weights = _history(
        [[[0.9, 0.9, 0.9, 2.0, 0.9], [0.9, 0.9, 0.9, 2.0, 0.9]]],
        masks=[[[0.0, 1.0, 1.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0, 1.0]]],
        weights=[[[1.0, 0.05, 1.0, 1.0, 1.0], [1.0, 0.05, 1.0, 1.0, 1.0]]],
    )
    out = ics(ranges, masks, weights, _rays(5), torch.tensor([[1.0, 0.0, 0.0]]))
    assert out.cache["ics_active_mask"].tolist() == [[False, False, False, False, False]]

    ranges, masks, weights = _history([[[1.0], [1.0]]])
    _, permissive = _attenuator(a_max=10.0)
    out = permissive(ranges, masks, weights, _rays(1), torch.tensor([[0.1, 0.0, 0.0]]))
    assert out.cache["ics_active_mask"].item() is True
    assert math.isclose(out.metrics["ics_beta"].item(), 1.0, rel_tol=1e-6)


def test_range_rate_cache_and_optional_filter():
    rays = _rays(1)
    ranges, masks, weights = _history([[[0.9], [0.7]]])
    command = torch.tensor([[0.0, 1.0, 0.0]])

    _, default_ics = _attenuator(use_range_rate_filter=False)
    default_out = default_ics(ranges, masks, weights, rays, command, dt=0.1)
    assert torch.allclose(default_out.cache["ics_range_rate_estimate"], torch.tensor([[-2.0]]))
    assert default_out.metrics["ics_beta"].item() == 1.0

    _, rate_ics = _attenuator(use_range_rate_filter=True)
    rate_out = rate_ics(ranges, masks, weights, rays, command, dt=0.1)
    assert rate_out.cache["ics_active_mask"].item() is True
    assert rate_out.metrics["ics_beta"].item() < 1.0

    one_frame = _history([[[0.7]]])
    one_out = rate_ics(*one_frame, rays, command)
    assert torch.allclose(one_out.cache["ics_range_rate_estimate"], torch.zeros(1, 1))


def test_command_clipping_uses_unclipped_beta_and_preserves_direction():
    _, ics = _attenuator(velocity_limit=0.5)
    ranges, masks, weights = _history([[[0.9], [0.9]]])
    command = torch.tensor([[2.0, 0.0, 0.0]])
    out = ics(ranges, masks, weights, _rays(1), command)

    expected_beta = math.sqrt(2.0 * (0.9 - 0.5)) / 2.0
    assert math.isclose(out.metrics["ics_beta"].item(), expected_beta, rel_tol=1e-5)
    assert torch.allclose(out.v_final_b, torch.tensor([[0.5, 0.0, 0.0]]), atol=1e-6)
    assert out.metrics["ics_command_speed"].shape == (1, 1)
    assert out.metrics["ics_final_speed"].shape == (1, 1)
    assert out.metrics["ics_clip_ratio"].shape == (1, 1)
    assert out.metrics["ics_clip_ratio"].item() < 1.0


def test_downward_attenuator_disabled_preserves_ics_output():
    params = {
        "d_safe": 0.1,
        "emergency_clearance": 0.05,
        "active_horizon_margin": 0.0,
        "velocity_limit": 3.0,
    }
    mod, baseline = _attenuator(**params, downward_attenuation_enabled=False)
    _, disabled = _attenuator(
        **params,
        downward_attenuation_enabled=False,
        downward_ray_min_z=1.0,
        downward_clearance_margin=0.1,
    )
    ranges, masks, weights = _history([[[0.3], [0.3]]])
    rays = torch.tensor([[0.0, 0.0, -1.0]])
    command = torch.tensor([[0.4, 0.0, -1.0]])

    baseline_out = baseline(ranges, masks, weights, rays, command)
    disabled_out = disabled(ranges, masks, weights, rays, command)

    assert set(disabled_out.metrics) == set(mod.ICS_METRIC_KEYS)
    assert torch.allclose(disabled_out.v_final_b, baseline_out.v_final_b)
    for key in baseline_out.metrics:
        assert torch.allclose(disabled_out.metrics[key], baseline_out.metrics[key])


def test_downward_attenuator_enabled_reduces_only_downward_z_from_mid360_rays():
    _, ics = _attenuator(
        d_safe=0.1,
        emergency_clearance=0.05,
        active_horizon_margin=0.0,
        velocity_limit=3.0,
        downward_attenuation_enabled=True,
        downward_ray_min_z=0.25,
    )
    ranges, masks, weights = _history([[[0.3, 0.3], [0.3, 0.3]]])
    rays = torch.tensor([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    command = torch.tensor([[0.4, 0.0, -1.0]])

    out = ics(ranges, masks, weights, rays, command)

    expected_z = -math.sqrt(2.0 * (0.3 - 0.05))
    assert torch.allclose(out.v_final_b[..., :2], torch.tensor([[0.4, 0.0]]))
    assert math.isclose(out.v_final_b[..., 2].item(), expected_z, rel_tol=1e-5)
    assert out.cache["ics_downward_active"].item() == 1.0
    assert out.cache["ics_downward_beta"].item() < 1.0
    assert out.metrics["ics_downward_active"].item() == 1.0
    assert out.metrics["ics_downward_has_ray"].item() == 1.0
    assert out.metrics["ics_downward_beta"].item() == out.cache["ics_downward_beta"].item()
    assert math.isclose(out.metrics["ics_downward_pre_z"].item(), -1.0, rel_tol=1e-6)
    assert math.isclose(
        out.metrics["ics_downward_post_z"].item(),
        out.v_final_b[..., 2].item(),
        rel_tol=1e-6,
    )
    assert out.metrics["ics_downward_z_delta_abs"].item() > 0.0
    assert 0.0 < out.metrics["ics_downward_attenuation_ratio"].item() < 1.0

    upward = ics(ranges, masks, weights, rays, torch.tensor([[0.4, 0.0, 1.0]]))
    assert torch.allclose(upward.v_final_b, torch.tensor([[0.4, 0.0, 1.0]]))


def test_privileged_height_safety_filter_is_default_equivalent_and_downward_only():
    mod = _load_safety_filter()
    disabled = mod.PrivilegedHeightFloorSafetyFilter(
        mod.PrivilegedHeightSafetyFilterConfig(enabled=False),
        device="cpu",
    )
    enabled = mod.PrivilegedHeightFloorSafetyFilter(
        mod.PrivilegedHeightSafetyFilterConfig(
            enabled=True,
            height_floor=0.5,
            attenuation_band=0.3,
            min_downward_scale=0.0,
        ),
        device="cpu",
    )
    command = torch.tensor([[0.2, -0.1, -1.0], [0.2, -0.1, 1.0]])
    height = torch.tensor([[0.56], [0.56]])

    disabled_out = disabled(command, height)
    enabled_out = enabled(command, height)

    assert torch.allclose(disabled_out.v_final_b, command)
    assert torch.allclose(enabled_out.v_final_b[..., :2], command[..., :2])
    assert math.isclose(enabled_out.v_final_b[0, 2].item(), -0.2, rel_tol=1e-6)
    assert enabled_out.v_final_b[1, 2].item() == 1.0
    assert enabled_out.metrics["safety_filter_height_active"].reshape(-1).tolist() == [1.0, 0.0]


def test_history_accessor_copy_ordering_and_source_wrapper():
    obs_mod = _load_observation()
    builder = obs_mod.MID360ObservationBuilder(
        obs_mod.ObservationConfig(history_len=2, lidar_hbeams=1, lidar_vbeams=2, lidar_range=10.0),
        device="cpu",
    )
    state = torch.zeros(1, 13)
    state[:, 3] = 1.0
    frame1 = builder.build(
        torch.ones(1, 2, 3),
        torch.zeros(1, 3),
        state,
        torch.zeros(1, 3),
        0.1,
        1,
        torch.zeros(1, 3),
    )
    builder.build_history(frame1)
    frame2 = builder.build(
        torch.ones(1, 2, 3) * 2.0,
        torch.zeros(1, 3),
        state,
        torch.zeros(1, 3),
        0.1,
        1,
        torch.zeros(1, 3),
    )
    builder.build_history(frame2)

    copied = builder.get_history(copy=True)
    borrowed = builder.get_history(copy=False)
    assert copied["range_history"].shape == (1, 2, 1, 2)
    assert torch.allclose(copied["range_history"][:, 0], frame1["range"])
    assert torch.allclose(copied["range_history"][:, -1], frame2["range"])
    copied["range_history"][:] = 99.0
    assert not torch.allclose(builder.get_history(copy=False)["range_history"], copied["range_history"])
    borrowed["range_history"][:] = 7.0
    assert torch.allclose(builder.get_history(copy=False)["range_history"], torch.full((1, 2, 1, 2), 7.0))

    env_source = open(ENV_PATH, encoding="utf-8").read()
    assert "def get_instinctrl_range_history(self, copy: bool = True)" in env_source
    assert "return self._obs_builder.get_history(copy=copy)" in env_source


def test_source_level_safety_and_integration_contracts():
    ics_source = open(ICS_PATH, encoding="utf-8").read().lower()
    for token in ["surface_", "map", "odom", "slam", "dynamic_obstacle", "root"]:
        assert token not in ics_source

    env_source = open(ENV_PATH, encoding="utf-8").read()
    actor_block = env_source.split("# -----------------Network Input Final--------------", 1)[1]
    actor_block = actor_block.split("# ============================================", 1)[0]
    assert '"lidar_grid": obs_hist["lidar_grid"]' in actor_block
    assert '"state_vec": obs_hist["state_vec"]' in actor_block
    for token in ['"ics_', '"anchor_', '"observability_', '"map', '"odom', '"root_state', '"safety_filter']:
        assert token not in actor_block

    train_source = open(TRAIN_PATH, encoding="utf-8").read()
    assert "v_final_body = ics_out.v_final_b" in train_source
    assert "safety_out = safety_filter(v_final_body, root_height_w)" in train_source
    assert "env.set_prev_issued_action_body(v_final_body)" in train_source
    assert "v_final_world = adapter(v_final_body, drone_quat)" in train_source
    assert train_source.index("v_final_body = ics_out.v_final_b") < train_source.index(
        "v_final_world = adapter(v_final_body, drone_quat)"
    )
