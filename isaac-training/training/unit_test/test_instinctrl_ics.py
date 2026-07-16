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
        {"residual_margin": -0.1},
        {"collision_clearance_threshold": -0.1},
        {"residual_margin": float("nan")},
        {"collision_clearance_threshold": float("inf")},
    ]
    for kwargs in bad_kwargs:
        try:
            mod.ICSConfig(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {kwargs}")


def test_residual_preemption_defaults_and_namespace_loading_are_off():
    mod = _load_ics()
    cfg = mod.ICSConfig()
    loaded = mod.ICSConfig.from_namespace(type("Cfg", (), {})())
    for value in (cfg, loaded):
        assert value.residual_preemption_enabled is False
        assert value.residual_margin == 0.0
        assert value.collision_clearance_threshold == 0.3


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


def test_disabled_residual_preemption_is_neutral_and_preserves_legacy_output():
    _, baseline = _attenuator(
        d_safe=0.8,
        clearance_margin=0.15,
        a_max=2.0,
        residual_preemption_enabled=False,
        collision_clearance_threshold=0.3,
    )
    ranges, masks, weights = _history([[[1.2], [1.0]]])
    out = baseline(ranges, masks, weights, _rays(1), torch.tensor([[0.1, 0.0, 0.0]]), dt=0.1)

    assert out.metrics["ics_beta"].item() == 1.0
    assert out.metrics["ics_emergency"].item() == 0.0
    assert out.metrics["ics_residual_preemption_trigger"].item() == 0.0
    assert out.metrics["ics_residual_preemption_range_rate_available"].item() == 0.0
    assert torch.count_nonzero(out.cache["ics_residual_preemption_command_closing"]).item() == 0
    assert torch.count_nonzero(out.cache["ics_residual_preemption_range_closing"]).item() == 0
    assert out.cache["ics_residual_preemption_source"].item() == 0
    assert not out.cache["ics_residual_preemption_range_rate_available"].item()
    assert not out.cache["ics_residual_preemption_eligible"].item()
    assert not out.cache["ics_residual_preemption_beam_trigger"].item()


def test_disabled_residual_preemption_matches_literal_legacy_golden_matrix():
    # Frozen from 6f6dee3; do not obtain expectations from Git at test time.
    # These are complete legacy snapshots: every public metric and every cache
    # tensor is checked with zero tolerance for each behavioral class.
    base = {
        "d_safe": 0.5, "emergency_clearance": 0.2, "a_max": 1.0,
        "velocity_limit": 3.0, "active_horizon_margin": 0.5,
        "latency_sec": 0.0, "clearance_margin": 0.0,
        "approach_eps": 0.0, "range_rate_eps": 0.0,
        "residual_preemption_enabled": False,
    }
    downward_metrics_off = {
        "ics_downward_beta": [[1.0]], "ics_downward_active": [[0.0]],
        "ics_downward_has_ray": [[0.0]], "ics_downward_min_clearance": [[0.0]],
        "ics_downward_pre_z": [[0.0]], "ics_downward_post_z": [[0.0]],
        "ics_downward_z_delta_abs": [[0.0]], "ics_downward_attenuation_ratio": [[0.0]],
    }
    downward_cache_off = dict(downward_metrics_off)
    cases = {
        "empty_active": (base, [[[2.0], [2.0]]], _rays(1), [[1.0, 0.0, 0.0]], None, {
            "v_final_b": [[1.0, 0.0, 0.0]],
            "metrics": {**downward_metrics_off, "ics_beta": [[1.0]], "ics_active_beam_count": [[0.0]], "ics_min_clearance": [[2.0]], "ics_worst_margin": [[0.0]], "ics_emergency": [[0.0]], "ics_command_speed": [[1.0]], "ics_brake_speed": [[0.0]], "ics_final_speed": [[1.0]], "ics_clip_ratio": [[1.0]]},
            "cache": {**downward_cache_off, "ics_active_mask": [[False]], "ics_approach_speed": [[1.0]], "ics_closing_speed": [[1.0]], "ics_range_rate_estimate": [[0.0]], "ics_safety_margin": [[0.7320507764816284]], "ics_worst_beam_index": [-1], "ics_effective_clearance": [[2.0]]},
        }),
        "normal": (base, [[[0.9], [0.9]]], _rays(1), [[1.0, 0.0, 0.0]], None, {
            "v_final_b": [[0.8944271802902222, 0.0, 0.0]],
            "metrics": {**downward_metrics_off, "ics_beta": [[0.8944271802902222]], "ics_active_beam_count": [[1.0]], "ics_min_clearance": [[0.8999999761581421]], "ics_worst_margin": [[-0.10557281970977783]], "ics_emergency": [[0.0]], "ics_command_speed": [[1.0]], "ics_brake_speed": [[0.0]], "ics_final_speed": [[0.8944271802902222]], "ics_clip_ratio": [[1.0]]},
            "cache": {**downward_cache_off, "ics_active_mask": [[True]], "ics_approach_speed": [[1.0]], "ics_closing_speed": [[1.0]], "ics_range_rate_estimate": [[0.0]], "ics_safety_margin": [[-0.10557281970977783]], "ics_worst_beam_index": [0], "ics_effective_clearance": [[0.8999999761581421]]},
        }),
        "emergency": (base, [[[0.15], [0.15]]], _rays(1), [[1.0, 0.0, 0.0]], None, {
            "v_final_b": [[0.0, 0.0, 0.0]],
            "metrics": {**downward_metrics_off, "ics_beta": [[0.0]], "ics_active_beam_count": [[1.0]], "ics_min_clearance": [[0.15000000596046448]], "ics_worst_margin": [[-1.0]], "ics_emergency": [[1.0]], "ics_command_speed": [[1.0]], "ics_brake_speed": [[0.0]], "ics_final_speed": [[0.0]], "ics_clip_ratio": [[1.0]]},
            "cache": {**downward_cache_off, "ics_active_mask": [[True]], "ics_approach_speed": [[1.0]], "ics_closing_speed": [[1.0]], "ics_range_rate_estimate": [[0.0]], "ics_safety_margin": [[-1.0]], "ics_worst_beam_index": [0], "ics_effective_clearance": [[0.15000000596046448]]},
        }),
        "downward": ({**base, "d_safe": 0.1, "emergency_clearance": 0.05, "active_horizon_margin": 0.0, "downward_attenuation_enabled": True, "downward_ray_min_z": 0.25}, [[[0.3, 0.3], [0.3, 0.3]]], torch.tensor([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]), [[0.4, 0.0, -1.0]], None, {
            "v_final_b": [[0.4000000059604645, 0.0, -0.7071067690849304]],
            "metrics": {"ics_beta": [[1.0]], "ics_active_beam_count": [[0.0]], "ics_min_clearance": [[0.30000001192092896]], "ics_worst_margin": [[0.0]], "ics_emergency": [[0.0]], "ics_command_speed": [[1.0770329236984253]], "ics_brake_speed": [[0.0]], "ics_final_speed": [[0.8124037981033325]], "ics_clip_ratio": [[1.0]], "ics_downward_beta": [[0.7071067690849304]], "ics_downward_active": [[1.0]], "ics_downward_has_ray": [[1.0]], "ics_downward_min_clearance": [[0.30000001192092896]], "ics_downward_pre_z": [[-1.0]], "ics_downward_post_z": [[-0.7071067690849304]], "ics_downward_z_delta_abs": [[0.2928932309150696]], "ics_downward_attenuation_ratio": [[0.2928932309150696]]},
            "cache": {"ics_active_mask": [[False, False]], "ics_approach_speed": [[1.0, 0.4000000059604645]], "ics_closing_speed": [[1.0, 0.4000000059604645]], "ics_range_rate_estimate": [[0.0, 0.0]], "ics_safety_margin": [[-0.36754441261291504, 0.23245558142662048]], "ics_worst_beam_index": [-1], "ics_effective_clearance": [[0.30000001192092896, 0.30000001192092896]], "ics_downward_beta": [[0.7071067690849304]], "ics_downward_active": [[1.0]], "ics_downward_has_ray": [[1.0]], "ics_downward_min_clearance": [[0.30000001192092896]], "ics_downward_pre_z": [[-1.0]], "ics_downward_post_z": [[-0.7071067690849304]], "ics_downward_z_delta_abs": [[0.2928932309150696]], "ics_downward_attenuation_ratio": [[0.2928932309150696]]},
        }),
        "clipping": ({**base, "velocity_limit": 0.5}, [[[0.9], [0.9]]], _rays(1), [[2.0, 0.0, 0.0]], None, {
            "v_final_b": [[0.5, 0.0, 0.0]],
            "metrics": {**downward_metrics_off, "ics_beta": [[0.4472135901451111]], "ics_active_beam_count": [[1.0]], "ics_min_clearance": [[0.8999999761581421]], "ics_worst_margin": [[-1.1055728197097778]], "ics_emergency": [[0.0]], "ics_command_speed": [[2.0]], "ics_brake_speed": [[0.0]], "ics_final_speed": [[0.5]], "ics_clip_ratio": [[0.55901700258255]]},
            "cache": {**downward_cache_off, "ics_active_mask": [[True]], "ics_approach_speed": [[2.0]], "ics_closing_speed": [[2.0]], "ics_range_rate_estimate": [[0.0]], "ics_safety_margin": [[-1.1055728197097778]], "ics_worst_beam_index": [0], "ics_effective_clearance": [[0.8999999761581421]]},
        }),
        "range_rate": ({**base, "use_range_rate_filter": True}, [[[0.9], [0.7]]], _rays(1), [[0.0, 1.0, 0.0]], 0.1, {
            "v_final_b": [[0.0, 0.3162277936935425, 0.0]],
            "metrics": {**downward_metrics_off, "ics_beta": [[0.3162277936935425]], "ics_active_beam_count": [[1.0]], "ics_min_clearance": [[0.699999988079071]], "ics_worst_margin": [[-1.367544412612915]], "ics_emergency": [[0.0]], "ics_command_speed": [[1.0]], "ics_brake_speed": [[0.0]], "ics_final_speed": [[0.3162277936935425]], "ics_clip_ratio": [[1.0]]},
            "cache": {**downward_cache_off, "ics_active_mask": [[True]], "ics_approach_speed": [[0.0]], "ics_closing_speed": [[1.9999998807907104]], "ics_range_rate_estimate": [[-1.9999998807907104]], "ics_safety_margin": [[-1.367544412612915]], "ics_worst_beam_index": [0], "ics_effective_clearance": [[0.699999988079071]]},
        }),
    }
    legacy_metric_keys = set(_load_ics().ICS_METRIC_KEYS) - {
        "ics_residual_preemption_trigger", "ics_residual_preemption_range_rate_available",
    }
    for _, (cfg, range_values, rays, command, dt, expected) in cases.items():
        _, ics = _attenuator(**cfg)
        out = ics(*_history(range_values), rays, torch.tensor(command), dt=dt)
        assert set(out.metrics) - {"ics_residual_preemption_trigger", "ics_residual_preemption_range_rate_available"} == legacy_metric_keys
        assert set(out.cache) - {
            "ics_residual_preemption_command_closing", "ics_residual_preemption_range_closing",
            "ics_residual_preemption_range_rate_available", "ics_residual_preemption_source",
            "ics_residual_preemption_required_stop", "ics_residual_preemption_residual",
            "ics_residual_preemption_eligible", "ics_residual_preemption_beam_trigger",
        } == set(expected["cache"])
        torch.testing.assert_close(out.v_final_b, torch.as_tensor(expected["v_final_b"]), rtol=0, atol=0)
        for group_name, actual in (("metrics", out.metrics), ("cache", out.cache)):
            for key, expected_value in expected[group_name].items():
                torch.testing.assert_close(actual[key], torch.as_tensor(expected_value), rtol=0, atol=0)
        assert out.metrics["ics_residual_preemption_trigger"].item() == 0.0
        assert out.metrics["ics_residual_preemption_range_rate_available"].item() == 0.0
        for key in (
            "ics_residual_preemption_command_closing", "ics_residual_preemption_range_closing",
            "ics_residual_preemption_required_stop", "ics_residual_preemption_residual",
        ):
            assert torch.count_nonzero(out.cache[key]).item() == 0
        for key in (
            "ics_residual_preemption_range_rate_available", "ics_residual_preemption_source",
            "ics_residual_preemption_eligible", "ics_residual_preemption_beam_trigger",
        ):
            assert torch.count_nonzero(out.cache[key]).item() == 0


def test_residual_preemption_uses_per_beam_range_rate_without_relabeling_emergency():
    _, legacy = _attenuator(
        d_safe=0.8,
        clearance_margin=0.15,
        a_max=2.0,
        residual_preemption_enabled=False,
        collision_clearance_threshold=0.3,
    )
    _, enabled = _attenuator(
        d_safe=0.8,
        clearance_margin=0.15,
        a_max=2.0,
        residual_preemption_enabled=True,
        residual_margin=0.0,
        collision_clearance_threshold=0.3,
    )
    history = _history([[[1.2], [1.0]]])
    command = torch.tensor([[0.1, 0.0, 0.0]])
    legacy_out = legacy(*history, _rays(1), command, dt=0.1)
    out = enabled(*history, _rays(1), command, dt=0.1)

    assert legacy_out.metrics["ics_beta"].item() == 1.0
    assert out.metrics["ics_beta"].item() == 0.0
    assert out.metrics["ics_emergency"].item() == 0.0
    assert out.metrics["ics_residual_preemption_trigger"].item() == 1.0
    assert out.metrics["ics_residual_preemption_range_rate_available"].item() == 1.0
    assert math.isclose(out.cache["ics_residual_preemption_command_closing"].item(), 0.1, rel_tol=1e-6)
    assert math.isclose(out.cache["ics_residual_preemption_range_closing"].item(), 2.0, rel_tol=1e-6)
    assert out.cache["ics_residual_preemption_source"].item() == 2
    assert math.isclose(out.cache["ics_residual_preemption_required_stop"].item(), 1.0, rel_tol=1e-6)
    assert math.isclose(out.cache["ics_residual_preemption_residual"].item(), -0.3, abs_tol=1e-6)
    assert out.cache["ics_residual_preemption_eligible"].item()
    assert out.cache["ics_residual_preemption_beam_trigger"].item()


def test_residual_preemption_handles_zero_command_and_missing_or_invalid_rate_evidence():
    _, ics = _attenuator(
        d_safe=0.1,
        emergency_clearance=0.05,
        residual_preemption_enabled=True,
        collision_clearance_threshold=0.3,
    )
    zero = torch.zeros(1, 3)
    closing = _history([[[1.2], [1.0]]])
    closing_out = ics(*closing, _rays(1), zero, dt=0.1)
    assert closing_out.metrics["ics_residual_preemption_trigger"].item() == 1.0
    assert closing_out.cache["ics_residual_preemption_source"].item() == 2
    assert torch.allclose(closing_out.v_final_b, zero)

    opening = _history([[[1.0], [1.2]]])
    opening_out = ics(*opening, _rays(1), zero, dt=0.1)
    assert opening_out.metrics["ics_residual_preemption_trigger"].item() == 0.0
    assert opening_out.metrics["ics_residual_preemption_range_rate_available"].item() == 1.0
    assert opening_out.cache["ics_residual_preemption_source"].item() == 0

    one_frame = _history([[[1.0]]])
    one_frame_out = ics(*one_frame, _rays(1), zero)
    assert one_frame_out.metrics["ics_residual_preemption_range_rate_available"].item() == 0.0
    assert one_frame_out.metrics["ics_residual_preemption_trigger"].item() == 0.0

    for masks, weights in (
        ([[[1.0], [0.0]]], None),
        ([[[0.0], [1.0]]], None),
        (None, [[[0.0], [1.0]]]),
        (None, [[[1.0], [0.0]]]),
    ):
        ranges, mask_history, weight_history = _history([[[1.2], [1.0]]], masks, weights)
        out = ics(ranges, mask_history, weight_history, _rays(1), zero, dt=0.1)
        assert out.metrics["ics_residual_preemption_range_rate_available"].item() == 0.0
        assert out.metrics["ics_residual_preemption_trigger"].item() == 0.0


def test_residual_preemption_uses_command_evidence_when_rate_is_unavailable():
    _, ics = _attenuator(
        d_safe=0.1,
        emergency_clearance=0.05,
        residual_preemption_enabled=True,
        collision_clearance_threshold=0.3,
    )
    ranges, masks, weights = _history([[[0.4], [0.4]]], masks=[[[0.0], [1.0]]])
    out = ics(ranges, masks, weights, _rays(1), torch.tensor([[1.0, 0.0, 0.0]]), dt=0.1)
    assert out.metrics["ics_residual_preemption_range_rate_available"].item() == 0.0
    assert out.metrics["ics_residual_preemption_trigger"].item() == 1.0
    assert out.cache["ics_residual_preemption_source"].item() == 1


def test_residual_preemption_leaves_eligible_positive_residual_unchanged():
    _, ics = _attenuator(
        d_safe=0.1,
        emergency_clearance=0.05,
        residual_preemption_enabled=True,
        collision_clearance_threshold=0.3,
        residual_margin=0.0,
    )
    ranges, masks, weights = _history([[[3.0], [3.0]]])
    out = ics(ranges, masks, weights, _rays(1), torch.tensor([[1.0, 0.0, 0.0]]))
    assert out.cache["ics_residual_preemption_eligible"].item()
    assert out.cache["ics_residual_preemption_residual"].item() > 0.0
    assert out.metrics["ics_residual_preemption_trigger"].item() == 0.0
    assert out.metrics["ics_beta"].item() == 1.0


def test_residual_preemption_empty_active_set_and_equal_source():
    _, ics = _attenuator(
        d_safe=0.1,
        emergency_clearance=0.05,
        residual_preemption_enabled=True,
        collision_clearance_threshold=0.3,
    )
    ranges, masks, weights = _history([[[2.0], [1.0]]])
    out = ics(ranges, masks, weights, _rays(1), torch.tensor([[1.0, 0.0, 0.0]]), dt=1.0)
    assert out.cache["ics_residual_preemption_source"].item() == 3

    far, far_masks, far_weights = _history([[[2.0], [2.0]]])
    empty = ics(far, far_masks, far_weights, _rays(1), torch.zeros(1, 3))
    assert empty.metrics["ics_residual_preemption_trigger"].item() == 0.0
    assert empty.cache["ics_residual_preemption_source"].item() == 0


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
    for key in [
        "ics_residual_preemption_trigger",
        "ics_residual_preemption_range_rate_available",
        "ics_residual_preemption_command_closing",
        "ics_residual_preemption_range_closing",
        "ics_residual_preemption_source",
        "ics_residual_preemption_required_stop",
        "ics_residual_preemption_residual",
        "ics_residual_preemption_eligible",
        "ics_residual_preemption_beam_trigger",
    ]:
        assert key not in actor_block

    train_source = open(TRAIN_PATH, encoding="utf-8").read()
    assert "v_final_body = ics_out.v_final_b" in train_source
    assert "safety_out = safety_filter(v_final_body, root_height_w)" in train_source
    assert "env.set_prev_issued_action_body(v_final_body)" in train_source
    assert "v_final_world = adapter(v_final_body, drone_quat)" in train_source
    assert train_source.index("v_final_body = ics_out.v_final_b") < train_source.index(
        "v_final_world = adapter(v_final_body, drone_quat)"
    )
