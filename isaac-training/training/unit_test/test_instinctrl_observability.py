import importlib.util
import math
import os
import sys

import torch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPTS = os.path.join(ROOT, "training", "scripts")
OBS_PATH = os.path.join(SCRIPTS, "instinctRL", "observability.py")
ENV_PATH = os.path.join(SCRIPTS, "env.py")


def _load_observability():
    spec = importlib.util.spec_from_file_location("instinctrl_observability_test", OBS_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _logger(**kwargs):
    mod = _load_observability()
    cfg = mod.ObservabilityConfig(**kwargs)
    return mod, mod.RangeJacobianObservabilityLogger(cfg, device="cpu")


def _base_inputs(num_envs=1):
    rays = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ])
    valid = torch.ones(num_envs, rays.shape[0], dtype=torch.bool)
    weight = torch.ones(num_envs, rays.shape[0])
    return rays, valid, weight


def test_config_validation():
    mod = _load_observability()
    mod.ObservabilityConfig(mode="offline", rank_tol=1e-4, condition_number_cap=10.0)
    for kwargs in [
        {"mode": "bad"},
        {"rank_tol": 0.0},
        {"condition_number_cap": 0.0},
        {"condition_number_cap": float("inf")},
        {"min_valid_fraction": -0.1},
        {"min_valid_fraction": 1.1},
        {"log_interval": 0},
    ]:
        try:
            mod.ObservabilityConfig(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {kwargs}")


def test_proxy_mode_uses_normalized_negative_ray_directions_and_weights():
    mod, logger = _logger(mode="proxy")
    rays = torch.tensor([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]])
    valid = torch.tensor([[True, True, False]])
    weight = torch.tensor([[1.0, 0.25, 1.0]])
    out = logger.compute(ray_directions_b=rays, valid_mask=valid, reliability_weight=weight)

    expected_j = torch.tensor([[[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]])
    assert isinstance(out, mod.ObservabilityOutput)
    assert torch.allclose(out.cache["jacobian_rows"], expected_j)
    assert torch.allclose(out.cache["weighted_jacobian_rows"][0, 0], expected_j[0, 0])
    assert torch.allclose(out.cache["weighted_jacobian_rows"][0, 1], expected_j[0, 1] * 0.5)
    assert out.cache["effective_row_mask"].tolist() == [[True, True, False]]
    assert out.metrics["observability_is_proxy"].item() == 1.0
    assert out.metrics["observability_mode_code"].item() == mod.OBS_MODE_PROXY


def test_normal_mode_uses_negative_unit_normals_and_sqrt_weights():
    mod, logger = _logger(mode="offline", use_surface_normals=True, use_finite_difference=False)
    rays, valid, weight = _base_inputs()
    normals = torch.tensor([[[2.0, 0.0, 0.0], [0.0, -3.0, 0.0], [0.0, 0.0, 4.0], [0.0, 0.0, 0.0]]])
    weight[0, 1] = 0.25
    out = logger.compute(
        ray_directions_b=rays,
        valid_mask=valid,
        reliability_weight=weight,
        surface_normals_b=normals,
    )
    assert out.metrics["observability_mode_code"].item() == mod.OBS_MODE_NORMAL
    assert out.metrics["observability_is_proxy"].item() == 0.0
    assert torch.allclose(out.cache["jacobian_rows"][0, 0], torch.tensor([-1.0, 0.0, 0.0]))
    assert torch.allclose(out.cache["jacobian_rows"][0, 1], torch.tensor([0.0, 1.0, 0.0]))
    assert torch.allclose(out.cache["weighted_jacobian_rows"][0, 1], torch.tensor([0.0, 0.5, 0.0]))
    assert out.cache["effective_row_mask"].tolist() == [[True, True, True, False]]


def test_finite_difference_estimator_recovers_known_jacobian_and_sign():
    mod, logger = _logger(mode="offline", use_finite_difference=True, use_surface_normals=True)
    rays, valid, weight = _base_inputs()
    j_true = torch.tensor([[
        [1.0, 0.0, 0.0],
        [0.0, -2.0, 0.0],
        [0.0, 0.0, 3.0],
        [1.0, 2.0, 3.0],
    ]])
    perturb = torch.eye(3)
    delta = torch.einsum("kd,nrd->nkr", perturb, j_true)
    normals = torch.ones(1, 4, 3)
    out = logger.compute(
        ray_directions_b=rays,
        valid_mask=valid,
        reliability_weight=weight,
        surface_normals_b=normals,
        fd_perturbations_b=perturb,
        fd_range_delta=delta,
    )
    assert out.metrics["observability_mode_code"].item() == mod.OBS_MODE_FINITE_DIFFERENCE
    assert out.metrics["observability_is_proxy"].item() == 0.0
    assert torch.allclose(out.cache["jacobian_rows"], j_true, atol=1e-6)

    perturb_over = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ])
    delta_over = torch.einsum("kd,nrd->nkr", perturb_over, j_true)
    out_over = logger.compute(
        ray_directions_b=rays,
        valid_mask=valid,
        reliability_weight=weight,
        fd_perturbations_b=perturb_over,
        fd_range_delta=delta_over,
    )
    assert torch.allclose(out_over.cache["jacobian_rows"], j_true, atol=1e-5)


def test_mode_precedence_and_malformed_inputs():
    mod, _ = _logger()
    rays, valid, weight = _base_inputs()
    normals = torch.ones(1, 4, 3)
    perturb = torch.eye(3)
    j_true = torch.ones(1, 4, 3)
    delta = torch.einsum("kd,nrd->nkr", perturb, j_true)

    _, proxy_logger = _logger(mode="proxy", use_finite_difference=True, use_surface_normals=True)
    out = proxy_logger.compute(
        ray_directions_b=rays,
        valid_mask=valid,
        reliability_weight=weight,
        surface_normals_b=normals,
        fd_perturbations_b=perturb,
        fd_range_delta=delta,
    )
    assert out.metrics["observability_mode_code"].item() == mod.OBS_MODE_PROXY

    _, normal_logger = _logger(mode="offline", use_surface_normals=True, use_finite_difference=True)
    out = normal_logger.compute(
        ray_directions_b=rays,
        valid_mask=valid,
        reliability_weight=weight,
        surface_normals_b=normals,
    )
    assert out.metrics["observability_mode_code"].item() == mod.OBS_MODE_NORMAL

    _, fallback_logger = _logger(mode="offline", use_surface_normals=True, use_finite_difference=True)
    out = fallback_logger.compute(ray_directions_b=rays, valid_mask=valid, reliability_weight=weight)
    assert out.metrics["observability_mode_code"].item() == mod.OBS_MODE_PROXY
    assert out.metrics["observability_is_proxy"].item() == 1.0

    for kwargs in [
        {"fd_perturbations_b": torch.eye(2), "fd_range_delta": torch.zeros(1, 2, 4)},
        {"fd_perturbations_b": torch.ones(3, 3), "fd_range_delta": torch.zeros(1, 3, 4)},
        {"surface_normals_b": torch.ones(1, 4, 2)},
        {"surface_normals_b": torch.full((1, 4, 3), float("nan"))},
    ]:
        try:
            normal_logger.compute(ray_directions_b=rays, valid_mask=valid, reliability_weight=weight, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("malformed supplied observability input must fail")


def test_svd_rank_degenerate_metrics_and_condition_cap():
    mod, logger = _logger(mode="proxy", rank_tol=1e-4, condition_number_cap=100.0)
    rays_full = torch.eye(3)
    valid = torch.ones(1, 3, dtype=torch.bool)
    weight = torch.ones(1, 3)
    out = logger.compute(ray_directions_b=rays_full, valid_mask=valid, reliability_weight=weight)
    assert out.metrics["observability_rank"].item() == 3.0
    assert out.metrics["observability_sigma_min"].item() > 0
    assert out.metrics["observability_condition_number"].item() <= 100.0

    rays_rank2 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    out = logger.compute(ray_directions_b=rays_rank2, valid_mask=valid, reliability_weight=weight)
    assert out.metrics["observability_rank"].item() == 2.0
    assert abs(out.metrics["observability_sigma_min"].item()) < 1e-5

    rays_rank1 = torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [-3.0, 0.0, 0.0]])
    out = logger.compute(ray_directions_b=rays_rank1, valid_mask=valid, reliability_weight=weight)
    assert out.metrics["observability_rank"].item() == 1.0

    valid_few = torch.tensor([[True, True, False]])
    out = logger.compute(ray_directions_b=rays_full, valid_mask=valid_few, reliability_weight=weight)
    assert out.metrics["observability_rank"].item() == 0.0
    assert out.metrics["observability_sigma_min"].item() == 0.0
    assert out.metrics["observability_sigma_max"].item() == 0.0
    assert out.metrics["observability_condition_number"].item() == 100.0
    assert torch.allclose(out.cache["observability_weak_direction"], torch.zeros(1, 3))
    assert torch.isfinite(torch.cat([v.float().reshape(-1) for v in out.metrics.values()])).all()


def test_weak_direction_and_drift_projection_are_cache_only_and_sign_invariant():
    _, logger = _logger(mode="proxy")
    rays = torch.eye(3)
    valid = torch.ones(1, 3, dtype=torch.bool)
    weight = torch.ones(1, 3)
    drift = torch.tensor([[0.0, 0.0, 2.0]])
    out = logger.compute(ray_directions_b=rays, valid_mask=valid, reliability_weight=weight, drift_b=drift)
    assert "observability_weak_direction" in out.cache
    assert "observability_weak_direction" not in out.metrics
    assert out.metrics["observability_drift_norm"].item() == 2.0
    assert out.metrics["observability_drift_projection"].item() >= 0.0

    no_drift = logger.compute(ray_directions_b=rays, valid_mask=valid, reliability_weight=weight)
    assert no_drift.metrics["observability_drift_norm"].item() == 0.0
    assert no_drift.metrics["observability_drift_projection"].item() == 0.0

    try:
        logger.compute(ray_directions_b=rays, valid_mask=valid, reliability_weight=weight, drift_b=torch.zeros(1, 2))
    except ValueError:
        pass
    else:
        raise AssertionError("bad drift shape must fail")
    try:
        logger.compute(ray_directions_b=rays, valid_mask=valid, reliability_weight=weight, drift_b=torch.full((1, 3), float("nan")))
    except ValueError:
        pass
    else:
        raise AssertionError("non-finite drift must fail")


def test_public_metrics_shapes_cache_boundary_and_scenario_id():
    mod, logger = _logger(mode="proxy")
    rays, valid, weight = _base_inputs(num_envs=2)
    out = logger.compute(
        ray_directions_b=rays,
        valid_mask=valid,
        reliability_weight=weight,
        scenario_id=torch.tensor([7, 8]),
    )
    expected = set(mod.OBSERVABILITY_METRIC_KEYS) | {"observability_scenario_id"}
    assert set(out.metrics) == expected
    for key, value in out.metrics.items():
        assert value.shape == (2, 1), key
    assert out.metrics["observability_mode_code"].dtype == torch.long
    assert out.metrics["observability_scenario_id"].dtype == torch.long
    for dense in ["jacobian_rows", "weighted_jacobian_rows", "singular_values", "observability_weak_direction", "effective_row_mask"]:
        assert dense in out.cache
        assert dense not in out.metrics


def test_source_level_actor_contract_for_observability_env_integration():
    source = open(ENV_PATH, encoding="utf-8").read()
    actor_block = source.split("# -----------------Network Input Final--------------", 1)[1]
    actor_block = actor_block.split("# ============================================", 1)[0]
    assert '"lidar_grid": obs_hist["lidar_grid"]' in actor_block
    assert '"state_vec": obs_hist["state_vec"]' in actor_block
    forbidden = ['"observability_', '"jacobian', '"normal', '"map', '"odom', '"root_state']
    for token in forbidden:
        assert token not in actor_block
