import importlib.util
import math
import os
import sys
from types import SimpleNamespace

import torch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPTS = os.path.join(ROOT, "training", "scripts")
ANCHOR_PATH = os.path.join(SCRIPTS, "instinctRL", "anchor.py")
ENV_PATH = os.path.join(SCRIPTS, "env.py")


def _load_anchor():
    spec = importlib.util.spec_from_file_location("instinctrl_anchor_test", ANCHOR_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _manager(num_envs=2, **kwargs):
    mod = _load_anchor()
    cfg = mod.AnchorConfig(lidar_hbeams=2, lidar_vbeams=3, **kwargs)
    return mod, mod.MeasurementSpaceAnchorManager(cfg, num_envs=num_envs, device="cpu")


def _frame(num_envs=2, value=1.0, mask=True, weight=1.0):
    r = torch.full((num_envs, 2, 3), float(value))
    m = torch.full((num_envs, 2, 3), bool(mask), dtype=torch.bool)
    w = torch.full((num_envs, 2, 3), float(weight))
    v = torch.zeros(num_envs, 3)
    return r, m, w, v


def _step(manager, value=1.0, mask=True, weight=1.0, v_cmd=None):
    r, m, w, v = _frame(manager.state.active.shape[0], value, mask, weight)
    if v_cmd is not None:
        v = v_cmd
    return manager.step(r, m, w, v)


def test_config_validation_and_canonical_key_policy():
    mod = _load_anchor()
    mod.AnchorConfig(eps_enter=0.05, eps_exit=0.15, min_valid_anchor_fraction=0.10)
    mod.AnchorConfig(eps_enter=0.05, eps_exit=0.15, min_valid_anchor_fraction=1.0)
    for kwargs in [
        {"eps_enter": 0.2, "eps_exit": 0.2},
        {"min_valid_anchor_fraction": 0.0},
        {"min_valid_anchor_fraction": -0.1},
        {"min_valid_anchor_fraction": 1.1},
        {"min_valid_anchor_fraction": float("nan")},
        {"huber_delta": 0.0},
    ]:
        try:
            mod.AnchorConfig(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {kwargs}")

    bad_cfg = SimpleNamespace(min_valid_fraction=0.1)
    try:
        mod.AnchorConfig.from_namespace(bad_cfg, lidar_hbeams=2, lidar_vbeams=3)
    except ValueError as exc:
        assert "min_valid_anchor_fraction" in str(exc)
    else:
        raise AssertionError("unsupported min_valid_fraction alias must fail")


def test_huber_loss_is_pure_per_element_standard_huber():
    mod = _load_anchor()
    x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=torch.float64)
    out = mod.huber_loss(x, 1.0)
    expected = torch.tensor([1.5, 0.125, 0.0, 0.125, 1.5], dtype=torch.float64)
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert out.device == x.device
    assert torch.allclose(out, expected)
    try:
        mod.huber_loss(x, 0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("delta <= 0 must fail")


def test_activation_hysteresis_boundaries_and_hold_steps():
    mod, manager = _manager(num_envs=3)
    r, m, w, _ = _frame(3)
    v = torch.tensor([
        [0.05, 0.0, 0.0],
        [0.10, 0.0, 0.0],
        [0.00, 0.0, 0.0],
    ])
    out = manager.step(r, m, w, v)
    assert out.metrics["anchor_active"].squeeze(-1).tolist() == [1.0, 0.0, 1.0]
    assert out.metrics["anchor_hold_steps"].squeeze(-1).tolist() == [1.0, 0.0, 1.0]
    assert out.metrics["anchor_activation_count"].squeeze(-1).tolist() == [1.0, 0.0, 1.0]

    v_hold = torch.tensor([
        [0.149, 0.0, 0.0],
        [0.00, 0.0, 0.0],
        [0.149, 0.0, 0.0],
    ])
    out = manager.step(r + 0.1, m, w, v_hold)
    assert out.metrics["anchor_active"].squeeze(-1).tolist() == [1.0, 1.0, 1.0]
    assert out.metrics["anchor_hold_steps"].squeeze(-1).tolist() == [2.0, 1.0, 2.0]
    assert out.metrics["anchor_activation_count"].squeeze(-1).tolist() == [1.0, 1.0, 1.0]

    v_exit = torch.tensor([
        [0.15, 0.0, 0.0],
        [0.20, 0.0, 0.0],
        [0.149, 0.0, 0.0],
    ])
    out = manager.step(r + 0.2, m, w, v_exit)
    assert out.metrics["anchor_active"].squeeze(-1).tolist() == [0.0, 0.0, 1.0]
    assert out.metrics["anchor_reset_reason"].squeeze(-1).tolist() == [
        mod.ANCHOR_RESET_COMMAND,
        mod.ANCHOR_RESET_COMMAND,
        mod.ANCHOR_RESET_NONE,
    ]
    assert out.metrics["anchor_hold_steps"].squeeze(-1).tolist() == [0.0, 0.0, 3.0]


def test_capture_stores_frozen_reference_masks_and_weights():
    _, manager = _manager(num_envs=1)
    r, m, w, v = _frame(1, value=2.0)
    m[0, 0, 1] = False
    w[0, 1, 2] = 0.0
    out = manager.step(r, m, w, v)
    assert out.metrics["anchor_active"].item() == 1.0
    assert torch.allclose(manager.state.r_star, r)
    assert manager.state.m_star.dtype == torch.bool
    assert torch.equal(manager.state.m_star, m)
    assert torch.allclose(manager.state.w_star, w)

    _step(manager, value=5.0)
    assert torch.allclose(manager.state.r_star, r)
    assert torch.equal(manager.state.m_star, m)
    assert torch.allclose(manager.state.w_star, w)


def test_masked_error_loss_valid_fraction_and_weight_semantics():
    _, manager = _manager(num_envs=1, huber_delta=1.0, min_valid_anchor_fraction=0.01)
    r, m, w, v = _frame(1, value=1.0)
    w[0, 0, 0] = 0.0
    manager.step(r, m, w, v)

    r2 = torch.full((1, 2, 3), 3.0)
    m2 = torch.ones(1, 2, 3)
    w2 = torch.ones(1, 2, 3)
    w2[0, 0, 1] = 0.5
    out = manager.step(r2, m2, w2, torch.zeros(1, 3))

    expected_error = torch.full((1, 2, 3), 2.0)
    expected_error[0, 0, 1] = 1.0
    assert torch.allclose(out.cache["anchor_error"], expected_error)
    assert out.cache["usable_anchor_mask"].sum().item() == 5
    assert math.isclose(out.metrics["anchor_valid_fraction"].item(), 5.0 / 6.0, rel_tol=1e-6)
    # Huber(2)=1.5 on four beams, Huber(1)=0.5 on one beam, fixed denominator 6.
    assert math.isclose(out.metrics["anchor_loss"].item(), (4 * 1.5 + 0.5) / 6.0, rel_tol=1e-6)
    assert math.isclose(out.metrics["anchor_error_mean"].item(), (4 * 2.0 + 1.0) / 5.0, rel_tol=1e-6)
    assert math.isclose(out.metrics["anchor_error_max"].item(), 2.0, rel_tol=1e-6)


def test_invalid_reset_uses_post_transition_public_metrics():
    mod, manager = _manager(num_envs=1, min_valid_anchor_fraction=0.5)
    _step(manager, value=1.0)
    m = torch.zeros(1, 2, 3, dtype=torch.bool)
    out = manager.step(torch.ones(1, 2, 3), m, torch.ones(1, 2, 3), torch.zeros(1, 3))

    assert out.metrics["anchor_reset_reason"].item() == mod.ANCHOR_RESET_INVALID
    assert out.metrics["anchor_active"].item() == 0.0
    assert out.metrics["anchor_loss"].item() == 0.0
    assert out.metrics["anchor_error_mean"].item() == 0.0
    assert out.metrics["anchor_error_max"].item() == 0.0
    assert out.metrics["anchor_hold_steps"].item() == 0.0
    assert not manager.state.active.item()


def test_reset_priority_and_activation_count_rules():
    mod, manager = _manager(num_envs=5, min_valid_anchor_fraction=0.5)
    _step(manager, value=1.0)
    assert manager.state.activation_count.tolist() == [1, 1, 1, 1, 1]

    r, m, w, v = _frame(5, value=2.0)
    m[2:] = False
    v[0] = torch.tensor([0.2, 0.0, 0.0])
    v[2] = torch.tensor([0.2, 0.0, 0.0])
    explicit = torch.tensor([False, True, False, False, False])
    episode = torch.tensor([True, False, False, False, False])
    out = manager.step(r, m, w, v, explicit_reset_mask=explicit, episode_reset_mask=episode)

    assert out.metrics["anchor_reset_reason"].squeeze(-1).tolist() == [
        mod.ANCHOR_RESET_EPISODE,
        mod.ANCHOR_RESET_EXPLICIT,
        mod.ANCHOR_RESET_COMMAND,
        mod.ANCHOR_RESET_INVALID,
        mod.ANCHOR_RESET_INVALID,
    ]
    assert manager.state.activation_count.tolist() == [0, 1, 1, 1, 1]

    manager.reset(torch.tensor([1]), reason=mod.ANCHOR_RESET_EPISODE)
    assert manager.state.activation_count.tolist() == [0, 0, 1, 1, 1]
    manager.reset(env_ids=None, reason=mod.ANCHOR_RESET_EPISODE)
    assert manager.state.activation_count.tolist() == [0, 0, 0, 0, 0]

    # Pending direct episode reset still outranks same-step explicit masks.
    out = manager.step(
        r,
        torch.ones_like(m, dtype=torch.bool),
        w,
        torch.zeros(5, 3),
        explicit_reset_mask=torch.ones(5, dtype=torch.bool),
    )
    assert out.metrics["anchor_reset_reason"].squeeze(-1).tolist() == [
        mod.ANCHOR_RESET_EPISODE,
        mod.ANCHOR_RESET_EPISODE,
        mod.ANCHOR_RESET_EPISODE,
        mod.ANCHOR_RESET_EPISODE,
        mod.ANCHOR_RESET_EPISODE,
    ]


def test_structural_mask_controls_denominators_and_padding():
    mod = _load_anchor()
    structural = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.float32)
    cfg = mod.AnchorConfig(lidar_hbeams=2, lidar_vbeams=3, min_valid_anchor_fraction=0.01, huber_delta=1.0)
    manager = mod.MeasurementSpaceAnchorManager(cfg, num_envs=1, device="cpu", structural_mask=structural)
    assert manager.structural_mask.dtype == torch.bool
    assert manager.structural_mask.sum().item() == 3

    _step(manager, value=1.0)
    out = _step(manager, value=3.0)
    assert math.isclose(out.metrics["anchor_valid_fraction"].item(), 1.0)
    assert math.isclose(out.metrics["anchor_loss"].item(), 1.5)
    assert out.cache["usable_anchor_mask"].sum().item() == 3

    for bad in [torch.ones(1, 2, 3), torch.ones(3, 2), torch.tensor([[float("nan"), 1, 1], [1, 1, 1]])]:
        try:
            mod.MeasurementSpaceAnchorManager(cfg, num_envs=1, device="cpu", structural_mask=bad)
        except ValueError:
            pass
        else:
            raise AssertionError("bad structural mask must fail")


def test_fail_fast_validation_and_weight_clamping():
    _, manager = _manager(num_envs=1)
    r, m, w, v = _frame(1)
    manager.step(r, m, w * 2.0, v)
    assert torch.all(manager.state.w_star == 1.0)

    bad_cases = [
        (r.reshape(1, 6), m, w, v, ValueError),
        (r, m.reshape(1, 6), w, v, ValueError),
        (r, m, w.reshape(1, 6), v, ValueError),
        (r, m, w, torch.zeros(1, 2), ValueError),
        (torch.full_like(r, float("nan")), m, w, v, ValueError),
        (r, torch.full_like(r, float("nan")), w, v, ValueError),
        (r, m, torch.full_like(w, float("inf")), v, ValueError),
    ]
    for bad_r, bad_m, bad_w, bad_v, exc_type in bad_cases:
        try:
            manager.step(bad_r, bad_m, bad_w, bad_v)
        except exc_type:
            pass
        else:
            raise AssertionError("bad input should fail fast")

    manager.step(r, m, w, torch.zeros(1, 1, 3))
    try:
        manager.step(r, m, w, v, explicit_reset_mask=torch.zeros(1, 1, dtype=torch.bool))
    except ValueError:
        pass
    else:
        raise AssertionError("[N,1] reset mask must fail")
    try:
        manager.step(r, m, w, v, explicit_reset_mask=torch.zeros(1))
    except TypeError:
        pass
    else:
        raise AssertionError("non-bool reset mask must fail")


def test_step_output_metrics_cache_and_actor_contract_boundary():
    mod, manager = _manager(num_envs=2)
    out = _step(manager, value=1.0)
    assert isinstance(out, mod.AnchorStepOutput)
    assert set(out.metrics) == set(mod.ANCHOR_METRIC_KEYS)
    for key, value in out.metrics.items():
        assert value.shape == (2, 1), key
    assert out.metrics["anchor_reset_reason"].dtype == torch.long
    assert "anchor_error" in out.cache
    assert "usable_anchor_mask" in out.cache
    assert out.cache["anchor_error"].shape == (2, 2, 3)
    assert out.cache["usable_anchor_mask"].shape == (2, 2, 3)
    assert not any(key in out.metrics for key in out.cache)


def test_env_integration_source_keeps_dense_anchor_out_of_actor_obs_and_info():
    source = open(ENV_PATH, encoding="utf-8").read()
    for key in [
        "anchor_active",
        "anchor_loss",
        "anchor_valid_fraction",
        "anchor_error_mean",
        "anchor_error_max",
        "anchor_hold_steps",
        "anchor_activation_count",
        "anchor_reset_reason",
    ]:
        assert key in source
    assert "self.anchor_outputs = anchor_out.cache" in source
    assert "for key, value in anchor_out.metrics.items()" in source

    actor_block = source.split("# -----------------Network Input Final--------------", 1)[1]
    actor_block = actor_block.split("# ============================================", 1)[0]
    assert '"lidar_grid": obs_hist["lidar_grid"]' in actor_block
    assert '"state_vec": obs_hist["state_vec"]' in actor_block
    assert "anchor_" not in actor_block
