import importlib.util
import os
import sys
from types import SimpleNamespace

import torch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPTS = os.path.join(ROOT, "training", "scripts")
MODULE_PATH = os.path.join(SCRIPTS, "instinctRL", "mid360_pattern.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("instinctrl_mid360_pattern_test", MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_mid360_ray_count_shape_and_unit_directions():
    mod = _load_module()
    cfg = mod.LivoxMid360RayPatternCfg(horizontal_res=10.0, num_vertical_lines=12)
    starts, dirs = cfg.func(cfg, "cpu")
    assert starts.shape == (36 * 12, 3)
    assert dirs.shape == (36 * 12, 3)
    assert torch.allclose(dirs.norm(dim=-1), torch.ones(36 * 12), atol=1e-5)


def test_mid360_ray_order_is_deterministic():
    mod = _load_module()
    cfg = mod.LivoxMid360RayPatternCfg(horizontal_res=5.0, num_vertical_lines=15)
    _, dirs_a = cfg.func(cfg, "cpu")
    _, dirs_b = cfg.func(cfg, "cpu")
    assert torch.allclose(dirs_a, dirs_b)
    assert mod.ray_order_hash(dirs_a) == mod.ray_order_hash(dirs_b)


def test_create_cfg_uses_sensor_fov_and_count():
    mod = _load_module()
    sensor = SimpleNamespace(
        lidar_range=40.0,
        lidar_vfov=[-7.0, 52.0],
        lidar_hres=1.0,
        lidar_vbeams=59,
    )
    cfg = mod.create_mid360_pattern_cfg(sensor)
    assert cfg.num_horizontal_rays == 360
    assert cfg.num_vertical_lines == 59
    assert cfg.num_rays == 360 * 59
