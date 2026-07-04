import importlib.util
import math
import os
import sys

import torch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPTS = os.path.join(ROOT, "training", "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)


def _load_adapter():
    path = os.path.join(SCRIPTS, "instinctRL", "command_adapter.py")
    spec = importlib.util.spec_from_file_location("instinctrl_command_adapter_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.BodyToWorldVelocityAdapter


def _yaw_quat(deg):
    half = math.radians(deg) * 0.5
    return torch.tensor([[math.cos(half), 0.0, 0.0, math.sin(half)]])


def _rpy_quat(roll, pitch, yaw):
    r, p, y = [math.radians(v) * 0.5 for v in (roll, pitch, yaw)]
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return torch.tensor([[
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]])


def test_identity_keeps_body_velocity():
    adapter = _load_adapter()()
    body = torch.tensor([[1.0, -2.0, 0.5]])
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    assert torch.allclose(adapter(body, quat), body, atol=1e-6)


def test_yaw_90_rotates_body_x_to_world_y():
    adapter = _load_adapter()()
    world = adapter(torch.tensor([[1.0, 0.0, 0.0]]), _yaw_quat(90.0))
    assert torch.allclose(world, torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-5)


def test_roll_pitch_case_matches_rotation_matrix_expectation():
    adapter = _load_adapter()()
    quat = _rpy_quat(30.0, 20.0, 0.0)
    body_z = torch.tensor([[0.0, 0.0, 1.0]])
    expected = torch.tensor([[0.29619813, -0.5, 0.8137977]])
    assert torch.allclose(adapter(body_z, quat), expected, atol=1e-5)
