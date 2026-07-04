import importlib.util
import os
import sys

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
