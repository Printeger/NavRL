import importlib.util
import os
import sys

import torch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPTS = os.path.join(ROOT, "training", "scripts")
MODULE_PATH = os.path.join(SCRIPTS, "instinctRL", "observation.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("instinctrl_observation_test", MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _builder(history_len=2):
    mod = _load_module()
    cfg = mod.ObservationConfig(
        history_len=history_len,
        lidar_hbeams=2,
        lidar_vbeams=2,
        lidar_range=10.0,
        tau_staleness=0.5,
    )
    return mod.MID360ObservationBuilder(cfg, device="cpu")


def _state(num_envs):
    state = torch.zeros(num_envs, 13)
    state[:, 3] = 1.0
    state[:, 10:13] = torch.tensor([0.1, 0.2, 0.3])
    return state


def test_raw_range_mask_and_weight_bounds():
    builder = _builder()
    ray_hits = torch.tensor([[
        [1.0, 0.0, 0.0],
        [float("inf"), 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [0.001, 0.0, 0.0],
    ]])
    obs = builder.build(
        ray_hits,
        torch.zeros(1, 3),
        _state(1),
        torch.zeros(1, 3),
        dt=0.02,
        num_envs=1,
        prev_action=torch.zeros(1, 3),
    )
    assert obs["range"].shape == (1, 2, 2)
    assert obs["range"][0, 0, 0].item() == 1.0
    assert obs["mask"].reshape(-1).tolist() == [1.0, 0.0, 0.0, 0.0]
    assert torch.all((obs["weight"] >= 0.0) & (obs["weight"] <= 1.0))
    assert obs["weight"].reshape(-1)[1:].sum().item() == 0.0


def test_prev_action_is_required_and_recorded_in_history():
    builder = _builder()
    ray_hits = torch.ones(1, 4, 3)
    try:
        builder.build(ray_hits, torch.zeros(1, 3), _state(1), torch.zeros(1, 3), 0.02, 1)
    except ValueError:
        pass
    else:
        raise AssertionError("prev_action must be explicit")

    prev_action = torch.tensor([[0.3, -0.2, 0.1]])
    obs = builder.build(
        ray_hits,
        torch.zeros(1, 3),
        _state(1),
        torch.zeros(1, 3),
        0.02,
        1,
        prev_action=prev_action,
    )
    hist = builder.build_history(obs)
    state_hist = hist["state_vec"].reshape(1, 2, 13)
    assert torch.allclose(state_hist[0, -1, 9:12], prev_action[0])


def test_timestamp_monotonicity_stale_flag_and_history_rollover():
    builder = _builder()
    ray_hits = torch.ones(1, 4, 3)
    obs1 = builder.build(ray_hits, torch.zeros(1, 3), _state(1), torch.zeros(1, 3), 0.02, 1, torch.zeros(1, 3))
    hist1 = builder.build_history(obs1)
    obs2 = builder.build(ray_hits * 2.0, torch.zeros(1, 3), _state(1), torch.zeros(1, 3), 0.02, 1, torch.ones(1, 3))
    hist2 = builder.build_history(obs2)
    obs3 = builder.build(ray_hits * 3.0, torch.zeros(1, 3), _state(1), torch.zeros(1, 3), 0.0, 1, torch.ones(1, 3))

    assert obs2["sim_time"].item() > obs1["sim_time"].item()
    assert obs3["is_stale"].item() is True
    assert torch.allclose(hist2["lidar_grid"][:, 0], hist1["lidar_grid"][:, 3])


def test_reset_selected_env_history_clears_only_that_env():
    builder = _builder()
    ray_hits = torch.ones(2, 4, 3)
    obs1 = builder.build(ray_hits, torch.zeros(2, 3), _state(2), torch.zeros(2, 3), 0.02, 2, torch.ones(2, 3))
    builder.build_history(obs1)
    builder.reset_history(torch.tensor([1]))
    obs2 = builder.build(ray_hits * 2.0, torch.zeros(2, 3), _state(2), torch.zeros(2, 3), 0.02, 2, torch.full((2, 3), 2.0))
    hist = builder.build_history(obs2)["state_vec"].reshape(2, 2, 13)

    assert torch.allclose(hist[0, 0, 9:12], torch.ones(3))
    assert torch.allclose(hist[1, 0], torch.zeros(13))
