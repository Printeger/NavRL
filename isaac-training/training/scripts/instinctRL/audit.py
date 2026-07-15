"""
instinctRL Audit Module
=======================
Platform, actor-input, rollout, and checkpoint audit checks.
"""

import os
import torch
from typing import Dict, List, Tuple, Optional


# Forbidden substrings that must NOT appear in actor observation keys
FORBIDDEN_ACTOR_KEY_PATTERNS = [
    "pose", "pos", "position", "odom", "velocity",
    "vel_g", "vel_w", "root_state", "map", "slam",
    "privileged", "direction", "distance", "dynamic_obstacle",
    "r5e", "r5e1", "r5e2", "r5e3", "evidence1", "evidence2", "evidence3", "r5f", "r5g", "r5h", "collision_window",
    "governor", "ics", "clearance", "height", "root",
    "world_z", "root_height", "safety_filter",
]


def _get_path(container, path, default=None):
    try:
        if hasattr(container, "get"):
            value = container.get(path, None)
            if value is not None:
                return value
        current = container
        for key in path:
            current = current[key]
        return current
    except Exception:
        return default


def check_platform_lock(cfg) -> Tuple[bool, str]:
    """
    Verify that the platform is locked to TASLAB_UAV + Livox MID360.

    Returns:
        (passed, message)
    """
    messages = []

    # Drone model check
    model_name = getattr(cfg.drone, "model_name", None)
    if model_name != "TaslabUAV":
        messages.append(
            f"PLATFORM AUDIT FAIL: drone.model_name={model_name}, expected 'TaslabUAV'"
        )
    else:
        messages.append("PLATFORM AUDIT PASS: drone.model_name='TaslabUAV'")

    # MID360 sensor check — verify MID360-specific FOV, not generic
    lidar_vfov = getattr(cfg.sensor, "lidar_vfov", None)
    lidar_range = getattr(cfg.sensor, "lidar_range", None)
    if lidar_vfov and abs(lidar_vfov[0] - (-7.0)) < 0.5 and abs(lidar_vfov[1] - 52.0) < 0.5:
        messages.append("PLATFORM AUDIT PASS: sensor matches Livox MID360 FOV [-7°, 52°]")
    else:
        messages.append(
            f"PLATFORM AUDIT WARN: sensor FOV={lidar_vfov}, expected [-7°, 52°] for MID360"
        )
    if lidar_range and abs(lidar_range - 40.0) < 1.0:
        messages.append("PLATFORM AUDIT PASS: lidar_range=40m (MID360)")
    else:
        messages.append(
            f"PLATFORM AUDIT WARN: lidar_range={lidar_range}, expected 40m for MID360"
        )

    passed = "FAIL" not in " ".join(messages)
    return passed, " | ".join(messages)


def check_actor_input(tensordict) -> Tuple[bool, str]:
    """
    Scan actor observation keys for forbidden substrings.

    The actor must not receive pose, odometry, explicit velocity,
    map, SLAM state, or privileged simulator state.

    Args:
        tensordict: The observation TensorDict passed to the policy.

    Returns:
        (passed, message)
    """
    violations = []

    def _scan_keys(d, path=""):
        if hasattr(d, "keys"):
            for k in d.keys():
                full_path = f"{path}/{k}" if path else str(k)
                key_lower = str(k).lower()
                for pattern in FORBIDDEN_ACTOR_KEY_PATTERNS:
                    if pattern in key_lower:
                        violations.append(f"  '{full_path}' contains forbidden pattern '{pattern}'")
                if hasattr(d[k], "keys"):
                    _scan_keys(d[k], full_path)

    try:
        # Navigate to agents.observation
        obs = _get_path(tensordict, ("agents", "observation"))
        if obs is not None:
            _scan_keys(obs, "agents.observation")
    except Exception as e:
        violations.append(f"  Error scanning tensordict: {e}")

    if violations:
        return False, "ACTOR INPUT AUDIT FAIL:\n" + "\n".join(violations)
    return True, "ACTOR INPUT AUDIT PASS: no forbidden fields in actor observation"


def check_actor_schema(tensordict, history_len: int) -> Tuple[bool, str]:
    """Verify instinctRL-B actor observation keys and low-dimensional schema."""
    try:
        obs = _get_path(tensordict, ("agents", "observation"))
        if obs is None:
            return False, "ACTOR SCHEMA AUDIT FAIL: missing agents.observation"
        keys = set(obs.keys())
        expected = {"lidar_grid", "state_vec"}
        if keys != expected:
            return False, f"ACTOR SCHEMA AUDIT FAIL: keys={sorted(keys)}, expected={sorted(expected)}"
        state_dim = obs["state_vec"].shape[-1]
        expected_state_dim = history_len * 13
        if state_dim != expected_state_dim:
            return False, (
                f"ACTOR SCHEMA AUDIT FAIL: state_vec dim={state_dim}, "
                f"expected {expected_state_dim} (history * 13)"
            )
        lidar_channels = obs["lidar_grid"].shape[-3]
        expected_channels = history_len * 3
        if lidar_channels != expected_channels:
            return False, (
                f"ACTOR SCHEMA AUDIT FAIL: lidar_grid channels={lidar_channels}, "
                f"expected {expected_channels} (history * range/mask/weight)"
            )
    except Exception as exc:
        return False, f"ACTOR SCHEMA AUDIT FAIL: {exc}"
    return True, "ACTOR SCHEMA AUDIT PASS: lidar_grid + state_vec only, expected history schema"


def check_action_type(action: torch.Tensor, expected_dim: int = 3) -> Tuple[bool, str]:
    """
    Verify action is velocity-command based (3-dim), not CTBR (4-dim) or motor thrust.

    Args:
        action: Action tensor, shape [..., action_dim]
        expected_dim: Expected action dimension (3 for velocity)

    Returns:
        (passed, message)
    """
    if action is None:
        return True, "ACTION TYPE AUDIT SKIP: no action to check"

    actual_dim = action.shape[-1]
    if actual_dim == expected_dim:
        return True, f"ACTION TYPE AUDIT PASS: {actual_dim}-dim velocity command"
    elif actual_dim == 4:
        return False, (
            f"ACTION TYPE AUDIT FAIL: {actual_dim}-dim action detected. "
            f"CTBR/body-rate/motor-thrust actions are forbidden. Expected {expected_dim}-dim velocity."
        )
    else:
        return False, (
            f"ACTION TYPE AUDIT FAIL: unexpected action dim={actual_dim}. "
            f"Expected {expected_dim}-dim velocity command."
        )


def require_actor_contract(tensordict, history_len: int) -> None:
    """Raise if actor observation schema or forbidden-key scan fails."""
    actor_ok, actor_msg = check_actor_input(tensordict)
    if not actor_ok:
        raise RuntimeError(actor_msg)
    schema_ok, schema_msg = check_actor_schema(tensordict, history_len)
    if not schema_ok:
        raise RuntimeError(schema_msg)


def _require_finite(name: str, tensor: torch.Tensor) -> None:
    if tensor is None:
        raise RuntimeError(f"AUDIT FAIL: missing tensor {name}")
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"AUDIT FAIL: tensor {name} contains non-finite values")


def audit_policy_init(policy, tensordict, cfg) -> Dict[str, bool]:
    """Run policy initialization audit and fail hard on actor/governor violations."""
    history_len = getattr(getattr(cfg, "instinctRL", None), "observation", None)
    history_len = getattr(history_len, "history_len", 4)
    require_actor_contract(tensordict, history_len)
    with torch.no_grad():
        out = policy(tensordict.clone())
    _require_finite("agents.action", out["agents", "action"])
    if getattr(policy, "learned_governor", False):
        for key in ("governor_alpha", "governor_v_corr", "governor_v_gov_b"):
            _require_finite(key, out[key])
        alpha = out["governor_alpha"]
        if not ((alpha >= 0.0).all() and (alpha <= 1.0).all()):
            raise RuntimeError("AUDIT FAIL: governor_alpha outside [0, 1]")
        action_normalized = out["agents", "action_normalized"]
        if action_normalized.shape[-1] != 4:
            raise RuntimeError(
                f"AUDIT FAIL: learned governor action dim={action_normalized.shape[-1]}, expected 4"
            )
    return {"policy_init": True, "actor_contract": True}


def audit_rollout_batch(data, cfg) -> Dict[str, bool]:
    """Validate a collected rollout contains clean actor obs, finite rewards, and PPO keys."""
    history_len = getattr(getattr(cfg, "instinctRL", None), "observation", None)
    history_len = getattr(history_len, "history_len", 4)
    require_actor_contract(data, history_len)
    if "next" in data.keys():
        require_actor_contract(data["next"], history_len)
    for key in (("agents", "action_normalized"), ("sample_log_prob",), ("state_value",)):
        _require_finite(str(key), _get_path(data, key))
    _require_finite("next.agents.reward", _get_path(data, ("next", "agents", "reward")))
    action_normalized = _get_path(data, ("agents", "action_normalized"))
    gov_cfg = getattr(getattr(getattr(cfg, "algo", None), "instinctRL", None), "governor", None)
    if getattr(gov_cfg, "alpha_mode", "fixed") == "learned" and action_normalized.shape[-1] != 4:
        raise RuntimeError(
            f"AUDIT FAIL: learned governor rollout action dim={action_normalized.shape[-1]}, expected 4"
        )
    return {"rollout_batch": True}


def audit_checkpoint_file(path: str) -> Dict[str, bool]:
    """Validate that a checkpoint file exists, is non-empty, and can be loaded by torch."""
    if not os.path.exists(path):
        raise RuntimeError(f"CHECKPOINT AUDIT FAIL: missing file {path}")
    if os.path.getsize(path) <= 0:
        raise RuntimeError(f"CHECKPOINT AUDIT FAIL: empty file {path}")
    try:
        torch.load(path, map_location="cpu")
    except Exception as exc:
        raise RuntimeError(f"CHECKPOINT AUDIT FAIL: torch.load failed: {exc}") from exc
    return {"checkpoint_file": True}


def run_audit(
    cfg,
    actor_tensordict=None,
    action: Optional[torch.Tensor] = None,
) -> Dict[str, bool]:
    """
    Run all instinctRL-A audit checks.

    Returns:
        Dict with check names as keys and pass/fail as bool values.
    """
    results = {}

    platform_ok, platform_msg = check_platform_lock(cfg)
    results["platform_lock"] = platform_ok
    print(f"[instinctRL audit] {platform_msg}")

    if actor_tensordict is not None:
        actor_ok, actor_msg = check_actor_input(actor_tensordict)
        results["actor_input"] = actor_ok
        print(f"[instinctRL audit] {actor_msg}")

    if action is not None:
        action_ok, action_msg = check_action_type(action)
        results["action_type"] = action_ok
        print(f"[instinctRL audit] {action_msg}")

    all_pass = all(results.values()) if results else True
    if not all_pass:
        failed = [k for k, v in results.items() if not v]
        raise RuntimeError(
            f"[instinctRL audit] AUDIT FAILED: {failed}. "
            f"Aborting — platform/sensor/actor contract violated."
        )

    print("[instinctRL audit] All checks passed.")
    return results
