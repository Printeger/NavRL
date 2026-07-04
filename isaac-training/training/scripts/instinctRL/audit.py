"""
instinctRL Audit Module
=======================
Staged platform, actor-input, and action-type audit checks.

Runs at environment construction (instinctRL-A).
Full rollout/eval/checkpoint/ROS hooks deferred to instinctRL-F.

instinctRL-0 / instinctRL-A
"""

import torch
from typing import Dict, List, Tuple, Optional


# Forbidden substrings that must NOT appear in actor observation keys
FORBIDDEN_ACTOR_KEY_PATTERNS = [
    "pose", "pos", "position", "odom", "velocity",
    "vel_g", "vel_w", "root_state", "map", "slam",
    "privileged", "direction", "distance", "dynamic_obstacle",
]


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
        obs = tensordict.get(("agents", "observation"), None)
        if obs is not None:
            _scan_keys(obs, "agents.observation")
    except Exception as e:
        violations.append(f"  Error scanning tensordict: {e}")

    if violations:
        return False, "ACTOR INPUT AUDIT FAIL:\n" + "\n".join(violations)
    return True, "ACTOR INPUT AUDIT PASS: no forbidden fields in actor observation"


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
