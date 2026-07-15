"""Task semantics and handbook metrics for instinctRL.

This module is intentionally torch-only so train/eval semantics can be tested
without launching Isaac Sim.
"""

import math
from typing import Dict, Iterable, List, Optional, Sequence

import torch


COMMAND_MODE_NORMAL = 0
COMMAND_MODE_AGGRESSIVE = 1
COMMAND_MODE_ADVERSARIAL = 2
COMMAND_MODE_OSCILLATION = 3
COMMAND_MODE_RECOVERY = 4
COMMAND_MODE_COUNT = 5

TERMINATION_NONE = 0
TERMINATION_BELOW_BOUND = 1
TERMINATION_ABOVE_BOUND = 2
TERMINATION_COLLISION = 3
TERMINATION_TIMEOUT = 4

R5H_CONDITION_NAMES = (
    "collision",
    "ics_violation",
    "downward_active",
    "low_beta",
    "emergency",
    "near_floor",
)

R5H_CONCENTRATION_VALUE_NAMES = (
    "ics_beta",
    "ics_downward_beta",
    "ics_active_beam_count",
    "ics_downward_min_clearance",
    "min_clearance",
    "v_cmd_xy_norm",
    "v_cmd_z_abs",
    "v_gov_xy_norm",
    "v_gov_z_abs",
    "v_final_xy_norm",
    "v_final_z_abs",
    "actual_xy_speed",
    "actual_z_abs",
)

R5H_CONCENTRATION_SAMPLE_VALUE_NAMES = (
    "ics_downward_min_clearance",
    "min_clearance",
)

R5H_STATION_VALUE_NAMES = (
    "actual_speed_xy",
    "actual_speed_z_abs",
    "v_gov_speed_xy",
    "v_gov_speed_z_abs",
    "v_final_speed_xy",
    "v_final_speed_z_abs",
    "actual_v_gov_mismatch_xy",
    "actual_v_gov_mismatch_z_abs",
    "actual_v_final_mismatch_xy",
    "actual_v_final_mismatch_z_abs",
    "alpha",
    "v_corr_norm",
    "v_corr_z_abs",
    "prev_action_speed_xy",
    "prev_action_speed_z_abs",
    "prev_action_v_final_mismatch_xy",
    "prev_action_v_final_mismatch_z_abs",
    "prev_action_actual_mismatch_xy",
    "prev_action_actual_mismatch_z_abs",
    "prev_action_v_final_alignment_xy",
)

R5H_ANCHOR_CONDITION_NAMES = ("active", "valid", "high_loss")

R5H_ANCHOR_VALUE_NAMES = (
    "station_drift",
    "null_speed",
    "actual_xy_speed",
    "actual_z_abs",
    "anchor_error",
    "anchor_loss",
)

R5H_TRACKING_FIELD_NAMES = (
    "r5h_tracking_active",
    "r5h_tracking_pre_ics_preservation",
    "r5h_tracking_post_ics_preservation",
    "r5h_tracking_governor_preservation_loss",
    "r5h_tracking_post_ics_preservation_loss",
    "r5h_tracking_horizontal_active",
    "r5h_tracking_horizontal_pre_ics_preservation",
    "r5h_tracking_horizontal_post_ics_preservation",
    "r5h_tracking_horizontal_governor_preservation_loss",
    "r5h_tracking_horizontal_post_ics_preservation_loss",
    "r5h_tracking_vertical_active",
    "r5h_tracking_vertical_pre_ics_preservation",
    "r5h_tracking_vertical_post_ics_preservation",
    "r5h_tracking_vertical_governor_preservation_loss",
    "r5h_tracking_vertical_post_ics_preservation_loss",
)

R5H_DIAGNOSTIC_FIELDS = (
    tuple(f"r5h_{condition}" for condition in R5H_CONDITION_NAMES)
    + tuple(
        f"r5h_{value_name}_when_{condition}"
        for condition in R5H_CONDITION_NAMES
        for value_name in R5H_CONCENTRATION_VALUE_NAMES
    )
    + tuple(
        f"r5h_{value_name}_sample_when_{condition}"
        for condition in R5H_CONDITION_NAMES
        for value_name in R5H_CONCENTRATION_SAMPLE_VALUE_NAMES
    )
    + ("r5h_station_null_command",)
    + tuple(
        f"r5h_station_null_{value_name}"
        for value_name in R5H_STATION_VALUE_NAMES
    )
    + tuple(f"r5h_anchor_{condition}" for condition in R5H_ANCHOR_CONDITION_NAMES)
    + tuple(
        f"r5h_anchor_{value_name}_when_{condition}"
        for condition in R5H_ANCHOR_CONDITION_NAMES
        for value_name in R5H_ANCHOR_VALUE_NAMES
    )
    + R5H_TRACKING_FIELD_NAMES
)

R5H_COLLISION_WINDOW_STEPS = (25, 50)
R5H_COLLISION_WINDOW_VALUE_FIELDS = (
    "min_clearance",
    "ics_beta",
    "ics_downward_beta",
    "ics_active_beam_count",
    "v_cmd_xy_norm",
    "v_cmd_z_abs",
    "v_gov_xy_norm",
    "v_gov_z_abs",
    "v_final_xy_norm",
    "v_final_z_abs",
    "actual_xy_speed",
    "actual_z_abs",
    "near_floor",
    "downward_active",
)


DEFAULT_COMMAND_CURRICULUM = (
    (0, (0.55, 0.0, 0.0, 0.15, 0.30)),
    (500_000, (0.45, 0.20, 0.05, 0.20, 0.10)),
    (2_000_000, (0.35, 0.25, 0.15, 0.20, 0.05)),
)

STATION_FIRST_COMMAND_CURRICULUM = (
    (0, (0.20, 0.0, 0.0, 0.10, 0.70)),
    (1_000_000, (0.45, 0.10, 0.02, 0.18, 0.25)),
    (3_000_000, (0.45, 0.20, 0.05, 0.20, 0.10)),
)

DIAGNOSTIC_MIXED_COMMAND_CURRICULUM = (
    (0, (0.45, 0.20, 0.05, 0.20, 0.10)),
)

COMMAND_CURRICULUM_PROFILES = {
    "default": DEFAULT_COMMAND_CURRICULUM,
    "legacy": DEFAULT_COMMAND_CURRICULUM,
    "station_first": STATION_FIRST_COMMAND_CURRICULUM,
    "diagnostic_mixed": DIAGNOSTIC_MIXED_COMMAND_CURRICULUM,
}


def _as_vector(value: torch.Tensor, name: str) -> torch.Tensor:
    if value.dim() == 3 and value.shape[-2] == 1:
        value = value.squeeze(-2)
    if value.dim() != 2 or value.shape[-1] != 3:
        raise ValueError(f"{name} must have shape [N,3] or [N,1,3]")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    return value


def _as_scalar(
    value: Optional[torch.Tensor],
    name: str,
    N: int,
    default: float = 0.0,
    *,
    device=None,
) -> torch.Tensor:
    if value is None:
        return torch.full((N, 1), default, dtype=torch.float32, device=device)
    if value.dim() == 1:
        value = value.reshape(N, 1)
    elif value.dim() == 3 and value.shape == (N, 1, 1):
        value = value.reshape(N, 1)
    elif value.dim() == 2 and value.shape == (N, 1):
        pass
    else:
        raise ValueError(f"{name} must have shape [N], [N,1], or [N,1,1]")
    if not torch.isfinite(value.float()).all():
        value = torch.nan_to_num(value.float(), nan=default, posinf=default, neginf=default)
    return value.float().to(device=device)


def _as_vector_flat(value: torch.Tensor, name: str) -> torch.Tensor:
    if value.dim() >= 3 and value.shape[-2:] == (1, 3):
        value = value.squeeze(-2)
    if value.dim() < 2 or value.shape[-1] != 3:
        raise ValueError(f"{name} must have trailing shape [3] or [1,3]")
    if not torch.isfinite(value.float()).all():
        raise ValueError(f"{name} must be finite")
    return value.float().reshape(-1, 3)


def _as_scalar_flat(
    value: Optional[torch.Tensor],
    name: str,
    N: int,
    default: float = 0.0,
    *,
    device=None,
) -> torch.Tensor:
    if value is None:
        return torch.full((N, 1), default, dtype=torch.float32, device=device)
    if value.numel() != N:
        raise ValueError(f"{name} must have {N} elements after flattening")
    value = value.float().reshape(N, 1)
    if not torch.isfinite(value).all():
        value = torch.nan_to_num(value, nan=default, posinf=default, neginf=default)
    return value.to(device=device)


def world_to_body_velocity(velocity_w: torch.Tensor, quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Rotate a world-frame velocity into the TASLAB body/governor frame."""
    velocity = _as_vector(velocity_w, "velocity_w")
    if quat_wxyz.dim() == 3 and quat_wxyz.shape[-2] == 1:
        quat_wxyz = quat_wxyz.squeeze(-2)
    if quat_wxyz.dim() != 2 or quat_wxyz.shape[-1] != 4:
        raise ValueError("quat_wxyz must have shape [N,4] or [N,1,4]")
    if quat_wxyz.shape[0] != velocity.shape[0]:
        raise ValueError("quat_wxyz batch size must match velocity_w")
    if not torch.isfinite(quat_wxyz).all():
        raise ValueError("quat_wxyz must be finite")

    q_w = quat_wxyz[:, 0]
    q_vec = quat_wxyz[:, 1:]
    a = velocity * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.linalg.cross(q_vec, velocity, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * (q_vec * velocity).sum(dim=-1, keepdim=True) * 2.0
    return a - b + c


def compute_termination_stats(
    *,
    below_bound: torch.Tensor,
    above_bound: torch.Tensor,
    collision: torch.Tensor,
    truncated: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Return explicit, mutually interpretable termination diagnostics."""
    below = below_bound.reshape(-1, 1).bool()
    above = above_bound.reshape(-1, 1).bool()
    coll = collision.reshape(-1, 1).bool()
    trunc = truncated.reshape(-1, 1).bool()
    code = torch.full_like(below, TERMINATION_NONE, dtype=torch.long)
    code = torch.where(trunc, torch.full_like(code, TERMINATION_TIMEOUT), code)
    code = torch.where(above, torch.full_like(code, TERMINATION_ABOVE_BOUND), code)
    code = torch.where(below, torch.full_like(code, TERMINATION_BELOW_BOUND), code)
    code = torch.where(coll, torch.full_like(code, TERMINATION_COLLISION), code)
    return {
        "terminated_below_bound": below.float(),
        "terminated_above_bound": above.float(),
        "terminated_collision": coll.float(),
        "truncated_timeout": trunc.float(),
        "termination_reason_code": code,
    }


def _normalize_probabilities(probabilities: Sequence[float]) -> List[float]:
    if len(probabilities) != COMMAND_MODE_COUNT:
        raise ValueError(f"command probabilities must have {COMMAND_MODE_COUNT} entries")
    values = [float(value) for value in probabilities]
    if any((not math.isfinite(value)) or value < 0.0 for value in values):
        raise ValueError("command probabilities must be finite and non-negative")
    total = sum(values)
    if total <= 0.0:
        raise ValueError("command probabilities must have positive total")
    return [value / total for value in values]


def command_curriculum_probabilities(
    frame_count: int,
    schedule: Optional[Iterable[Sequence[object]]] = None,
    *,
    profile: str = "default",
) -> List[float]:
    """Select command-mode probabilities for the current training frame."""
    if schedule is None:
        try:
            schedule = COMMAND_CURRICULUM_PROFILES[str(profile)]
        except KeyError as exc:
            supported = ", ".join(sorted(COMMAND_CURRICULUM_PROFILES))
            raise ValueError(
                f"Unsupported command curriculum profile {profile!r}; "
                f"expected one of {supported}"
            ) from exc
    frame = int(frame_count)
    selected = None
    for threshold, probabilities in schedule:
        if frame >= int(threshold):
            selected = probabilities
    if selected is None:
        selected = DEFAULT_COMMAND_CURRICULUM[0][1]
    return _normalize_probabilities(selected)


def command_mode_one_hot(mode: torch.Tensor) -> torch.Tensor:
    """Return one-hot command-mode diagnostics as float tensor [N,5]."""
    mode = mode.reshape(-1).long().clamp(0, COMMAND_MODE_COUNT - 1)
    return torch.nn.functional.one_hot(mode, COMMAND_MODE_COUNT).float()


def nearest_obstacle_vector_from_scan(
    *,
    ranges: torch.Tensor,
    mask: torch.Tensor,
    ray_directions_b: torch.Tensor,
    fallback_distance: float = 1.0,
) -> torch.Tensor:
    """Approximate nearest obstacle vector in body frame from MID360 returns."""
    N = ranges.shape[0]
    flat_range = ranges.reshape(N, -1)
    flat_mask = mask.reshape(N, -1).bool()
    valid_range = torch.where(
        flat_mask,
        flat_range,
        torch.full_like(flat_range, float("inf")),
    )
    min_range, index = valid_range.min(dim=1)
    ray_dirs = ray_directions_b.reshape(-1, 3).to(ranges.device)
    nearest_dir = ray_dirs[index]
    fallback = torch.tensor([1.0, 0.0, 0.0], dtype=ranges.dtype, device=ranges.device).expand(N, 3)
    finite = torch.isfinite(min_range).unsqueeze(-1)
    distance = torch.where(
        torch.isfinite(min_range),
        min_range,
        torch.full_like(min_range, float(fallback_distance)),
    )
    return torch.where(finite, nearest_dir * distance.unsqueeze(-1), fallback)


def compute_handbook_step_metrics(
    *,
    v_cmd_b: torch.Tensor,
    actual_velocity_b: torch.Tensor,
    v_final_b: torch.Tensor,
    min_clearance: torch.Tensor,
    height_w: Optional[torch.Tensor] = None,
    ics_beta: Optional[torch.Tensor] = None,
    ics_emergency: Optional[torch.Tensor] = None,
    anchor_active: Optional[torch.Tensor] = None,
    anchor_error_mean: Optional[torch.Tensor] = None,
    anchor_error_max: Optional[torch.Tensor] = None,
    anchor_loss: Optional[torch.Tensor] = None,
    collision: Optional[torch.Tensor] = None,
    d_safe: float = 0.8,
    height_floor: float = 0.5,
    height_ceiling: float = 4.0,
    command_eps: float = 1e-3,
) -> Dict[str, torch.Tensor]:
    """Per-step metrics that map directly to handbook eval summaries."""
    v_cmd = _as_vector(v_cmd_b, "v_cmd_b")
    N = v_cmd.shape[0]
    actual = _as_vector(actual_velocity_b, "actual_velocity_b")
    v_final = _as_vector(v_final_b, "v_final_b")
    device = v_cmd.device
    min_clearance = _as_scalar(min_clearance, "min_clearance", N, default=float("inf"), device=device)
    beta = _as_scalar(ics_beta, "ics_beta", N, default=1.0, device=device).clamp(0.0, 1.0)
    emergency = _as_scalar(ics_emergency, "ics_emergency", N, default=0.0, device=device).clamp(0.0, 1.0)
    active = _as_scalar(anchor_active, "anchor_active", N, default=0.0, device=device).clamp(0.0, 1.0)
    anchor_mean = _as_scalar(anchor_error_mean, "anchor_error_mean", N, default=0.0, device=device).clamp_min(0.0)
    anchor_max = _as_scalar(anchor_error_max, "anchor_error_max", N, default=0.0, device=device).clamp_min(0.0)
    anchor_loss = _as_scalar(anchor_loss, "anchor_loss", N, default=0.0, device=device).clamp_min(0.0)
    collision = _as_scalar(collision, "collision", N, default=0.0, device=device).clamp(0.0, 1.0)
    floor = float(height_floor)
    ceiling = float(height_ceiling)
    if not math.isfinite(floor) or floor < 0.0:
        raise ValueError("height_floor must be finite and >= 0")
    if not math.isfinite(ceiling) or ceiling < floor:
        raise ValueError("height_ceiling must be finite and >= height_floor")
    height = _as_scalar(height_w, "height_w", N, default=floor, device=device)

    command_active = (v_cmd.norm(dim=-1, keepdim=True) > command_eps).float()
    null_command = 1.0 - command_active
    actual_error_sq = ((actual - v_cmd) ** 2).sum(dim=-1, keepdim=True) * command_active
    proxy_error_sq = ((v_final - v_cmd) ** 2).sum(dim=-1, keepdim=True) * command_active
    command_norm = v_cmd.norm(dim=-1, keepdim=True)
    preservation = torch.where(
        command_norm > command_eps,
        (v_final.norm(dim=-1, keepdim=True) / command_norm.clamp_min(command_eps)).clamp(0.0, 2.0),
        torch.ones_like(command_norm),
    )
    command_safe_gate = command_active * (beta >= 0.999).float() * (1.0 - emergency)
    amplification = (preservation - 1.0).clamp_min(0.0) * command_safe_gate
    horizontal_cmd_norm = v_cmd[..., :2].norm(dim=-1, keepdim=True)
    horizontal_final_norm = v_final[..., :2].norm(dim=-1, keepdim=True)
    horizontal_safe_gate = (
        (horizontal_cmd_norm > command_eps).float()
        * (beta >= 0.999).float()
        * (1.0 - emergency)
    )
    horizontal_preservation = torch.where(
        horizontal_cmd_norm > command_eps,
        horizontal_final_norm / horizontal_cmd_norm.clamp_min(command_eps),
        torch.ones_like(horizontal_cmd_norm),
    ).clamp(0.0, 2.0)
    horizontal_amplification = (
        horizontal_preservation - 1.0
    ).clamp_min(0.0) * horizontal_safe_gate
    vertical_cmd_abs = v_cmd[..., 2:3].abs()
    vertical_final_abs = v_final[..., 2:3].abs()
    vertical_safe_gate = (
        (vertical_cmd_abs > command_eps).float()
        * (beta >= 0.999).float()
        * (1.0 - emergency)
    )
    vertical_preservation = torch.where(
        vertical_cmd_abs > command_eps,
        vertical_final_abs / vertical_cmd_abs.clamp_min(command_eps),
        torch.ones_like(vertical_cmd_abs),
    ).clamp(0.0, 2.0)
    vertical_amplification = (
        vertical_preservation - 1.0
    ).clamp_min(0.0) * vertical_safe_gate
    intervention = (beta < 0.999).float()
    violation = ((min_clearance < float(d_safe)) & (emergency < 0.5)).float()
    height_floor_violation = (floor - height).clamp_min(0.0)
    height_ceiling_margin = ceiling - height
    height_ceiling_violation = (height - ceiling).clamp_min(0.0)
    return {
        "tracking_actual_error_sq": actual_error_sq,
        "tracking_proxy_error_sq": proxy_error_sq,
        "command_preservation_ratio": preservation,
        "null_command_speed": actual.norm(dim=-1, keepdim=True) * null_command,
        "null_command_output_speed": v_final.norm(dim=-1, keepdim=True) * null_command,
        "command_amplification": amplification,
        "command_amplification_active": (amplification > 0.0).float(),
        "command_amplification_horizontal": horizontal_amplification,
        "command_amplification_horizontal_active": (horizontal_amplification > 0.0).float(),
        "command_amplification_vertical": vertical_amplification,
        "command_amplification_vertical_active": (vertical_amplification > 0.0).float(),
        "height_world_z": height,
        "height_floor_violation": height_floor_violation,
        "height_ceiling_violation": height_ceiling_violation,
        "height_ceiling_margin": height_ceiling_margin,
        "v_cmd_z": v_cmd[..., 2:3],
        "v_final_b_z": v_final[..., 2:3],
        "anchor_active": active,
        "anchor_error_mean": anchor_mean,
        "anchor_error_max": anchor_max,
        "anchor_loss": anchor_loss,
        "safety_min_clearance": min_clearance,
        "safety_collision": collision,
        "ics_beta": beta,
        "ics_intervention": intervention,
        "ics_emergency": emergency,
        "ics_violation": violation,
    }


def compute_r5e_mechanism_step_metrics(
    *,
    v_cmd_b: torch.Tensor,
    actual_velocity_b: torch.Tensor,
    v_gov_b: torch.Tensor,
    v_final_b: torch.Tensor,
    height_world_z: torch.Tensor,
    min_clearance: torch.Tensor,
    ics_beta: Optional[torch.Tensor] = None,
    ics_emergency: Optional[torch.Tensor] = None,
    d_safe: float = 0.8,
    height_floor: float = 0.5,
    command_eps: float = 1e-3,
) -> Dict[str, torch.Tensor]:
    """R5E eval-only mechanism diagnostics.

    Conditional quantities are returned as masked numerators plus explicit mask
    counts. Eval aggregation owns the divisions so inactive steps do not pull
    conditional means toward zero.
    """
    v_cmd = _as_vector_flat(v_cmd_b, "v_cmd_b")
    N = v_cmd.shape[0]
    device = v_cmd.device
    actual = _as_vector_flat(actual_velocity_b, "actual_velocity_b").to(device=device)
    gov = _as_vector_flat(v_gov_b, "v_gov_b").to(device=device)
    final = _as_vector_flat(v_final_b, "v_final_b").to(device=device)
    if actual.shape[0] != N or gov.shape[0] != N or final.shape[0] != N:
        raise ValueError("R5E vector inputs must flatten to the same length")

    height = _as_scalar_flat(height_world_z, "height_world_z", N, device=device)
    clearance = _as_scalar_flat(
        min_clearance,
        "min_clearance",
        N,
        default=float("inf"),
        device=device,
    )
    beta = _as_scalar_flat(ics_beta, "ics_beta", N, default=1.0, device=device).clamp(0.0, 1.0)
    emergency = _as_scalar_flat(
        ics_emergency,
        "ics_emergency",
        N,
        default=0.0,
        device=device,
    ).clamp(0.0, 1.0)

    eps = float(command_eps)
    if not math.isfinite(eps) or eps < 0.0:
        raise ValueError("command_eps must be finite and >= 0")
    floor = float(height_floor)
    if not math.isfinite(floor) or floor < 0.0:
        raise ValueError("height_floor must be finite and >= 0")
    safe = float(d_safe)
    if not math.isfinite(safe) or safe < 0.0:
        raise ValueError("d_safe must be finite and >= 0")

    command_norm = v_cmd.norm(dim=-1, keepdim=True)
    command_active = (command_norm > eps).float()
    null_command = 1.0 - command_active
    gov_norm = gov.norm(dim=-1, keepdim=True)
    final_norm = final.norm(dim=-1, keepdim=True)
    pre_ics_preservation = torch.where(
        command_norm > eps,
        gov_norm / command_norm.clamp_min(eps),
        torch.zeros_like(command_norm),
    ).clamp(0.0, 2.0)
    post_ics_preservation = torch.where(
        command_norm > eps,
        final_norm / command_norm.clamp_min(eps),
        torch.zeros_like(command_norm),
    ).clamp(0.0, 2.0)

    horizontal_cmd_norm = v_cmd[..., :2].norm(dim=-1, keepdim=True)
    horizontal_final_norm = final[..., :2].norm(dim=-1, keepdim=True)
    horizontal_active = (horizontal_cmd_norm > eps).float()
    horizontal_preservation = torch.where(
        horizontal_cmd_norm > eps,
        horizontal_final_norm / horizontal_cmd_norm.clamp_min(eps),
        torch.zeros_like(horizontal_cmd_norm),
    ).clamp(0.0, 2.0)

    vertical_cmd_abs = v_cmd[..., 2:3].abs()
    vertical_final_abs = final[..., 2:3].abs()
    vertical_active = (vertical_cmd_abs > eps).float()
    vertical_abs_preservation = torch.where(
        vertical_cmd_abs > eps,
        vertical_final_abs / vertical_cmd_abs.clamp_min(eps),
        torch.zeros_like(vertical_cmd_abs),
    ).clamp(0.0, 2.0)

    near_floor = (height <= floor + 0.10).float()
    ics_violation = ((clearance < safe) & (emergency < 0.5)).float()
    inactive_clearance = torch.full_like(clearance, float("nan"))

    return {
        "r5e_null_command": null_command,
        "r5e_null_actual_speed_xy": actual[..., :2].norm(dim=-1, keepdim=True) * null_command,
        "r5e_null_actual_speed_z_abs": actual[..., 2:3].abs() * null_command,
        "r5e_null_output_speed_xy": final[..., :2].norm(dim=-1, keepdim=True) * null_command,
        "r5e_null_output_speed_z_abs": final[..., 2:3].abs() * null_command,
        "r5e_command_active": command_active,
        "r5e_command_preservation_pre_ics": pre_ics_preservation * command_active,
        "r5e_command_preservation_post_ics": post_ics_preservation * command_active,
        "r5e_command_preservation_ics_loss": (
            pre_ics_preservation - post_ics_preservation
        ).clamp_min(0.0) * command_active,
        "r5e_command_horizontal_active": horizontal_active,
        "r5e_command_preservation_horizontal": horizontal_preservation * horizontal_active,
        "r5e_command_vertical_active": vertical_active,
        "r5e_command_preservation_vertical_abs": vertical_abs_preservation * vertical_active,
        "r5e_near_floor": near_floor,
        "r5e_near_floor_v_cmd_z": v_cmd[..., 2:3] * near_floor,
        "r5e_near_floor_v_gov_z": gov[..., 2:3] * near_floor,
        "r5e_near_floor_v_final_z": final[..., 2:3] * near_floor,
        "r5e_near_floor_ics_beta": beta * near_floor,
        "r5e_near_floor_clearance": torch.where(
            near_floor.bool(),
            clearance,
            inactive_clearance,
        ),
        "r5e_ics_violation_near_floor": ics_violation * near_floor,
    }


def compute_r5g_station_anchor_step_metrics(
    *,
    v_cmd_b: torch.Tensor,
    actual_velocity_b: torch.Tensor,
    v_final_b: torch.Tensor,
    station_drift: torch.Tensor,
    anchor_active: Optional[torch.Tensor] = None,
    anchor_valid_fraction: Optional[torch.Tensor] = None,
    anchor_error_mean: Optional[torch.Tensor] = None,
    anchor_loss: Optional[torch.Tensor] = None,
    observability_valid_fraction: Optional[torch.Tensor] = None,
    command_eps: float = 1e-3,
    min_anchor_valid_fraction: float = 0.1,
    anchor_loss_high_threshold: float = 0.05,
    observability_min_valid_fraction: float = 0.01,
) -> Dict[str, torch.Tensor]:
    """R5G eval-only station/null and anchor root-cause diagnostics."""
    v_cmd = _as_vector_flat(v_cmd_b, "v_cmd_b")
    N = v_cmd.shape[0]
    device = v_cmd.device
    actual = _as_vector_flat(actual_velocity_b, "actual_velocity_b").to(device=device)
    final = _as_vector_flat(v_final_b, "v_final_b").to(device=device)
    if actual.shape[0] != N or final.shape[0] != N:
        raise ValueError("R5G station vector inputs must flatten to the same length")

    drift = _as_scalar_flat(station_drift, "station_drift", N, device=device).clamp_min(0.0)
    active = _as_scalar_flat(anchor_active, "anchor_active", N, default=0.0, device=device).clamp(0.0, 1.0)
    valid_fraction = _as_scalar_flat(
        anchor_valid_fraction,
        "anchor_valid_fraction",
        N,
        default=0.0,
        device=device,
    ).clamp(0.0, 1.0)
    anchor_error = _as_scalar_flat(
        anchor_error_mean,
        "anchor_error_mean",
        N,
        default=0.0,
        device=device,
    ).clamp_min(0.0)
    anchor_loss_value = _as_scalar_flat(
        anchor_loss,
        "anchor_loss",
        N,
        default=0.0,
        device=device,
    ).clamp_min(0.0)
    observability_valid = _as_scalar_flat(
        observability_valid_fraction,
        "observability_valid_fraction",
        N,
        default=0.0,
        device=device,
    ).clamp(0.0, 1.0)

    eps = float(command_eps)
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("command_eps must be finite and > 0")
    min_anchor_valid = float(min_anchor_valid_fraction)
    if not math.isfinite(min_anchor_valid) or not (0.0 <= min_anchor_valid <= 1.0):
        raise ValueError("min_anchor_valid_fraction must be finite and in [0, 1]")
    high_loss_threshold = float(anchor_loss_high_threshold)
    if not math.isfinite(high_loss_threshold) or high_loss_threshold < 0.0:
        raise ValueError("anchor_loss_high_threshold must be finite and >= 0")
    min_observability_valid = float(observability_min_valid_fraction)
    if not math.isfinite(min_observability_valid) or not (0.0 <= min_observability_valid <= 1.0):
        raise ValueError("observability_min_valid_fraction must be finite and in [0, 1]")

    command_active = (v_cmd.norm(dim=-1, keepdim=True) > eps).float()
    null_command = 1.0 - command_active
    actual_xy = actual[..., :2]
    final_xy = final[..., :2]
    actual_xy_norm = actual_xy.norm(dim=-1, keepdim=True)
    final_xy_norm = final_xy.norm(dim=-1, keepdim=True)
    mismatch_xy = (actual_xy - final_xy).norm(dim=-1, keepdim=True)
    mismatch_z_abs = (actual[..., 2:3] - final[..., 2:3]).abs()
    alignment_active = null_command * (actual_xy_norm > eps).float() * (final_xy_norm > eps).float()
    alignment_xy = torch.where(
        alignment_active.bool(),
        (actual_xy * final_xy).sum(dim=-1, keepdim=True)
        / (actual_xy_norm * final_xy_norm).clamp_min(eps),
        torch.zeros_like(actual_xy_norm),
    ).clamp(-1.0, 1.0)
    output_xy_active = null_command * (final_xy_norm > eps).float()
    ratio_xy = torch.where(
        output_xy_active.bool(),
        actual_xy_norm / final_xy_norm.clamp_min(eps),
        torch.zeros_like(actual_xy_norm),
    ).clamp(0.0, 100.0)

    anchor_is_active = (active >= 0.5).float()
    anchor_is_valid = anchor_is_active * (valid_fraction >= min_anchor_valid).float()
    anchor_is_invalid = anchor_is_active * (1.0 - (valid_fraction >= min_anchor_valid).float())
    anchor_high_loss = anchor_is_active * (anchor_loss_value > high_loss_threshold).float()
    observability_good = anchor_is_active * (observability_valid >= min_observability_valid).float()
    observability_poor = anchor_is_active * (1.0 - (observability_valid >= min_observability_valid).float())

    result = {
        "r5g_station_null_command": null_command,
        "r5g_station_null_actual_speed_xy": actual_xy_norm * null_command,
        "r5g_station_null_output_speed_xy": final_xy_norm * null_command,
        "r5g_station_null_mismatch_xy": mismatch_xy * null_command,
        "r5g_station_null_mismatch_z_abs": mismatch_z_abs * null_command,
        "r5g_station_null_alignment_xy": alignment_xy * alignment_active,
        "r5g_station_null_alignment_xy_active": alignment_active,
        "r5g_station_null_actual_output_xy_ratio": ratio_xy * output_xy_active,
        "r5g_station_null_output_xy_active": output_xy_active,
    }
    for name, mask in (
        ("active", anchor_is_active),
        ("valid", anchor_is_valid),
        ("invalid", anchor_is_invalid),
        ("high_loss", anchor_high_loss),
        ("obs_valid", observability_good),
        ("obs_poor", observability_poor),
    ):
        result[f"r5g_anchor_{name}"] = mask
        result[f"r5g_anchor_station_drift_when_{name}"] = drift * mask
        result[f"r5g_anchor_error_when_{name}"] = anchor_error * mask
        result[f"r5g_anchor_loss_when_{name}"] = anchor_loss_value * mask
    return result


def compute_r5g_downward_step_metrics(
    *,
    downward_active: torch.Tensor,
    downward_has_ray: Optional[torch.Tensor] = None,
    downward_beta: Optional[torch.Tensor] = None,
    downward_min_clearance: Optional[torch.Tensor] = None,
    downward_pre_z: Optional[torch.Tensor] = None,
    downward_post_z: Optional[torch.Tensor] = None,
    downward_z_delta_abs: Optional[torch.Tensor] = None,
    downward_attenuation_ratio: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """R5G eval-only scalar diagnostics for MID360 downward attenuation."""
    active = downward_active.float().reshape(-1, 1).clamp(0.0, 1.0)
    N = active.shape[0]
    device = active.device
    has_ray = _as_scalar_flat(
        downward_has_ray,
        "downward_has_ray",
        N,
        default=0.0,
        device=device,
    ).clamp(0.0, 1.0)
    beta = _as_scalar_flat(downward_beta, "downward_beta", N, default=1.0, device=device).clamp(0.0, 1.0)
    clearance = _as_scalar_flat(
        downward_min_clearance,
        "downward_min_clearance",
        N,
        default=0.0,
        device=device,
    )
    pre_z = _as_scalar_flat(downward_pre_z, "downward_pre_z", N, default=0.0, device=device)
    post_z = _as_scalar_flat(downward_post_z, "downward_post_z", N, default=0.0, device=device)
    delta = _as_scalar_flat(
        downward_z_delta_abs,
        "downward_z_delta_abs",
        N,
        default=0.0,
        device=device,
    ).clamp_min(0.0)
    attenuation_ratio = _as_scalar_flat(
        downward_attenuation_ratio,
        "downward_attenuation_ratio",
        N,
        default=0.0,
        device=device,
    ).clamp(0.0, 1.0)
    inactive = torch.full_like(active, float("nan"))
    return {
        "r5g_downward_active": active,
        "r5g_downward_has_ray": has_ray,
        "r5g_downward_beta_when_active": beta * active,
        "r5g_downward_min_clearance_when_active": torch.where(active.bool(), clearance, inactive),
        "r5g_downward_pre_z_when_active": pre_z * active,
        "r5g_downward_post_z_when_active": post_z * active,
        "r5g_downward_z_delta_abs_when_active": delta * active,
        "r5g_downward_attenuation_ratio_when_active": attenuation_ratio * active,
    }


def _condition_sample(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    inactive = torch.full_like(value, float("nan"))
    return torch.where(mask.bool(), value, inactive)


def compute_r5h_mechanism_step_metrics(
    *,
    v_cmd_b: torch.Tensor,
    actual_velocity_b: torch.Tensor,
    v_gov_b: torch.Tensor,
    v_final_b: torch.Tensor,
    min_clearance: torch.Tensor,
    height_world_z: Optional[torch.Tensor] = None,
    ics_beta: Optional[torch.Tensor] = None,
    ics_emergency: Optional[torch.Tensor] = None,
    ics_violation: Optional[torch.Tensor] = None,
    ics_active_beam_count: Optional[torch.Tensor] = None,
    ics_downward_active: Optional[torch.Tensor] = None,
    ics_downward_beta: Optional[torch.Tensor] = None,
    ics_downward_min_clearance: Optional[torch.Tensor] = None,
    collision: Optional[torch.Tensor] = None,
    governor_alpha: Optional[torch.Tensor] = None,
    governor_v_corr: Optional[torch.Tensor] = None,
    prev_action_b: Optional[torch.Tensor] = None,
    station_drift: Optional[torch.Tensor] = None,
    anchor_active: Optional[torch.Tensor] = None,
    anchor_valid_fraction: Optional[torch.Tensor] = None,
    anchor_error_mean: Optional[torch.Tensor] = None,
    anchor_loss: Optional[torch.Tensor] = None,
    command_eps: float = 1e-3,
    height_floor: float = 0.5,
    d_safe: float = 0.8,
    low_beta_threshold: float = 0.999,
    min_anchor_valid_fraction: float = 0.1,
    anchor_loss_high_threshold: float = 0.05,
) -> Dict[str, torch.Tensor]:
    """R5H eval-only mechanism diagnostics.

    Returned tensors are per-step scalars shaped ``[N, 1]``. Conditional
    means are encoded as masked numerators plus explicit mask fields; eval
    summary code owns all divisions and quantiles.
    """
    v_cmd = _as_vector_flat(v_cmd_b, "v_cmd_b")
    N = v_cmd.shape[0]
    device = v_cmd.device
    actual = _as_vector_flat(actual_velocity_b, "actual_velocity_b").to(device=device)
    gov = _as_vector_flat(v_gov_b, "v_gov_b").to(device=device)
    final = _as_vector_flat(v_final_b, "v_final_b").to(device=device)
    if actual.shape[0] != N or gov.shape[0] != N or final.shape[0] != N:
        raise ValueError("R5H vector inputs must flatten to the same length")

    clearance = _as_scalar_flat(
        min_clearance,
        "min_clearance",
        N,
        default=float("inf"),
        device=device,
    )
    height = _as_scalar_flat(
        height_world_z,
        "height_world_z",
        N,
        default=float("inf"),
        device=device,
    )
    beta = _as_scalar_flat(ics_beta, "ics_beta", N, default=1.0, device=device).clamp(0.0, 1.0)
    emergency = _as_scalar_flat(
        ics_emergency,
        "ics_emergency",
        N,
        default=0.0,
        device=device,
    ).clamp(0.0, 1.0)
    violation = _as_scalar_flat(
        ics_violation,
        "ics_violation",
        N,
        default=0.0,
        device=device,
    ).clamp(0.0, 1.0)
    active_beams = _as_scalar_flat(
        ics_active_beam_count,
        "ics_active_beam_count",
        N,
        default=0.0,
        device=device,
    ).clamp_min(0.0)
    downward_active = _as_scalar_flat(
        ics_downward_active,
        "ics_downward_active",
        N,
        default=0.0,
        device=device,
    ).clamp(0.0, 1.0)
    downward_beta = _as_scalar_flat(
        ics_downward_beta,
        "ics_downward_beta",
        N,
        default=1.0,
        device=device,
    ).clamp(0.0, 1.0)
    downward_clearance = _as_scalar_flat(
        ics_downward_min_clearance,
        "ics_downward_min_clearance",
        N,
        default=float("nan"),
        device=device,
    )
    collision_mask = _as_scalar_flat(
        collision,
        "collision",
        N,
        default=0.0,
        device=device,
    ).clamp(0.0, 1.0)
    alpha = _as_scalar_flat(
        governor_alpha,
        "governor_alpha",
        N,
        default=0.0,
        device=device,
    ).clamp(0.0, 1.0)
    v_corr = (
        _as_vector_flat(governor_v_corr, "governor_v_corr").to(device=device)
        if governor_v_corr is not None
        else torch.zeros_like(v_cmd)
    )
    previous_action = (
        _as_vector_flat(prev_action_b, "prev_action_b").to(device=device)
        if prev_action_b is not None
        else torch.zeros_like(v_cmd)
    )
    drift = _as_scalar_flat(
        station_drift,
        "station_drift",
        N,
        default=0.0,
        device=device,
    ).clamp_min(0.0)
    anchor_is_active_raw = _as_scalar_flat(
        anchor_active,
        "anchor_active",
        N,
        default=0.0,
        device=device,
    ).clamp(0.0, 1.0)
    valid_fraction = _as_scalar_flat(
        anchor_valid_fraction,
        "anchor_valid_fraction",
        N,
        default=0.0,
        device=device,
    ).clamp(0.0, 1.0)
    anchor_error = _as_scalar_flat(
        anchor_error_mean,
        "anchor_error_mean",
        N,
        default=0.0,
        device=device,
    ).clamp_min(0.0)
    anchor_loss_value = _as_scalar_flat(
        anchor_loss,
        "anchor_loss",
        N,
        default=0.0,
        device=device,
    ).clamp_min(0.0)

    eps = float(command_eps)
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("command_eps must be finite and > 0")
    floor = float(height_floor)
    if not math.isfinite(floor) or floor < 0.0:
        raise ValueError("height_floor must be finite and >= 0")
    safe = float(d_safe)
    if not math.isfinite(safe) or safe < 0.0:
        raise ValueError("d_safe must be finite and >= 0")
    beta_threshold = float(low_beta_threshold)
    if not math.isfinite(beta_threshold) or not (0.0 <= beta_threshold <= 1.0):
        raise ValueError("low_beta_threshold must be finite and in [0, 1]")
    min_anchor_valid = float(min_anchor_valid_fraction)
    if not math.isfinite(min_anchor_valid) or not (0.0 <= min_anchor_valid <= 1.0):
        raise ValueError("min_anchor_valid_fraction must be finite and in [0, 1]")
    high_loss_threshold = float(anchor_loss_high_threshold)
    if not math.isfinite(high_loss_threshold) or high_loss_threshold < 0.0:
        raise ValueError("anchor_loss_high_threshold must be finite and >= 0")

    command_norm = v_cmd.norm(dim=-1, keepdim=True)
    command_active = (command_norm > eps).float()
    null_command = 1.0 - command_active
    near_floor = (height <= floor + 0.10).float()
    derived_violation = ((clearance < safe) & (emergency < 0.5)).float()
    violation = torch.maximum(violation, derived_violation)

    value_map = {
        "ics_beta": beta,
        "ics_downward_beta": downward_beta,
        "ics_active_beam_count": active_beams,
        "ics_downward_min_clearance": downward_clearance,
        "min_clearance": clearance,
        "v_cmd_xy_norm": v_cmd[..., :2].norm(dim=-1, keepdim=True),
        "v_cmd_z_abs": v_cmd[..., 2:3].abs(),
        "v_gov_xy_norm": gov[..., :2].norm(dim=-1, keepdim=True),
        "v_gov_z_abs": gov[..., 2:3].abs(),
        "v_final_xy_norm": final[..., :2].norm(dim=-1, keepdim=True),
        "v_final_z_abs": final[..., 2:3].abs(),
        "actual_xy_speed": actual[..., :2].norm(dim=-1, keepdim=True),
        "actual_z_abs": actual[..., 2:3].abs(),
    }
    condition_map = {
        "collision": (collision_mask >= 0.5).float(),
        "ics_violation": (violation >= 0.5).float(),
        "downward_active": (downward_active >= 0.5).float(),
        "low_beta": (beta < beta_threshold).float(),
        "emergency": (emergency >= 0.5).float(),
        "near_floor": near_floor,
    }

    result: Dict[str, torch.Tensor] = {}
    for condition, mask in condition_map.items():
        result[f"r5h_{condition}"] = mask
        for value_name, value in value_map.items():
            result[f"r5h_{value_name}_when_{condition}"] = value * mask
        for value_name in R5H_CONCENTRATION_SAMPLE_VALUE_NAMES:
            result[f"r5h_{value_name}_sample_when_{condition}"] = _condition_sample(
                value_map[value_name],
                mask,
            )

    gov_xy = gov[..., :2]
    final_xy = final[..., :2]
    actual_xy = actual[..., :2]
    prev_xy = previous_action[..., :2]
    prev_norm = prev_xy.norm(dim=-1, keepdim=True)
    final_norm_xy = final_xy.norm(dim=-1, keepdim=True)
    alignment_active = null_command * (prev_norm > eps).float() * (final_norm_xy > eps).float()
    prev_final_alignment = torch.where(
        alignment_active.bool(),
        (prev_xy * final_xy).sum(dim=-1, keepdim=True)
        / (prev_norm.clamp_min(eps) * final_norm_xy.clamp_min(eps)),
        torch.zeros_like(final_norm_xy),
    ).clamp(-1.0, 1.0)
    result["r5h_station_null_command"] = null_command
    station_values = {
        "actual_speed_xy": actual_xy.norm(dim=-1, keepdim=True),
        "actual_speed_z_abs": actual[..., 2:3].abs(),
        "v_gov_speed_xy": gov_xy.norm(dim=-1, keepdim=True),
        "v_gov_speed_z_abs": gov[..., 2:3].abs(),
        "v_final_speed_xy": final_xy.norm(dim=-1, keepdim=True),
        "v_final_speed_z_abs": final[..., 2:3].abs(),
        "actual_v_gov_mismatch_xy": (actual_xy - gov_xy).norm(dim=-1, keepdim=True),
        "actual_v_gov_mismatch_z_abs": (actual[..., 2:3] - gov[..., 2:3]).abs(),
        "actual_v_final_mismatch_xy": (actual_xy - final_xy).norm(dim=-1, keepdim=True),
        "actual_v_final_mismatch_z_abs": (actual[..., 2:3] - final[..., 2:3]).abs(),
        "alpha": alpha,
        "v_corr_norm": v_corr.norm(dim=-1, keepdim=True),
        "v_corr_z_abs": v_corr[..., 2:3].abs(),
        "prev_action_speed_xy": prev_norm,
        "prev_action_speed_z_abs": previous_action[..., 2:3].abs(),
        "prev_action_v_final_mismatch_xy": (prev_xy - final_xy).norm(dim=-1, keepdim=True),
        "prev_action_v_final_mismatch_z_abs": (
            previous_action[..., 2:3] - final[..., 2:3]
        ).abs(),
        "prev_action_actual_mismatch_xy": (prev_xy - actual_xy).norm(dim=-1, keepdim=True),
        "prev_action_actual_mismatch_z_abs": (
            previous_action[..., 2:3] - actual[..., 2:3]
        ).abs(),
        "prev_action_v_final_alignment_xy": prev_final_alignment,
    }
    for value_name, value in station_values.items():
        result[f"r5h_station_null_{value_name}"] = value * null_command

    anchor_is_active = (anchor_is_active_raw >= 0.5).float()
    anchor_conditions = {
        "active": anchor_is_active,
        "valid": anchor_is_active * (valid_fraction >= min_anchor_valid).float(),
        "high_loss": anchor_is_active * (anchor_loss_value > high_loss_threshold).float(),
    }
    anchor_values = {
        "station_drift": drift,
        "null_speed": actual.norm(dim=-1, keepdim=True) * null_command,
        "actual_xy_speed": actual_xy.norm(dim=-1, keepdim=True),
        "actual_z_abs": actual[..., 2:3].abs(),
        "anchor_error": anchor_error,
        "anchor_loss": anchor_loss_value,
    }
    for condition, mask in anchor_conditions.items():
        result[f"r5h_anchor_{condition}"] = mask
        for value_name, value in anchor_values.items():
            result[f"r5h_anchor_{value_name}_when_{condition}"] = value * mask

    gov_norm = gov.norm(dim=-1, keepdim=True)
    final_norm = final.norm(dim=-1, keepdim=True)
    pre_ics_preservation = torch.where(
        command_active.bool(),
        gov_norm / command_norm.clamp_min(eps),
        torch.zeros_like(command_norm),
    ).clamp(0.0, 2.0)
    post_ics_preservation = torch.where(
        command_active.bool(),
        final_norm / command_norm.clamp_min(eps),
        torch.zeros_like(command_norm),
    ).clamp(0.0, 2.0)
    result["r5h_tracking_active"] = command_active
    result["r5h_tracking_pre_ics_preservation"] = pre_ics_preservation * command_active
    result["r5h_tracking_post_ics_preservation"] = post_ics_preservation * command_active
    result["r5h_tracking_governor_preservation_loss"] = (
        1.0 - pre_ics_preservation
    ).clamp_min(0.0) * command_active
    result["r5h_tracking_post_ics_preservation_loss"] = (
        pre_ics_preservation - post_ics_preservation
    ).clamp_min(0.0) * command_active

    horizontal_cmd = v_cmd[..., :2].norm(dim=-1, keepdim=True)
    horizontal_gov = gov_xy.norm(dim=-1, keepdim=True)
    horizontal_final = final_xy.norm(dim=-1, keepdim=True)
    horizontal_active = (horizontal_cmd > eps).float()
    horizontal_pre = torch.where(
        horizontal_active.bool(),
        horizontal_gov / horizontal_cmd.clamp_min(eps),
        torch.zeros_like(horizontal_cmd),
    ).clamp(0.0, 2.0)
    horizontal_post = torch.where(
        horizontal_active.bool(),
        horizontal_final / horizontal_cmd.clamp_min(eps),
        torch.zeros_like(horizontal_cmd),
    ).clamp(0.0, 2.0)
    result["r5h_tracking_horizontal_active"] = horizontal_active
    result["r5h_tracking_horizontal_pre_ics_preservation"] = horizontal_pre * horizontal_active
    result["r5h_tracking_horizontal_post_ics_preservation"] = horizontal_post * horizontal_active
    result["r5h_tracking_horizontal_governor_preservation_loss"] = (
        1.0 - horizontal_pre
    ).clamp_min(0.0) * horizontal_active
    result["r5h_tracking_horizontal_post_ics_preservation_loss"] = (
        horizontal_pre - horizontal_post
    ).clamp_min(0.0) * horizontal_active

    vertical_cmd = v_cmd[..., 2:3].abs()
    vertical_gov = gov[..., 2:3].abs()
    vertical_final = final[..., 2:3].abs()
    vertical_active = (vertical_cmd > eps).float()
    vertical_pre = torch.where(
        vertical_active.bool(),
        vertical_gov / vertical_cmd.clamp_min(eps),
        torch.zeros_like(vertical_cmd),
    ).clamp(0.0, 2.0)
    vertical_post = torch.where(
        vertical_active.bool(),
        vertical_final / vertical_cmd.clamp_min(eps),
        torch.zeros_like(vertical_cmd),
    ).clamp(0.0, 2.0)
    result["r5h_tracking_vertical_active"] = vertical_active
    result["r5h_tracking_vertical_pre_ics_preservation"] = vertical_pre * vertical_active
    result["r5h_tracking_vertical_post_ics_preservation"] = vertical_post * vertical_active
    result["r5h_tracking_vertical_governor_preservation_loss"] = (
        1.0 - vertical_pre
    ).clamp_min(0.0) * vertical_active
    result["r5h_tracking_vertical_post_ics_preservation_loss"] = (
        vertical_pre - vertical_post
    ).clamp_min(0.0) * vertical_active

    return result


def compute_vertical_channel_step_metrics(
    *,
    v_cmd_z: torch.Tensor,
    v_corr_z: torch.Tensor,
    v_gov_z: torch.Tensor,
    v_final_z: torch.Tensor,
    station_drift: Optional[torch.Tensor] = None,
    command_preservation_ratio: Optional[torch.Tensor] = None,
    command_amplification_vertical: Optional[torch.Tensor] = None,
    ics_beta: Optional[torch.Tensor] = None,
    ics_emergency: Optional[torch.Tensor] = None,
    v_corr_limit: float = 0.0,
    command_eps: float = 1e-3,
    saturation_tol: float = 1e-4,
) -> Dict[str, torch.Tensor]:
    """Per-step vertical governor diagnostics for mechanism diagnosis.

    The returned tensors are all shaped [N, 1]. Conditional quantities are
    returned as masked numerators; callers should divide by the corresponding
    mask sum when computing "when active" means.
    """
    cmd = _as_scalar(v_cmd_z, "v_cmd_z", v_cmd_z.reshape(-1).shape[0], device=v_cmd_z.device)
    N = cmd.shape[0]
    device = cmd.device
    corr = _as_scalar(v_corr_z, "v_corr_z", N, device=device)
    gov = _as_scalar(v_gov_z, "v_gov_z", N, device=device)
    final = _as_scalar(v_final_z, "v_final_z", N, device=device)
    drift = _as_scalar(station_drift, "station_drift", N, default=0.0, device=device)
    preservation = _as_scalar(
        command_preservation_ratio,
        "command_preservation_ratio",
        N,
        default=0.0,
        device=device,
    )
    amplification = _as_scalar(
        command_amplification_vertical,
        "command_amplification_vertical",
        N,
        default=0.0,
        device=device,
    )
    beta = _as_scalar(ics_beta, "ics_beta", N, default=1.0, device=device).clamp(0.0, 1.0)
    emergency = _as_scalar(
        ics_emergency,
        "ics_emergency",
        N,
        default=0.0,
        device=device,
    ).clamp(0.0, 1.0)

    eps = float(command_eps)
    if not math.isfinite(eps) or eps < 0.0:
        raise ValueError("command_eps must be finite and >= 0")
    tol = float(saturation_tol)
    if not math.isfinite(tol) or tol < 0.0:
        raise ValueError("saturation_tol must be finite and >= 0")
    limit = float(v_corr_limit)
    if not math.isfinite(limit) or limit < 0.0:
        raise ValueError("v_corr_limit must be finite and >= 0")

    command_active = (cmd.abs() > eps).float()
    command_null = 1.0 - command_active
    corr_abs = corr.abs()
    corr_active = (corr_abs > eps).float()
    saturated = (
        corr_abs >= max(limit - tol, 0.0)
    ).float() * float(limit > 0.0)
    gov_minus_cmd = gov - cmd
    final_minus_cmd = final - cmd
    ics_delta = final - gov
    reinforces = ((cmd * corr) > 0.0).float() * command_active
    opposes = ((cmd * corr) < 0.0).float() * command_active
    null_corr_active = command_null * corr_active
    tracking_corr_active = command_active * corr_active

    return {
        "vertical_command_active": command_active,
        "vertical_command_null": command_null,
        "vertical_corr_z": corr,
        "vertical_corr_z_abs": corr_abs,
        "vertical_corr_z_positive": (corr > 0.0).float(),
        "vertical_corr_z_negative": (corr < 0.0).float(),
        "vertical_corr_z_saturated": saturated,
        "vertical_gov_minus_cmd_z": gov_minus_cmd,
        "vertical_gov_minus_cmd_z_abs": gov_minus_cmd.abs(),
        "vertical_final_minus_cmd_z": final_minus_cmd,
        "vertical_final_minus_cmd_z_abs": final_minus_cmd.abs(),
        "vertical_ics_delta_z": ics_delta,
        "vertical_ics_delta_z_abs": ics_delta.abs(),
        "vertical_corr_reinforces_command": reinforces,
        "vertical_corr_opposes_command": opposes,
        "vertical_null_corr_active": null_corr_active,
        "vertical_null_corr_abs": corr_abs * command_null,
        "vertical_null_station_drift_when_corr_active": drift * null_corr_active,
        "vertical_tracking_corr_active": tracking_corr_active,
        "vertical_tracking_amplification_when_corr_active": amplification
        * tracking_corr_active,
        "vertical_tracking_preservation_when_corr_active": preservation
        * tracking_corr_active,
        "vertical_ics_beta": beta,
        "vertical_ics_emergency": emergency,
    }
