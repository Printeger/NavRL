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
