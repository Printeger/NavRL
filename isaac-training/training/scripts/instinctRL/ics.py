"""
instinctRL-E ICS-inspired command attenuation.

The deployed path attenuates the body-frame governor command from MID360
range history only:

    v_final_b = beta * v_gov_b

Dense per-beam tensors are returned in the internal cache. Public metrics are
scalar tensors intended for env info/debug streams.
"""

from dataclasses import dataclass
import math
from typing import Dict, Optional

import torch


ICS_METRIC_KEYS = (
    "ics_beta",
    "ics_active_beam_count",
    "ics_min_clearance",
    "ics_worst_margin",
    "ics_emergency",
    "ics_command_speed",
    "ics_brake_speed",
    "ics_final_speed",
    "ics_clip_ratio",
)


@dataclass
class ICSConfig:
    """Configuration for range-history attenuation."""

    enabled: bool = False
    d_safe: float = 0.8
    emergency_clearance: float = 0.25
    active_horizon_margin: float = 0.5
    a_max: float = 2.0
    latency_sec: float = 0.0
    clearance_margin: float = 0.0
    min_reliability: float = 0.1
    approach_eps: float = 1e-3
    range_rate_eps: float = 1e-3
    use_range_rate_filter: bool = False
    velocity_limit: float = 2.0
    range_rate_mode: str = "finite_difference"
    empty_active_set_beta: float = 1.0
    brake_mode: str = "zero"
    eps: float = 1e-6
    history_dt: float = 1.0

    def __post_init__(self):
        if self.range_rate_mode != "finite_difference":
            raise ValueError("range_rate_mode must be 'finite_difference'")
        if self.brake_mode != "zero":
            raise ValueError("brake_mode must be 'zero'")
        for name in (
            "d_safe",
            "emergency_clearance",
            "active_horizon_margin",
            "a_max",
            "latency_sec",
            "clearance_margin",
            "min_reliability",
            "approach_eps",
            "range_rate_eps",
            "velocity_limit",
            "empty_active_set_beta",
            "eps",
            "history_dt",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            setattr(self, name, value)
        if self.d_safe <= 0.0:
            raise ValueError("d_safe must be > 0")
        if self.emergency_clearance < 0.0:
            raise ValueError("emergency_clearance must be >= 0")
        if self.emergency_clearance > self.d_safe:
            raise ValueError("emergency_clearance must be <= d_safe")
        if self.active_horizon_margin < 0.0:
            raise ValueError("active_horizon_margin must be >= 0")
        if self.a_max <= 0.0:
            raise ValueError("a_max must be > 0")
        if self.latency_sec < 0.0:
            raise ValueError("latency_sec must be >= 0")
        if self.clearance_margin < 0.0:
            raise ValueError("clearance_margin must be >= 0")
        if not (0.0 < self.min_reliability <= 1.0):
            raise ValueError("min_reliability must satisfy 0.0 < value <= 1.0")
        if self.approach_eps < 0.0:
            raise ValueError("approach_eps must be >= 0")
        if self.range_rate_eps < 0.0:
            raise ValueError("range_rate_eps must be >= 0")
        if self.velocity_limit <= 0.0:
            raise ValueError("velocity_limit must be > 0")
        if not (0.0 <= self.empty_active_set_beta <= 1.0):
            raise ValueError("empty_active_set_beta must satisfy 0.0 <= value <= 1.0")
        if self.eps <= 0.0:
            raise ValueError("eps must be > 0")
        if self.history_dt <= 0.0:
            raise ValueError("history_dt must be > 0")

    @classmethod
    def from_namespace(cls, cfg) -> "ICSConfig":
        return cls(
            enabled=bool(getattr(cfg, "enabled", False)),
            d_safe=float(getattr(cfg, "d_safe", 0.8)),
            emergency_clearance=float(getattr(cfg, "emergency_clearance", 0.25)),
            active_horizon_margin=float(getattr(cfg, "active_horizon_margin", 0.5)),
            a_max=float(getattr(cfg, "a_max", 2.0)),
            latency_sec=float(getattr(cfg, "latency_sec", 0.0)),
            clearance_margin=float(getattr(cfg, "clearance_margin", 0.0)),
            min_reliability=float(getattr(cfg, "min_reliability", 0.1)),
            approach_eps=float(getattr(cfg, "approach_eps", 1e-3)),
            range_rate_eps=float(getattr(cfg, "range_rate_eps", 1e-3)),
            use_range_rate_filter=bool(getattr(cfg, "use_range_rate_filter", False)),
            velocity_limit=float(getattr(cfg, "velocity_limit", 2.0)),
            range_rate_mode=str(getattr(cfg, "range_rate_mode", "finite_difference")),
            empty_active_set_beta=float(getattr(cfg, "empty_active_set_beta", 1.0)),
            brake_mode=str(getattr(cfg, "brake_mode", "zero")),
        )


@dataclass
class ICSOutput:
    """Attenuated body command with public metrics and dense internal cache."""

    v_final_b: torch.Tensor
    metrics: Dict[str, torch.Tensor]
    cache: Dict[str, torch.Tensor]


class RangeHistoryICSAttenuator:
    """Compute body-frame command attenuation from MID360 history."""

    def __init__(self, config: ICSConfig, *, device: str = "cuda:0"):
        self.cfg = config
        self.device = torch.device(device)

    def __call__(
        self,
        range_history: torch.Tensor,
        mask_history: torch.Tensor,
        weight_history: torch.Tensor,
        ray_directions_b: torch.Tensor,
        v_gov_b: torch.Tensor,
        *,
        dt: Optional[float] = None,
        history_dt: Optional[float] = None,
    ) -> ICSOutput:
        return self.forward(
            range_history,
            mask_history,
            weight_history,
            ray_directions_b,
            v_gov_b,
            dt=dt,
            history_dt=history_dt,
        )

    def forward(
        self,
        range_history: torch.Tensor,
        mask_history: torch.Tensor,
        weight_history: torch.Tensor,
        ray_directions_b: torch.Tensor,
        v_gov_b: torch.Tensor,
        *,
        dt: Optional[float] = None,
        history_dt: Optional[float] = None,
    ) -> ICSOutput:
        ranges = self._validate_history("range_history", range_history)
        masks = self._validate_history("mask_history", mask_history)
        weights = self._validate_history("weight_history", weight_history)
        if masks.shape != ranges.shape or weights.shape != ranges.shape:
            raise ValueError("range_history, mask_history, and weight_history must have matching shapes")

        N, L, R = ranges.shape
        rays = self._validate_ray_directions(ray_directions_b, N, R)
        command, output_shape = self._validate_command(v_gov_b, N)
        step_dt = self._resolve_history_dt(dt, history_dt)

        latest_range = ranges[:, -1]
        latest_mask = self._mask_to_bool(masks[:, -1])
        latest_weight = weights[:, -1].clamp(0.0, 1.0)
        reliable = latest_weight >= self.cfg.min_reliability

        rate = self._range_rate_estimate(ranges, masks, step_dt)
        approach_speed = torch.einsum("nd,nrd->nr", command, rays).clamp_min(0.0)
        command_speed = command.norm(dim=-1, keepdim=True)

        effective_clearance = (
            latest_range
            - self.cfg.latency_sec * command_speed
            - self.cfg.clearance_margin
        )
        clearance_over_safe = (effective_clearance - self.cfg.d_safe).clamp_min(0.0)
        v_safe = torch.sqrt(2.0 * self.cfg.a_max * clearance_over_safe)

        range_closing_speed = (-rate).clamp_min(0.0)
        if self.cfg.use_range_rate_filter:
            closing_evidence = (
                (approach_speed > self.cfg.approach_eps)
                | (range_closing_speed > self.cfg.range_rate_eps)
            )
            closing_speed = torch.maximum(approach_speed, range_closing_speed)
        else:
            closing_evidence = approach_speed > self.cfg.approach_eps
            closing_speed = approach_speed

        active_mask = (
            latest_mask
            & reliable
            & closing_evidence
            & (effective_clearance <= self.cfg.d_safe + self.cfg.active_horizon_margin)
        )

        ratio = v_safe / closing_speed.clamp_min(self.cfg.eps)
        ratio = ratio.clamp(0.0, 1.0)
        active_ratio = torch.where(active_mask, ratio, torch.ones_like(ratio))
        beta = active_ratio.min(dim=1, keepdim=True).values
        active_count = active_mask.sum(dim=1, keepdim=True)
        empty_beta = torch.full_like(beta, self.cfg.empty_active_set_beta)
        beta = torch.where(active_count > 0, beta, empty_beta)

        reliable_latest = latest_mask & reliable
        min_clearance = torch.where(
            reliable_latest,
            effective_clearance,
            torch.full_like(effective_clearance, float("inf")),
        ).min(dim=1, keepdim=True).values
        finite_min = torch.isfinite(min_clearance)
        min_clearance = torch.where(finite_min, min_clearance, torch.zeros_like(min_clearance))
        emergency = reliable_latest & (effective_clearance < self.cfg.emergency_clearance)
        emergency_any = emergency.any(dim=1, keepdim=True)
        beta = torch.where(emergency_any, torch.zeros_like(beta), beta).clamp(0.0, 1.0)

        brake = torch.zeros_like(command)
        v_att = beta * command + (1.0 - beta) * brake
        v_final, clip_ratio = self._clip_command(v_att)
        final_speed = v_final.norm(dim=-1, keepdim=True)

        safety_margin = v_safe - closing_speed
        worst_margin = torch.where(
            active_mask,
            safety_margin,
            torch.full_like(safety_margin, float("inf")),
        ).min(dim=1, keepdim=True).values
        worst_margin = torch.where(
            torch.isfinite(worst_margin),
            worst_margin,
            torch.zeros_like(worst_margin),
        )
        worst_beam_index = torch.where(
            active_count.squeeze(-1) > 0,
            torch.where(active_mask, safety_margin, torch.full_like(safety_margin, float("inf"))).argmin(dim=1),
            torch.full((N,), -1, dtype=torch.long, device=self.device),
        )

        metrics = {
            "ics_beta": beta,
            "ics_active_beam_count": active_count.to(command.dtype),
            "ics_min_clearance": min_clearance,
            "ics_worst_margin": worst_margin,
            "ics_emergency": emergency_any.to(command.dtype),
            "ics_command_speed": command_speed,
            "ics_brake_speed": brake.norm(dim=-1, keepdim=True),
            "ics_final_speed": final_speed,
            "ics_clip_ratio": clip_ratio,
        }
        for key, value in list(metrics.items()):
            metrics[key] = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)

        cache = {
            "ics_active_mask": active_mask,
            "ics_approach_speed": approach_speed,
            "ics_closing_speed": closing_speed,
            "ics_range_rate_estimate": rate,
            "ics_safety_margin": safety_margin,
            "ics_worst_beam_index": worst_beam_index,
            "ics_effective_clearance": effective_clearance,
        }
        return ICSOutput(v_final_b=v_final.reshape(output_shape), metrics=metrics, cache=cache)

    def _validate_history(self, name: str, value: torch.Tensor) -> torch.Tensor:
        if value.device != self.device:
            raise ValueError(f"{name} must be on attenuator device")
        if value.dim() == 4:
            N, L, H, V = value.shape
            flat = value.reshape(N, L, H * V)
        elif value.dim() == 3:
            flat = value
        else:
            raise ValueError(f"{name} must have shape [N,L,H,V] or [N,L,R]")
        if flat.shape[0] <= 0 or flat.shape[1] <= 0 or flat.shape[2] <= 0:
            raise ValueError(f"{name} dimensions must be positive")
        if not torch.isfinite(flat).all():
            raise ValueError(f"{name} must be finite")
        return flat

    def _validate_ray_directions(self, ray_directions_b: torch.Tensor, N: int, R: int) -> torch.Tensor:
        if ray_directions_b.device != self.device:
            raise ValueError("ray_directions_b must be on attenuator device")
        if ray_directions_b.dim() == 2 and ray_directions_b.shape == (R, 3):
            rays = ray_directions_b.unsqueeze(0).expand(N, R, 3)
        elif ray_directions_b.dim() == 3 and ray_directions_b.shape == (N, R, 3):
            rays = ray_directions_b
        else:
            raise ValueError("ray_directions_b must have shape [R,3] or [N,R,3]")
        if not torch.isfinite(rays).all():
            raise ValueError("ray_directions_b must be finite")
        norms = rays.norm(dim=-1, keepdim=True)
        if (norms <= self.cfg.eps).any():
            raise ValueError("ray_directions_b must contain nonzero vectors")
        return rays / norms.clamp_min(self.cfg.eps)

    def _validate_command(self, v_gov_b: torch.Tensor, N: int):
        if v_gov_b.device != self.device:
            raise ValueError("v_gov_b must be on attenuator device")
        if v_gov_b.shape == (N, 3):
            command = v_gov_b
            output_shape = v_gov_b.shape
        elif v_gov_b.shape == (N, 1, 3):
            command = v_gov_b.squeeze(1)
            output_shape = v_gov_b.shape
        else:
            raise ValueError("v_gov_b must have shape [N,3] or [N,1,3]")
        if not torch.isfinite(command).all():
            raise ValueError("v_gov_b must be finite")
        return command, output_shape

    def _resolve_history_dt(self, dt: Optional[float], history_dt: Optional[float]) -> float:
        value = self.cfg.history_dt
        if history_dt is not None:
            value = float(history_dt)
        if dt is not None:
            value = float(dt)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("history dt must be finite and > 0")
        return value

    def _range_rate_estimate(self, ranges: torch.Tensor, masks: torch.Tensor, step_dt: float) -> torch.Tensor:
        N, L, R = ranges.shape
        if L < 2:
            return torch.zeros(N, R, dtype=ranges.dtype, device=self.device)
        latest_valid = self._mask_to_bool(masks[:, -1])
        previous_valid = self._mask_to_bool(masks[:, -2])
        rate = (ranges[:, -1] - ranges[:, -2]) / step_dt
        valid_pair = latest_valid & previous_valid
        return torch.where(valid_pair, rate, torch.zeros_like(rate))

    @staticmethod
    def _mask_to_bool(mask: torch.Tensor) -> torch.Tensor:
        if mask.dtype == torch.bool:
            return mask
        return mask > 0.0

    def _clip_command(self, command: torch.Tensor):
        speed = command.norm(dim=-1, keepdim=True)
        scale = (self.cfg.velocity_limit / speed.clamp_min(self.cfg.eps)).clamp_max(1.0)
        clipped = command * scale
        clip_ratio = torch.where(
            speed > self.cfg.velocity_limit,
            self.cfg.velocity_limit / speed.clamp_min(self.cfg.eps),
            torch.ones_like(speed),
        )
        return clipped, clip_ratio
