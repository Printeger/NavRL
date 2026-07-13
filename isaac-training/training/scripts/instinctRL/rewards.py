"""
instinctRL-F reward integration.

This module computes reward terms from actor-clean command/sensor signals by
default. Privileged actual velocity is optional and reward-only.
"""

from dataclasses import dataclass
import math
from typing import Dict, Optional

import torch


REWARD_COMPONENT_KEYS = (
    "reward_tracking",
    "reward_anchor",
    "reward_safety",
    "reward_ics_compliance",
    "reward_intervention",
    "reward_smoothness",
    "reward_null_command_speed",
    "reward_null_command_output",
    "reward_proxy_tracking",
    "reward_preservation_low",
    "reward_preservation_high",
    "reward_command_amplification",
    "reward_height_floor",
    "reward_height_ceiling",
    "reward_collision",
    "reward_total",
)


@dataclass
class RewardConfig:
    """Configuration for instinctRL reward terms."""

    enabled: bool = True
    tracking_weight: float = 1.0
    anchor_weight: float = 4.0
    safety_weight: float = 0.5
    ics_compliance_weight: float = 1.0
    intervention_weight: float = 0.05
    smoothness_weight: float = 0.05
    null_command_speed_weight: float = 2.0
    null_command_output_weight: float = 0.1
    null_output_anchor_loss_threshold: float = 0.05
    proxy_tracking_weight: float = 0.25
    preservation_low_weight: float = 0.5
    preservation_high_weight: float = 0.5
    preservation_lower: float = 0.75
    preservation_upper: float = 1.05
    command_amplification_weight: float = 0.5
    height_floor: float = 0.5
    height_floor_weight: float = 8.0
    height_ceiling: float = 4.0
    height_ceiling_weight: float = 0.0
    collision_weight: float = 10.0
    clearance_safe: float = 0.8
    clearance_margin: float = 0.2
    max_reward_abs: float = 20.0
    use_privileged_velocity_for_reward: bool = False
    min_anchor_valid_fraction: float = 0.1
    tracking_beta_gate: bool = True
    command_eps: float = 1e-3
    eps: float = 1e-6

    def __post_init__(self):
        for name in (
            "tracking_weight",
            "anchor_weight",
            "safety_weight",
            "ics_compliance_weight",
            "intervention_weight",
            "smoothness_weight",
            "null_command_speed_weight",
            "null_command_output_weight",
            "null_output_anchor_loss_threshold",
            "proxy_tracking_weight",
            "preservation_low_weight",
            "preservation_high_weight",
            "preservation_lower",
            "preservation_upper",
            "command_amplification_weight",
            "height_floor",
            "height_floor_weight",
            "height_ceiling",
            "height_ceiling_weight",
            "collision_weight",
            "clearance_safe",
            "clearance_margin",
            "max_reward_abs",
            "min_anchor_valid_fraction",
            "command_eps",
            "eps",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            setattr(self, name, value)
        for name in (
            "tracking_weight",
            "anchor_weight",
            "safety_weight",
            "ics_compliance_weight",
            "intervention_weight",
            "smoothness_weight",
            "null_command_speed_weight",
            "null_command_output_weight",
            "null_output_anchor_loss_threshold",
            "proxy_tracking_weight",
            "preservation_low_weight",
            "preservation_high_weight",
            "command_amplification_weight",
            "height_floor_weight",
            "height_ceiling_weight",
            "collision_weight",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be >= 0")
        if self.height_floor < 0.0:
            raise ValueError("height_floor must be >= 0")
        if self.height_ceiling < self.height_floor:
            raise ValueError("height_ceiling must be >= height_floor")
        if self.clearance_safe <= 0.0:
            raise ValueError("clearance_safe must be > 0")
        if self.clearance_margin < 0.0:
            raise ValueError("clearance_margin must be >= 0")
        if self.max_reward_abs <= 0.0:
            raise ValueError("max_reward_abs must be > 0")
        if self.preservation_lower < 0.0:
            raise ValueError("preservation_lower must be >= 0")
        if self.preservation_upper < self.preservation_lower:
            raise ValueError("preservation_upper must be >= preservation_lower")
        if not (0.0 <= self.min_anchor_valid_fraction <= 1.0):
            raise ValueError("min_anchor_valid_fraction must satisfy 0.0 <= value <= 1.0")
        if self.command_eps <= 0.0:
            raise ValueError("command_eps must be > 0")
        if self.eps <= 0.0:
            raise ValueError("eps must be > 0")

    @classmethod
    def from_namespace(cls, cfg) -> "RewardConfig":
        return cls(
            enabled=bool(getattr(cfg, "enabled", True)),
            tracking_weight=float(getattr(cfg, "tracking_weight", 1.0)),
            anchor_weight=float(getattr(cfg, "anchor_weight", 4.0)),
            safety_weight=float(getattr(cfg, "safety_weight", 0.5)),
            ics_compliance_weight=float(getattr(cfg, "ics_compliance_weight", 1.0)),
            intervention_weight=float(getattr(cfg, "intervention_weight", 0.05)),
            smoothness_weight=float(getattr(cfg, "smoothness_weight", 0.05)),
            null_command_speed_weight=float(getattr(cfg, "null_command_speed_weight", 2.0)),
            null_command_output_weight=float(getattr(cfg, "null_command_output_weight", 0.1)),
            null_output_anchor_loss_threshold=float(
                getattr(cfg, "null_output_anchor_loss_threshold", 0.05)
            ),
            proxy_tracking_weight=float(getattr(cfg, "proxy_tracking_weight", 0.25)),
            preservation_low_weight=float(getattr(cfg, "preservation_low_weight", 0.5)),
            preservation_high_weight=float(getattr(cfg, "preservation_high_weight", 0.5)),
            preservation_lower=float(getattr(cfg, "preservation_lower", 0.75)),
            preservation_upper=float(getattr(cfg, "preservation_upper", 1.05)),
            command_amplification_weight=float(
                getattr(cfg, "command_amplification_weight", 0.5)
            ),
            height_floor=float(getattr(cfg, "height_floor", 0.5)),
            height_floor_weight=float(getattr(cfg, "height_floor_weight", 8.0)),
            height_ceiling=float(getattr(cfg, "height_ceiling", 4.0)),
            height_ceiling_weight=float(getattr(cfg, "height_ceiling_weight", 0.0)),
            collision_weight=float(getattr(cfg, "collision_weight", 10.0)),
            clearance_safe=float(getattr(cfg, "clearance_safe", 0.8)),
            clearance_margin=float(getattr(cfg, "clearance_margin", 0.2)),
            max_reward_abs=float(getattr(cfg, "max_reward_abs", 20.0)),
            use_privileged_velocity_for_reward=bool(
                getattr(cfg, "use_privileged_velocity_for_reward", False)
            ),
            min_anchor_valid_fraction=float(getattr(cfg, "min_anchor_valid_fraction", 0.1)),
            tracking_beta_gate=bool(getattr(cfg, "tracking_beta_gate", True)),
            command_eps=float(getattr(cfg, "command_eps", 1e-3)),
            eps=float(getattr(cfg, "eps", 1e-6)),
        )


@dataclass
class RewardTerms:
    """Total reward plus public components and internal debug cache."""

    total: torch.Tensor
    components: Dict[str, torch.Tensor]
    cache: Dict[str, torch.Tensor]


class InstinctRLRewardComputer:
    """Compute instinctRL-F reward terms."""

    def __init__(self, config: RewardConfig, *, device: str = "cuda:0"):
        self.cfg = config
        self.device = torch.device(device)

    def compute(
        self,
        *,
        v_cmd_b: torch.Tensor,
        v_final_b: torch.Tensor,
        prev_v_final_b: torch.Tensor,
        min_clearance: Optional[torch.Tensor],
        collision: torch.Tensor,
        anchor_loss: Optional[torch.Tensor] = None,
        anchor_active: Optional[torch.Tensor] = None,
        anchor_valid_fraction: Optional[torch.Tensor] = None,
        ics_beta: Optional[torch.Tensor] = None,
        ics_emergency: Optional[torch.Tensor] = None,
        ics_active_beam_count: Optional[torch.Tensor] = None,
        actual_velocity_b: Optional[torch.Tensor] = None,
        height_w: Optional[torch.Tensor] = None,
    ) -> RewardTerms:
        v_cmd = self._vector("v_cmd_b", v_cmd_b)
        N = v_cmd.shape[0]
        v_final = self._vector("v_final_b", v_final_b, N)
        prev_v_final = self._vector("prev_v_final_b", prev_v_final_b, N)

        beta = self._scalar("ics_beta", ics_beta, N, default=1.0).clamp(0.0, 1.0)
        emergency = self._scalar("ics_emergency", ics_emergency, N, default=0.0).clamp(0.0, 1.0)
        anchor_loss = self._scalar("anchor_loss", anchor_loss, N, default=0.0).clamp_min(0.0)
        anchor_active = self._scalar("anchor_active", anchor_active, N, default=0.0).clamp(0.0, 1.0)
        anchor_valid_fraction = self._scalar(
            "anchor_valid_fraction", anchor_valid_fraction, N, default=0.0
        ).clamp(0.0, 1.0)
        min_clearance = self._clearance(min_clearance, N, dtype=v_cmd.dtype)
        collision_f = self._scalar("collision", collision, N, default=0.0).clamp(0.0, 1.0)
        active_beam_count = self._scalar(
            "ics_active_beam_count", ics_active_beam_count, N, default=0.0
        ).clamp_min(0.0)
        height_w = self._scalar("height_w", height_w, N, default=self.cfg.height_floor)

        tracking_signal = v_final
        actual_signal = v_final
        if actual_velocity_b is not None:
            actual_signal = self._vector("actual_velocity_b", actual_velocity_b, N)
        if self.cfg.use_privileged_velocity_for_reward and actual_velocity_b is not None:
            tracking_signal = actual_signal

        command_norm = v_cmd.norm(dim=-1, keepdim=True)
        command_active = (command_norm > self.cfg.command_eps).to(v_cmd.dtype)
        null_command = 1.0 - command_active
        tracking_error = (tracking_signal - v_cmd).norm(dim=-1, keepdim=True)
        raw_tracking = -self.cfg.tracking_weight * command_active * tracking_error
        if self.cfg.tracking_beta_gate:
            tracking_gate = beta * (1.0 - emergency)
        else:
            tracking_gate = torch.ones_like(beta)
        reward_tracking = raw_tracking * tracking_gate
        reward_ics_compliance = (
            self.cfg.ics_compliance_weight
            * (-raw_tracking)
            * (1.0 - tracking_gate).clamp(0.0, 1.0)
        )

        anchor_valid = (anchor_valid_fraction >= self.cfg.min_anchor_valid_fraction).to(v_cmd.dtype)
        reward_anchor = -self.cfg.anchor_weight * anchor_loss * anchor_active * anchor_valid
        station_correction_allowed = (
            anchor_active
            * anchor_valid
            * (anchor_loss >= self.cfg.null_output_anchor_loss_threshold).to(v_cmd.dtype)
        )
        null_output_bias_gate = (1.0 - station_correction_allowed).clamp(0.0, 1.0)

        clearance_threshold = self.cfg.clearance_safe + self.cfg.clearance_margin
        clearance_violation = (clearance_threshold - min_clearance).clamp_min(0.0)
        reward_safety = -self.cfg.safety_weight * clearance_violation

        reward_intervention = -self.cfg.intervention_weight * (1.0 - beta)
        reward_smoothness = -self.cfg.smoothness_weight * (
            v_final - prev_v_final
        ).norm(dim=-1, keepdim=True)
        reward_null_command_speed = (
            -self.cfg.null_command_speed_weight
            * null_command
            * actual_signal.norm(dim=-1, keepdim=True)
        )
        reward_null_command_output = (
            -self.cfg.null_command_output_weight
            * null_command
            * null_output_bias_gate
            * v_final.norm(dim=-1, keepdim=True)
        )
        command_safe_gate = (
            command_active
            * (beta >= 0.999).to(v_cmd.dtype)
            * (1.0 - emergency)
        )
        reward_proxy_tracking = (
            -self.cfg.proxy_tracking_weight
            * command_safe_gate
            * (v_final - v_cmd).norm(dim=-1, keepdim=True)
        )
        preservation_ratio = v_final.norm(dim=-1, keepdim=True) / command_norm.clamp_min(
            self.cfg.command_eps
        )
        preservation_low_violation = (
            self.cfg.preservation_lower - preservation_ratio
        ).clamp_min(0.0) * command_safe_gate
        preservation_high_violation = (
            preservation_ratio - self.cfg.preservation_upper
        ).clamp_min(0.0) * command_safe_gate
        reward_preservation_low = (
            -self.cfg.preservation_low_weight * preservation_low_violation
        )
        reward_preservation_high = (
            -self.cfg.preservation_high_weight * preservation_high_violation
        )
        amplification = (preservation_ratio - 1.0).clamp_min(0.0)
        reward_command_amplification = (
            -self.cfg.command_amplification_weight * command_safe_gate * amplification
        )
        height_floor_violation = (self.cfg.height_floor - height_w).clamp_min(0.0)
        reward_height_floor = (
            -self.cfg.height_floor_weight * height_floor_violation.square()
        )
        height_ceiling_margin = self.cfg.height_ceiling - height_w
        height_ceiling_violation = (height_w - self.cfg.height_ceiling).clamp_min(0.0)
        reward_height_ceiling = (
            -self.cfg.height_ceiling_weight * height_ceiling_violation.square()
        )
        reward_collision = -self.cfg.collision_weight * collision_f

        components = {
            "reward_tracking": reward_tracking,
            "reward_anchor": reward_anchor,
            "reward_safety": reward_safety,
            "reward_ics_compliance": reward_ics_compliance,
            "reward_intervention": reward_intervention,
            "reward_smoothness": reward_smoothness,
            "reward_null_command_speed": reward_null_command_speed,
            "reward_null_command_output": reward_null_command_output,
            "reward_proxy_tracking": reward_proxy_tracking,
            "reward_preservation_low": reward_preservation_low,
            "reward_preservation_high": reward_preservation_high,
            "reward_command_amplification": reward_command_amplification,
            "reward_height_floor": reward_height_floor,
            "reward_height_ceiling": reward_height_ceiling,
            "reward_collision": reward_collision,
        }
        total_raw = sum(components.values())
        scale = torch.where(
            total_raw.abs() > self.cfg.max_reward_abs,
            self.cfg.max_reward_abs / total_raw.abs().clamp_min(self.cfg.eps),
            torch.ones_like(total_raw),
        )
        components = {key: value * scale for key, value in components.items()}
        total = sum(components.values())
        components["reward_total"] = total

        for key, value in list(components.items()):
            components[key] = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        total = components["reward_total"]

        cache = {
            "tracking_error_norm": tracking_error,
            "tracking_gate": tracking_gate,
            "null_command_speed": actual_signal.norm(dim=-1, keepdim=True) * null_command,
            "null_command_output_speed": v_final.norm(dim=-1, keepdim=True) * null_command,
            "null_command_output_bias_gate": null_output_bias_gate * null_command,
            "command_preservation_ratio": preservation_ratio * command_active,
            "preservation_low_violation": preservation_low_violation,
            "preservation_high_violation": preservation_high_violation,
            "command_amplification": amplification * command_safe_gate,
            "height_floor_violation": height_floor_violation,
            "height_ceiling_violation": height_ceiling_violation,
            "height_ceiling_margin": height_ceiling_margin,
            "clearance_violation": clearance_violation,
            "reward_clip_scale": scale,
            "ics_active_beam_count": active_beam_count,
        }
        return RewardTerms(total=total, components=components, cache=cache)

    def _vector(self, name: str, value: torch.Tensor, N: Optional[int] = None) -> torch.Tensor:
        if value is None:
            raise ValueError(f"{name} is required")
        if value.device != self.device:
            raise ValueError(f"{name} must be on reward computer device")
        if value.dim() == 3 and value.shape[1] == 1 and value.shape[-1] == 3:
            value = value.squeeze(1)
        if value.dim() != 2 or value.shape[-1] != 3:
            raise ValueError(f"{name} must have shape [N,3] or [N,1,3]")
        if N is not None and value.shape[0] != N:
            raise ValueError(f"{name} batch size must match")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must be finite")
        return value

    def _scalar(
        self,
        name: str,
        value: Optional[torch.Tensor],
        N: int,
        *,
        default: float,
    ) -> torch.Tensor:
        if value is None:
            return torch.full((N, 1), default, dtype=torch.float32, device=self.device)
        if value.device != self.device:
            raise ValueError(f"{name} must be on reward computer device")
        if value.dim() == 1:
            value = value.reshape(N, 1)
        elif value.dim() == 2 and value.shape == (1, 1) and N > 1:
            value = value.expand(N, 1)
        elif value.dim() == 2 and value.shape == (N, 1):
            pass
        elif value.dim() == 3 and value.shape == (N, 1, 1):
            value = value.squeeze(1)
        else:
            raise ValueError(f"{name} must have shape [N] or [N,1]")
        if value.shape != (N, 1):
            raise ValueError(f"{name} batch size must match")
        if value.dtype == torch.bool:
            value = value.to(dtype=torch.float32)
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must be finite")
        return value.float()

    def _clearance(
        self,
        value: Optional[torch.Tensor],
        N: int,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        threshold = self.cfg.clearance_safe + self.cfg.clearance_margin
        if value is None:
            return torch.full((N, 1), threshold, dtype=dtype, device=self.device)
        if value.device != self.device:
            raise ValueError("min_clearance must be on reward computer device")
        if value.dim() == 1:
            value = value.reshape(N, 1)
        elif value.dim() == 2 and value.shape == (1, 1) and N > 1:
            value = value.expand(N, 1)
        elif value.dim() == 2 and value.shape == (N, 1):
            pass
        else:
            raise ValueError("min_clearance must have shape [N] or [N,1]")
        if value.shape != (N, 1):
            raise ValueError("min_clearance batch size must match")
        value = value.to(dtype=dtype)
        return torch.nan_to_num(value, nan=threshold, posinf=threshold, neginf=0.0)
