"""
instinctRL R5F safety-filter helpers.

The privileged height-floor filter is sim/eval-only. It may be used at the
controller boundary for diagnosis, but root height must never enter actor
observation or a Paper-1 deployable actor-method claim.
"""

from dataclasses import dataclass
import math
from typing import Dict

import torch


@dataclass
class PrivilegedHeightSafetyFilterConfig:
    """Config for a default-off root-height floor filter."""

    enabled: bool = False
    height_floor: float = 0.5
    attenuation_band: float = 0.3
    min_downward_scale: float = 0.0
    eps: float = 1e-6

    def __post_init__(self):
        for name in ("height_floor", "attenuation_band", "min_downward_scale", "eps"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            setattr(self, name, value)
        if self.height_floor < 0.0:
            raise ValueError("height_floor must be >= 0")
        if self.attenuation_band <= 0.0:
            raise ValueError("attenuation_band must be > 0")
        if not 0.0 <= self.min_downward_scale <= 1.0:
            raise ValueError(
                "min_downward_scale must satisfy 0.0 <= value <= 1.0"
            )
        if self.eps <= 0.0:
            raise ValueError("eps must be > 0")

    @classmethod
    def from_namespace(cls, cfg) -> "PrivilegedHeightSafetyFilterConfig":
        return cls(
            enabled=bool(getattr(cfg, "privileged_height_floor_enabled", False)),
            height_floor=float(getattr(cfg, "height_floor", 0.5)),
            attenuation_band=float(getattr(cfg, "attenuation_band", 0.3)),
            min_downward_scale=float(getattr(cfg, "min_downward_scale", 0.0)),
            eps=float(getattr(cfg, "eps", 1e-6)),
        )


@dataclass
class SafetyFilterOutput:
    """Filtered body command plus boundary-only diagnostics."""

    v_final_b: torch.Tensor
    metrics: Dict[str, torch.Tensor]


class PrivilegedHeightFloorSafetyFilter:
    """Attenuate only downward body-z command from root height at the boundary."""

    def __init__(self, config: PrivilegedHeightSafetyFilterConfig, *, device: str = "cuda:0"):
        self.cfg = config
        self.device = torch.device(device)

    def __call__(
        self,
        v_final_b: torch.Tensor,
        root_height_w: torch.Tensor,
    ) -> SafetyFilterOutput:
        return self.forward(v_final_b, root_height_w)

    def forward(
        self,
        v_final_b: torch.Tensor,
        root_height_w: torch.Tensor,
    ) -> SafetyFilterOutput:
        command, output_shape = self._validate_command(v_final_b)
        height = self._validate_height(root_height_w, command.shape[0], command.dtype)
        ones = torch.ones(command.shape[0], 1, dtype=command.dtype, device=self.device)
        zeros = torch.zeros_like(ones)
        if not self.cfg.enabled:
            return SafetyFilterOutput(
                v_final_b=command.reshape(output_shape),
                metrics={
                    "safety_filter_height_scale": ones,
                    "safety_filter_height_active": zeros,
                    "safety_filter_height_margin": height - self.cfg.height_floor,
                },
            )

        margin = height - self.cfg.height_floor
        scale = (margin / self.cfg.attenuation_band).clamp(
            self.cfg.min_downward_scale,
            1.0,
        )
        downward = command[..., 2:3] < 0.0
        filtered_z = torch.where(downward, command[..., 2:3] * scale, command[..., 2:3])
        filtered = torch.cat((command[..., :2], filtered_z), dim=-1)
        active = downward & (scale < 1.0)
        return SafetyFilterOutput(
            v_final_b=filtered.reshape(output_shape),
            metrics={
                "safety_filter_height_scale": scale,
                "safety_filter_height_active": active.to(command.dtype),
                "safety_filter_height_margin": margin,
            },
        )

    def _validate_command(self, value: torch.Tensor):
        if value.device != self.device:
            raise ValueError("v_final_b must be on safety filter device")
        if value.dim() == 3 and value.shape[-2:] == (1, 3):
            command = value.squeeze(-2)
            output_shape = value.shape
        elif value.dim() == 2 and value.shape[-1] == 3:
            command = value
            output_shape = value.shape
        else:
            raise ValueError("v_final_b must have shape [N,3] or [N,1,3]")
        if not torch.isfinite(command).all():
            raise ValueError("v_final_b must be finite")
        return command, output_shape

    def _validate_height(self, value: torch.Tensor, N: int, dtype: torch.dtype) -> torch.Tensor:
        if value.device != self.device:
            raise ValueError("root_height_w must be on safety filter device")
        if value.dim() == 1:
            height = value.reshape(N, 1)
        elif value.dim() == 2 and value.shape == (N, 1):
            height = value
        elif value.dim() == 3 and value.shape == (N, 1, 1):
            height = value.squeeze(1)
        else:
            raise ValueError("root_height_w must have shape [N], [N,1], or [N,1,1]")
        if height.shape != (N, 1):
            raise ValueError("root_height_w batch size must match v_final_b")
        if not torch.isfinite(height).all():
            raise ValueError("root_height_w must be finite")
        return height.to(dtype=dtype)
