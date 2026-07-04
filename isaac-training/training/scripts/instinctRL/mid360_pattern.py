"""
Orbit-compatible Livox MID360 ray pattern for instinctRL.

The RayCaster API only requires a pattern config with a ``func(cfg, device)``
callable returning ray starts and ray directions.  This module keeps that small
adapter local to instinctRL while reusing the existing MID360 generator.
"""

from __future__ import annotations

import hashlib
import math
import os
import sys
from dataclasses import dataclass
from typing import Callable, Sequence

import torch


_TRAINING_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_ENVS_ROOT = os.path.join(_TRAINING_ROOT, "envs")
if _ENVS_ROOT not in sys.path:
    sys.path.insert(0, _ENVS_ROOT)

from livox_mid360 import LivoxMid360Config, create_livox_mid360_pattern  # noqa: E402


def _mid360_pattern(cfg: "LivoxMid360RayPatternCfg", device: str) -> tuple[torch.Tensor, torch.Tensor]:
    livox_cfg = LivoxMid360Config(
        max_range=cfg.max_range,
        min_range=cfg.min_range,
        horizontal_fov=cfg.horizontal_fov,
        vertical_fov_min=cfg.vertical_fov_min,
        vertical_fov_max=cfg.vertical_fov_max,
        horizontal_res=cfg.horizontal_res,
        num_vertical_lines=cfg.num_vertical_lines,
        enable_dynamic_scan=False,
        enable_occlusion_mask=False,
        enable_noise=False,
        mount_pitch=0.0,
        mount_roll=0.0,
        mount_yaw=0.0,
        mount_position=(0.0, 0.0, 0.0),
    )
    ray_starts, ray_directions = create_livox_mid360_pattern(livox_cfg, device=device)
    # Orbit RayCaster applies sensor offsets in-place during initialization.
    # The Livox helper returns origins via expand(), so clone before handing
    # tensors to RayCaster to avoid overlapping-memory writes.
    return ray_starts.clone().contiguous(), ray_directions.clone().contiguous()


@dataclass
class LivoxMid360RayPatternCfg:
    """Configuration object accepted by Orbit's RayCaster pattern interface."""

    func: Callable[["LivoxMid360RayPatternCfg", str], tuple[torch.Tensor, torch.Tensor]] = _mid360_pattern
    max_range: float = 40.0
    min_range: float = 0.1
    horizontal_fov: float = 360.0
    vertical_fov_min: float = -7.0
    vertical_fov_max: float = 52.0
    horizontal_res: float = 1.0
    num_vertical_lines: int = 59

    @property
    def num_horizontal_rays(self) -> int:
        return int(self.horizontal_fov / self.horizontal_res)

    @property
    def num_rays(self) -> int:
        return self.num_horizontal_rays * self.num_vertical_lines


def create_mid360_pattern_cfg(sensor_cfg) -> LivoxMid360RayPatternCfg:
    vfov = getattr(sensor_cfg, "lidar_vfov", [-7.0, 52.0])
    return LivoxMid360RayPatternCfg(
        max_range=float(getattr(sensor_cfg, "lidar_range", 40.0)),
        vertical_fov_min=float(vfov[0]),
        vertical_fov_max=float(vfov[1]),
        horizontal_res=float(getattr(sensor_cfg, "lidar_hres", 1.0)),
        num_vertical_lines=int(getattr(sensor_cfg, "lidar_vbeams", 59)),
    )


def mount_quat_wxyz(sensor_cfg) -> tuple[float, float, float, float]:
    roll = math.radians(float(getattr(sensor_cfg, "lidar_mount_roll", 0.0)))
    pitch = math.radians(float(getattr(sensor_cfg, "lidar_mount_pitch", 0.0)))
    yaw = math.radians(float(getattr(sensor_cfg, "lidar_mount_yaw", 0.0)))

    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)

    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def mount_position(sensor_cfg) -> tuple[float, float, float]:
    position: Sequence[float] = getattr(sensor_cfg, "lidar_mount_position", (0.0, 0.0, 0.0))
    return (float(position[0]), float(position[1]), float(position[2]))


def ray_order_hash(ray_directions: torch.Tensor) -> str:
    data = ray_directions.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()[:16]
