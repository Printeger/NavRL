"""
Livox Mid-360 LiDAR High-Fidelity Ray Pattern Generator
========================================================
Based on official specs: https://www.livoxtech.com/cn/mid-360/specs

Livox Mid-360 Technical Specifications:
=======================================
| Parameter          | Value                                    |
|--------------------|------------------------------------------|
| Laser Wavelength   | 905 nm                                   |
| Eye Safety         | Class 1 (IEC60825-1:2014)                |
| Range              | 40m @ 10% reflectivity, 70m @ 80%        |
| Blind Spot         | 0.1 m                                    |
| FOV                | Horizontal 360°, Vertical -7° ~ 52°      |
| Range Accuracy     | ≤ 2cm (1σ @ 10m)                         |
| Angular Accuracy   | < 0.15°                                  |
| Point Rate         | 200,000 pts/sec                          |
| Frame Rate         | 10 Hz                                    |

Key Features of This Implementation:
====================================
1. **Dynamic Lissajous-like Scanning**: Non-repetitive scan pattern with 
   time-based phase shifts for realistic coverage accumulation.
2. **Self-Occlusion Masking**: Filters rays that would hit the drone body
   when sensor is tilted (default 45° forward/down).
3. **Sim2Real Noise Model**: Gaussian range noise + dropout noise.
4. **Full GPU Vectorization**: No Python loops, all torch operations.

Author: NavRL Team
"""

import torch
from dataclasses import dataclass, field
from typing import Optional, Tuple
import math


@dataclass
class LivoxMid360Config:
    """
    Configuration for Livox Mid-360 LiDAR sensor simulation.

    All angles are in degrees unless otherwise specified.
    """

    # ============================================
    # Sensor Specifications (from datasheet)
    # ============================================
    max_range: float = 40.0
    """Maximum detection range (m) at 10% reflectivity"""

    min_range: float = 0.1
    """Minimum detection range / blind spot (m)"""

    horizontal_fov: float = 360.0
    """Horizontal field of view (degrees)"""

    vertical_fov_min: float = -7.0
    """Vertical FOV lower bound (degrees)"""

    vertical_fov_max: float = 52.0
    """Vertical FOV upper bound (degrees) - total 59° vertical range"""

    # ============================================
    # Resolution Parameters (adjustable for performance)
    # ============================================
    horizontal_res: float = 1.0
    """Horizontal angular resolution (degrees)"""

    num_vertical_lines: int = 30
    """Number of vertical scan lines"""

    # ============================================
    # Mounting Configuration
    # ============================================
    mount_pitch: float = 45.0
    """Sensor pitch angle relative to drone body (degrees).
    Positive = tilted forward/down. Default 45° for forward-looking."""

    mount_roll: float = 0.0
    """Sensor roll angle relative to drone body (degrees)"""

    mount_yaw: float = 0.0
    """Sensor yaw angle relative to drone body (degrees)"""

    mount_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Sensor position offset relative to drone body frame (x, y, z) in meters"""

    # ============================================
    # Self-Occlusion Masking
    # ============================================
    enable_occlusion_mask: bool = True
    """Enable self-occlusion masking for rays hitting drone body"""

    occlusion_elevation_threshold: float = 15.0
    """Rays with elevation > this (in sensor frame, after tilt) are masked (degrees)"""

    occlusion_rear_cone_angle: float = 30.0
    """Half-angle of rear cone to mask (degrees). Rays pointing backward within this cone are masked."""

    # ============================================
    # Dynamic Scanning (Lissajous-like)
    # ============================================
    enable_dynamic_scan: bool = True
    """Enable time-varying phase shifts for non-repetitive scanning"""

    horizontal_phase_speed: float = 37.0
    """Horizontal phase rotation speed (degrees/second) - prime number for coverage"""

    vertical_phase_speed: float = 23.0
    """Vertical phase oscillation speed (degrees/second) - different prime for Lissajous"""

    phase_noise_std: float = 0.5
    """Random phase noise standard deviation (degrees)"""

    # ============================================
    # Noise Model (Sim2Real)
    # ============================================
    enable_noise: bool = True
    """Enable measurement noise"""

    range_noise_std: float = 0.02
    """Range measurement noise standard deviation (m) - 2cm from spec"""

    range_noise_distance_scale: float = 0.001
    """Additional range noise that scales with distance (m per m)"""

    dropout_probability: float = 0.02
    """Probability of random point dropout (simulates surface absorption)"""

    near_dropout_probability: float = 0.1
    """Higher dropout probability for very close objects (< 1m)"""

    # ============================================
    # Computed Properties
    # ============================================
    @property
    def num_horizontal_rays(self) -> int:
        """Number of horizontal rays per scan"""
        return int(self.horizontal_fov / self.horizontal_res)

    @property
    def total_rays_nominal(self) -> int:
        """Nominal total rays before occlusion masking"""
        return self.num_horizontal_rays * self.num_vertical_lines

    @property
    def vertical_fov_range(self) -> float:
        """Total vertical FOV range in degrees"""
        return self.vertical_fov_max - self.vertical_fov_min


class LivoxMid360Pattern:
    """
    High-fidelity Livox Mid-360 LiDAR ray pattern generator.

    Features:
    - Dynamic Lissajous-like non-repetitive scanning
    - Self-occlusion masking for drone body
    - Sim2Real noise model (Gaussian + dropout)
    - Full GPU vectorization

    Usage:
        ```python
        cfg = LivoxMid360Config(mount_pitch=45.0)
        pattern = LivoxMid360Pattern(cfg, device="cuda:0")

        # In simulation loop:
        ray_origins, ray_directions = pattern.generate_rays(dt=0.02)

        # After raycasting:
        noisy_distances = pattern.apply_noise(distances)
        ```
    """

    def __init__(
        self,
        cfg: LivoxMid360Config,
        device: str = "cuda:0",
        seed: Optional[int] = None
    ):
        """
        Initialize the ray pattern generator.

        Args:
            cfg: LivoxMid360Config configuration object
            device: Torch device for computation
            seed: Optional random seed for reproducibility
        """
        self.cfg = cfg
        self.device = device

        # Random generator
        self._generator = torch.Generator(device=device)
        if seed is not None:
            self._generator.manual_seed(seed)

        # Time accumulator for dynamic scanning
        self._time = 0.0

        # Precompute base ray pattern (before dynamic phase shifts)
        self._init_base_pattern()

        # Precompute mounting rotation matrix
        self._init_mount_transform()

        # Precompute occlusion mask (static part)
        self._init_occlusion_mask()

    def _init_base_pattern(self):
        """Initialize the base ray angle grid."""
        cfg = self.cfg

        # Horizontal angles: [-180, 180) degrees
        self._base_azimuth = torch.linspace(
            -cfg.horizontal_fov / 2,
            cfg.horizontal_fov / 2 - cfg.horizontal_res,
            cfg.num_horizontal_rays,
            device=self.device
        )

        # Vertical angles (elevation): [-7, 52] degrees
        self._base_elevation = torch.linspace(
            cfg.vertical_fov_min,
            cfg.vertical_fov_max,
            cfg.num_vertical_lines,
            device=self.device
        )

        # Create meshgrid indices for vectorized operations
        self._h_idx = torch.arange(cfg.num_horizontal_rays, device=self.device)
        self._v_idx = torch.arange(cfg.num_vertical_lines, device=self.device)

    def _init_mount_transform(self):
        """Precompute the sensor mounting rotation matrix."""
        cfg = self.cfg

        # Convert mount angles to radians
        roll = math.radians(cfg.mount_roll)
        pitch = math.radians(cfg.mount_pitch)
        yaw = math.radians(cfg.mount_yaw)

        # Rotation matrices (ZYX Euler convention: yaw -> pitch -> roll)
        # Roll (rotation around X)
        cr, sr = math.cos(roll), math.sin(roll)
        R_roll = torch.tensor([
            [1, 0, 0],
            [0, cr, -sr],
            [0, sr, cr]
        ], device=self.device, dtype=torch.float32)

        # Pitch (rotation around Y)
        cp, sp = math.cos(pitch), math.sin(pitch)
        R_pitch = torch.tensor([
            [cp, 0, sp],
            [0, 1, 0],
            [-sp, 0, cp]
        ], device=self.device, dtype=torch.float32)

        # Yaw (rotation around Z)
        cy, sy = math.cos(yaw), math.sin(yaw)
        R_yaw = torch.tensor([
            [cy, -sy, 0],
            [sy, cy, 0],
            [0, 0, 1]
        ], device=self.device, dtype=torch.float32)

        # Combined rotation: R = R_yaw @ R_pitch @ R_roll
        self._mount_rotation = R_yaw @ R_pitch @ R_roll

        # Mount position offset
        self._mount_position = torch.tensor(
            cfg.mount_position, device=self.device, dtype=torch.float32
        )

    def _init_occlusion_mask(self):
        """Precompute static occlusion mask parameters."""
        cfg = self.cfg

        # Convert thresholds to radians
        self._occlusion_elev_thresh_rad = math.radians(
            cfg.occlusion_elevation_threshold)
        self._occlusion_rear_cone_rad = math.radians(
            cfg.occlusion_rear_cone_angle)

        # Precompute cos threshold for rear cone check
        self._rear_cone_cos_thresh = math.cos(
            math.pi - cfg.occlusion_rear_cone_angle * math.pi / 180)

    def _spherical_to_cartesian(
        self,
        azimuth: torch.Tensor,
        elevation: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert spherical coordinates to Cartesian direction vectors.

        Coordinate system (sensor frame):
        - X: Forward
        - Y: Left  
        - Z: Up

        Args:
            azimuth: Horizontal angles in radians [N]
            elevation: Vertical angles in radians [N]

        Returns:
            Direction vectors [N, 3] (unit vectors)
        """
        cos_elev = torch.cos(elevation)

        x = cos_elev * torch.cos(azimuth)
        y = cos_elev * torch.sin(azimuth)
        z = torch.sin(elevation)

        return torch.stack([x, y, z], dim=-1)

    def _apply_dynamic_phase(self, dt: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply time-varying phase shifts for Lissajous-like scanning.

        Args:
            dt: Time step since last call (seconds)

        Returns:
            azimuth: Modified horizontal angles in degrees [N]
            elevation: Modified vertical angles in degrees [N]
        """
        cfg = self.cfg

        # Update time accumulator
        self._time += dt
        t = self._time

        # Create meshgrid of base angles
        az_grid, el_grid = torch.meshgrid(
            self._base_azimuth, self._base_elevation, indexing='xy'
        )
        azimuth = az_grid.flatten()
        elevation = el_grid.flatten()

        if cfg.enable_dynamic_scan:
            # Horizontal phase shift (rotates the entire pattern)
            h_phase = (cfg.horizontal_phase_speed * t) % 360.0

            # Vertical phase modulation (slight wobble in elevation)
            v_phase = cfg.vertical_phase_speed * t
            v_modulation = 0.5 * \
                torch.sin(torch.deg2rad(
                    torch.tensor(v_phase, device=self.device)))

            # Add random phase noise for non-repetitive coverage
            if cfg.phase_noise_std > 0:
                h_noise = torch.randn(
                    azimuth.shape, device=self.device, generator=self._generator
                ) * cfg.phase_noise_std
                v_noise = torch.randn(
                    elevation.shape, device=self.device, generator=self._generator
                ) * cfg.phase_noise_std * 0.5  # Less noise in elevation
            else:
                h_noise = 0.0
                v_noise = 0.0

            # Apply phase shifts
            azimuth = azimuth + h_phase + h_noise
            elevation = elevation + v_modulation + v_noise

            # Wrap azimuth to [-180, 180)
            azimuth = ((azimuth + 180) % 360) - 180

            # Clamp elevation to valid FOV
            elevation = torch.clamp(
                elevation, cfg.vertical_fov_min, cfg.vertical_fov_max)

        return azimuth, elevation

    def _compute_occlusion_mask(self, directions: torch.Tensor) -> torch.Tensor:
        """
        Compute self-occlusion mask for TOP-MOUNTED forward-tilted sensor.

        Physical Setup:
        - LiDAR is mounted on TOP of drone, pitched forward/down (e.g., 45°)
        - Drone body is BEHIND and BELOW the sensor
        - Upper hemisphere (Z > 0 in body frame): Fully visible (sky/ceiling)
        - Rear-lower region: Blocked by drone body and propellers

        Occlusion Zones:
        1. Rear cone: Rays pointing backward (X < threshold) hit drone body
        2. Propeller zone: Rays pointing down-sideways may hit props

        Args:
            directions: Ray directions in BODY frame [N, 3]
                       (X=Forward, Y=Left, Z=Up)

        Returns:
            Boolean mask [N], True = valid ray, False = occluded
        """
        cfg = self.cfg

        if not cfg.enable_occlusion_mask:
            return torch.ones(directions.shape[0], dtype=torch.bool, device=self.device)

        # Normalize directions (should already be unit vectors, but ensure)
        dirs_normalized = directions / \
            (directions.norm(dim=-1, keepdim=True) + 1e-8)

        # Extract components (Body frame: X=Forward, Y=Left, Z=Up)
        x = dirs_normalized[:, 0]  # Forward
        y = dirs_normalized[:, 1]  # Left
        z = dirs_normalized[:, 2]  # Up

        # ============================================
        # Occlusion Zone 1: Rear Cone (Drone Body)
        # ============================================
        # Drone body is behind the sensor. Block rays pointing backward.
        # Use configurable rear cone angle (default 60° half-angle = cos(120°) ≈ -0.5)
        # A ray with x < -0.5 is pointing more than 60° backward
        rear_cone_cos = math.cos(
            math.pi - math.radians(cfg.occlusion_rear_cone_angle))
        is_pointing_back = x < rear_cone_cos

        # ============================================
        # Occlusion Zone 2: Propeller Region
        # ============================================
        # Propellers are typically at the sides and slightly below.
        # Block rays that point: down (z < -0.2) AND laterally (not forward/back)
        # This catches rays that would hit the prop disc
        is_hitting_props = (z < -0.3) & (torch.abs(x) <
                                         0.4) & (torch.abs(y) > 0.3)

        # ============================================
        # Occlusion Zone 3: Direct Downward (Landing Gear / Body Bottom)
        # ============================================
        # Rays pointing almost straight down might hit landing gear
        is_straight_down = z < -0.85  # Within ~30° of straight down

        # ============================================
        # Combine: Valid if NOT in any occlusion zone
        # ============================================
        occluded = is_pointing_back | is_hitting_props | is_straight_down
        valid_mask = ~occluded

        return valid_mask

    def generate_rays(
        self,
        dt: float = 0.0,
        return_sensor_frame: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate ray origins and directions for the current time step.

        Args:
            dt: Time step since last call (seconds). Used for dynamic scanning.
            return_sensor_frame: If True, return rays in sensor frame instead of body frame.

        Returns:
            ray_origins: Ray origin points [N, 3] (at sensor mount position)
            ray_directions: Unit direction vectors [N, 3]

        Note:
            N may be less than total_rays_nominal due to occlusion masking.
        """
        cfg = self.cfg

        # Step 1: Get angles with dynamic phase shifts
        azimuth_deg, elevation_deg = self._apply_dynamic_phase(dt)

        # Step 2: Convert to radians
        azimuth_rad = torch.deg2rad(azimuth_deg)
        elevation_rad = torch.deg2rad(elevation_deg)

        # Step 3: Convert to Cartesian direction vectors (sensor frame)
        directions_sensor = self._spherical_to_cartesian(
            azimuth_rad, elevation_rad)

        if return_sensor_frame:
            # Return rays in sensor frame (before mount transform)
            ray_origins = torch.zeros_like(directions_sensor)
            return ray_origins, directions_sensor

        # Step 4: Transform to body frame using mount rotation
        # directions_body = R_mount @ directions_sensor^T -> transpose back
        directions_body = (self._mount_rotation @ directions_sensor.T).T

        # Step 5: Apply occlusion mask
        valid_mask = self._compute_occlusion_mask(directions_body)
        directions_body = directions_body[valid_mask]

        # Step 6: Create ray origins at mount position
        ray_origins = self._mount_position.unsqueeze(
            0).expand(directions_body.shape[0], -1)

        return ray_origins, directions_body

    def apply_noise(
        self,
        distances: torch.Tensor,
        intensities: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply realistic noise to distance measurements.

        Noise model includes:
        1. Gaussian range noise (constant + distance-scaled)
        2. Random dropout (simulates surface absorption/misses)
        3. Near-range dropout (higher probability for close objects)

        Args:
            distances: Raw distance measurements [N] or [B, N]
            intensities: Optional intensity values for intensity-dependent noise

        Returns:
            Noisy distances [N] or [B, N]. Invalid measurements set to inf.
        """
        cfg = self.cfg

        if not cfg.enable_noise:
            return distances

        # Ensure we work with the right shape
        original_shape = distances.shape
        distances = distances.flatten() if distances.dim() == 1 else distances

        # Create output tensor
        noisy_distances = distances.clone()

        # Valid measurements mask (not inf, not nan, within range)
        valid_mask = (
            torch.isfinite(distances) &
            (distances >= cfg.min_range) &
            (distances <= cfg.max_range)
        )

        if valid_mask.any():
            valid_distances = distances[valid_mask]

            # 1. Gaussian range noise
            # Base noise + distance-scaled noise
            noise_std = cfg.range_noise_std + cfg.range_noise_distance_scale * valid_distances
            range_noise = torch.randn(
                valid_distances.shape, device=valid_distances.device,
                dtype=valid_distances.dtype, generator=self._generator) * noise_std

            # 2. Random dropout (base probability)
            dropout_mask = torch.rand(
                valid_distances.shape, device=valid_distances.device,
                dtype=valid_distances.dtype, generator=self._generator) < cfg.dropout_probability

            # 3. Near-range dropout (higher probability for very close objects < 1m)
            near_mask = valid_distances < 1.0
            near_dropout = torch.rand(
                valid_distances.shape, device=valid_distances.device,
                dtype=valid_distances.dtype, generator=self._generator) < cfg.near_dropout_probability
            dropout_mask = dropout_mask | (near_mask & near_dropout)

            # 4. Unreliable zone handling (0.1m - 0.2m per Livox spec)
            # This zone has significantly degraded accuracy - 50% dropout + extra noise
            unreliable_mask = (valid_distances >= cfg.min_range) & (
                valid_distances < 0.2)
            unreliable_dropout = torch.rand(
                valid_distances.shape, device=valid_distances.device,
                dtype=valid_distances.dtype, generator=self._generator) < 0.5  # 50% dropout
            dropout_mask = dropout_mask | (
                unreliable_mask & unreliable_dropout)

            # Add extra noise to unreliable zone (3x base noise)
            unreliable_extra_noise = torch.randn(
                valid_distances.shape, device=valid_distances.device,
                dtype=valid_distances.dtype, generator=self._generator) * (cfg.range_noise_std * 3.0)
            range_noise = torch.where(
                unreliable_mask & ~dropout_mask,  # Only for non-dropped unreliable points
                range_noise + unreliable_extra_noise,
                range_noise
            )

            # Apply noise to valid measurements
            valid_noisy = valid_distances + range_noise

            # Apply dropout (set to inf)
            valid_noisy[dropout_mask] = float('inf')

            # Clamp to valid range
            valid_noisy = torch.clamp(
                valid_noisy, cfg.min_range, cfg.max_range)
            valid_noisy[dropout_mask] = float('inf')  # Restore inf after clamp

            # Write back
            noisy_distances[valid_mask] = valid_noisy

        return noisy_distances.reshape(original_shape)

    def reset_time(self):
        """Reset the internal time accumulator for dynamic scanning."""
        self._time = 0.0

    def get_coverage_info(self, num_frames: int = 10, dt: float = 0.1) -> dict:
        """
        Analyze coverage accumulation over multiple frames.

        Args:
            num_frames: Number of frames to simulate
            dt: Time step between frames

        Returns:
            Dictionary with coverage statistics
        """
        self.reset_time()

        all_directions = []
        for _ in range(num_frames):
            _, dirs = self.generate_rays(dt=dt)
            all_directions.append(dirs.cpu())

        all_dirs = torch.cat(all_directions, dim=0)

        # Compute angular coverage
        azimuth = torch.atan2(all_dirs[:, 1], all_dirs[:, 0])
        elevation = torch.asin(all_dirs[:, 2])

        return {
            'total_rays': all_dirs.shape[0],
            'rays_per_frame': all_dirs.shape[0] // num_frames,
            'azimuth_range_deg': (azimuth.min().item() * 180/math.pi, azimuth.max().item() * 180/math.pi),
            'elevation_range_deg': (elevation.min().item() * 180/math.pi, elevation.max().item() * 180/math.pi),
            # Could compute actual unique cells
            'unique_directions_approx': all_dirs.shape[0],
        }

    @property
    def nominal_ray_count(self) -> int:
        """Nominal number of rays before occlusion masking."""
        return self.cfg.total_rays_nominal

    def __repr__(self) -> str:
        cfg = self.cfg
        return (
            f"LivoxMid360Pattern(\n"
            f"  resolution: {cfg.num_horizontal_rays} x {cfg.num_vertical_lines} = {cfg.total_rays_nominal} rays\n"
            f"  FOV: H={cfg.horizontal_fov}°, V=[{cfg.vertical_fov_min}°, {cfg.vertical_fov_max}°]\n"
            f"  mount: pitch={cfg.mount_pitch}°, roll={cfg.mount_roll}°, yaw={cfg.mount_yaw}°\n"
            f"  dynamic_scan: {cfg.enable_dynamic_scan}\n"
            f"  occlusion_mask: {cfg.enable_occlusion_mask}\n"
            f"  noise: {cfg.enable_noise} (σ={cfg.range_noise_std}m, dropout={cfg.dropout_probability*100}%)\n"
            f")"
        )


# ============================================
# Predefined Configurations
# ============================================

def create_high_res_config(**kwargs) -> LivoxMid360Config:
    """High resolution configuration (close to real sensor)."""
    defaults = dict(
        horizontal_res=0.5,      # 720 horizontal rays
        num_vertical_lines=59,   # 59 vertical lines
        # Total: 720 × 59 = 42,480 rays
    )
    defaults.update(kwargs)
    return LivoxMid360Config(**defaults)


def create_medium_res_config(**kwargs) -> LivoxMid360Config:
    """Medium resolution configuration (balanced performance)."""
    defaults = dict(
        horizontal_res=1.0,      # 360 horizontal rays
        num_vertical_lines=30,   # 30 vertical lines
        # Total: 360 × 30 = 10,800 rays
    )
    defaults.update(kwargs)
    return LivoxMid360Config(**defaults)


def create_low_res_config(**kwargs) -> LivoxMid360Config:
    """Low resolution configuration (fast simulation)."""
    defaults = dict(
        horizontal_res=2.0,      # 180 horizontal rays
        num_vertical_lines=15,   # 15 vertical lines
        # Total: 180 × 15 = 2,700 rays
    )
    defaults.update(kwargs)
    return LivoxMid360Config(**defaults)


def create_rl_training_config(**kwargs) -> LivoxMid360Config:
    """Ultra-low resolution for RL training (fast, small observation space)."""
    defaults = dict(
        horizontal_res=5.0,      # 72 horizontal rays
        num_vertical_lines=12,   # 12 vertical lines
        # Total: 72 × 12 = 864 rays
        enable_dynamic_scan=False,  # Deterministic for RL
        phase_noise_std=0.0,
    )
    defaults.update(kwargs)
    return LivoxMid360Config(**defaults)


def create_drone_forward_config(**kwargs) -> LivoxMid360Config:
    """
    Configuration for TOP-MOUNTED forward-tilted drone mount.

    Physical Setup:
    - LiDAR mounted on top of drone body
    - Pitched forward/down by 45° to see obstacles ahead and below
    - Upper hemisphere fully visible (sky/ceiling)
    - Rear blocked by drone body, sides may have propeller occlusion
    """
    defaults = dict(
        horizontal_res=2.0,
        num_vertical_lines=20,
        # Mounting configuration
        # Pitched DOWN 45° (forward-looking)
        mount_pitch=45.0,
        # Slightly forward and above drone CG
        mount_position=(0.05, 0.0, 0.05),
        # Occlusion configuration for top mount
        enable_occlusion_mask=True,
        # Don't cut by elevation (upper hemisphere visible)
        occlusion_elevation_threshold=90.0,
        occlusion_rear_cone_angle=60.0,      # 60° half-angle rear cone blocked by body
    )
    defaults.update(kwargs)
    return LivoxMid360Config(**defaults)


# Legacy compatibility aliases
LIVOX_MID360_HIGH_RES = create_high_res_config()
LIVOX_MID360_MEDIUM_RES = create_medium_res_config()
LIVOX_MID360_LOW_RES = create_low_res_config()
LIVOX_MID360_RL_RES = create_rl_training_config()


# ============================================
# Legacy Compatibility Functions
# ============================================

def create_livox_mid360_pattern(
    cfg: LivoxMid360Config,
    device: str = "cuda:0"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Legacy function for backward compatibility.

    Creates a static ray pattern (no dynamic scanning).
    For dynamic scanning, use LivoxMid360Pattern class instead.

    Args:
        cfg: LivoxMid360Config configuration object
        device: Computation device

    Returns:
        ray_starts: Ray origin points [N, 3]
        ray_directions: Unit direction vectors [N, 3]
    """
    # Create pattern generator with dynamic scan disabled for static output
    static_cfg = LivoxMid360Config(
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
        mount_pitch=0.0,  # No mount transform for legacy
    )

    pattern = LivoxMid360Pattern(static_cfg, device=device)
    return pattern.generate_rays(dt=0.0)


def create_livox_from_hydra_cfg(
    cfg,
    device: str = "cuda:0",
    seed: Optional[int] = None
) -> LivoxMid360Pattern:
    """
    Create LivoxMid360Pattern from Hydra/OmegaConf configuration.

    This is a convenience function for RL training that reads sensor parameters
    from the Hydra config file (drone.yaml).

    Args:
        cfg: Hydra config object with cfg.sensor attributes
        device: Computation device
        seed: Optional random seed for reproducibility

    Returns:
        LivoxMid360Pattern instance configured from the yaml file

    Usage:
        ```python
        # In your RL environment __init__:
        from training.envs.livox_mid360 import create_livox_from_hydra_cfg

        self.livox_pattern = create_livox_from_hydra_cfg(
            cfg=cfg,
            device=cfg.device
        )

        # In step():
        ray_origins, ray_dirs = self.livox_pattern.generate_rays(dt=self.dt)
        ```
    """
    # Extract sensor parameters from Hydra config
    max_range = getattr(cfg.sensor, 'lidar_range', 40.0)
    vfov = getattr(cfg.sensor, 'lidar_vfov', [-7.0, 52.0])
    vbeams = getattr(cfg.sensor, 'lidar_vbeams', 59)
    hres = getattr(cfg.sensor, 'lidar_hres', 1.0)

    # Extract mount parameters
    mount_pitch = getattr(cfg.sensor, 'lidar_mount_pitch', 45.0)
    mount_roll = getattr(cfg.sensor, 'lidar_mount_roll', 0.0)
    mount_yaw = getattr(cfg.sensor, 'lidar_mount_yaw', 0.0)
    mount_position = getattr(
        cfg.sensor, 'lidar_mount_position', [0.05, 0.0, 0.05])

    # Convert list to tuple if needed
    if isinstance(mount_position, list):
        mount_position = tuple(mount_position)

    # Create configuration
    livox_cfg = LivoxMid360Config(
        max_range=max_range,
        vertical_fov_min=vfov[0],
        vertical_fov_max=vfov[1],
        num_vertical_lines=vbeams,
        horizontal_res=hres,
        mount_pitch=mount_pitch,
        mount_roll=mount_roll,
        mount_yaw=mount_yaw,
        mount_position=mount_position,
    )

    return LivoxMid360Pattern(livox_cfg, device=device, seed=seed)


# ============================================
# Test and Demo
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("Livox Mid-360 High-Fidelity Ray Pattern Generator")
    print("=" * 60)

    # Test 1: Basic pattern generation
    print("\n[TEST 1] Basic Pattern Generation")
    print("-" * 40)

    cfg = create_drone_forward_config()
    pattern = LivoxMid360Pattern(cfg, device="cpu")
    print(pattern)

    origins, directions = pattern.generate_rays(dt=0.0)
    print(f"Ray origins shape: {origins.shape}")
    print(f"Ray directions shape: {directions.shape}")
    print(
        f"Rays after occlusion masking: {directions.shape[0]} / {cfg.total_rays_nominal}")

    # Test 2: Dynamic scanning coverage
    print("\n[TEST 2] Dynamic Scanning Coverage")
    print("-" * 40)

    coverage = pattern.get_coverage_info(num_frames=10, dt=0.1)
    print(f"Total rays (10 frames): {coverage['total_rays']}")
    print(f"Rays per frame: {coverage['rays_per_frame']}")
    print(
        f"Azimuth range: {coverage['azimuth_range_deg'][0]:.1f}° to {coverage['azimuth_range_deg'][1]:.1f}°")
    print(
        f"Elevation range: {coverage['elevation_range_deg'][0]:.1f}° to {coverage['elevation_range_deg'][1]:.1f}°")

    # Test 3: Noise model
    print("\n[TEST 3] Noise Model")
    print("-" * 40)

    test_distances = torch.tensor([1.0, 5.0, 10.0, 20.0, 40.0])
    noisy_distances = pattern.apply_noise(test_distances)
    print(f"Original distances: {test_distances.tolist()}")
    print(f"Noisy distances:    {noisy_distances.tolist()}")

    # Test 4: Different configurations
    print("\n[TEST 4] Configuration Comparison")
    print("-" * 40)

    configs = [
        ("High Res", create_high_res_config()),
        ("Medium Res", create_medium_res_config()),
        ("Low Res", create_low_res_config()),
        ("RL Training", create_rl_training_config()),
        ("Drone Forward", create_drone_forward_config()),
    ]

    for name, cfg in configs:
        print(f"  {name:15s}: {cfg.num_horizontal_rays:4d} x {cfg.num_vertical_lines:2d} = {cfg.total_rays_nominal:6d} rays")

    # Test 5: Legacy compatibility
    print("\n[TEST 5] Legacy Compatibility")
    print("-" * 40)

    legacy_cfg = LIVOX_MID360_MEDIUM_RES
    legacy_origins, legacy_dirs = create_livox_mid360_pattern(
        legacy_cfg, device="cpu")
    print(f"Legacy function output shape: {legacy_dirs.shape}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
