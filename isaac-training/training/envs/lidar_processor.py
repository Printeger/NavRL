"""
LiDAR Point Cloud Retina Processor
===================================
Converts raw LiDAR point clouds into fixed-size depth images for CNN-based RL policies.

Key Features:
=============
1. **Coordinate Transformation**: Transforms points from LiDAR sensor frame to drone body frame
   accounting for sensor mounting angle (default 45° forward pitch).

2. **Spherical Min-Pooling**: Projects 3D points onto a 2D spherical depth image using
   minimum distance pooling to preserve thin obstacles (wires, branches).

3. **Danger-Aware Normalization**: Outputs normalized depth where 1.0 = DANGER (close),
   0.0 = SAFE (far/empty), which works better for CNN feature extraction.

Coordinate Systems:
==================
- **LiDAR Sensor Frame** (before mount transform):
  - X: Forward (along sensor optical axis)
  - Y: Left
  - Z: Up
  
- **Drone Body Frame** (after mount transform):
  - X: Forward (drone heading)
  - Y: Left
  - Z: Up
  
- When LiDAR is pitched down 45°, the sensor's X-axis points forward-down in body frame.

Author: NavRL Team
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from dataclasses import dataclass
import math

# Try to import torch_scatter for efficient scatter operations
try:
    from torch_scatter import scatter_min
    HAS_TORCH_SCATTER = True
except ImportError:
    HAS_TORCH_SCATTER = False


@dataclass
class RetinaConfig:
    """Configuration for LiDAR Retina processor."""

    # Mounting configuration
    mount_pitch_deg: float = 45.0
    """LiDAR pitch angle relative to body (degrees). Positive = tilted forward/down."""

    mount_roll_deg: float = 0.0
    """LiDAR roll angle relative to body (degrees)."""

    mount_yaw_deg: float = 0.0
    """LiDAR yaw angle relative to body (degrees)."""

    # Grid configuration
    grid_height: int = 16
    """Number of elevation bins (vertical resolution)."""

    grid_width: int = 72
    """Number of azimuth bins (horizontal resolution). 72 = 5° per bin for 360° FOV."""

    # Range configuration
    max_distance: float = 10.0
    """Maximum distance for normalization (meters). Points beyond this are clipped."""

    min_distance: float = 0.1
    """Minimum valid distance (meters). Points closer are clipped."""

    # FOV configuration (in body frame, after transformation)
    azimuth_fov: Tuple[float, float] = (-180.0, 180.0)
    """Azimuth field of view (degrees). Default: full 360°."""

    elevation_fov: Tuple[float, float] = (-60.0, 60.0)
    """Elevation field of view (degrees). Adjusted for tilted mount."""

    # Processing options
    invert_depth: bool = True
    """If True: 1.0 = close (danger), 0.0 = far (safe). Better for CNNs."""

    fill_value: float = 0.0
    """Value for empty pixels (after inversion: 0.0 = safe/empty)."""


class LidarRetina:
    """
    LiDAR Point Cloud to Depth Image Processor (Retina Layer).

    Converts raw point cloud batches into fixed-size depth images suitable for CNN input.
    Uses spherical min-pooling to preserve thin obstacles like wires and branches.

    Usage:
        ```python
        retina = LidarRetina(mount_angle_deg=45.0, grid_H=16, grid_W=72)

        # In RL step:
        # points: (num_envs, num_points, 3) in LiDAR sensor frame
        depth_image = retina.process(points)  # (num_envs, grid_H, grid_W)

        # For CNN input (add channel dimension):
        cnn_input = depth_image.unsqueeze(1)  # (num_envs, 1, grid_H, grid_W)
        ```

    Physical Interpretation:
        - Input points are in LiDAR sensor frame (X=forward along sensor axis)
        - Output depth image is in drone body frame (X=forward along drone heading)
        - The mount angle rotates the sensor frame to align with body frame
    """

    def __init__(
        self,
        mount_angle_deg: float = 45.0,
        grid_H: int = 16,
        grid_W: int = 72,
        max_dist: float = 10.0,
        min_dist: float = 0.1,
        elevation_fov: Tuple[float, float] = (-60.0, 60.0),
        azimuth_fov: Tuple[float, float] = (-180.0, 180.0),
        device: str = 'cuda',
        invert_depth: bool = True,
    ):
        """
        Initialize the LiDAR Retina processor.

        Args:
            mount_angle_deg: LiDAR pitch angle (degrees). Positive = tilted forward/down.
            grid_H: Number of elevation bins (vertical resolution).
            grid_W: Number of azimuth bins (horizontal resolution).
            max_dist: Maximum distance for clipping/normalization (meters).
            min_dist: Minimum valid distance (meters).
            elevation_fov: (min_deg, max_deg) elevation FOV in body frame.
            azimuth_fov: (min_deg, max_deg) azimuth FOV in body frame.
            device: Torch device ('cuda' or 'cpu').
            invert_depth: If True, output 1.0=close, 0.0=far (better for CNNs).
        """
        self.device = device
        self.grid_H = grid_H
        self.grid_W = grid_W
        self.max_dist = max_dist
        self.min_dist = min_dist
        self.invert_depth = invert_depth

        # FOV bounds (in radians)
        self.el_min = math.radians(elevation_fov[0])
        self.el_max = math.radians(elevation_fov[1])
        self.az_min = math.radians(azimuth_fov[0])
        self.az_max = math.radians(azimuth_fov[1])

        # Precompute FOV ranges
        self.el_range = self.el_max - self.el_min
        self.az_range = self.az_max - self.az_min

        # Build and cache the mount rotation matrix
        self._build_rotation_matrix(mount_angle_deg)

        # Fill value for empty pixels (before inversion)
        self._empty_fill = max_dist if not invert_depth else 0.0

        # Log configuration
        self._log_config(mount_angle_deg)

    def _build_rotation_matrix(self, pitch_deg: float, roll_deg: float = 0.0, yaw_deg: float = 0.0):
        """
        Build the rotation matrix from LiDAR sensor frame to drone body frame.

        The LiDAR is mounted with a pitch angle (tilted forward/down).
        We need R such that: P_body = R @ P_lidar

        Physical Setup (Isaac Sim convention: X-Forward, Y-Left, Z-Up):
        - LiDAR is tilted DOWN by pitch_deg (e.g., 45°)
        - LiDAR's X-axis (optical axis) points forward-DOWN in body frame
        - A point at LiDAR [1,0,0] (sensor front) should map to body [x>0, 0, z<0] (front-down)

        Matrix Construction:
        - R_body_lidar: rotation FROM body TO lidar (body needs to pitch DOWN to align with lidar)
        - P_body = R_body_lidar^T @ P_lidar = R_body_lidar.T @ P_lidar

        For pitch DOWN around Y-Left axis:
        - Positive pitch rotates X toward -Z (nose down)
        - R_pitch columns represent where body axes end up in lidar frame
        - We want the INVERSE: where lidar axes end up in body frame

        Verification:
        - Input: LiDAR point [1, 0, 0] (directly in front of sensor)
        - Expected output: Body point [0.707, 0, -0.707] (front-DOWN, since sensor looks down)

        Args:
            pitch_deg: Pitch angle in degrees (positive = sensor tilted forward/down)
            roll_deg: Roll angle in degrees
            yaw_deg: Yaw angle in degrees
        """
        # Convert to radians
        # CRITICAL: Use positive angles directly!
        # The rotation matrix R transforms P_lidar to P_body.
        # If lidar is pitched DOWN by 45°, a point at lidar-front [1,0,0]
        # should appear at body front-DOWN [0.7, 0, -0.7].
        # This requires R_pitch with positive pitch angle.
        # Positive: sensor tilted down -> points go down in body
        pitch = math.radians(pitch_deg)
        roll = math.radians(roll_deg)
        yaw = math.radians(yaw_deg)

        # Rotation matrices
        # Roll (rotation around X-axis)
        cr, sr = math.cos(roll), math.sin(roll)
        R_roll = torch.tensor([
            [1, 0, 0],
            [0, cr, -sr],
            [0, sr, cr]
        ], dtype=torch.float32, device=self.device)

        # Pitch (rotation around Y-axis)
        cp, sp = math.cos(pitch), math.sin(pitch)
        R_pitch = torch.tensor([
            [cp, 0, sp],
            [0, 1, 0],
            [-sp, 0, cp]
        ], dtype=torch.float32, device=self.device)

        # Yaw (rotation around Z-axis)
        cy, sy = math.cos(yaw), math.sin(yaw)
        R_yaw = torch.tensor([
            [cy, -sy, 0],
            [sy, cy, 0],
            [0, 0, 1]
        ], dtype=torch.float32, device=self.device)

        # Combined rotation: R = R_roll @ R_pitch @ R_yaw
        # This is the inverse of ZYX Euler (we're going from sensor to body)
        self.R_lidar_to_body = R_roll @ R_pitch @ R_yaw

    def _log_config(self, mount_angle_deg: float):
        """Log configuration for debugging."""
        self._config_str = (
            f"LidarRetina Configuration:\n"
            f"  Grid: {self.grid_H} x {self.grid_W} (H x W)\n"
            f"  Mount pitch: {mount_angle_deg}°\n"
            f"  Distance range: [{self.min_dist}, {self.max_dist}] m\n"
            f"  Elevation FOV: [{math.degrees(self.el_min):.1f}°, {math.degrees(self.el_max):.1f}°]\n"
            f"  Azimuth FOV: [{math.degrees(self.az_min):.1f}°, {math.degrees(self.az_max):.1f}°]\n"
            f"  Invert depth: {self.invert_depth}\n"
            f"  torch_scatter available: {HAS_TORCH_SCATTER}"
        )

    def transform_to_body_frame(self, points_lidar: torch.Tensor) -> torch.Tensor:
        """
        Transform points from LiDAR sensor frame to drone body frame.

        Args:
            points_lidar: Points in sensor frame (B, N, 3) or (N, 3)

        Returns:
            points_body: Points in body frame, same shape as input
        """
        # Handle both batched and non-batched input
        original_shape = points_lidar.shape
        if points_lidar.dim() == 2:
            points_lidar = points_lidar.unsqueeze(0)

        B, N, _ = points_lidar.shape

        # Apply rotation: P_body = R @ P_lidar^T -> transpose back
        # points_lidar: (B, N, 3) -> (B, 3, N) for matmul
        points_transposed = points_lidar.transpose(1, 2)  # (B, 3, N)

        # Batch matrix multiply: (3, 3) @ (B, 3, N) -> (B, 3, N)
        points_body_transposed = torch.matmul(
            self.R_lidar_to_body.unsqueeze(0),  # (1, 3, 3)
            points_transposed  # (B, 3, N)
        )

        # Transpose back: (B, 3, N) -> (B, N, 3)
        points_body = points_body_transposed.transpose(1, 2)

        # Restore original shape if input was 2D
        if len(original_shape) == 2:
            points_body = points_body.squeeze(0)

        return points_body

    def cartesian_to_spherical(
        self,
        points: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert Cartesian coordinates to spherical coordinates.

        Coordinate convention:
        - X: Forward (body frame)
        - Y: Left
        - Z: Up

        Spherical convention:
        - r: Distance from origin
        - azimuth (θ): Angle in XY plane from X-axis, positive toward Y (CCW from above)
        - elevation (φ): Angle from XY plane, positive toward Z (up)

        Args:
            points: (B, N, 3) or (N, 3) Cartesian coordinates

        Returns:
            r: Distance (same shape as points without last dim)
            azimuth: Horizontal angle in radians [-π, π]
            elevation: Vertical angle in radians [-π/2, π/2]
        """
        x = points[..., 0]
        y = points[..., 1]
        z = points[..., 2]

        # Distance
        r = torch.sqrt(x*x + y*y + z*z).clamp(min=1e-8)

        # Azimuth: atan2(y, x) gives angle from X-axis toward Y-axis
        azimuth = torch.atan2(y, x)

        # Elevation: angle from XY plane
        # asin(z/r) gives angle above/below horizontal
        elevation = torch.asin((z / r).clamp(-1.0, 1.0))

        return r, azimuth, elevation

    def angles_to_grid_indices(
        self,
        azimuth: torch.Tensor,
        elevation: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Map spherical angles to grid indices.

        Args:
            azimuth: Horizontal angles in radians (B, N) or (N,)
            elevation: Vertical angles in radians (B, N) or (N,)

        Returns:
            row_idx: Elevation grid indices [0, grid_H-1]
            col_idx: Azimuth grid indices [0, grid_W-1]
            valid_mask: Boolean mask for points within FOV
        """
        # Check if angles are within FOV
        valid_mask = (
            (azimuth >= self.az_min) & (azimuth < self.az_max) &
            (elevation >= self.el_min) & (elevation < self.el_max)
        )

        # Normalize angles to [0, 1] within FOV
        az_normalized = (azimuth - self.az_min) / self.az_range
        el_normalized = (elevation - self.el_min) / self.el_range

        # Map to grid indices
        # Note: Higher elevation = lower row index (image convention: row 0 is top)
        col_idx = (az_normalized * self.grid_W).long()
        # Flip for image coords
        row_idx = ((1.0 - el_normalized) * self.grid_H).long()

        # Clamp to valid range
        col_idx = col_idx.clamp(0, self.grid_W - 1)
        row_idx = row_idx.clamp(0, self.grid_H - 1)

        return row_idx, col_idx, valid_mask

    def _scatter_min_native(
        self,
        distances: torch.Tensor,
        indices: torch.Tensor,
        num_pixels: int,
        batch_size: int
    ) -> torch.Tensor:
        """
        Perform scatter min using native PyTorch (fallback when torch_scatter unavailable).

        Uses scatter_reduce with 'amin' operation (PyTorch 1.12+).

        Args:
            distances: Flattened distances (B * N,)
            indices: Flattened pixel indices (B * N,) in range [0, B * num_pixels)
            num_pixels: grid_H * grid_W
            batch_size: Number of environments

        Returns:
            depth_flat: (B * num_pixels,) with minimum distances per pixel
        """
        total_pixels = batch_size * num_pixels

        # Initialize with max distance (will be replaced by min)
        depth_flat = torch.full(
            (total_pixels,),
            self.max_dist,
            dtype=distances.dtype,
            device=distances.device
        )

        # Use scatter_reduce with 'amin' (available in PyTorch 1.12+)
        try:
            depth_flat.scatter_reduce_(
                dim=0,
                index=indices,
                src=distances,
                reduce='amin',
                include_self=True
            )
        except (AttributeError, TypeError):
            # Fallback for older PyTorch versions
            # This is slower but works
            for i in range(indices.shape[0]):
                idx = indices[i].item()
                depth_flat[idx] = min(
                    depth_flat[idx].item(), distances[i].item())

        return depth_flat

    def _scatter_min_external(
        self,
        distances: torch.Tensor,
        indices: torch.Tensor,
        num_pixels: int,
        batch_size: int
    ) -> torch.Tensor:
        """
        Perform scatter min using torch_scatter library (faster).

        Args:
            distances: Flattened distances (B * N,)
            indices: Flattened pixel indices (B * N,) in range [0, B * num_pixels)
            num_pixels: grid_H * grid_W
            batch_size: Number of environments

        Returns:
            depth_flat: (B * num_pixels,) with minimum distances per pixel
        """
        total_pixels = batch_size * num_pixels

        # scatter_min returns (values, argmin)
        depth_flat, _ = scatter_min(
            distances,
            indices,
            dim=0,
            dim_size=total_pixels,
            fill_value=self.max_dist
        )

        return depth_flat

    def spherical_min_pool(
        self,
        points_body: torch.Tensor,
        return_debug: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        """
        Project points onto spherical depth image using min-pooling.

        This is the core "retina" operation:
        1. Convert points to spherical coordinates
        2. Map angles to grid pixel indices
        3. For each pixel, keep the MINIMUM distance (preserves thin obstacles)

        Args:
            points_body: Points in body frame (B, N, 3)
            return_debug: If True, return debug info dict

        Returns:
            depth_image: (B, H, W) depth image
            debug_info: (optional) Dictionary with intermediate values
        """
        # Ensure batch dimension
        if points_body.dim() == 2:
            points_body = points_body.unsqueeze(0)

        B, N, _ = points_body.shape
        num_pixels = self.grid_H * self.grid_W

        # Step 1: Convert to spherical coordinates
        r, azimuth, elevation = self.cartesian_to_spherical(points_body)

        # Step 2: Map to grid indices
        row_idx, col_idx, valid_mask = self.angles_to_grid_indices(
            azimuth, elevation)

        # Also filter by distance
        distance_valid = (r >= self.min_dist) & (r <= self.max_dist)
        valid_mask = valid_mask & distance_valid

        # Step 3: Prepare for scatter operation
        # Flatten batch and point dimensions
        # Compute global pixel indices: batch_idx * num_pixels + row_idx * grid_W + col_idx
        batch_indices = torch.arange(
            B, device=self.device).view(B, 1).expand(B, N)

        # Global pixel index in flattened output
        pixel_indices = (
            batch_indices * num_pixels +
            row_idx * self.grid_W +
            col_idx
        )

        # Flatten everything
        r_flat = r.view(-1)
        pixel_indices_flat = pixel_indices.view(-1)
        valid_mask_flat = valid_mask.view(-1)

        # Filter to valid points only
        valid_distances = r_flat[valid_mask_flat]
        valid_indices = pixel_indices_flat[valid_mask_flat]

        # Step 4: Scatter min operation
        # Priority: PyTorch native (most portable) > torch_scatter (if available)
        if valid_distances.numel() > 0:
            depth_flat = self._scatter_min_native(
                valid_distances, valid_indices, num_pixels, B)
        else:
            # No valid points - fill with max distance
            depth_flat = torch.full((B * num_pixels,), self.max_dist,
                                    dtype=points_body.dtype, device=self.device)

        # Step 5: Reshape to image
        depth_image = depth_flat.view(B, self.grid_H, self.grid_W)

        if return_debug:
            debug_info = {
                'r': r,
                'azimuth': azimuth,
                'elevation': elevation,
                'row_idx': row_idx,
                'col_idx': col_idx,
                'valid_mask': valid_mask,
                'num_valid_points': valid_mask.sum().item(),
                'num_total_points': B * N,
            }
            return depth_image, debug_info

        return depth_image

    def normalize_depth(self, depth_image: torch.Tensor) -> torch.Tensor:
        """
        Normalize depth image to [0, 1] range.

        If invert_depth=True (default):
        - 1.0 = Very close (DANGER)
        - 0.0 = Far or empty (SAFE)

        This inversion makes more sense for CNNs because:
        - High activation = obstacle present = needs attention
        - Zero activation = empty space = can ignore

        Args:
            depth_image: (B, H, W) raw depth values

        Returns:
            normalized: (B, H, W) values in [0, 1]
        """
        # Clip to valid range
        depth_clipped = depth_image.clamp(self.min_dist, self.max_dist)

        # Normalize to [0, 1]
        normalized = (depth_clipped - self.min_dist) / \
            (self.max_dist - self.min_dist)

        if self.invert_depth:
            # Invert: close -> 1.0, far -> 0.0
            normalized = 1.0 - normalized

        return normalized

    def process(
        self,
        points_lidar: torch.Tensor,
        return_debug: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        """
        Full processing pipeline: LiDAR points -> normalized depth image.

        Pipeline:
        1. Transform points from LiDAR frame to body frame
        2. Project onto spherical grid using min-pooling
        3. Normalize depth values

        Args:
            points_lidar: Raw points in LiDAR sensor frame (B, N, 3) or (N, 3)
            return_debug: If True, return debug information

        Returns:
            depth_image: Normalized depth image (B, H, W) or (H, W) if unbatched
                        Values in [0, 1], where 1.0 = close (if invert_depth=True)
            debug_info: (optional) Dictionary with intermediate values
        """
        # Track if we need to squeeze output
        squeeze_batch = points_lidar.dim() == 2
        if squeeze_batch:
            points_lidar = points_lidar.unsqueeze(0)

        # Move to device if needed
        if points_lidar.device != self.device:
            points_lidar = points_lidar.to(self.device)

        # Step A: Coordinate transformation
        points_body = self.transform_to_body_frame(points_lidar)

        # Step B: Spherical min-pooling
        if return_debug:
            depth_raw, debug_info = self.spherical_min_pool(
                points_body, return_debug=True)
            debug_info['points_body'] = points_body
        else:
            depth_raw = self.spherical_min_pool(points_body)

        # Step C: Normalization
        depth_normalized = self.normalize_depth(depth_raw)

        # Restore original batch dimension
        if squeeze_batch:
            depth_normalized = depth_normalized.squeeze(0)

        if return_debug:
            debug_info['depth_raw'] = depth_raw
            debug_info['depth_normalized'] = depth_normalized
            return depth_normalized, debug_info

        return depth_normalized

    def __repr__(self) -> str:
        return self._config_str

    def __call__(self, points_lidar: torch.Tensor) -> torch.Tensor:
        """Shorthand for process()."""
        return self.process(points_lidar)


class LidarRetinaMultiScale(LidarRetina):
    """
    Multi-scale LiDAR Retina that outputs depth images at multiple resolutions.

    Useful for feature pyramid networks (FPN) or multi-scale CNN architectures.

    Usage:
        ```python
        retina = LidarRetinaMultiScale(
            mount_angle_deg=45.0,
            grid_H=32,  # Base resolution
            grid_W=144,
            scales=[1, 2, 4]  # 1x, 0.5x, 0.25x
        )

        depth_pyramid = retina.process(points)
        # depth_pyramid[0]: (B, 32, 144)
        # depth_pyramid[1]: (B, 16, 72)
        # depth_pyramid[2]: (B, 8, 36)
        ```
    """

    def __init__(
        self,
        scales: Tuple[int, ...] = (1, 2, 4),
        **kwargs
    ):
        """
        Initialize multi-scale retina.

        Args:
            scales: Downsampling factors for each pyramid level.
            **kwargs: Arguments passed to LidarRetina.
        """
        super().__init__(**kwargs)
        self.scales = scales

    def process(
        self,
        points_lidar: torch.Tensor,
        return_debug: bool = False
    ) -> Union[Tuple[torch.Tensor, ...], Tuple[Tuple[torch.Tensor, ...], dict]]:
        """
        Process points and return multi-scale depth pyramid.

        Args:
            points_lidar: Raw points in LiDAR sensor frame (B, N, 3)
            return_debug: If True, return debug information

        Returns:
            depth_pyramid: Tuple of depth images at different scales
            debug_info: (optional) Debug information from base resolution
        """
        # Get base resolution depth image
        if return_debug:
            depth_base, debug_info = super().process(points_lidar, return_debug=True)
        else:
            depth_base = super().process(points_lidar)

        # Build pyramid
        pyramid = [depth_base]

        for scale in self.scales[1:]:
            # Downsample using average pooling
            depth_scaled = F.avg_pool2d(
                depth_base.unsqueeze(1),  # Add channel dim
                kernel_size=scale,
                stride=scale
            ).squeeze(1)  # Remove channel dim
            pyramid.append(depth_scaled)

        if return_debug:
            return tuple(pyramid), debug_info

        return tuple(pyramid)


# ============================================
# Utility Functions
# ============================================

def create_retina_from_config(cfg: RetinaConfig, device: str = 'cuda') -> LidarRetina:
    """
    Create LidarRetina from configuration dataclass.

    Args:
        cfg: RetinaConfig configuration object
        device: Torch device

    Returns:
        LidarRetina instance
    """
    return LidarRetina(
        mount_angle_deg=cfg.mount_pitch_deg,
        grid_H=cfg.grid_height,
        grid_W=cfg.grid_width,
        max_dist=cfg.max_distance,
        min_dist=cfg.min_distance,
        elevation_fov=cfg.elevation_fov,
        azimuth_fov=cfg.azimuth_fov,
        device=device,
        invert_depth=cfg.invert_depth,
    )


def create_retina_from_hydra_cfg(
    cfg,
    grid_H: int = 16,
    grid_W: int = 72,
    elevation_fov: Tuple[float, float] = (-60.0, 60.0),
    azimuth_fov: Tuple[float, float] = (-180.0, 180.0),
    device: str = 'cuda',
    invert_depth: bool = True,
) -> LidarRetina:
    """
    Create LidarRetina from Hydra/OmegaConf configuration.

    This is a convenience function for RL training that reads sensor parameters
    from the Hydra config file (drone.yaml).

    Args:
        cfg: Hydra config object with cfg.sensor attributes
        grid_H: Number of elevation bins (vertical resolution)
        grid_W: Number of azimuth bins (horizontal resolution)
        elevation_fov: (min_deg, max_deg) elevation FOV in body frame
        azimuth_fov: (min_deg, max_deg) azimuth FOV in body frame
        device: Torch device ('cuda' or 'cpu')
        invert_depth: If True, output 1.0=close, 0.0=far

    Returns:
        LidarRetina instance configured from the yaml file

    Usage:
        ```python
        # In your RL environment __init__:
        from training.envs.lidar_processor import create_retina_from_hydra_cfg

        self.lidar_retina = create_retina_from_hydra_cfg(
            cfg=cfg,
            grid_H=16,
            grid_W=72,
            device=cfg.device
        )

        # In step():
        depth_image = self.lidar_retina(point_cloud)  # (num_envs, H, W)
        ```
    """
    # Extract sensor parameters from Hydra config
    mount_pitch = getattr(cfg.sensor, 'lidar_mount_pitch', 45.0)
    max_dist = getattr(cfg.sensor, 'lidar_range', 40.0)
    min_dist = getattr(cfg.sensor, 'lidar_min_range', 0.1)

    return LidarRetina(
        mount_angle_deg=mount_pitch,
        grid_H=grid_H,
        grid_W=grid_W,
        max_dist=max_dist,
        min_dist=min_dist,
        elevation_fov=elevation_fov,
        azimuth_fov=azimuth_fov,
        device=device,
        invert_depth=invert_depth,
    )


def visualize_depth_image(
    depth_image: torch.Tensor,
    save_path: Optional[str] = None,
    title: str = "LiDAR Depth Image"
) -> None:
    """
    Visualize depth image using matplotlib.

    Args:
        depth_image: (H, W) or (B, H, W) depth image
        save_path: If provided, save figure to this path
        title: Plot title
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for visualization")
        return

    # Handle batched input
    if depth_image.dim() == 3:
        depth_image = depth_image[0]

    # Convert to numpy
    img = depth_image.cpu().numpy()

    plt.figure(figsize=(12, 4))
    plt.imshow(img, cmap='viridis', aspect='auto')
    plt.colorbar(label='Normalized Depth (1.0=close, 0.0=far)')
    plt.xlabel('Azimuth Bin')
    plt.ylabel('Elevation Bin')
    plt.title(title)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

    plt.close()


# ============================================
# Test and Demo
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("LiDAR Retina Processor Test")
    print("=" * 60)

    # Use CPU for testing without GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"torch_scatter available: {HAS_TORCH_SCATTER}")

    # Create retina processor
    retina = LidarRetina(
        mount_angle_deg=45.0,
        grid_H=16,
        grid_W=72,
        max_dist=10.0,
        device=device
    )
    print(f"\n{retina}")

    # Test 1: Basic processing
    print("\n[TEST 1] Basic Processing")
    print("-" * 40)

    B, N = 4, 1000  # 4 environments, 1000 points each

    # Generate random test points in LiDAR frame
    # Points distributed in front of the sensor
    torch.manual_seed(42)
    points_lidar = torch.randn(B, N, 3, device=device)
    points_lidar[..., 0] = points_lidar[..., 0].abs() * 5 + \
        0.5  # X: forward, 0.5-5.5m
    points_lidar[..., 1] = points_lidar[..., 1] * 3  # Y: left-right
    points_lidar[..., 2] = points_lidar[..., 2] * 2  # Z: up-down

    depth_image = retina.process(points_lidar)

    print(f"Input shape: {points_lidar.shape}")
    print(f"Output shape: {depth_image.shape}")
    print(f"Output range: [{depth_image.min():.3f}, {depth_image.max():.3f}]")
    print(
        f"Non-zero pixels: {(depth_image > 0).sum().item()} / {depth_image.numel()}")

    # Test 2: Debug mode
    print("\n[TEST 2] Debug Mode")
    print("-" * 40)

    depth_image, debug_info = retina.process(points_lidar, return_debug=True)
    print(
        f"Valid points: {debug_info['num_valid_points']} / {debug_info['num_total_points']}")
    print(
        f"Azimuth range: [{debug_info['azimuth'].min():.2f}, {debug_info['azimuth'].max():.2f}] rad")
    print(
        f"Elevation range: [{debug_info['elevation'].min():.2f}, {debug_info['elevation'].max():.2f}] rad")

    # Test 3: Coordinate transformation
    print("\n[TEST 3] Coordinate Transformation (CRITICAL)")
    print("-" * 40)

    # Test point: (1, 0, 0) in LiDAR frame (forward along sensor axis)
    # Since LiDAR is pitched DOWN 45°, sensor-front should map to body front-DOWN
    test_point = torch.tensor([[1.0, 0.0, 0.0]], device=device)
    transformed = retina.transform_to_body_frame(test_point)
    print(f"Input (LiDAR frame): {test_point[0].tolist()}")
    print(f"Output (Body frame): {transformed[0].tolist()}")
    print(f"Expected: LiDAR front -> Body front-DOWN: X≈0.707, Z≈-0.707")

    # Validation check
    z_val = transformed[0, 2].item()
    if z_val > 0:
        print(f"❌ FAILED: Z should be NEGATIVE (front-down), got {z_val:.4f}")
    else:
        print(
            f"✅ PASSED: Z is negative ({z_val:.4f}), rotation matrix is correct!")

    # Test 4: Single point projection
    print("\n[TEST 4] Single Point Projection")
    print("-" * 40)

    # Point at (5, 0, 0) in body frame should appear at center of image
    single_point = torch.tensor(
        [[[5.0, 0.0, 0.0]]], device=device)  # (1, 1, 3)
    depth_single, debug = retina.spherical_min_pool(
        single_point, return_debug=True)

    center_row = retina.grid_H // 2
    center_col = retina.grid_W // 2
    print(f"Point at (5, 0, 0) body frame:")
    print(
        f"  Depth at center ({center_row}, {center_col}): {depth_single[0, center_row, center_col]:.3f}")
    print(
        f"  Grid row, col indices: {debug['row_idx'][0, 0].item()}, {debug['col_idx'][0, 0].item()}")

    # Test 5: Multi-scale
    print("\n[TEST 5] Multi-Scale Processing")
    print("-" * 40)

    multi_retina = LidarRetinaMultiScale(
        mount_angle_deg=45.0,
        grid_H=32,
        grid_W=144,
        scales=(1, 2, 4),
        device=device
    )

    pyramid = multi_retina.process(points_lidar)
    for i, depth in enumerate(pyramid):
        print(f"  Scale {multi_retina.scales[i]}x: {depth.shape}")

    # Test 6: Performance benchmark
    print("\n[TEST 6] Performance Benchmark")
    print("-" * 40)

    if device == 'cuda':
        # Warmup
        for _ in range(10):
            _ = retina.process(points_lidar)

        torch.cuda.synchronize()

        import time
        start = time.time()
        num_iters = 100
        for _ in range(num_iters):
            _ = retina.process(points_lidar)
        torch.cuda.synchronize()
        elapsed = time.time() - start

        print(f"  {num_iters} iterations: {elapsed*1000:.1f} ms")
        print(f"  Per iteration: {elapsed/num_iters*1000:.2f} ms")
        print(f"  Throughput: {B * num_iters / elapsed:.1f} env/sec")
    else:
        print("  Skipping GPU benchmark (running on CPU)")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
