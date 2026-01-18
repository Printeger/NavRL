"""
Universal Arena Generator for Bio-Inspired UAV Reflex Policy Training
======================================================================
Generates procedural geometric constraints within a fixed Arena (10m x 10m x 4m).
Covers 5 abstract geometric modes representing fundamental physical constraints.

Modes:
  A: Discrete (3D Lattice Forest) - Floating obstacles with Poisson sampling
  B: Continuous (3D Ant Nest) - Multi-story maze with ramps/platforms
  C: Restricted Channels - Tunnels/Windows/Vertical Shafts
  D: Vertical Constraints (Lethal Sandwich) - Cave ceilings + hanging hazards
  E: Dynamic (Shooting Gallery) - Moving obstacles

Features:
  - Global Gravity Tilt (random arena rotation)
  - Solvability checking
  - Curriculum labels for RL

Author: NavRL Team
"""

try:
    import torch
except ImportError:
    torch = None  # Optional for standalone testing

import numpy as np
import random
import math
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque


# =============================================================================
# Enums and Data Classes
# =============================================================================

class ArenaMode(Enum):
    """Arena generation modes"""
    LATTICE_FOREST = "A"      # Discrete 3D obstacles
    ANT_NEST = "B"            # Multi-story maze
    RESTRICTED_CHANNEL = "C"  # Tunnels/Windows/Shafts
    LETHAL_SANDWICH = "D"     # Ceiling constraints + hanging hazards
    SHOOTING_GALLERY = "E"    # Dynamic obstacles


class ChannelOrientation(Enum):
    """Sub-types for Mode C: Restricted Channels"""
    HORIZONTAL = "horizontal"  # Window/horizontal pipe (60%)
    VERTICAL = "vertical"      # Vertical shaft/manhole (30%)
    SLOPED = "sloped"          # 45-degree tilted tunnel (10%)


class SandwichSubType(Enum):
    """Sub-types for Mode D: Lethal Sandwich"""
    CAVE = "cave"              # Varying ceiling heights
    HANGING_HAZARDS = "hazards"  # Thin wires/vines hanging down


@dataclass
class ObstaclePrimitive:
    """Represents a single obstacle primitive"""
    prim_type: str  # "cylinder", "cube", "sphere"
    position: Tuple[float, float, float]
    scale: Tuple[float, float, float]
    rotation: Tuple[float, float, float, float] = (
        1.0, 0.0, 0.0, 0.0)  # quaternion wxyz
    color: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    is_dynamic: bool = False
    motion_type: str = "static"  # "static", "linear", "sine"
    motion_params: Dict = field(default_factory=dict)
    is_hazard: bool = False  # For thin wires/vines


@dataclass
class GapInfo:
    """Information about a traversable gap"""
    center: Tuple[float, float, float]
    size: Tuple[float, float, float]  # width, height, depth
    orientation: str  # "horizontal", "vertical", "sloped"
    normal: Tuple[float, float, float]  # Normal vector of the gap plane


@dataclass
class ArenaConfig:
    """Arena generation configuration"""
    size_x: float = 10.0
    size_y: float = 10.0
    size_z: float = 4.0
    drone_radius: float = 0.25
    drone_height: float = 0.15
    start_pos: Tuple[float, float, float] = (-4.0, 0.0, 1.5)
    goal_pos: Tuple[float, float, float] = (4.0, 0.0, 1.5)

    # Gravity tilt parameters
    max_tilt_angle: float = 30.0  # degrees

    # Channel mode parameters
    channel_weights: Tuple[float, float, float] = (0.6, 0.3, 0.1)  # H, V, S


@dataclass
class CurriculumLabels:
    """
    Labels for RL curriculum and reward shaping.

    COORDINATE SYSTEM CONVENTIONS:
    - All geometric calculations (distances, gap_center, etc.) use ARENA LOCAL coordinates
    - local_start/local_goal: Effective spawn points in arena local frame (use for reward calculations)
    - transformed_start/transformed_goal: World coordinates after gravity tilt (use for Sim spawning only)
    - RL Policy should use local coordinates; coordinate transforms are handled by Sim wrapper
    """
    mode: str
    sub_mode: Optional[str]
    difficulty: float
    target_velocity: Tuple[float, float, float]
    nearest_obstacle_dist: float  # Distance in LOCAL coords
    gap_center: Optional[Tuple[float, float, float]]  # LOCAL coords
    gap_size: Optional[Tuple[float, float, float]]
    requires_vertical_flight: bool
    gravity_tilt: Tuple[float, float]  # roll, pitch in degrees

    # Dense reward shaping (LOCAL coords)
    safe_flight_corridor_radius: float = 1.0  # Min clearance along direct path

    # LOCAL coordinates - use these for reward calculations
    local_start: Optional[Tuple[float, float, float]] = None
    local_goal: Optional[Tuple[float, float, float]] = None

    # WORLD coordinates after tilt - use these ONLY for Sim spawning
    transformed_start: Optional[Tuple[float, float, float]] = None
    transformed_goal: Optional[Tuple[float, float, float]] = None


@dataclass
class ArenaResult:
    """Result of arena generation"""
    obstacles: List[ObstaclePrimitive]
    config: ArenaConfig
    mode: ArenaMode
    sub_mode: Optional[str]
    difficulty: float

    # Metadata
    solvable: bool
    complexity: float
    gaps: List[GapInfo]

    # Gravity tilt (quaternion)
    gravity_tilt_quat: Tuple[float, float, float, float]
    gravity_tilt_euler: Tuple[float, float]  # roll, pitch degrees

    # Curriculum labels
    labels: CurriculumLabels

    # Suggested path waypoints
    suggested_path: List[Tuple[float, float, float]
                         ] = field(default_factory=list)


# =============================================================================
# Main Generator Class
# =============================================================================

class UniversalArenaGenerator:
    """
    Universal Arena Generator for Drone Training.

    Generates 10m x 10m x 4m training arenas with various geometric constraints.
    Supports curriculum learning through difficulty parameter.
    """

    def __init__(self, cfg: Optional[ArenaConfig] = None, seed: Optional[int] = None):
        """
        Initialize the generator.

        Args:
            cfg: Arena configuration (uses defaults if None)
            seed: Random seed for reproducibility
        """
        self.cfg = cfg or ArenaConfig()
        self.seed = seed

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Current arena state (for dynamic updates)
        self._current_result: Optional[ArenaResult] = None
        self._sim_time: float = 0.0

        # Mode generators
        self._mode_generators = {
            ArenaMode.LATTICE_FOREST: self._generate_lattice_forest,
            ArenaMode.ANT_NEST: self._generate_ant_nest,
            ArenaMode.RESTRICTED_CHANNEL: self._generate_restricted_channel,
            ArenaMode.LETHAL_SANDWICH: self._generate_lethal_sandwich,
            ArenaMode.SHOOTING_GALLERY: self._generate_shooting_gallery,
        }

    def reset(
        self,
        env_ids: Optional[List[int]] = None,
        mode: Optional[ArenaMode] = None,
        mode_weights: Optional[List[float]] = None,
        difficulty: float = 0.5,
        sub_mode: Optional[str] = None,
    ) -> ArenaResult:
        """
        Generate a new arena configuration.

        Args:
            env_ids: Environment IDs to reset (for batched envs, currently unused)
            mode: Specific mode to use, or None for weighted random
            mode_weights: Weights for random mode selection [A, B, C, D, E]
            difficulty: Difficulty level (0.0 - 1.0)
            sub_mode: Specific sub-mode for modes C and D

        Returns:
            ArenaResult with obstacles and metadata
        """
        # Select mode
        if mode is None:
            if mode_weights is None:
                mode_weights = [0.2, 0.2, 0.2, 0.2, 0.2]
            modes = list(ArenaMode)
            mode = random.choices(modes, weights=mode_weights, k=1)[0]

        # Generate arena
        generator = self._mode_generators[mode]
        result = generator(difficulty, sub_mode)

        # Store for dynamic updates
        self._current_result = result
        self._sim_time = 0.0

        return result

    def update_dynamic_obstacles(self, dt: float) -> List[Tuple[float, float, float]]:
        """
        Update positions of dynamic obstacles with realistic motion patterns.

        Improvements over simple harmonic motion:
        - Random pauses (birds landing, pendulums slowing)
        - Variable speed (acceleration/deceleration)
        - Occasional direction changes
        - Speed modulation over time

        Args:
            dt: Time step in seconds

        Returns:
            List of new positions for all obstacles
        """
        if self._current_result is None:
            return []

        self._sim_time += dt

        new_positions = []
        for obs in self._current_result.obstacles:
            if not obs.is_dynamic:
                new_positions.append(obs.position)
                continue

            px, py, pz = obs.position
            params = obs.motion_params
            t = self._sim_time

            # Get dynamic motion modifiers
            pause_duration = params.get("pause_duration", 0.0)
            pause_interval = params.get("pause_interval", 0.0)
            speed_variation = params.get("speed_variation", 0.0)

            # Check if in pause state
            if pause_interval > 0:
                cycle_time = t % pause_interval
                if cycle_time < pause_duration:
                    # During pause - don't move
                    new_positions.append(obs.position)
                    continue
                # Adjust time to account for pauses
                num_complete_pauses = int(t / pause_interval)
                t = t - num_complete_pauses * pause_duration

            # Apply speed variation (smooth modulation)
            if speed_variation > 0:
                speed_mult = 1.0 + speed_variation * math.sin(0.3 * t)
            else:
                speed_mult = 1.0

            if obs.motion_type == "linear":
                axis = params.get("axis", "x")
                speed = params.get("speed", 1.0) * speed_mult
                amplitude = params.get("amplitude", 2.0)
                phase = params.get("phase", 0.0)

                offset = amplitude * math.sin(speed * t + phase)

                if axis == "x":
                    new_positions.append((px + offset, py, pz))
                elif axis == "y":
                    new_positions.append((px, py + offset, pz))
                else:
                    new_positions.append((px, py, pz + offset * 0.5))

            elif obs.motion_type == "sine":
                freq = params.get("frequency", 0.5) * speed_mult
                amplitude = params.get("amplitude", 2.0)
                phase = params.get("phase", 0.0)
                axis = params.get("axis", "x")

                offset = amplitude * math.sin(2 * math.pi * freq * t + phase)

                if axis == "x":
                    new_positions.append((px + offset, py, pz))
                elif axis == "y":
                    new_positions.append((px, py + offset, pz))
                else:
                    new_positions.append((px, py, pz + offset))

            elif obs.motion_type == "circular":
                freq = params.get("frequency", 0.3) * speed_mult
                radius = params.get("radius", 1.5)
                phase = params.get("phase", 0.0)

                angle = 2 * math.pi * freq * t + phase
                new_positions.append((
                    px + radius * math.cos(angle),
                    py + radius * math.sin(angle),
                    pz
                ))

            elif obs.motion_type == "erratic":
                # New motion type: more unpredictable, bird-like movement
                base_speed = params.get("speed", 1.0)
                amplitude = params.get("amplitude", 2.0)

                # Multiple overlapping sine waves for irregular motion
                offset_x = amplitude * 0.7 * \
                    math.sin(base_speed * t + params.get("phase_x", 0))
                offset_y = amplitude * 0.5 * \
                    math.sin(base_speed * 1.3 * t + params.get("phase_y", 1.5))
                offset_z = amplitude * 0.3 * \
                    math.sin(base_speed * 0.7 * t + params.get("phase_z", 3.0))

                new_positions.append(
                    (px + offset_x, py + offset_y, pz + offset_z))

            else:
                new_positions.append(obs.position)

        return new_positions

        return new_positions

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _compute_gap_size(self, difficulty: float) -> float:
        """
        Compute minimum gap size based on difficulty.

        difficulty 0.0 -> gap = 2.0m (easy)
        difficulty 1.0 -> gap = 0.4m (very tight)
        """
        min_gap = 0.4
        max_gap = 2.0
        return max_gap - (max_gap - min_gap) * difficulty

    def _compute_gravity_tilt(self, apply_tilt: bool = True) -> Tuple[Tuple[float, float, float, float], Tuple[float, float]]:
        """
        Compute random gravity tilt.

        Returns:
            (quaternion_wxyz, (roll_deg, pitch_deg))
        """
        if not apply_tilt:
            return (1.0, 0.0, 0.0, 0.0), (0.0, 0.0)

        max_angle = math.radians(self.cfg.max_tilt_angle)
        roll = random.uniform(-max_angle, max_angle)
        pitch = random.uniform(-max_angle, max_angle)

        # Convert to quaternion
        cr, sr = math.cos(roll / 2), math.sin(roll / 2)
        cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)

        w = cr * cp
        x = sr * cp
        y = cr * sp
        z = -sr * sp

        return (w, x, y, z), (math.degrees(roll), math.degrees(pitch))

    def _transform_endpoints_by_tilt(
        self,
        tilt_quat: Tuple[float, float, float, float],
        local_start: Optional[Tuple[float, float, float]] = None,
        local_goal: Optional[Tuple[float, float, float]] = None,
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """
        Transform start and goal positions by the gravity tilt quaternion.

        When the arena is tilted, the start/goal positions should also be 
        transformed so they remain valid positions relative to the tilted ground.

        Args:
            tilt_quat: Quaternion (w, x, y, z) representing the tilt
            local_start: Local frame start position (defaults to cfg.start_pos)
            local_goal: Local frame goal position (defaults to cfg.goal_pos)

        Returns:
            (transformed_start, transformed_goal) in world coordinates
        """
        # Use provided local coords or fall back to config defaults
        start_local = local_start if local_start else self.cfg.start_pos
        goal_local = local_goal if local_goal else self.cfg.goal_pos

        w, qx, qy, qz = tilt_quat

        def rotate_point(p: Tuple[float, float, float]) -> Tuple[float, float, float]:
            """Rotate a point by quaternion"""
            px, py, pz = p

            # Quaternion rotation: q * p * q^-1
            # For unit quaternion, q^-1 = conjugate
            # Using formula: p' = p + 2*w*(q_vec x p) + 2*(q_vec x (q_vec x p))

            # Cross product q_vec x p
            cx1 = qy * pz - qz * py
            cy1 = qz * px - qx * pz
            cz1 = qx * py - qy * px

            # Cross product q_vec x (q_vec x p)
            cx2 = qy * cz1 - qz * cy1
            cy2 = qz * cx1 - qx * cz1
            cz2 = qx * cy1 - qy * cx1

            # Final rotated position
            return (
                px + 2 * (w * cx1 + cx2),
                py + 2 * (w * cy1 + cy2),
                pz + 2 * (w * cz1 + cz2)
            )

        new_start = rotate_point(start_local)
        new_goal = rotate_point(goal_local)

        # Clamp to arena bounds with margin
        margin = 0.5
        new_start = (
            max(-self.cfg.size_x/2 + margin,
                min(self.cfg.size_x/2 - margin, new_start[0])),
            max(-self.cfg.size_y/2 + margin,
                min(self.cfg.size_y/2 - margin, new_start[1])),
            max(0.3, min(self.cfg.size_z - 0.3, new_start[2]))
        )
        new_goal = (
            max(-self.cfg.size_x/2 + margin,
                min(self.cfg.size_x/2 - margin, new_goal[0])),
            max(-self.cfg.size_y/2 + margin,
                min(self.cfg.size_y/2 - margin, new_goal[1])),
            max(0.3, min(self.cfg.size_z - 0.3, new_goal[2]))
        )

        return new_start, new_goal

    def _is_near_endpoints(
        self,
        x: float,
        y: float,
        z: float = None,
        margin: float = 1.5,
        check_start: Optional[Tuple[float, float, float]] = None,
        check_goal: Optional[Tuple[float, float, float]] = None,
    ) -> bool:
        """
        Check if position is too close to start or goal (in LOCAL frame).

        Args:
            x, y, z: Position to check (local coordinates)
            margin: Minimum distance from endpoints
            check_start: Override start position (use for vertical shaft mode)
            check_goal: Override goal position (use for vertical shaft mode)
        """
        start = check_start if check_start else self.cfg.start_pos
        goal = check_goal if check_goal else self.cfg.goal_pos

        if z is None:
            d_start = math.sqrt((x - start[0])**2 + (y - start[1])**2)
            d_goal = math.sqrt((x - goal[0])**2 + (y - goal[1])**2)
        else:
            d_start = math.sqrt(
                (x - start[0])**2 + (y - start[1])**2 + (z - start[2])**2)
            d_goal = math.sqrt(
                (x - goal[0])**2 + (y - goal[1])**2 + (z - goal[2])**2)

        return d_start < margin or d_goal < margin

    def _check_solvability(
        self,
        obstacles: List[ObstaclePrimitive],
        gaps: List[GapInfo],
        min_gap_size: float,
        local_start: Optional[Tuple[float, float, float]] = None,
        local_goal: Optional[Tuple[float, float, float]] = None,
    ) -> Tuple[bool, float]:
        """
        Check if arena is solvable and compute complexity.

        Args:
            obstacles: List of obstacles (local coordinates)
            gaps: List of gap info
            min_gap_size: Minimum gap size
            local_start: Effective start position (local coords)
            local_goal: Effective goal position (local coords)

        Returns:
            (solvable, complexity)
        """
        drone_size = max(self.cfg.drone_radius * 2, self.cfg.drone_height)

        # Check if any gap is large enough
        if gaps:
            max_gap = max(min(g.size[0], g.size[1]) for g in gaps)

            # Special bypass for Vertical Shaft mode:
            # The 2D BFS cannot correctly evaluate 3D vertical paths.
            # Since vertical shafts are explicitly carved with known geometry,
            # trust the gap size check for vertical orientations.
            has_vertical_gap = any(g.orientation == "vertical" for g in gaps)

            if has_vertical_gap:
                # Trust explicit vertical shaft geometry - gap size determines solvability
                solvable = max_gap > drone_size
            else:
                # For horizontal/sloped gaps, gap size check is sufficient
                solvable = max_gap > drone_size
        else:
            # No explicit gaps (Lattice Forest, Ant Nest, etc.) - use 2D BFS
            solvable = self._check_path_exists(
                obstacles, local_start, local_goal)

        # Complexity based on obstacle density and gap tightness
        area = self.cfg.size_x * self.cfg.size_y
        density = len(obstacles) / area
        gap_ratio = min_gap_size / 2.0 if min_gap_size > 0 else 0.5

        complexity = min(1.0, density * 5.0 + (1.0 - gap_ratio) * 0.5)

        return solvable, complexity

    def _check_path_exists(
        self,
        obstacles: List[ObstaclePrimitive],
        local_start: Optional[Tuple[float, float, float]] = None,
        local_goal: Optional[Tuple[float, float, float]] = None,
    ) -> bool:
        """
        Simple 2D BFS path check in LOCAL coordinates.

        Args:
            obstacles: List of obstacles (local coordinates)
            local_start: Start position in local frame (defaults to cfg.start_pos)
            local_goal: Goal position in local frame (defaults to cfg.goal_pos)
        """
        resolution = 0.4
        grid_w = int(self.cfg.size_x / resolution)
        grid_h = int(self.cfg.size_y / resolution)

        grid = np.zeros((grid_w, grid_h), dtype=bool)
        margin = self.cfg.drone_radius

        for obs in obstacles:
            if obs.prim_type in ["cylinder", "sphere"]:
                ox, oy, _ = obs.position
                radius = obs.scale[0] + margin
                for ix in range(grid_w):
                    for iy in range(grid_h):
                        wx = -self.cfg.size_x / 2 + ix * resolution
                        wy = -self.cfg.size_y / 2 + iy * resolution
                        if math.sqrt((wx - ox)**2 + (wy - oy)**2) < radius:
                            grid[ix, iy] = True
            elif obs.prim_type == "cube":
                ox, oy, _ = obs.position
                sx, sy, _ = obs.scale
                for ix in range(grid_w):
                    for iy in range(grid_h):
                        wx = -self.cfg.size_x / 2 + ix * resolution
                        wy = -self.cfg.size_y / 2 + iy * resolution
                        if abs(wx - ox) < sx + margin and abs(wy - oy) < sy + margin:
                            grid[ix, iy] = True

        # BFS using provided local start/goal or defaults
        start = local_start if local_start else self.cfg.start_pos
        goal = local_goal if local_goal else self.cfg.goal_pos

        start_g = (
            max(0, min(grid_w - 1,
                int((start[0] + self.cfg.size_x / 2) / resolution))),
            max(0, min(grid_h - 1,
                int((start[1] + self.cfg.size_y / 2) / resolution))),
        )
        goal_g = (
            max(0, min(grid_w - 1,
                int((goal[0] + self.cfg.size_x / 2) / resolution))),
            max(0, min(grid_h - 1,
                int((goal[1] + self.cfg.size_y / 2) / resolution))),
        )

        queue = deque([start_g])
        visited = {start_g}
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1),
                      (1, 1), (-1, 1), (1, -1), (-1, -1)]

        while queue:
            cx, cy = queue.popleft()
            if (cx, cy) == goal_g:
                return True

            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < grid_w and 0 <= ny < grid_h:
                    if (nx, ny) not in visited and not grid[nx, ny]:
                        visited.add((nx, ny))
                        queue.append((nx, ny))

        return False

    def _create_labels(
        self,
        mode: ArenaMode,
        sub_mode: Optional[str],
        difficulty: float,
        gaps: List[GapInfo],
        obstacles: List[ObstaclePrimitive],
        gravity_euler: Tuple[float, float],
        local_start: Optional[Tuple[float, float, float]] = None,
        local_goal: Optional[Tuple[float, float, float]] = None,
        transformed_start: Optional[Tuple[float, float, float]] = None,
        transformed_goal: Optional[Tuple[float, float, float]] = None,
    ) -> CurriculumLabels:
        """
        Create curriculum labels for RL.

        IMPORTANT: All geometric calculations use LOCAL coordinates.
        transformed_start/goal are only for Sim spawning reference.

        Args:
            local_start: Effective start in local frame (for distance calcs)
            local_goal: Effective goal in local frame (for distance calcs)
            transformed_start: World coords for Sim spawning
            transformed_goal: World coords for Sim spawning
        """

        # Target velocity based on mode
        if mode == ArenaMode.RESTRICTED_CHANNEL and sub_mode == "vertical":
            target_vel = (0.0, 0.0, 1.5)  # Vertical flight
            requires_vertical = True
        else:
            target_vel = (2.0, 0.0, 0.0)  # Forward flight
            requires_vertical = False

        # Use LOCAL coordinates for all geometric calculations
        # This ensures distance metrics are consistent regardless of tilt
        start_local = local_start if local_start else self.cfg.start_pos
        goal_local = local_goal if local_goal else self.cfg.goal_pos

        # Compute nearest obstacle distance using LOCAL coords
        # (obstacles are stored in local frame, so this is correct)
        min_dist = float('inf')
        for obs in obstacles:
            # Both obs.position and start_local are in local frame
            d = math.sqrt(
                sum((a - b)**2 for a, b in zip(obs.position, start_local)))
            # Subtract approximate obstacle size
            obs_radius = max(obs.scale)
            min_dist = min(min_dist, d - obs_radius)

        # Safe flight corridor radius using LOCAL coords
        safe_corridor = self._compute_safe_corridor_radius(
            obstacles, start_local, goal_local
        )

        # Gap info (already in local coords)
        gap_center = gaps[0].center if gaps else None
        gap_size = gaps[0].size if gaps else None

        return CurriculumLabels(
            mode=mode.value,
            sub_mode=sub_mode,
            difficulty=difficulty,
            target_velocity=target_vel,
            nearest_obstacle_dist=min_dist if min_dist != float(
                'inf') else 10.0,
            gap_center=gap_center,  # LOCAL coords
            gap_size=gap_size,
            requires_vertical_flight=requires_vertical,
            gravity_tilt=gravity_euler,
            safe_flight_corridor_radius=safe_corridor,
            # LOCAL coords for RL reward calculations
            local_start=start_local,
            local_goal=goal_local,
            # WORLD coords for Sim spawning only
            transformed_start=transformed_start,
            transformed_goal=transformed_goal,
        )

    def _compute_safe_corridor_radius(
        self,
        obstacles: List[ObstaclePrimitive],
        start: Tuple[float, float, float],
        goal: Tuple[float, float, float]
    ) -> float:
        """
        Compute the minimum clearance radius along the direct path from start to goal.
        All inputs should be in LOCAL coordinates.
        """
        if not obstacles:
            return 5.0  # Large safe corridor if no obstacles

        # Sample points along the direct path
        num_samples = 20
        min_clearance = float('inf')

        for i in range(num_samples + 1):
            t = i / num_samples
            # Point on direct line from start to goal
            px = start[0] + t * (goal[0] - start[0])
            py = start[1] + t * (goal[1] - start[1])
            pz = start[2] + t * (goal[2] - start[2])

            # Find minimum distance to any obstacle from this point
            for obs in obstacles:
                ox, oy, oz = obs.position
                dist = math.sqrt((px - ox)**2 + (py - oy)**2 + (pz - oz)**2)
                # Subtract obstacle radius (approximate)
                obs_radius = max(obs.scale)
                clearance = dist - obs_radius
                min_clearance = min(min_clearance, clearance)

        return max(0.1, min_clearance)  # Ensure positive value

    # =========================================================================
    # Mode A: Lattice Forest (3D Discrete Obstacles)
    # =========================================================================

    def _generate_lattice_forest(self, difficulty: float, sub_mode: Optional[str] = None) -> ArenaResult:
        """
        Generate 3D lattice of floating obstacles using Poisson sampling.
        Simulates: Forests, floating mines, tree canopies.
        """
        obstacles = []
        gaps = []
        gap_size = self._compute_gap_size(difficulty)

        # Local start/goal (uses default config positions)
        # Must be defined BEFORE obstacle placement for endpoint checking
        local_start = self.cfg.start_pos
        local_goal = self.cfg.goal_pos

        # Poisson disk sampling for 2D base positions
        min_distance = gap_size + 0.4
        points_2d = self._poisson_disk_sampling(
            width=self.cfg.size_x - 2,
            height=self.cfg.size_y - 2,
            min_dist=min_distance,
            num_samples=int(25 * (0.5 + difficulty * 0.5)) + 5,
        )

        # Generate 3D lattice - obstacles at multiple heights
        num_layers = int(2 + difficulty * 2)  # 2-4 vertical layers
        layer_heights = np.linspace(0.5, self.cfg.size_z - 0.5, num_layers)

        for px, py in points_2d:
            if self._is_near_endpoints(px, py, margin=1.8,
                                       check_start=local_start, check_goal=local_goal):
                continue

            # Randomly select which layers this column has obstacles
            active_layers = random.sample(
                range(num_layers), k=random.randint(1, num_layers))

            for layer_idx in active_layers:
                pz = layer_heights[layer_idx]

                # Random obstacle type
                if random.random() < 0.6:
                    # Cylinder (tree trunk or pillar)
                    radius = random.uniform(0.1, 0.25 + 0.15 * difficulty)
                    height = random.uniform(0.5, 1.5)
                    obstacles.append(ObstaclePrimitive(
                        prim_type="cylinder",
                        position=(px, py, pz),
                        scale=(radius, radius, height / 2),
                        color=(0.4, 0.5 + random.uniform(0, 0.2), 0.3),
                    ))
                else:
                    # Cube (floating block)
                    size = random.uniform(0.2, 0.4 + 0.2 * difficulty)
                    obstacles.append(ObstaclePrimitive(
                        prim_type="cube",
                        position=(px, py, pz),
                        scale=(size / 2, size / 2, size / 2),
                        color=(0.5, 0.4, 0.3),
                    ))

        # Add some purely floating obstacles (not on columns)
        num_floaters = int(5 + 10 * difficulty)
        for _ in range(num_floaters):
            fx = random.uniform(-4, 4)
            fy = random.uniform(-4, 4)
            fz = random.uniform(1.0, 3.0)

            if self._is_near_endpoints(fx, fy, fz, margin=1.5,
                                       check_start=local_start, check_goal=local_goal):
                continue

            size = random.uniform(0.15, 0.3)
            obstacles.append(ObstaclePrimitive(
                prim_type="sphere",
                position=(fx, fy, fz),
                scale=(size, size, size),
                color=(0.6, 0.3, 0.3),
            ))

        # Compute gravity tilt
        tilt_quat, tilt_euler = self._compute_gravity_tilt(apply_tilt=True)

        # Transform to world coordinates for Sim spawning
        transformed_start, transformed_goal = local_start, local_goal
        if tilt_euler != (0.0, 0.0):
            transformed_start, transformed_goal = self._transform_endpoints_by_tilt(
                tilt_quat, local_start, local_goal)

        # Check solvability using LOCAL coordinates
        solvable, complexity = self._check_solvability(
            obstacles, gaps, gap_size, local_start, local_goal)

        # Create labels with both local and world coords
        labels = self._create_labels(
            ArenaMode.LATTICE_FOREST, None, difficulty, gaps, obstacles, tilt_euler,
            local_start=local_start, local_goal=local_goal,
            transformed_start=transformed_start, transformed_goal=transformed_goal
        )

        return ArenaResult(
            obstacles=obstacles,
            config=self.cfg,
            mode=ArenaMode.LATTICE_FOREST,
            sub_mode=None,
            difficulty=difficulty,
            solvable=solvable,
            complexity=complexity,
            gaps=gaps,
            gravity_tilt_quat=tilt_quat,
            gravity_tilt_euler=tilt_euler,
            labels=labels,
            # suggested_path uses WORLD coords for visualization
            suggested_path=[transformed_start, (0, 0, 1.5), transformed_goal],
        )

    def _poisson_disk_sampling(
        self,
        width: float,
        height: float,
        min_dist: float,
        num_samples: int = 30,
    ) -> List[Tuple[float, float]]:
        """Poisson disk sampling for evenly distributed random points."""
        cell_size = min_dist / math.sqrt(2)
        grid_w = max(1, int(math.ceil(width / cell_size)))
        grid_h = max(1, int(math.ceil(height / cell_size)))
        grid = [[None for _ in range(grid_h)] for _ in range(grid_w)]

        points = []
        active = []

        # First point
        x0 = random.uniform(-width / 2, width / 2)
        y0 = random.uniform(-height / 2, height / 2)
        points.append((x0, y0))
        active.append((x0, y0))

        gx = int((x0 + width / 2) / cell_size)
        gy = int((y0 + height / 2) / cell_size)
        if 0 <= gx < grid_w and 0 <= gy < grid_h:
            grid[gx][gy] = (x0, y0)

        while active and len(points) < num_samples * 3:
            idx = random.randint(0, len(active) - 1)
            px, py = active[idx]
            found = False

            for _ in range(30):
                angle = random.uniform(0, 2 * math.pi)
                dist = random.uniform(min_dist, 2 * min_dist)
                nx = px + dist * math.cos(angle)
                ny = py + dist * math.sin(angle)

                if not (-width / 2 <= nx <= width / 2 and -height / 2 <= ny <= height / 2):
                    continue

                gx = int((nx + width / 2) / cell_size)
                gy = int((ny + height / 2) / cell_size)

                if not (0 <= gx < grid_w and 0 <= gy < grid_h):
                    continue

                valid = True
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        ngx, ngy = gx + dx, gy + dy
                        if 0 <= ngx < grid_w and 0 <= ngy < grid_h:
                            neighbor = grid[ngx][ngy]
                            if neighbor is not None:
                                d = math.sqrt(
                                    (nx - neighbor[0])**2 + (ny - neighbor[1])**2)
                                if d < min_dist:
                                    valid = False
                                    break
                    if not valid:
                        break

                if valid:
                    points.append((nx, ny))
                    active.append((nx, ny))
                    grid[gx][gy] = (nx, ny)
                    found = True
                    break

            if not found:
                active.pop(idx)

        return points[:num_samples]

    # =========================================================================
    # Mode B: Ant Nest (3D Multi-Story Maze)
    # =========================================================================

    def _generate_ant_nest(self, difficulty: float, sub_mode: Optional[str] = None) -> ArenaResult:
        """
        Generate multi-story maze with ramps and platforms.
        Simulates: Indoor environments, parking garages, multi-level structures.
        """
        obstacles = []
        gaps = []
        gap_size = self._compute_gap_size(difficulty)

        # Local start/goal (uses default config positions)
        # Must be defined BEFORE obstacle placement for endpoint checking
        local_start = self.cfg.start_pos
        local_goal = self.cfg.goal_pos

        # Maze parameters
        cell_size = max(1.2, gap_size + 0.3)
        grid_w = int(self.cfg.size_x / cell_size)
        grid_h = int(self.cfg.size_y / cell_size)

        # Ensure odd dimensions
        grid_w = grid_w if grid_w % 2 == 1 else grid_w - 1
        grid_h = grid_h if grid_h % 2 == 1 else grid_h - 1
        grid_w = max(5, grid_w)
        grid_h = max(5, grid_h)

        # Generate base maze
        maze = self._recursive_backtracker_maze(grid_w, grid_h)

        # Multi-story parameters
        num_floors = int(2 + difficulty)  # 2-3 floors
        floor_heights = np.linspace(0, self.cfg.size_z - 1.5, num_floors)
        wall_height = (self.cfg.size_z - 0.5) / num_floors

        offset_x = -self.cfg.size_x / 2
        offset_y = -self.cfg.size_y / 2

        for floor_idx, floor_z in enumerate(floor_heights):
            # Add floor/ceiling slabs with holes
            if floor_idx > 0:
                # Floor slab with random holes for vertical movement
                slab_thickness = 0.15
                hole_positions = [(random.randint(1, grid_w - 2), random.randint(1, grid_h - 2))
                                  for _ in range(2 + floor_idx)]

                for i in range(grid_w):
                    for j in range(grid_h):
                        if maze[i][j] == 0:  # Passage
                            wx = offset_x + (i + 0.5) * cell_size
                            wy = offset_y + (j + 0.5) * cell_size

                            if self._is_near_endpoints(wx, wy, margin=1.2,
                                                       check_start=local_start, check_goal=local_goal):
                                continue

                            # Check if this is a hole position
                            is_hole = any(abs(i - hx) <= 1 and abs(j - hy) <= 1
                                          for hx, hy in hole_positions)

                            if not is_hole:
                                obstacles.append(ObstaclePrimitive(
                                    prim_type="cube",
                                    position=(wx, wy, floor_z),
                                    scale=(cell_size / 2, cell_size /
                                           2, slab_thickness / 2),
                                    color=(0.4, 0.4, 0.45),
                                ))

            # Add walls for this floor
            for i in range(grid_w):
                for j in range(grid_h):
                    if maze[i][j] == 1:  # Wall
                        wx = offset_x + (i + 0.5) * cell_size
                        wy = offset_y + (j + 0.5) * cell_size

                        if self._is_near_endpoints(wx, wy, margin=1.0,
                                                   check_start=local_start, check_goal=local_goal):
                            continue

                        obstacles.append(ObstaclePrimitive(
                            prim_type="cube",
                            position=(wx, wy, floor_z + wall_height / 2),
                            scale=(cell_size / 2, cell_size /
                                   2, wall_height / 2),
                            color=(0.55, 0.55, 0.6),
                        ))

        # Add ramps between floors
        num_ramps = int(1 + difficulty)
        for _ in range(num_ramps):
            rx = random.uniform(-3, 3)
            ry = random.uniform(-3, 3)
            if self._is_near_endpoints(rx, ry, margin=1.5,
                                       check_start=local_start, check_goal=local_goal):
                continue

            ramp_length = 2.0
            ramp_width = 0.8
            ramp_angle = math.radians(25 + 10 * difficulty)  # 25-35 degrees

            # Create ramp as rotated cube
            ramp_height = ramp_length * math.sin(ramp_angle)
            rz = ramp_height / 2

            # Rotation quaternion for pitch
            cp, sp = math.cos(ramp_angle / 2), math.sin(ramp_angle / 2)

            obstacles.append(ObstaclePrimitive(
                prim_type="cube",
                position=(rx, ry, rz),
                scale=(ramp_length / 2, ramp_width / 2, 0.1),
                rotation=(cp, 0, sp, 0),
                color=(0.6, 0.5, 0.4),
            ))

        # Gravity tilt
        tilt_quat, tilt_euler = self._compute_gravity_tilt(apply_tilt=True)

        # Transform to world coordinates for Sim spawning
        transformed_start, transformed_goal = local_start, local_goal
        if tilt_euler != (0.0, 0.0):
            transformed_start, transformed_goal = self._transform_endpoints_by_tilt(
                tilt_quat, local_start, local_goal)

        # Check solvability using LOCAL coordinates
        solvable, complexity = self._check_solvability(
            obstacles, gaps, gap_size, local_start, local_goal)

        labels = self._create_labels(
            ArenaMode.ANT_NEST, None, difficulty, gaps, obstacles, tilt_euler,
            local_start=local_start, local_goal=local_goal,
            transformed_start=transformed_start, transformed_goal=transformed_goal
        )

        return ArenaResult(
            obstacles=obstacles,
            config=self.cfg,
            mode=ArenaMode.ANT_NEST,
            sub_mode=None,
            difficulty=difficulty,
            solvable=solvable,
            complexity=complexity,
            gaps=gaps,
            gravity_tilt_quat=tilt_quat,
            gravity_tilt_euler=tilt_euler,
            labels=labels,
            suggested_path=[transformed_start, (0, 0, 1.5), transformed_goal],
        )

    def _recursive_backtracker_maze(self, width: int, height: int) -> List[List[int]]:
        """Generate maze using recursive backtracker. 0=passage, 1=wall"""
        maze = [[1 for _ in range(height)] for _ in range(width)]

        stack = [(1, 1)]
        maze[1][1] = 0

        directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]

        while stack:
            x, y = stack[-1]
            random.shuffle(directions)
            found = False

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 < nx < width - 1 and 0 < ny < height - 1 and maze[nx][ny] == 1:
                    maze[x + dx // 2][y + dy // 2] = 0
                    maze[nx][ny] = 0
                    stack.append((nx, ny))
                    found = True
                    break

            if not found:
                stack.pop()

        return maze

    # =========================================================================
    # Mode C: Restricted Channels (Tunnels/Windows/Vertical Shafts)
    # =========================================================================

    def _generate_restricted_channel(self, difficulty: float, sub_mode: Optional[str] = None) -> ArenaResult:
        """
        Generate restricted passage with weighted orientation sampling.
        - Horizontal (60%): Window/horizontal pipe
        - Vertical (30%): Vertical shaft (manhole)
        - Sloped (10%): 45-degree tilted tunnel
        """
        obstacles = []
        gaps = []
        gap_size = self._compute_gap_size(difficulty)

        # Select orientation
        if sub_mode:
            orientation = ChannelOrientation(sub_mode)
        else:
            weights = self.cfg.channel_weights
            orientation = random.choices(
                [ChannelOrientation.HORIZONTAL,
                    ChannelOrientation.VERTICAL, ChannelOrientation.SLOPED],
                weights=weights,
                k=1
            )[0]

        # Extrusion depth (0.1m window to 5.0m tunnel)
        extrusion_depth = random.uniform(0.1, 0.5 + 4.5 * difficulty)

        # Apply gravity tilt only if NOT vertical shaft
        apply_tilt = orientation != ChannelOrientation.VERTICAL
        tilt_quat, tilt_euler = self._compute_gravity_tilt(
            apply_tilt=apply_tilt)

        # Default local start/goal from config
        local_start = self.cfg.start_pos
        local_goal = self.cfg.goal_pos

        if orientation == ChannelOrientation.HORIZONTAL:
            obstacles, gaps = self._create_horizontal_channel(
                gap_size, extrusion_depth, difficulty)
        elif orientation == ChannelOrientation.VERTICAL:
            # Vertical shaft returns NEW local start/goal positions
            obstacles, gaps, local_start, local_goal = self._create_vertical_shaft(
                gap_size, extrusion_depth, difficulty)
        else:  # SLOPED
            obstacles, gaps = self._create_sloped_channel(
                gap_size, extrusion_depth, difficulty)

        # Calculate world coordinates for Sim spawning
        transformed_start, transformed_goal = local_start, local_goal
        if apply_tilt and tilt_euler != (0.0, 0.0):
            # Transform the LOCAL positions to world
            transformed_start, transformed_goal = self._transform_endpoints_by_tilt(
                tilt_quat, local_start, local_goal)

        # Check solvability using LOCAL coordinates (correct geometry!)
        solvable, complexity = self._check_solvability(
            obstacles, gaps, gap_size, local_start, local_goal)

        labels = self._create_labels(
            ArenaMode.RESTRICTED_CHANNEL, orientation.value, difficulty, gaps, obstacles, tilt_euler,
            local_start=local_start, local_goal=local_goal,
            transformed_start=transformed_start, transformed_goal=transformed_goal
        )

        return ArenaResult(
            obstacles=obstacles,
            config=self.cfg,
            mode=ArenaMode.RESTRICTED_CHANNEL,
            sub_mode=orientation.value,
            difficulty=difficulty,
            solvable=solvable,
            complexity=complexity,
            gaps=gaps,
            gravity_tilt_quat=tilt_quat,
            gravity_tilt_euler=tilt_euler,
            labels=labels,
            # Use WORLD coords for visualization path
            suggested_path=[
                transformed_start,
                gaps[0].center if gaps else (0, 0, 1.5),
                transformed_goal
            ],
        )

    def _create_horizontal_channel(
        self, gap_size: float, depth: float, difficulty: float
    ) -> Tuple[List[ObstaclePrimitive], List[GapInfo]]:
        """Create a wall with a horizontal window/tunnel"""
        obstacles = []

        wall_x = 0.0
        wall_height = self.cfg.size_z
        wall_width = self.cfg.size_y

        # Hole parameters
        hole_w = gap_size + random.uniform(0, 0.3)
        hole_h = gap_size + random.uniform(0, 0.3)
        hole_y = random.uniform(-2, 2)
        hole_z = random.uniform(1.0, 2.5)

        # Create wall blocks around the hole
        block_size = 0.4
        ny = int(wall_width / block_size)
        nz = int(wall_height / block_size)

        for iy in range(ny):
            for iz in range(nz):
                by = -wall_width / 2 + (iy + 0.5) * block_size
                bz = (iz + 0.5) * block_size

                # Check if inside hole
                if abs(by - hole_y) < hole_w / 2 and abs(bz - hole_z) < hole_h / 2:
                    continue

                obstacles.append(ObstaclePrimitive(
                    prim_type="cube",
                    position=(wall_x, by, bz),
                    scale=(depth / 2, block_size / 2, block_size / 2),
                    color=(0.7, 0.65, 0.55),
                ))

        gap = GapInfo(
            center=(wall_x, hole_y, hole_z),
            size=(hole_w, hole_h, depth),
            orientation="horizontal",
            normal=(1.0, 0.0, 0.0),
        )

        return obstacles, [gap]

    def _create_vertical_shaft(
        self, gap_size: float, depth: float, difficulty: float
    ) -> Tuple[List[ObstaclePrimitive], List[GapInfo], Tuple[float, float, float], Tuple[float, float, float]]:
        """
        Create a proper vertical shaft that separates start from goal along Z-axis.

        The shaft creates two layers of floors (ceiling and floor) with a vertical 
        tube connecting them. Drone must fly vertically through the shaft.

        Key change: Start position is ABOVE the barrier, goal is BELOW (or vice versa).
        This forces the drone to actually fly through the vertical shaft.

        Returns:
            obstacles, gaps, new_start_pos, new_goal_pos
        """
        obstacles = []

        # Shaft parameters - centered in arena
        # Near center for proper vertical flight
        shaft_x = random.uniform(-0.5, 0.5)
        shaft_y = random.uniform(-0.5, 0.5)
        shaft_radius = gap_size / 2 + random.uniform(0, 0.15)

        # Two floor slabs: lower at z=1.0, upper at z=3.0
        lower_slab_z = 1.0
        upper_slab_z = 3.0
        slab_thickness = min(depth * 0.3, 0.5)  # Reasonable thickness

        # Start above upper slab, goal below lower slab (or randomize direction)
        if random.random() < 0.5:
            # Start high, goal low
            new_start = (shaft_x, shaft_y, upper_slab_z + 0.7)
            new_goal = (shaft_x, shaft_y, lower_slab_z - 0.5)
        else:
            # Start low, goal high
            new_start = (shaft_x, shaft_y, lower_slab_z - 0.5)
            new_goal = (shaft_x, shaft_y, upper_slab_z + 0.7)

        # Ensure goal is within arena bounds
        new_goal = (
            max(-4.5, min(4.5, new_goal[0])),
            max(-4.5, min(4.5, new_goal[1])),
            max(0.3, min(self.cfg.size_z - 0.3, new_goal[2]))
        )
        new_start = (
            max(-4.5, min(4.5, new_start[0])),
            max(-4.5, min(4.5, new_start[1])),
            max(0.3, min(self.cfg.size_z - 0.3, new_start[2]))
        )

        # Create UPPER floor slab with circular hole (blocks drone from going around)
        block_size = 0.5
        nx = int(self.cfg.size_x / block_size)
        ny = int(self.cfg.size_y / block_size)

        for ix in range(nx):
            for iy in range(ny):
                bx = -self.cfg.size_x / 2 + (ix + 0.5) * block_size
                by = -self.cfg.size_y / 2 + (iy + 0.5) * block_size

                # Check if inside shaft hole
                dist = math.sqrt((bx - shaft_x)**2 + (by - shaft_y)**2)
                if dist < shaft_radius + block_size * 0.7:
                    continue

                # Upper slab
                obstacles.append(ObstaclePrimitive(
                    prim_type="cube",
                    position=(bx, by, upper_slab_z),
                    scale=(block_size / 2, block_size / 2, slab_thickness / 2),
                    color=(0.5, 0.5, 0.55),
                ))

        # Create LOWER floor slab with circular hole
        for ix in range(nx):
            for iy in range(ny):
                bx = -self.cfg.size_x / 2 + (ix + 0.5) * block_size
                by = -self.cfg.size_y / 2 + (iy + 0.5) * block_size

                dist = math.sqrt((bx - shaft_x)**2 + (by - shaft_y)**2)
                if dist < shaft_radius + block_size * 0.7:
                    continue

                # Lower slab
                obstacles.append(ObstaclePrimitive(
                    prim_type="cube",
                    position=(bx, by, lower_slab_z),
                    scale=(block_size / 2, block_size / 2, slab_thickness / 2),
                    color=(0.45, 0.45, 0.5),
                ))

        # Create shaft tube walls connecting the two slabs
        shaft_height = upper_slab_z - lower_slab_z
        tube_segments = 12
        tube_radius = shaft_radius + 0.05
        wall_thickness = 0.08

        for i in range(tube_segments):
            angle = 2 * math.pi * i / tube_segments
            wx = shaft_x + tube_radius * math.cos(angle)
            wy = shaft_y + tube_radius * math.sin(angle)

            obstacles.append(ObstaclePrimitive(
                prim_type="cube",
                position=(wx, wy, (upper_slab_z + lower_slab_z) / 2),
                scale=(wall_thickness, wall_thickness, shaft_height / 2),
                color=(0.4, 0.4, 0.45),
            ))

        # Add difficulty: partial obstructions inside shaft
        if difficulty > 0.3:
            num_rings = int(1 + 2 * difficulty)
            ring_heights = np.linspace(
                lower_slab_z + 0.5, upper_slab_z - 0.5, num_rings + 2)[1:-1]

            for ring_z in ring_heights:
                # Partial ring - leaves gap for drone
                num_arcs = random.randint(2, 4)
                arc_length = 2 * math.pi / (num_arcs * 2)
                start_angle = random.uniform(0, math.pi)

                for arc_idx in range(num_arcs):
                    arc_center = start_angle + \
                        arc_idx * (2 * math.pi / num_arcs)
                    for j in range(3):
                        angle = arc_center + (j - 1) * arc_length / 3
                        ox = shaft_x + (shaft_radius * 0.5) * math.cos(angle)
                        oy = shaft_y + (shaft_radius * 0.5) * math.sin(angle)

                        obstacles.append(ObstaclePrimitive(
                            prim_type="sphere",
                            position=(ox, oy, ring_z),
                            scale=(0.08, 0.08, 0.08),
                            color=(0.6, 0.3, 0.3),
                            is_hazard=True,
                        ))

        gap = GapInfo(
            center=(shaft_x, shaft_y, (upper_slab_z + lower_slab_z) / 2),
            size=(shaft_radius * 2, shaft_radius * 2, shaft_height),
            orientation="vertical",
            normal=(0.0, 0.0, 1.0),
        )

        return obstacles, [gap], new_start, new_goal

    def _create_sloped_channel(
        self, gap_size: float, depth: float, difficulty: float
    ) -> Tuple[List[ObstaclePrimitive], List[GapInfo]]:
        """Create a 45-degree tilted tunnel/plate"""
        obstacles = []

        # Sloped plate parameters
        plate_angle = math.radians(45)
        plate_x = 0.0
        plate_y = 0.0
        plate_z = self.cfg.size_z / 2

        # Hole in the sloped plate
        hole_offset = random.uniform(-1, 1)
        hole_size = gap_size + random.uniform(0, 0.2)

        # Create sloped blocks
        block_size = 0.5
        n_blocks = int(6 / block_size)

        cp, sp = math.cos(plate_angle / 2), math.sin(plate_angle / 2)
        rotation = (cp, sp, 0, 0)  # Rotation around X axis

        for iu in range(-n_blocks, n_blocks + 1):
            for iv in range(-n_blocks, n_blocks + 1):
                u = iu * block_size
                v = iv * block_size

                # Check if inside hole
                if abs(u - hole_offset) < hole_size / 2 and abs(v) < hole_size / 2:
                    continue

                # Position on tilted plane
                bx = plate_x + u
                by = plate_y + v * math.cos(plate_angle)
                bz = plate_z + v * math.sin(plate_angle)

                if abs(bx) > 4.5 or abs(by) > 4.5 or bz < 0.3 or bz > 3.7:
                    continue

                obstacles.append(ObstaclePrimitive(
                    prim_type="cube",
                    position=(bx, by, bz),
                    scale=(block_size / 2, block_size / 2, depth / 2),
                    rotation=rotation,
                    color=(0.6, 0.55, 0.5),
                ))

        gap = GapInfo(
            center=(plate_x + hole_offset, plate_y, plate_z),
            size=(hole_size, hole_size, depth),
            orientation="sloped",
            normal=(0.0, -math.sin(plate_angle), math.cos(plate_angle)),
        )

        return obstacles, [gap]

    # =========================================================================
    # Mode D: Lethal Sandwich (Ceiling Constraints + Hanging Hazards)
    # =========================================================================

    def _generate_lethal_sandwich(self, difficulty: float, sub_mode: Optional[str] = None) -> ArenaResult:
        """
        Generate vertical constraints with two sub-types:
        - Cave: Varying ceiling heights
        - Hanging Hazards: Thin wires/vines hanging down
        """
        # Select sub-type
        if sub_mode:
            sandwich_type = SandwichSubType(sub_mode)
        else:
            sandwich_type = random.choice(
                [SandwichSubType.CAVE, SandwichSubType.HANGING_HAZARDS])

        # Local start/goal (uses default config positions)
        # Must be defined BEFORE helper functions so they can check endpoints correctly
        local_start = self.cfg.start_pos
        local_goal = self.cfg.goal_pos

        if sandwich_type == SandwichSubType.CAVE:
            obstacles, gaps = self._create_cave_ceiling(
                difficulty, local_start, local_goal)
        else:
            obstacles, gaps = self._create_hanging_hazards(
                difficulty, local_start, local_goal)

        gap_size = self._compute_gap_size(difficulty)
        tilt_quat, tilt_euler = self._compute_gravity_tilt(apply_tilt=True)

        # Transform to world coordinates for Sim spawning
        transformed_start, transformed_goal = local_start, local_goal
        if tilt_euler != (0.0, 0.0):
            transformed_start, transformed_goal = self._transform_endpoints_by_tilt(
                tilt_quat, local_start, local_goal)

        # Check solvability using LOCAL coordinates
        solvable, complexity = self._check_solvability(
            obstacles, gaps, gap_size, local_start, local_goal)

        labels = self._create_labels(
            ArenaMode.LETHAL_SANDWICH, sandwich_type.value, difficulty, gaps, obstacles, tilt_euler,
            local_start=local_start, local_goal=local_goal,
            transformed_start=transformed_start, transformed_goal=transformed_goal
        )

        return ArenaResult(
            obstacles=obstacles,
            config=self.cfg,
            mode=ArenaMode.LETHAL_SANDWICH,
            sub_mode=sandwich_type.value,
            difficulty=difficulty,
            solvable=solvable,
            complexity=complexity,
            gaps=gaps,
            gravity_tilt_quat=tilt_quat,
            gravity_tilt_euler=tilt_euler,
            labels=labels,
            suggested_path=[transformed_start, (0, 0, 0.8), transformed_goal],
        )

    def _create_cave_ceiling(
        self, difficulty: float,
        local_start: Optional[Tuple[float, float, float]] = None,
        local_goal: Optional[Tuple[float, float, float]] = None
    ) -> Tuple[List[ObstaclePrimitive], List[GapInfo]]:
        """Create varying ceiling heights (cave mode).

        Args:
            difficulty: Arena difficulty level [0, 1]
            local_start: Start position in arena coordinates (for endpoint check)
            local_goal: Goal position in arena coordinates (for endpoint check)
        """
        obstacles = []

        # Use provided local coords or fall back to config defaults
        check_start = local_start or self.cfg.start_pos
        check_goal = local_goal or self.cfg.goal_pos

        base_ceiling = self.cfg.size_z
        min_clearance = max(0.6, 1.5 - difficulty)

        block_size = 0.8
        nx = int(self.cfg.size_x / block_size)
        ny = int(self.cfg.size_y / block_size)

        # Generate height map using Perlin-like noise
        height_map = np.zeros((nx, ny))

        # Add multiple octaves of noise
        for octave in range(3):
            freq = 2 ** octave * 0.3
            amp = 1.0 / (octave + 1)
            for ix in range(nx):
                for iy in range(ny):
                    height_map[ix, iy] += amp * \
                        math.sin(freq * ix) * math.cos(freq * iy)

        # Normalize and scale
        height_map = (height_map - height_map.min()) / \
            (height_map.max() - height_map.min() + 1e-6)
        height_map = height_map * \
            (base_ceiling - min_clearance - 0.5) * difficulty

        for ix in range(nx):
            for iy in range(ny):
                drop = height_map[ix, iy]
                if drop < 0.2:
                    continue

                bx = -self.cfg.size_x / 2 + (ix + 0.5) * block_size
                by = -self.cfg.size_y / 2 + (iy + 0.5) * block_size

                if self._is_near_endpoints(bx, by, margin=1.5,
                                           check_start=check_start, check_goal=check_goal):
                    continue

                bz = base_ceiling - drop / 2
                obstacles.append(ObstaclePrimitive(
                    prim_type="cube",
                    position=(bx, by, bz),
                    scale=(block_size / 2, block_size / 2, drop / 2),
                    color=(0.45, 0.4, 0.35),
                ))

        return obstacles, []

    def _create_hanging_hazards(
        self, difficulty: float,
        local_start: Optional[Tuple[float, float, float]] = None,
        local_goal: Optional[Tuple[float, float, float]] = None
    ) -> Tuple[List[ObstaclePrimitive], List[GapInfo]]:
        """
        Create thin wires/vines hanging from ceiling.

        Wire physics considerations:
        - Real thin wires are 1-5mm diameter
        - Minimum 10mm radius (20mm diameter) for stable PhysX collision detection
        - Thinner wires cause raycast tunneling and physics instability

        Args:
            difficulty: Arena difficulty level [0, 1]
            local_start: Start position in arena coordinates (for endpoint check)
            local_goal: Goal position in arena coordinates (for endpoint check)
        """
        obstacles = []

        # Use provided local coords or fall back to config defaults
        check_start = local_start or self.cfg.start_pos
        check_goal = local_goal or self.cfg.goal_pos

        ceiling_z = self.cfg.size_z

        # Number of hanging hazards
        num_hazards = int(15 + 35 * difficulty)

        for _ in range(num_hazards):
            hx = random.uniform(-4.5, 4.5)
            hy = random.uniform(-4.5, 4.5)

            if self._is_near_endpoints(hx, hy, margin=1.2,
                                       check_start=check_start, check_goal=check_goal):
                continue

            # Wire parameters - PHYSICS-STABLE THIN WIRES
            # Minimum 10mm radius (20mm diameter) for stable PhysX collision detection
            # Thinner wires cause raycast tunneling and physics instability
            # Range: 10-15mm radius (20-30mm diameter) - thin but detectable
            wire_radius = random.uniform(0.01, 0.015)
            wire_length = random.uniform(1.0, 3.0 + difficulty)
            hz = ceiling_z - wire_length / 2

            obstacles.append(ObstaclePrimitive(
                prim_type="cylinder",
                position=(hx, hy, hz),
                scale=(wire_radius, wire_radius, wire_length / 2),
                # BRIGHT ORANGE-RED for visibility in Isaac Sim debug views
                # Original dark gray (0.3, 0.3, 0.35) was nearly invisible
                color=(1.0, 0.3, 0.0),
                is_hazard=True,
            ))

        # Add some thicker vines/cables (these are the medium-thickness ones)
        num_cables = int(5 + 10 * difficulty)
        for _ in range(num_cables):
            cx = random.uniform(-4, 4)
            cy = random.uniform(-4, 4)

            if self._is_near_endpoints(cx, cy, margin=1.5,
                                       check_start=check_start, check_goal=check_goal):
                continue

            # Thicker cables: 2-4cm diameter
            cable_radius = random.uniform(0.01, 0.02)
            cable_length = random.uniform(2.0, 3.5)
            cz = ceiling_z - cable_length / 2

            obstacles.append(ObstaclePrimitive(
                prim_type="cylinder",
                position=(cx, cy, cz),
                scale=(cable_radius, cable_radius, cable_length / 2),
                # BRIGHT YELLOW-GREEN for visibility (different from wires)
                # Original dark green (0.25, 0.35, 0.25) was hard to see
                color=(0.8, 1.0, 0.0),
                is_hazard=True,
            ))

        return obstacles, []

    # =========================================================================
    # Mode E: Shooting Gallery (Dynamic Obstacles)
    # =========================================================================

    def _generate_shooting_gallery(self, difficulty: float, sub_mode: Optional[str] = None) -> ArenaResult:
        """
        Generate empty room with moving spheres.
        Simulates: Birds, drones, moving machinery.

        Enhanced with realistic motion patterns:
        - Random pauses (birds landing/resting)
        - Variable speed (natural acceleration/deceleration)
        - Erratic motion type for bird-like unpredictability
        """
        obstacles = []
        gaps = []

        # Local start/goal (uses default config positions)
        # Must be defined BEFORE obstacle placement for endpoint checking
        local_start = self.cfg.start_pos
        local_goal = self.cfg.goal_pos

        num_movers = int(3 + 5 * difficulty)

        # Include erratic motion type for more realism
        motion_types = ["linear", "sine", "circular", "erratic"]

        for i in range(num_movers):
            # Starting position
            px = random.uniform(-3, 3)
            py = random.uniform(-3, 3)
            pz = random.uniform(1.0, 3.0)

            # Size
            radius = random.uniform(0.2, 0.4 + 0.2 * difficulty)

            # Motion type - weighted towards more natural patterns
            if random.random() < 0.3:
                motion = "erratic"  # 30% chance of bird-like erratic motion
            else:
                motion = random.choice(["linear", "sine", "circular"])

            # Base motion parameters with realistic additions
            base_params = {
                "phase": random.uniform(0, 2 * math.pi),
                # Random pause: 0-3 second pause every 5-15 seconds
                "pause_duration": random.uniform(0, 3.0) if random.random() < 0.4 else 0.0,
                "pause_interval": random.uniform(5.0, 15.0),
                # Speed variation: ±0-30% oscillation
                "speed_variation": random.uniform(0, 0.3) if random.random() < 0.5 else 0.0,
            }

            if motion == "linear":
                axis = random.choice(["x", "y", "z"])
                speed = random.uniform(0.5, 1.5 + difficulty)
                amplitude = random.uniform(1.5, 3.5)
                params = {
                    **base_params,
                    "axis": axis,
                    "speed": speed,
                    "amplitude": amplitude,
                }
            elif motion == "sine":
                axis = random.choice(["x", "y"])
                freq = random.uniform(0.2, 0.8)
                amplitude = random.uniform(1.5, 3.0)
                params = {
                    **base_params,
                    "axis": axis,
                    "frequency": freq,
                    "amplitude": amplitude,
                }
            elif motion == "circular":
                freq = random.uniform(0.1, 0.4)
                radius_motion = random.uniform(1.0, 2.5)
                params = {
                    **base_params,
                    "frequency": freq,
                    "radius": radius_motion,
                }
            else:  # erratic
                params = {
                    **base_params,
                    "speed": random.uniform(0.3, 1.0 + difficulty),
                    "amplitude": random.uniform(1.0, 2.5),
                    "phase_x": random.uniform(0, 2 * math.pi),
                    "phase_y": random.uniform(0, 2 * math.pi),
                    "phase_z": random.uniform(0, 2 * math.pi),
                }

            obstacles.append(ObstaclePrimitive(
                prim_type="sphere",
                position=(px, py, pz),
                scale=(radius, radius, radius),
                color=(0.9, 0.3, 0.2),
                is_dynamic=True,
                motion_type=motion,
                motion_params=params,
            ))

        # Add some static obstacles for reference
        num_static = int(2 + 3 * difficulty)
        for _ in range(num_static):
            sx = random.uniform(-4, 4)
            sy = random.uniform(-4, 4)
            sz = random.uniform(0.5, 1.5)

            if self._is_near_endpoints(sx, sy, margin=1.5,
                                       check_start=local_start, check_goal=local_goal):
                continue

            size = random.uniform(0.3, 0.6)
            obstacles.append(ObstaclePrimitive(
                prim_type="cube",
                position=(sx, sy, sz),
                scale=(size / 2, size / 2, size / 2),
                color=(0.5, 0.5, 0.55),
                is_dynamic=False,
            ))

        tilt_quat, tilt_euler = self._compute_gravity_tilt(apply_tilt=True)

        # Transform to world coordinates for Sim spawning
        transformed_start, transformed_goal = local_start, local_goal
        if tilt_euler != (0.0, 0.0):
            transformed_start, transformed_goal = self._transform_endpoints_by_tilt(
                tilt_quat, local_start, local_goal)

        gap_size = self._compute_gap_size(difficulty)
        # Check solvability using LOCAL coordinates
        solvable, complexity = self._check_solvability(
            obstacles, gaps, gap_size, local_start, local_goal)

        labels = self._create_labels(
            ArenaMode.SHOOTING_GALLERY, None, difficulty, gaps, obstacles, tilt_euler,
            local_start=local_start, local_goal=local_goal,
            transformed_start=transformed_start, transformed_goal=transformed_goal
        )

        return ArenaResult(
            obstacles=obstacles,
            config=self.cfg,
            mode=ArenaMode.SHOOTING_GALLERY,
            sub_mode=None,
            difficulty=difficulty,
            solvable=True,  # Dynamic - always timing-solvable
            complexity=complexity,
            gaps=gaps,
            gravity_tilt_quat=tilt_quat,
            gravity_tilt_euler=tilt_euler,
            labels=labels,
            suggested_path=[transformed_start or self.cfg.start_pos,
                            (0, 0, 1.5), transformed_goal or self.cfg.goal_pos],
        )


# =============================================================================
# Isaac Sim Scene Spawner
# =============================================================================

class ArenaSpawner:
    """
    Spawns ArenaResult obstacles into an Isaac Sim scene.
    """

    def __init__(self, stage, base_path: str = "/World/Arena"):
        """
        Initialize spawner.

        Args:
            stage: USD stage
            base_path: Base prim path for arena objects
        """
        self.stage = stage
        self.base_path = base_path
        self.spawned_prims = []
        self._arena_root = None

    def clear(self):
        """Remove all spawned obstacles"""
        for prim_path in self.spawned_prims:
            prim = self.stage.GetPrimAtPath(prim_path)
            if prim.IsValid():
                self.stage.RemovePrim(prim_path)
        self.spawned_prims = []

    def spawn(self, result: ArenaResult) -> List[str]:
        """
        Spawn all obstacles from ArenaResult into the scene.

        Returns:
            List of spawned prim paths
        """
        from pxr import UsdGeom, Gf, UsdPhysics  # UsdPhysics for collision

        self.clear()

        # Create arena root with gravity tilt
        arena_root_path = self.base_path
        arena_root = UsdGeom.Xform.Define(self.stage, arena_root_path)

        xform = UsdGeom.Xformable(arena_root.GetPrim())
        xform.ClearXformOpOrder()

        # Apply gravity tilt rotation
        if result.gravity_tilt_quat != (1.0, 0.0, 0.0, 0.0):
            q = result.gravity_tilt_quat
            rotate_op = xform.AddOrientOp()
            rotate_op.Set(Gf.Quatf(q[0], q[1], q[2], q[3]))

        self._arena_root = arena_root_path

        # Spawn obstacles
        for i, obs in enumerate(result.obstacles):
            prim_path = f"{self.base_path}/obstacle_{i}"

            if obs.prim_type == "cylinder":
                prim = UsdGeom.Cylinder.Define(self.stage, prim_path)
            elif obs.prim_type == "cube":
                prim = UsdGeom.Cube.Define(self.stage, prim_path)
            elif obs.prim_type == "sphere":
                prim = UsdGeom.Sphere.Define(self.stage, prim_path)
            else:
                continue

            # CRITICAL: Apply Physics Collision API so drone can collide with obstacles
            # Without this, obstacles are "ghost" geometry (visual only) in Isaac Sim
            UsdPhysics.CollisionAPI.Apply(prim.GetPrim())

            # Set transform
            xform = UsdGeom.Xformable(prim.GetPrim())
            xform.ClearXformOpOrder()

            translate_op = xform.AddTranslateOp()
            translate_op.Set(Gf.Vec3d(*obs.position))

            # Apply rotation if not identity
            if obs.rotation != (1.0, 0.0, 0.0, 0.0):
                q = obs.rotation
                orient_op = xform.AddOrientOp()
                orient_op.Set(Gf.Quatf(q[0], q[1], q[2], q[3]))

            scale_op = xform.AddScaleOp()
            scale_op.Set(Gf.Vec3d(*obs.scale))

            # Set color
            if hasattr(prim, 'GetDisplayColorAttr'):
                prim.GetDisplayColorAttr().Set([obs.color])

            self.spawned_prims.append(prim_path)

        return self.spawned_prims

    def update_positions(self, positions: List[Tuple[float, float, float]]):
        """Update positions of spawned obstacles (for dynamic mode)"""
        from pxr import UsdGeom, Gf

        for prim_path, pos in zip(self.spawned_prims, positions):
            prim = self.stage.GetPrimAtPath(prim_path)
            if prim.IsValid():
                xform = UsdGeom.Xformable(prim)
                ops = xform.GetOrderedXformOps()
                if ops:
                    ops[0].Set(Gf.Vec3d(*pos))


__all__ = [
    "ArenaMode",
    "ChannelOrientation",
    "SandwichSubType",
    "ObstaclePrimitive",
    "GapInfo",
    "ArenaConfig",
    "CurriculumLabels",
    "ArenaResult",
    "UniversalArenaGenerator",
    "ArenaSpawner",
]
