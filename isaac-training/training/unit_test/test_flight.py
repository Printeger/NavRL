#!/usr/bin/env python3
"""
UAV Flight Test in Universal Arena
====================================
结合 Universal Arena Generator 场景与无人机飞行控制和 LiDAR 传感器。

功能:
    - 使用 UniversalArenaGenerator 生成 5 种模式的训练场景
    - 无人机配备 Livox Mid-360 LiDAR 传感器
    - 实时点云可视化
    - 键盘控制飞行 + 场景切换

键盘控制:
    飞行控制:
        W/S     : 前进/后退 (X轴)
        A/D     : 左移/右移 (Y轴)
        Q/E     : 上升/下降 (Z轴)
        Z/X     : 左转/右转 (Yaw)
        F       : 飞向目标点
        O       : 返回起点 (仅目标位置)
        B       : 重置无人机 (位置+速度+姿态)

    场景控制:
        1-5     : 切换场景模式 (1=Lattice, 2=Ant Nest, 3=Channel, 4=Sandwich, 5=Shooting)
        6-8     : 切换子模式 (Mode 3: 6=Horiz, 7=Vert, 8=Sloped; Mode 4: 6=Cave, 7=Hazards)
        R       : 重新生成当前场景
        +/-     : 增加/减少难度
        G       : 开关重力倾斜

    显示控制:
        T       : 开关点云显示
        V       : 开关重力向量显示
        I       : 打印场景统计信息
        H       : 打印危险物详情 (Mode 4)
        P       : 暂停/继续动态障碍物

运行方式 (与 train.py 相同):
    conda activate NavRL
    cd /home/mint/rl_dev/NavRL/isaac-training
    
    # 默认参数运行 (从 cfg/train.yaml 读取配置)
    python training/unit_test/test_flight.py
    
    # 带可视化
    python training/unit_test/test_flight.py headless=False
    
    # 自定义 LiDAR 参数
    python training/unit_test/test_flight.py headless=False sensor.lidar_range=30
    
    # 使用不同无人机模型
    python training/unit_test/test_flight.py drone.model_name=Firefly

Author: NavRL Team
"""

import os
import sys
import math
import hydra
from omegaconf import DictConfig

# ============================================================================
# Path Setup (MUST be before any imports)
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_ROOT = os.path.dirname(SCRIPT_DIR)
SCRIPTS_PATH = os.path.join(TRAINING_ROOT, "scripts")
MODULE_PATH = os.path.join(TRAINING_ROOT, "envs", "universal_generator.py")
CFG_PATH = os.path.join(TRAINING_ROOT, "cfg")

if TRAINING_ROOT not in sys.path:
    sys.path.insert(0, TRAINING_ROOT)
if SCRIPTS_PATH not in sys.path:
    sys.path.insert(0, SCRIPTS_PATH)


@hydra.main(config_path=CFG_PATH, config_name="train", version_base=None)
def main(cfg: DictConfig):
    """Main function - all imports happen here after SimulationApp

    Args:
        cfg: Hydra configuration object (from train.yaml)
    """

    # =========================================================================
    # Step 1: Launch Isaac Sim (MUST be first before any omni/pxr imports)
    # =========================================================================
    print("=" * 70)
    print("🚁 UAV Flight Test in Universal Arena")
    print("=" * 70)
    print(f"[INFO] Headless mode: {cfg.headless}")
    print(f"[INFO] Device: {cfg.device}")
    print(f"[INFO] LiDAR range: {cfg.sensor.lidar_range} m")
    print(f"[INFO] Vertical FOV: {cfg.sensor.lidar_vfov}")
    print(f"[INFO] Vertical beams: {cfg.sensor.lidar_vbeams}")
    print(f"[INFO] Horizontal resolution: {cfg.sensor.lidar_hres}°")
    print(f"[INFO] Drone model: {cfg.drone.model_name}")
    print("-" * 70)
    print("[INFO] Launching Isaac Sim...")

    from omni.isaac.kit import SimulationApp
    simulation_app = SimulationApp({
        "headless": cfg.headless,
        "width": 1920,
        "height": 1080,
        "anti_aliasing": 1,
    })

    # =========================================================================
    # Step 2: Import dependencies (AFTER SimulationApp is created)
    # =========================================================================
    print("[INFO] Importing dependencies...")

    import importlib.util
    import torch
    import numpy as np
    import carb
    import carb.input
    import omni
    import omni.appwindow
    import omni.isaac.core.utils.prims as prim_utils
    from omni.isaac.core.simulation_context import SimulationContext
    from omni.isaac.debug_draw import _debug_draw
    from omni.isaac.orbit.sensors import RayCaster, RayCasterCfg, patterns
    from omni_drones.robots.drone import MultirotorBase
    from omni_drones.controllers import LeePositionController
    from pxr import UsdGeom, Gf, UsdLux

    # Import the generator module
    spec = importlib.util.spec_from_file_location(
        "universal_generator", MODULE_PATH)
    generator_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator_module)

    UniversalArenaGenerator = generator_module.UniversalArenaGenerator
    ArenaSpawner = generator_module.ArenaSpawner
    ArenaMode = generator_module.ArenaMode
    ArenaConfig = generator_module.ArenaConfig

    print("[INFO] Dependencies loaded successfully")

    # =========================================================================
    # Step 3: Create Simulation Context
    # =========================================================================
    print("[INFO] Creating simulation context...")

    sim_context = SimulationContext(
        stage_units_in_meters=1.0,
        physics_dt=0.02,
        rendering_dt=0.02,
        backend="torch",
        device=cfg.device,
    )
    stage = sim_context.stage
    device = cfg.device

    # =========================================================================
    # Step 4: Create Base Scene (Ground, Light)
    # =========================================================================
    print("[INFO] Creating base scene...")

    # Ground plane (large enough for arena)
    ground_path = "/World/GroundPlane"
    ground = UsdGeom.Mesh.Define(stage, ground_path)
    ground.CreatePointsAttr(
        [(-50, -50, 0), (50, -50, 0), (50, 50, 0), (-50, 50, 0)])
    ground.CreateFaceVertexCountsAttr([4])
    ground.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    ground.CreateDisplayColorAttr([(0.15, 0.15, 0.18)])

    # Dome Light (ambient/environment lighting)
    dome_light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome_light.CreateIntensityAttr(800.0)
    dome_light.CreateColorAttr(Gf.Vec3f(0.9, 0.9, 1.0))

    # Distant Light (sun-like directional light)
    distant_light = UsdLux.DistantLight.Define(stage, "/World/DistantLight")
    distant_light.CreateIntensityAttr(3000.0)
    distant_light.CreateColorAttr(Gf.Vec3f(1.0, 0.98, 0.95))
    distant_light.CreateAngleAttr(1.0)
    xf = UsdGeom.Xformable(distant_light.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddRotateXYZOp().Set(Gf.Vec3d(-45, 30, 0))

    # Sphere Light (fill light above arena)
    sphere_light = UsdLux.SphereLight.Define(stage, "/World/SphereLight")
    sphere_light.CreateIntensityAttr(5000.0)
    sphere_light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
    sphere_light.CreateRadiusAttr(0.5)
    xf = UsdGeom.Xformable(sphere_light.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(0, 0, 12))

    print("[INFO] Lighting configured")

    # Arena boundary markers (corners)
    for i, (cx, cy) in enumerate([(-5, -5), (5, -5), (5, 5), (-5, 5)]):
        marker = UsdGeom.Cylinder.Define(stage, f"/World/BoundaryMarker_{i}")
        xform = UsdGeom.Xformable(marker.GetPrim())
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(cx, cy, 0.1))
        xform.AddScaleOp().Set(Gf.Vec3d(0.1, 0.1, 0.1))
        marker.CreateDisplayColorAttr([(0.8, 0.8, 0.2)])

    # =========================================================================
    # Step 5: Create Start/Goal Markers
    # =========================================================================
    arena_cfg = ArenaConfig()

    # Start marker (green sphere)
    start_marker = UsdGeom.Sphere.Define(stage, "/World/StartMarker")
    xf = UsdGeom.Xformable(start_marker.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*arena_cfg.start_pos))
    xf.AddScaleOp().Set(Gf.Vec3d(0.25, 0.25, 0.25))
    start_marker.CreateDisplayColorAttr([(0.2, 0.9, 0.2)])

    # Goal marker (red sphere)
    goal_marker = UsdGeom.Sphere.Define(stage, "/World/GoalMarker")
    xf = UsdGeom.Xformable(goal_marker.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*arena_cfg.goal_pos))
    xf.AddScaleOp().Set(Gf.Vec3d(0.25, 0.25, 0.25))
    goal_marker.CreateDisplayColorAttr([(0.9, 0.2, 0.2)])

    # =========================================================================
    # Step 6: Create UAV
    # =========================================================================
    print("[INFO] Creating UAV...")

    # Create environment prim for drone
    if not prim_utils.is_prim_path_valid("/World/envs/env_0"):
        prim_utils.define_prim("/World/envs/env_0")

    # Use drone model from config
    model_name = cfg.drone.model_name
    if model_name not in MultirotorBase.REGISTRY:
        print(
            f"[WARNING] Model '{model_name}' not found in registry, using 'Hummingbird'")
        model_name = "Hummingbird"

    drone_model = MultirotorBase.REGISTRY[model_name]
    drone_cfg = drone_model.cfg_cls(force_sensor=False)
    drone = drone_model(cfg=drone_cfg)

    # Spawn drone at start position
    init_pos = arena_cfg.start_pos
    drone.spawn(translations=[(init_pos[0], init_pos[1], init_pos[2])])

    print(f"[INFO] UAV '{model_name}' created at {init_pos}")

    # =========================================================================
    # Step 7: Initialize Simulation and Drone
    # =========================================================================
    print("[INFO] Initializing physics...")
    sim_context.reset()
    drone.initialize()

    # =========================================================================
    # Step 8: Initialize Arena Generator and Spawner
    # =========================================================================
    print("[INFO] Initializing arena generator...")

    generator = UniversalArenaGenerator(arena_cfg, seed=42)
    spawner = ArenaSpawner(stage, base_path="/World/Arena")

    # =========================================================================
    # Step 9: Create LiDAR Sensor
    # =========================================================================
    print("[INFO] Creating Livox Mid-360 LiDAR...")

    # LiDAR parameters from config
    lidar_range = cfg.sensor.lidar_range
    lidar_vfov = cfg.sensor.lidar_vfov
    lidar_vbeams = cfg.sensor.lidar_vbeams
    lidar_hres = cfg.sensor.lidar_hres

    vertical_angles = torch.linspace(
        lidar_vfov[0], lidar_vfov[1], lidar_vbeams
    ).tolist()

    # Get drone base_link path
    drone_base_link = f"/World/envs/env_0/{model_name}_0/base_link"

    ray_caster_cfg = RayCasterCfg(
        prim_path=drone_base_link,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        attach_yaw_only=False,
        pattern_cfg=patterns.BpearlPatternCfg(
            horizontal_fov=360.0,
            horizontal_res=lidar_hres,
            vertical_ray_angles=vertical_angles,
        ),
        max_distance=lidar_range,
        # Use ground plane (RayCaster limitation: supports only 1 mesh prim)
        # Arena obstacles are Xform with children, not a single mesh
        mesh_prim_paths=["/World/GroundPlane"],
        debug_vis=False,  # We'll do custom visualization
    )

    lidar = RayCaster(ray_caster_cfg)
    lidar._initialize_impl()
    lidar_initialized = True

    print(
        f"[INFO] LiDAR initialized: {int(360/lidar_hres)} x {lidar_vbeams} = {int(360/lidar_hres) * lidar_vbeams} rays")
    print(f"[INFO] Note: LiDAR scans ground plane only (arena obstacles visible via physics)")

    print(f"[INFO] LiDAR configured (will initialize after arena spawn)")

    # =========================================================================
    # Step 10: Create Position Controller
    # =========================================================================
    print("[INFO] Creating flight controller...")

    controller = LeePositionController(
        g=9.81, uav_params=drone.params
    ).to(device)

    # =========================================================================
    # Step 11: Initialize Debug Draw
    # =========================================================================
    debug_draw = _debug_draw.acquire_debug_draw_interface()

    # =========================================================================
    # Step 12: State Variables
    # =========================================================================
    mode_names = {
        ArenaMode.LATTICE_FOREST: "1: Lattice Forest",
        ArenaMode.ANT_NEST: "2: Ant Nest (Maze)",
        ArenaMode.RESTRICTED_CHANNEL: "3: Restricted Channel",
        ArenaMode.LETHAL_SANDWICH: "4: Lethal Sandwich",
        ArenaMode.SHOOTING_GALLERY: "5: Shooting Gallery",
    }

    sub_mode_names = {
        ArenaMode.RESTRICTED_CHANNEL: ["horizontal", "vertical", "sloped"],
        ArenaMode.LETHAL_SANDWICH: ["cave", "hazards"],
    }

    # Arena state
    current_mode = ArenaMode.LATTICE_FOREST
    current_sub_mode = None
    current_difficulty = 0.5
    current_result = None
    apply_gravity_tilt = False  # Start with no tilt for easier flight
    paused = False

    # Flight state
    target_pos = torch.tensor(
        [init_pos[0], init_pos[1], init_pos[2]],
        device=device, dtype=torch.float32
    )
    target_yaw = torch.tensor([0.0], device=device)
    move_speed = 0.05
    yaw_speed = 0.02

    # Visualization state
    show_pointcloud = True
    show_gravity_vector = True
    point_size = 4.0

    # Simulation state
    sim_time = 0.0
    dt = sim_context.get_physics_dt()

    # =========================================================================
    # Step 13: Keyboard Input Setup
    # =========================================================================
    appwindow = omni.appwindow.get_default_app_window()
    input_interface = carb.input.acquire_input_interface()
    keyboard = appwindow.get_keyboard()
    key_pressed = {}

    def on_keyboard_event(event):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            key_pressed[event.input] = True
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            key_pressed[event.input] = False
        return True

    keyboard_sub = input_interface.subscribe_to_keyboard_events(
        keyboard, on_keyboard_event)

    # =========================================================================
    # Step 14: Helper Functions
    # =========================================================================
    def update_endpoint_markers(result):
        """Update start/goal markers based on arena result."""
        if result.labels.local_start:
            start_pos = result.labels.local_start
        else:
            start_pos = arena_cfg.start_pos

        if result.labels.local_goal:
            goal_pos = result.labels.local_goal
        else:
            goal_pos = arena_cfg.goal_pos

        # Update start marker
        start_prim = stage.GetPrimAtPath("/World/StartMarker")
        if start_prim:
            xf = UsdGeom.Xformable(start_prim)
            xf.ClearXformOpOrder()
            xf.AddTranslateOp().Set(Gf.Vec3d(*start_pos))
            xf.AddScaleOp().Set(Gf.Vec3d(0.25, 0.25, 0.25))

        # Update goal marker
        goal_prim = stage.GetPrimAtPath("/World/GoalMarker")
        if goal_prim:
            xf = UsdGeom.Xformable(goal_prim)
            xf.ClearXformOpOrder()
            xf.AddTranslateOp().Set(Gf.Vec3d(*goal_pos))
            xf.AddScaleOp().Set(Gf.Vec3d(0.25, 0.25, 0.25))

        return start_pos, goal_pos

    def regenerate_arena():
        nonlocal current_result, sim_time, target_pos

        print(f"\n{'='*60}")
        print(f"Generating: {mode_names[current_mode]}")
        print(f"Difficulty: {current_difficulty:.2f}")
        if current_sub_mode:
            print(f"Sub-mode: {current_sub_mode}")
        print(f"Gravity Tilt: {'ON' if apply_gravity_tilt else 'OFF'}")
        print(f"{'='*60}")

        # Configure gravity tilt
        original_max_tilt = arena_cfg.max_tilt_angle
        if not apply_gravity_tilt:
            arena_cfg.max_tilt_angle = 0.0

        generator.cfg = arena_cfg

        current_result = generator.reset(
            mode=current_mode,
            difficulty=current_difficulty,
            sub_mode=current_sub_mode,
        )

        arena_cfg.max_tilt_angle = original_max_tilt

        # Spawn obstacles
        spawner.spawn(current_result)

        # Update markers and get positions
        start_pos, goal_pos = update_endpoint_markers(current_result)

        # Move drone to start position
        target_pos[0] = start_pos[0]
        target_pos[1] = start_pos[1]
        target_pos[2] = start_pos[2]

        # Print stats
        result = current_result
        print(f"Obstacles: {len(result.obstacles)}")
        print(f"Solvable: {result.solvable}")
        print(f"Complexity: {result.complexity:.3f}")
        print(
            f"Tilt: roll={result.gravity_tilt_euler[0]:.1f}°, pitch={result.gravity_tilt_euler[1]:.1f}°")
        print(f"Start: {start_pos}, Goal: {goal_pos}")

        sim_time = 0.0

    def draw_gravity_vector():
        """Draw gravity vector visualization."""
        if not show_gravity_vector or current_result is None:
            return

        q = current_result.gravity_tilt_quat
        w, x, y, z = q

        # Standard gravity rotated by quaternion
        gx = -2 * (x * z - w * y)
        gy = -2 * (y * z + w * x)
        gz = -(1 - 2 * (x * x + y * y))

        # Normalize
        mag = math.sqrt(gx * gx + gy * gy + gz * gz)
        if mag > 0:
            gx, gy, gz = gx / mag, gy / mag, gz / mag

        # Draw from above arena
        origin = (0, 0, 5.0)
        endpoint = (origin[0] + gx * 2, origin[1] + gy * 2, origin[2] + gz * 2)

        debug_draw.draw_lines([origin], [endpoint],
                              [(1.0, 0.7, 0.0, 1.0)], [5])

    def draw_pointcloud(ray_hits, lidar_pos):
        """Draw LiDAR point cloud with distance-based coloring."""
        if not show_pointcloud:
            return

        if ray_hits is None or lidar_pos is None:
            return

        distances = (ray_hits - lidar_pos.unsqueeze(1)).norm(dim=-1)
        valid_mask = distances[0] < lidar_range
        valid_points = ray_hits[0][valid_mask]
        valid_distances = distances[0][valid_mask]

        if len(valid_points) == 0:
            return

        points_np = valid_points.cpu().numpy()
        dists_np = valid_distances.cpu().numpy()
        norm_dists = dists_np / lidar_range

        colors = []
        for d in norm_dists:
            if d < 0.25:
                colors.append((1.0, 0.2, 0.2, 1.0))  # Red (close)
            elif d < 0.5:
                colors.append((1.0, 0.6, 0.2, 1.0))  # Orange
            elif d < 0.75:
                colors.append((1.0, 1.0, 0.2, 1.0))  # Yellow
            else:
                colors.append((0.2, 1.0, 0.2, 1.0))  # Green (far)

        point_list = [tuple(p) for p in points_np]
        debug_draw.draw_points(point_list, colors, [
                               point_size] * len(point_list))

    def print_stats():
        """Print arena statistics."""
        if current_result is None:
            print("No arena generated")
            return

        r = current_result
        print("\n" + "="*60)
        print("ARENA & FLIGHT STATISTICS")
        print("="*60)
        print(f"Mode: {r.mode.value} ({mode_names[r.mode]})")
        print(f"Sub-mode: {r.sub_mode or 'N/A'}")
        print(f"Difficulty: {r.difficulty:.2f}")
        print(f"Solvable: {r.solvable}")
        print(f"Complexity: {r.complexity:.3f}")
        print(f"Obstacles: {len(r.obstacles)}")
        print(f"  - Static: {sum(1 for o in r.obstacles if not o.is_dynamic)}")
        print(f"  - Dynamic: {sum(1 for o in r.obstacles if o.is_dynamic)}")
        print(f"  - Hazards: {sum(1 for o in r.obstacles if o.is_hazard)}")
        print(
            f"Gravity Tilt: roll={r.gravity_tilt_euler[0]:.2f}°, pitch={r.gravity_tilt_euler[1]:.2f}°")

        if r.labels.local_start:
            print(
                f"Local Start: {tuple(f'{v:.2f}' for v in r.labels.local_start)}")
        if r.labels.local_goal:
            print(
                f"Local Goal: {tuple(f'{v:.2f}' for v in r.labels.local_goal)}")

        # Current drone position
        raw_state = drone.get_state()
        if raw_state.dim() == 3:
            drone_pos = raw_state[0, 0, :3]
        elif raw_state.dim() == 2:
            drone_pos = raw_state[0, :3]
        else:
            drone_pos = raw_state[:3]
        print(
            f"Drone Position: ({drone_pos[0]:.2f}, {drone_pos[1]:.2f}, {drone_pos[2]:.2f})")
        print(
            f"Target Position: ({target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f})")
        print("="*60 + "\n")

    def print_hazard_details():
        """Print hazard details for Mode D."""
        if current_result is None:
            print("No arena generated")
            return

        hazards = [o for o in current_result.obstacles if o.is_hazard]
        if not hazards:
            print(
                "\n[INFO] No hazards in current arena (try Mode 4 with sub-mode 7: hazards)")
            return

        print("\n" + "="*60)
        print("HAZARD DETAILS (Mode 4: Lethal Sandwich)")
        print("="*60)

        thin_wires = [h for h in hazards if h.scale[0] <= 0.015]
        thick_cables = [h for h in hazards if h.scale[0] > 0.015]

        print(f"Total hazards: {len(hazards)}")
        print(f"  - Thin wires: {len(thin_wires)}")
        print(f"  - Thick cables: {len(thick_cables)}")

        if thin_wires:
            s = thin_wires[0]
            print(f"Sample wire: pos={tuple(f'{v:.2f}' for v in s.position)}, "
                  f"r={s.scale[0]:.4f}m, color={tuple(f'{v:.2f}' for v in s.color)}")
        print("="*60 + "\n")

    # =========================================================================
    # Step 15: Initial Arena Generation
    # =========================================================================
    regenerate_arena()

    # =========================================================================
    # Step 16: Main Loop
    # =========================================================================
    print("\n" + "="*70)
    print("🎮 FLIGHT TEST RUNNING")
    print("="*70)
    print("Flight Controls:")
    print("  W/S     : Forward/Backward (X)")
    print("  A/D     : Left/Right (Y)")
    print("  Q/E     : Up/Down (Z)")
    print("  Z/X     : Yaw Left/Right")
    print("  F       : Fly to Goal")
    print("  O       : Return to Start (target only)")
    print("  B       : Reset Drone (position + velocity)")
    print("")
    print("Arena Controls:")
    print("  1-5     : Switch Mode (1=Lattice, 2=Maze, 3=Channel, 4=Sandwich, 5=Dynamic)")
    print("  6-8     : Sub-mode (Mode3: 6=H,7=V,8=S; Mode4: 6=Cave,7=Hazard)")
    print("  R       : Regenerate")
    print("  +/-     : Difficulty")
    print("  G       : Toggle Gravity Tilt")
    print("")
    print("Display:")
    print("  T       : Toggle Point Cloud")
    print("  V       : Toggle Gravity Vector")
    print("  I       : Print Stats")
    print("  H       : Hazard Details")
    print("  P       : Pause Dynamic Obstacles")
    print("  Ctrl+C  : Exit")
    print("="*70 + "\n")

    step = 0
    print_interval = 250

    try:
        while simulation_app.is_running():
            sim_time += dt

            # =================================================================
            # Handle Flight Controls
            # =================================================================
            # Movement
            if key_pressed.get(carb.input.KeyboardInput.W, False):
                target_pos[0] += move_speed
            if key_pressed.get(carb.input.KeyboardInput.S, False):
                target_pos[0] -= move_speed
            if key_pressed.get(carb.input.KeyboardInput.A, False):
                target_pos[1] += move_speed
            if key_pressed.get(carb.input.KeyboardInput.D, False):
                target_pos[1] -= move_speed
            if key_pressed.get(carb.input.KeyboardInput.Q, False):
                target_pos[2] += move_speed
            if key_pressed.get(carb.input.KeyboardInput.E, False):
                target_pos[2] -= move_speed

            # Yaw
            if key_pressed.get(carb.input.KeyboardInput.Z, False):
                target_yaw[0] += yaw_speed
            if key_pressed.get(carb.input.KeyboardInput.X, False):
                target_yaw[0] -= yaw_speed

            # Fly to goal
            if key_pressed.get(carb.input.KeyboardInput.F, False):
                if current_result and current_result.labels.local_goal:
                    goal = current_result.labels.local_goal
                    target_pos[0] = goal[0]
                    target_pos[1] = goal[1]
                    target_pos[2] = goal[2]
                    print(f"Flying to goal: {goal}")
                key_pressed[carb.input.KeyboardInput.F] = False

            # Return to start
            if key_pressed.get(carb.input.KeyboardInput.O, False):
                if current_result and current_result.labels.local_start:
                    start = current_result.labels.local_start
                    target_pos[0] = start[0]
                    target_pos[1] = start[1]
                    target_pos[2] = start[2]
                    print(f"Returning to start: {start}")
                key_pressed[carb.input.KeyboardInput.O] = False

            # Reset drone (B key - "Back to origin")
            if key_pressed.get(carb.input.KeyboardInput.B, False):
                if current_result and current_result.labels.local_start:
                    start = current_result.labels.local_start
                    # Reset target position
                    target_pos[0] = start[0]
                    target_pos[1] = start[1]
                    target_pos[2] = start[2]
                    # Reset yaw
                    target_yaw[0] = 0.0
                    # Reset drone physics state
                    drone.set_world_poses(
                        positions=torch.tensor(
                            [[start[0], start[1], start[2]]], device=device),
                        orientations=torch.tensor(
                            # Identity quaternion
                            [[1.0, 0.0, 0.0, 0.0]], device=device)
                    )
                    # Reset velocities to zero
                    drone.set_velocities(
                        # linear + angular velocities
                        velocities=torch.zeros((1, 6), device=device)
                    )
                    print(f"[RESET] Drone reset to start position: {start}")
                key_pressed[carb.input.KeyboardInput.B] = False

            # Clamp height
            target_pos[2] = torch.clamp(
                target_pos[2], 0.3, arena_cfg.size_z - 0.3)

            # =================================================================
            # Handle Arena Controls
            # =================================================================
            arena_changed = False

            # Mode selection (1-5)
            if key_pressed.get(carb.input.KeyboardInput.KEY_1, False):
                current_mode = ArenaMode.LATTICE_FOREST
                current_sub_mode = None
                arena_changed = True
                key_pressed[carb.input.KeyboardInput.KEY_1] = False
            elif key_pressed.get(carb.input.KeyboardInput.KEY_2, False):
                current_mode = ArenaMode.ANT_NEST
                current_sub_mode = None
                arena_changed = True
                key_pressed[carb.input.KeyboardInput.KEY_2] = False
            elif key_pressed.get(carb.input.KeyboardInput.KEY_3, False):
                current_mode = ArenaMode.RESTRICTED_CHANNEL
                current_sub_mode = None
                arena_changed = True
                key_pressed[carb.input.KeyboardInput.KEY_3] = False
            elif key_pressed.get(carb.input.KeyboardInput.KEY_4, False):
                current_mode = ArenaMode.LETHAL_SANDWICH
                current_sub_mode = None
                arena_changed = True
                key_pressed[carb.input.KeyboardInput.KEY_4] = False
            elif key_pressed.get(carb.input.KeyboardInput.KEY_5, False):
                current_mode = ArenaMode.SHOOTING_GALLERY
                current_sub_mode = None
                arena_changed = True
                key_pressed[carb.input.KeyboardInput.KEY_5] = False

            # Sub-mode selection (6-8)
            if key_pressed.get(carb.input.KeyboardInput.KEY_6, False):
                if current_mode in sub_mode_names:
                    subs = sub_mode_names[current_mode]
                    if len(subs) >= 1:
                        current_sub_mode = subs[0]
                        arena_changed = True
                key_pressed[carb.input.KeyboardInput.KEY_6] = False
            elif key_pressed.get(carb.input.KeyboardInput.KEY_7, False):
                if current_mode in sub_mode_names:
                    subs = sub_mode_names[current_mode]
                    if len(subs) >= 2:
                        current_sub_mode = subs[1]
                        arena_changed = True
                key_pressed[carb.input.KeyboardInput.KEY_7] = False
            elif key_pressed.get(carb.input.KeyboardInput.KEY_8, False):
                if current_mode in sub_mode_names:
                    subs = sub_mode_names[current_mode]
                    if len(subs) >= 3:
                        current_sub_mode = subs[2]
                        arena_changed = True
                key_pressed[carb.input.KeyboardInput.KEY_8] = False

            # Regenerate (R)
            if key_pressed.get(carb.input.KeyboardInput.R, False):
                arena_changed = True
                key_pressed[carb.input.KeyboardInput.R] = False

            # Difficulty (+/-)
            if key_pressed.get(carb.input.KeyboardInput.EQUAL, False):
                current_difficulty = min(1.0, current_difficulty + 0.1)
                arena_changed = True
                key_pressed[carb.input.KeyboardInput.EQUAL] = False
            if key_pressed.get(carb.input.KeyboardInput.MINUS, False):
                current_difficulty = max(0.0, current_difficulty - 0.1)
                arena_changed = True
                key_pressed[carb.input.KeyboardInput.MINUS] = False

            # Gravity tilt (G)
            if key_pressed.get(carb.input.KeyboardInput.G, False):
                apply_gravity_tilt = not apply_gravity_tilt
                print(f"Gravity tilt: {'ON' if apply_gravity_tilt else 'OFF'}")
                arena_changed = True
                key_pressed[carb.input.KeyboardInput.G] = False

            # =================================================================
            # Handle Display Controls
            # =================================================================
            # Toggle point cloud (T)
            if key_pressed.get(carb.input.KeyboardInput.T, False):
                show_pointcloud = not show_pointcloud
                print(f"Point cloud: {'ON' if show_pointcloud else 'OFF'}")
                key_pressed[carb.input.KeyboardInput.T] = False

            # Toggle gravity vector (V)
            if key_pressed.get(carb.input.KeyboardInput.V, False):
                show_gravity_vector = not show_gravity_vector
                print(
                    f"Gravity vector: {'ON' if show_gravity_vector else 'OFF'}")
                key_pressed[carb.input.KeyboardInput.V] = False

            # Print stats (I)
            if key_pressed.get(carb.input.KeyboardInput.I, False):
                print_stats()
                key_pressed[carb.input.KeyboardInput.I] = False

            # Hazard details (H)
            if key_pressed.get(carb.input.KeyboardInput.H, False):
                print_hazard_details()
                key_pressed[carb.input.KeyboardInput.H] = False

            # Pause (P)
            if key_pressed.get(carb.input.KeyboardInput.P, False):
                paused = not paused
                print(
                    f"Dynamic obstacles: {'PAUSED' if paused else 'RUNNING'}")
                key_pressed[carb.input.KeyboardInput.P] = False

            # =================================================================
            # Regenerate Arena if Changed
            # =================================================================
            if arena_changed:
                regenerate_arena()

            # =================================================================
            # Update Dynamic Obstacles (Mode E)
            # =================================================================
            if not paused and current_result and current_mode == ArenaMode.SHOOTING_GALLERY:
                new_positions = generator.update_dynamic_obstacles(dt)
                if new_positions:
                    spawner.update_positions(new_positions)

            # =================================================================
            # Flight Control
            # =================================================================
            # Get drone state
            raw_state = drone.get_state()
            if raw_state.dim() == 3:
                drone_state = raw_state[0, 0, :13]
            elif raw_state.dim() == 2:
                drone_state = raw_state[0, :13]
            else:
                drone_state = raw_state[:13]

            # Compute control action
            action = controller(
                drone_state,
                target_pos=target_pos,
                target_yaw=target_yaw
            )

            # Apply action
            drone.apply_action(action)

            # =================================================================
            # Physics Step
            # =================================================================
            sim_context.step()

            # =================================================================
            # Update LiDAR
            # =================================================================
            lidar.update(dt)

            # =================================================================
            # Visualization
            # =================================================================
            debug_draw.clear_points()
            debug_draw.clear_lines()

            # Draw point cloud
            ray_hits = lidar.data.ray_hits_w
            lidar_pos = lidar.data.pos_w
            draw_pointcloud(ray_hits, lidar_pos)

            # Draw gravity vector
            draw_gravity_vector()

            # =================================================================
            # Periodic Status Print
            # =================================================================
            if step % print_interval == 0:
                pos = drone_state[:3]
                if ray_hits is not None and lidar_pos is not None:
                    distances = (
                        ray_hits - lidar_pos.unsqueeze(1)).norm(dim=-1)
                    valid_mask = distances < lidar_range
                    num_hits = valid_mask.sum().item()
                    if num_hits > 0:
                        min_dist = distances[valid_mask].min().item()
                    else:
                        min_dist = float('inf')
                else:
                    num_hits = 0
                    min_dist = float('inf')

                print(f"[{mode_names[current_mode]}] "
                      f"Pos: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) | "
                      f"Target: ({target_pos[0]:.1f}, {target_pos[1]:.1f}, {target_pos[2]:.1f}) | "
                      f"LiDAR: {num_hits} pts, min={min_dist:.1f}m")

            step += 1

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user (Ctrl+C)")

    # =========================================================================
    # Cleanup
    # =========================================================================
    input_interface.unsubscribe_to_keyboard_events(keyboard, keyboard_sub)
    debug_draw.clear_points()
    debug_draw.clear_lines()
    spawner.clear()
    simulation_app.close()

    print("\n" + "="*70)
    print("✅ Flight test finished!")
    print("="*70)


if __name__ == "__main__":
    main()
