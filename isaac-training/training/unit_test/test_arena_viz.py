#!/usr/bin/env python3
"""
Interactive Visualization for Universal Arena Generator
=======================================================
Spawns procedural arenas in Isaac Sim GUI for visual inspection.

Updated Features:
    - Coordinate system display (local vs world frames)
    - Dynamic start/goal marker updates for vertical shaft mode
    - Hazard visibility inspection (bright colors for thin wires)
    - Physics collision verification

Key Bindings:
    A-E      : Switch modes (A=Lattice, B=Ant Nest, C=Channel, D=Sandwich, E=Shooting)
    1-3      : Switch sub-modes (for Mode C: 1=Horiz, 2=Vert, 3=Sloped; Mode D: 1=Cave, 2=Hazards)
    R        : Regenerate arena with current settings
    +/-      : Increase/decrease difficulty
    G        : Toggle gravity tilt on/off
    SPACE    : Pause/resume dynamic obstacles
    T        : Toggle gravity vector visualization
    S        : Print solvability and stats (with coordinate info)
    H        : Print hazard details (for Mode D)

Instructions:
    Run with Isaac Sim's Python:
        cd /home/mint/rl_dev
        ./python.sh NavRL/isaac-training/training/unit_test/test_arena_viz.py

Author: NavRL Team
"""

import os
import sys
import math

# ============================================================================
# Path Setup (MUST be before any imports)
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_ROOT = os.path.dirname(SCRIPT_DIR)
MODULE_PATH = os.path.join(TRAINING_ROOT, "envs", "universal_generator.py")

if TRAINING_ROOT not in sys.path:
    sys.path.insert(0, TRAINING_ROOT)


def main():
    """Main visualization function - all imports happen here after SimulationApp"""

    # =========================================================================
    # Step 1: Launch Isaac Sim (MUST be first before any omni/pxr imports)
    # =========================================================================
    print("=" * 70)
    print("🏗️  Universal Arena Generator - Enhanced Visualization Demo")
    print("=" * 70)
    print("[INFO] Launching Isaac Sim...")

    from omni.isaac.kit import SimulationApp
    simulation_app = SimulationApp(
        {"headless": False, "width": 1600, "height": 900})

    # =========================================================================
    # Step 2: Import dependencies (AFTER SimulationApp is created)
    # =========================================================================
    print("[INFO] Importing dependencies...")

    import importlib.util
    import omni
    import carb
    import carb.input
    import omni.appwindow
    from omni.isaac.core import World
    from omni.isaac.core.simulation_context import SimulationContext
    from omni.isaac.debug_draw import _debug_draw
    import omni.isaac.core.utils.prims as prim_utils
    from pxr import UsdGeom, Gf

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
    )
    stage = sim_context.stage

    # =========================================================================
    # Step 4: Create Base Scene (Ground, Light, Markers)
    # =========================================================================
    print("[INFO] Creating base scene...")

    # Ground plane
    ground_path = "/World/GroundPlane"
    ground = UsdGeom.Mesh.Define(stage, ground_path)
    ground.CreatePointsAttr(
        [(-50, -50, 0), (50, -50, 0), (50, 50, 0), (-50, 50, 0)])
    ground.CreateFaceVertexCountsAttr([4])
    ground.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    ground.CreateDisplayColorAttr([(0.2, 0.2, 0.22)])

    # =========== Add Lighting ===========
    from pxr import UsdLux

    # Dome Light (ambient/environment lighting)
    dome_light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome_light.CreateIntensityAttr(1000.0)
    # Slightly blue-ish ambient
    dome_light.CreateColorAttr(Gf.Vec3f(0.9, 0.9, 1.0))

    # Distant Light (sun-like directional light)
    distant_light = UsdLux.DistantLight.Define(stage, "/World/DistantLight")
    distant_light.CreateIntensityAttr(3000.0)
    distant_light.CreateColorAttr(Gf.Vec3f(1.0, 0.98, 0.95))  # Warm sunlight
    distant_light.CreateAngleAttr(1.0)  # Soft shadows
    xf = UsdGeom.Xformable(distant_light.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddRotateXYZOp().Set(Gf.Vec3d(-45, 30, 0))  # Angled from above

    # Sphere Light (point light above arena for fill)
    sphere_light = UsdLux.SphereLight.Define(stage, "/World/SphereLight")
    sphere_light.CreateIntensityAttr(5000.0)
    sphere_light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
    sphere_light.CreateRadiusAttr(0.5)
    xf = UsdGeom.Xformable(sphere_light.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(0, 0, 10))  # Above the arena

    print("[INFO] Lighting added: DomeLight + DistantLight + SphereLight")

    # Arena boundary markers (corners)
    for i, (cx, cy) in enumerate([(-5, -5), (5, -5), (5, 5), (-5, 5)]):
        marker = UsdGeom.Cylinder.Define(stage, f"/World/BoundaryMarker_{i}")
        xform = UsdGeom.Xformable(marker.GetPrim())
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(cx, cy, 0.1))
        xform.AddScaleOp().Set(Gf.Vec3d(0.1, 0.1, 0.1))
        marker.CreateDisplayColorAttr([(0.8, 0.8, 0.2)])

    # Start marker (green)
    cfg = ArenaConfig()
    start_marker = UsdGeom.Sphere.Define(stage, "/World/StartMarker")
    xf = UsdGeom.Xformable(start_marker.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*cfg.start_pos))
    xf.AddScaleOp().Set(Gf.Vec3d(0.2, 0.2, 0.2))
    start_marker.CreateDisplayColorAttr([(0.2, 0.9, 0.2)])

    # Goal marker (red)
    goal_marker = UsdGeom.Sphere.Define(stage, "/World/GoalMarker")
    xf = UsdGeom.Xformable(goal_marker.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*cfg.goal_pos))
    xf.AddScaleOp().Set(Gf.Vec3d(0.2, 0.2, 0.2))
    goal_marker.CreateDisplayColorAttr([(0.9, 0.2, 0.2)])

    # Camera
    camera_path = "/World/ObserverCamera"
    camera = UsdGeom.Camera.Define(stage, camera_path)
    xf = UsdGeom.Xformable(camera.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(0, -20, 15))
    xf.AddRotateXYZOp().Set(Gf.Vec3d(60, 0, 0))

    # Reset simulation
    sim_context.reset()

    # =========================================================================
    # Step 5: Initialize Generator and Spawner
    # =========================================================================
    print("[INFO] Initializing generator...")

    generator = UniversalArenaGenerator(cfg, seed=42)
    spawner = ArenaSpawner(stage, base_path="/World/Arena")
    debug_draw = _debug_draw.acquire_debug_draw_interface()

    # =========================================================================
    # Step 6: State Variables
    # =========================================================================
    mode_names = {
        ArenaMode.LATTICE_FOREST: "A: 3D Lattice Forest",
        ArenaMode.ANT_NEST: "B: 3D Ant Nest (Multi-Story Maze)",
        ArenaMode.RESTRICTED_CHANNEL: "C: Restricted Channels",
        ArenaMode.LETHAL_SANDWICH: "D: Lethal Sandwich",
        ArenaMode.SHOOTING_GALLERY: "E: Shooting Gallery (Dynamic)",
    }

    sub_mode_names = {
        ArenaMode.RESTRICTED_CHANNEL: ["horizontal", "vertical", "sloped"],
        ArenaMode.LETHAL_SANDWICH: ["cave", "hazards"],
    }

    current_mode = ArenaMode.LATTICE_FOREST
    current_sub_mode = None
    current_difficulty = 0.5
    current_result = None
    apply_gravity_tilt = True
    paused = False
    sim_time = 0.0
    show_gravity_vector = True

    # =========================================================================
    # Step 7: Keyboard Input Setup
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
    # Step 8: Helper Functions
    # =========================================================================
    def update_endpoint_markers(result):
        """Update start/goal marker positions based on arena result."""
        # Use LOCAL coordinates for display (arena frame)
        # This shows where the drone should start/end relative to the arena
        if result.labels.local_start:
            start_pos = result.labels.local_start
        else:
            start_pos = cfg.start_pos

        if result.labels.local_goal:
            goal_pos = result.labels.local_goal
        else:
            goal_pos = cfg.goal_pos

        # Update start marker (green)
        start_prim = stage.GetPrimAtPath("/World/StartMarker")
        if start_prim:
            xf = UsdGeom.Xformable(start_prim)
            xf.ClearXformOpOrder()
            xf.AddTranslateOp().Set(Gf.Vec3d(*start_pos))
            xf.AddScaleOp().Set(Gf.Vec3d(0.2, 0.2, 0.2))

        # Update goal marker (red)
        goal_prim = stage.GetPrimAtPath("/World/GoalMarker")
        if goal_prim:
            xf = UsdGeom.Xformable(goal_prim)
            xf.ClearXformOpOrder()
            xf.AddTranslateOp().Set(Gf.Vec3d(*goal_pos))
            xf.AddScaleOp().Set(Gf.Vec3d(0.2, 0.2, 0.2))

    def regenerate_arena():
        nonlocal current_result, sim_time

        print(f"\n{'='*50}")
        print(f"Generating: {mode_names[current_mode]}")
        print(f"Difficulty: {current_difficulty:.2f}")
        if current_sub_mode:
            print(f"Sub-mode: {current_sub_mode}")
        print(
            f"Gravity Tilt: {'Enabled' if apply_gravity_tilt else 'Disabled'}")
        print(f"{'='*50}")

        # Temporarily disable gravity tilt if requested
        original_max_tilt = cfg.max_tilt_angle
        if not apply_gravity_tilt:
            cfg.max_tilt_angle = 0.0

        generator.cfg = cfg

        current_result = generator.reset(
            mode=current_mode,
            difficulty=current_difficulty,
            sub_mode=current_sub_mode,
        )

        cfg.max_tilt_angle = original_max_tilt

        # Spawn obstacles
        spawner.spawn(current_result)

        # Update start/goal markers to reflect actual positions (NEW)
        update_endpoint_markers(current_result)

        # Print stats
        result = current_result
        print(f"Obstacles: {len(result.obstacles)}")
        print(f"Solvable: {result.solvable}")
        print(f"Complexity: {result.complexity:.3f}")
        print(
            f"Gravity Tilt: roll={result.gravity_tilt_euler[0]:.1f}°, pitch={result.gravity_tilt_euler[1]:.1f}°")

        if result.gaps:
            gap = result.gaps[0]
            print(
                f"Gap: center={gap.center}, size={gap.size}, orient={gap.orientation}")

        # Print coordinate info for vertical shaft mode
        if current_mode == ArenaMode.RESTRICTED_CHANNEL and current_sub_mode == "vertical":
            print(f"[Vertical Shaft] Local Start: {result.labels.local_start}")
            print(f"[Vertical Shaft] Local Goal: {result.labels.local_goal}")

        # Reset animation
        sim_time = 0.0
        draw_gravity_vector()

    def draw_gravity_vector():
        debug_draw.clear_lines()

        if not show_gravity_vector or current_result is None:
            return

        # Compute gravity direction from tilt
        q = current_result.gravity_tilt_quat
        w, x, y, z = q

        # Standard gravity = (0, 0, -9.81), Rotate by quaternion
        gx = -2 * (x * z - w * y)
        gy = -2 * (y * z + w * x)
        gz = -(1 - 2 * (x * x + y * y))

        # Normalize
        mag = math.sqrt(gx * gx + gy * gy + gz * gz)
        if mag > 0:
            gx, gy, gz = gx / mag, gy / mag, gz / mag

        # Draw from origin
        origin = (0, 0, 3.5)
        endpoint = (origin[0] + gx * 2, origin[1] + gy * 2, origin[2] + gz * 2)

        # Yellow-orange color for gravity
        debug_draw.draw_lines([origin], [endpoint], [
                              (1.0, 0.7, 0.0, 1.0)], [5])

        # Arrow head
        arrow_size = 0.3
        perp1 = (-gy, gx, 0)
        head1 = (
            endpoint[0] - gx * arrow_size + perp1[0] * arrow_size,
            endpoint[1] - gy * arrow_size + perp1[1] * arrow_size,
            endpoint[2] - gz * arrow_size,
        )
        head2 = (
            endpoint[0] - gx * arrow_size - perp1[0] * arrow_size,
            endpoint[1] - gy * arrow_size - perp1[1] * arrow_size,
            endpoint[2] - gz * arrow_size,
        )
        debug_draw.draw_lines([endpoint, endpoint], [head1, head2],
                              [(1.0, 0.7, 0.0, 1.0), (1.0, 0.7, 0.0, 1.0)], [3, 3])

    def print_stats():
        if current_result is None:
            print("No arena generated")
            return

        r = current_result
        print("\n" + "="*60)
        print("ARENA STATISTICS")
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

        # Coordinate System Info (NEW)
        print(f"\nCoordinate System:")
        if r.labels.local_start:
            print(
                f"  Local Start (Arena Frame):  {tuple(f'{v:.2f}' for v in r.labels.local_start)}")
        if r.labels.local_goal:
            print(
                f"  Local Goal (Arena Frame):   {tuple(f'{v:.2f}' for v in r.labels.local_goal)}")
        if r.labels.transformed_start:
            print(
                f"  World Start (After Tilt):   {tuple(f'{v:.2f}' for v in r.labels.transformed_start)}")
        if r.labels.transformed_goal:
            print(
                f"  World Goal (After Tilt):    {tuple(f'{v:.2f}' for v in r.labels.transformed_goal)}")

        print(f"\nCurriculum Labels:")
        print(f"  - Target Velocity: {r.labels.target_velocity}")
        print(f"  - Nearest Obstacle: {r.labels.nearest_obstacle_dist:.2f}m")
        print(
            f"  - Safe Corridor Radius: {r.labels.safe_flight_corridor_radius:.2f}m")
        print(
            f"  - Vertical Flight Required: {r.labels.requires_vertical_flight}")
        if r.gaps:
            print(f"Gap Info:")
            for i, gap in enumerate(r.gaps):
                print(
                    f"  Gap {i}: center={gap.center}, size={gap.size}, orient={gap.orientation}")
        print("="*60 + "\n")

    def print_hazard_details():
        """Print detailed hazard information (for Mode D: Lethal Sandwich)."""
        if current_result is None:
            print("No arena generated")
            return

        hazards = [o for o in current_result.obstacles if o.is_hazard]
        if not hazards:
            print(
                "\n[INFO] No hazards in current arena (try Mode D with sub-mode 2: hazards)")
            return

        print("\n" + "="*60)
        print("HAZARD DETAILS (Mode D: Lethal Sandwich)")
        print("="*60)

        # Categorize hazards
        thin_wires = [h for h in hazards if h.scale[0] <= 0.015]
        thick_cables = [h for h in hazards if h.scale[0] > 0.015]

        print(f"Total hazards: {len(hazards)}")
        print(f"  - Thin wires (r ≤ 0.015m): {len(thin_wires)}")
        print(f"  - Thicker cables (r > 0.015m): {len(thick_cables)}")

        if thin_wires:
            sample = thin_wires[0]
            print(f"\nSample thin wire:")
            print(f"  Position: {tuple(f'{v:.2f}' for v in sample.position)}")
            print(f"  Radius: {sample.scale[0]:.4f}m")
            print(f"  Length: {sample.scale[2]*2:.2f}m")
            print(
                f"  Color (RGB): ({sample.color[0]:.2f}, {sample.color[1]:.2f}, {sample.color[2]:.2f})")

        if thick_cables:
            sample = thick_cables[0]
            print(f"\nSample thick cable:")
            print(f"  Position: {tuple(f'{v:.2f}' for v in sample.position)}")
            print(f"  Radius: {sample.scale[0]:.4f}m")
            print(f"  Length: {sample.scale[2]*2:.2f}m")
            print(
                f"  Color (RGB): ({sample.color[0]:.2f}, {sample.color[1]:.2f}, {sample.color[2]:.2f})")

        # Color brightness check
        bright_colors = all(
            h.color[0] >= 0.8 or h.color[1] >= 0.8 for h in hazards)
        print(f"\nAll hazards have bright visible colors: {bright_colors}")
        print("="*60 + "\n")

    # =========================================================================
    # Step 9: Initial Generation
    # =========================================================================
    regenerate_arena()

    # =========================================================================
    # Step 10: Main Loop
    # =========================================================================
    print("\n" + "="*70)
    print("🎬 VISUALIZATION RUNNING (Updated Universal Generator)")
    print("="*70)
    print("Controls:")
    print("  A-E    : Switch modes (A=Lattice, B=Ant Nest, C=Channel, D=Sandwich, E=Shooting)")
    print("  1-3    : Switch sub-modes (Mode C: 1=Horiz, 2=Vert, 3=Sloped; Mode D: 1=Cave, 2=Hazards)")
    print("  R      : Regenerate arena")
    print("  +/-    : Increase/decrease difficulty")
    print("  G      : Toggle gravity tilt")
    print("  T      : Toggle gravity vector display")
    print("  SPACE  : Pause/resume dynamic obstacles")
    print("  S      : Print detailed statistics (includes coordinate info)")
    print("  H      : Print hazard details (Mode D)")
    print("  Ctrl+C : Exit")
    print("="*70 + "\n")

    dt = 0.02
    step = 0

    try:
        while simulation_app.is_running():
            # -----------------------------------------------------------------
            # Handle keyboard input
            # -----------------------------------------------------------------
            mode_changed = False

            # Mode keys A-E
            if key_pressed.get(carb.input.KeyboardInput.A, False):
                current_mode = ArenaMode.LATTICE_FOREST
                current_sub_mode = None
                mode_changed = True
                key_pressed[carb.input.KeyboardInput.A] = False
            elif key_pressed.get(carb.input.KeyboardInput.B, False):
                current_mode = ArenaMode.ANT_NEST
                current_sub_mode = None
                mode_changed = True
                key_pressed[carb.input.KeyboardInput.B] = False
            elif key_pressed.get(carb.input.KeyboardInput.C, False):
                current_mode = ArenaMode.RESTRICTED_CHANNEL
                current_sub_mode = None
                mode_changed = True
                key_pressed[carb.input.KeyboardInput.C] = False
            elif key_pressed.get(carb.input.KeyboardInput.D, False):
                current_mode = ArenaMode.LETHAL_SANDWICH
                current_sub_mode = None
                mode_changed = True
                key_pressed[carb.input.KeyboardInput.D] = False
            elif key_pressed.get(carb.input.KeyboardInput.E, False):
                current_mode = ArenaMode.SHOOTING_GALLERY
                current_sub_mode = None
                mode_changed = True
                key_pressed[carb.input.KeyboardInput.E] = False

            # Sub-mode keys 1-3
            if key_pressed.get(carb.input.KeyboardInput.KEY_1, False):
                if current_mode in sub_mode_names:
                    subs = sub_mode_names[current_mode]
                    if len(subs) >= 1:
                        current_sub_mode = subs[0]
                        mode_changed = True
                key_pressed[carb.input.KeyboardInput.KEY_1] = False
            elif key_pressed.get(carb.input.KeyboardInput.KEY_2, False):
                if current_mode in sub_mode_names:
                    subs = sub_mode_names[current_mode]
                    if len(subs) >= 2:
                        current_sub_mode = subs[1]
                        mode_changed = True
                key_pressed[carb.input.KeyboardInput.KEY_2] = False
            elif key_pressed.get(carb.input.KeyboardInput.KEY_3, False):
                if current_mode in sub_mode_names:
                    subs = sub_mode_names[current_mode]
                    if len(subs) >= 3:
                        current_sub_mode = subs[2]
                        mode_changed = True
                key_pressed[carb.input.KeyboardInput.KEY_3] = False

            # R to regenerate
            if key_pressed.get(carb.input.KeyboardInput.R, False):
                mode_changed = True
                key_pressed[carb.input.KeyboardInput.R] = False

            # +/- for difficulty
            if key_pressed.get(carb.input.KeyboardInput.NUMPAD_ADD, False) or \
               key_pressed.get(carb.input.KeyboardInput.EQUAL, False):
                current_difficulty = min(1.0, current_difficulty + 0.1)
                mode_changed = True
                key_pressed[carb.input.KeyboardInput.NUMPAD_ADD] = False
                key_pressed[carb.input.KeyboardInput.EQUAL] = False

            if key_pressed.get(carb.input.KeyboardInput.NUMPAD_SUBTRACT, False) or \
               key_pressed.get(carb.input.KeyboardInput.MINUS, False):
                current_difficulty = max(0.0, current_difficulty - 0.1)
                mode_changed = True
                key_pressed[carb.input.KeyboardInput.NUMPAD_SUBTRACT] = False
                key_pressed[carb.input.KeyboardInput.MINUS] = False

            # G for gravity tilt toggle
            if key_pressed.get(carb.input.KeyboardInput.G, False):
                apply_gravity_tilt = not apply_gravity_tilt
                print(
                    f"Gravity tilt: {'Enabled' if apply_gravity_tilt else 'Disabled'}")
                mode_changed = True
                key_pressed[carb.input.KeyboardInput.G] = False

            # T for gravity vector toggle
            if key_pressed.get(carb.input.KeyboardInput.T, False):
                show_gravity_vector = not show_gravity_vector
                print(
                    f"Gravity vector: {'Visible' if show_gravity_vector else 'Hidden'}")
                draw_gravity_vector()
                key_pressed[carb.input.KeyboardInput.T] = False

            # Space for pause/resume
            if key_pressed.get(carb.input.KeyboardInput.SPACE, False):
                paused = not paused
                print(f"Animation: {'Paused' if paused else 'Running'}")
                key_pressed[carb.input.KeyboardInput.SPACE] = False

            # S for stats
            if key_pressed.get(carb.input.KeyboardInput.S, False):
                print_stats()
                key_pressed[carb.input.KeyboardInput.S] = False

            # H for hazard details (NEW)
            if key_pressed.get(carb.input.KeyboardInput.H, False):
                print_hazard_details()
                key_pressed[carb.input.KeyboardInput.H] = False

            # -----------------------------------------------------------------
            # Regenerate if mode changed
            # -----------------------------------------------------------------
            if mode_changed:
                regenerate_arena()

            # -----------------------------------------------------------------
            # Update dynamic obstacles (Mode E)
            # -----------------------------------------------------------------
            if not paused and current_result and current_mode == ArenaMode.SHOOTING_GALLERY:
                sim_time += dt
                new_positions = generator.update_dynamic_obstacles(dt)
                if new_positions:
                    spawner.update_positions(new_positions)

            # -----------------------------------------------------------------
            # Physics step
            # -----------------------------------------------------------------
            sim_context.step()

            # -----------------------------------------------------------------
            # Console status (every 5 seconds)
            # -----------------------------------------------------------------
            if step % 250 == 0:
                name = mode_names[current_mode]
                obs_count = len(
                    current_result.obstacles) if current_result else 0
                print(
                    f"  [{name}] Diff: {current_difficulty:.1f} | Obstacles: {obs_count}")

            step += 1

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user (Ctrl+C)")

    # =========================================================================
    # Cleanup
    # =========================================================================
    input_interface.unsubscribe_to_keyboard_events(keyboard, keyboard_sub)
    debug_draw.clear_lines()
    spawner.clear()
    simulation_app.close()

    print("\n" + "="*70)
    print("✅ Visualization finished!")
    print("="*70)


if __name__ == "__main__":
    main()
