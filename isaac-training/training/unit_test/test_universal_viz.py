"""
Universal Arena Generator - Interactive Visualization Test
============================================================
Cycles through all 5 arena modes (A-E) every 5 seconds with debug visualization.

Run with:
    cd /home/mint/rl_dev/NavRL/isaac-training/training
    python unit_test/test_universal_viz.py

Controls:
    1-5: Switch to specific mode (A-E)
    R: Regenerate current mode
    +/-: Increase/decrease difficulty
    Space: Pause/resume cycling
    Ctrl+C: Exit
"""

import os
import sys
import time

# Add scripts path
HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_PATH = os.path.join(os.path.dirname(HERE), "scripts")
sys.path.insert(0, SCRIPTS_PATH)


def main():
    """Main visualization function"""

    # =========================================================================
    # Step 1: Launch Isaac Sim
    # =========================================================================
    print("=" * 70)
    print("🏗️  Universal Arena Generator - Visualization Demo")
    print("=" * 70)
    print("[INFO] Launching Isaac Sim...")

    from omni.isaac.kit import SimulationApp
    sim_app = SimulationApp({"headless": False, "anti_aliasing": 1})

    # =========================================================================
    # Step 2: Import dependencies (after SimulationApp)
    # =========================================================================
    print("[INFO] Importing dependencies...")

    import torch
    import numpy as np
    import omni.isaac.orbit.sim as sim_utils
    from omni.isaac.core.simulation_context import SimulationContext
    from omni.isaac.orbit.assets import AssetBaseCfg
    from omni.isaac.debug_draw import _debug_draw
    import omni.isaac.core.utils.prims as prim_utils
    from pxr import UsdGeom, Gf
    import carb.input
    import omni.appwindow

    # Import our generator
    from universal_generator import (
        UniversalArenaGenerator,
        ArenaSpawner,
        ArenaMode,
        ArenaResult,
    )

    print("[INFO] Dependencies loaded successfully")

    # =========================================================================
    # Step 3: Create Simulation Context
    # =========================================================================
    print("[INFO] Creating simulation context...")

    dt = 0.02  # 50 Hz
    sim_context = SimulationContext(
        stage_units_in_meters=1.0,
        physics_dt=dt,
        rendering_dt=dt,
        backend="torch",
        physics_prim_path="/physicsScene",
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )

    # =========================================================================
    # Step 4: Create Base Scene (Ground, Light)
    # =========================================================================
    print("[INFO] Creating base scene...")

    # Ground plane (larger for arena)
    ground_cfg = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(20.0, 20.0)),
    )
    ground_cfg.spawn.func(ground_cfg.prim_path, ground_cfg.spawn)

    # Light
    light_cfg = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(
            color=(1.0, 1.0, 1.0), intensity=3000.0),
    )
    light_cfg.spawn.func(light_cfg.prim_path, light_cfg.spawn)

    # Arena container
    if not prim_utils.is_prim_path_valid("/World/Arena"):
        prim_utils.define_prim("/World/Arena")

    # Start and Goal markers
    stage = sim_context.stage

    # Start marker (Green sphere)
    start_path = "/World/Markers/Start"
    prim_utils.define_prim("/World/Markers")
    start_prim = UsdGeom.Sphere.Define(stage, start_path)
    xform = UsdGeom.Xformable(start_prim.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(-4.0, 0.0, 1.5))
    xform.AddScaleOp().Set(Gf.Vec3d(0.3, 0.3, 0.3))
    start_prim.GetDisplayColorAttr().Set([(0.0, 1.0, 0.0)])

    # Goal marker (Blue sphere)
    goal_path = "/World/Markers/Goal"
    goal_prim = UsdGeom.Sphere.Define(stage, goal_path)
    xform = UsdGeom.Xformable(goal_prim.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(4.0, 0.0, 1.5))
    xform.AddScaleOp().Set(Gf.Vec3d(0.3, 0.3, 0.3))
    goal_prim.GetDisplayColorAttr().Set([(0.0, 0.5, 1.0)])

    # Reset simulation
    sim_context.reset()

    # =========================================================================
    # Step 5: Initialize Generator and Spawner
    # =========================================================================
    print("[INFO] Initializing generator...")

    generator = UniversalArenaGenerator(seed=42)
    spawner = ArenaSpawner(stage, base_path="/World/Arena")
    debug_draw = _debug_draw.acquire_debug_draw_interface()

    # =========================================================================
    # Step 6: Mode Configuration
    # =========================================================================
    modes = [
        ArenaMode.POISSON_FOREST,
        ArenaMode.MAZE_LOGIC,
        ArenaMode.CHEESE_SLICE,
        ArenaMode.SANDWICH,
        ArenaMode.DYNAMIC_GALLERY,
    ]

    mode_info = {
        ArenaMode.POISSON_FOREST: ("🌲 Mode A: POISSON FOREST", (0.3, 0.7, 0.3)),
        ArenaMode.MAZE_LOGIC: ("🧱 Mode B: MAZE LOGIC", (0.6, 0.6, 0.8)),
        ArenaMode.CHEESE_SLICE: ("🧀 Mode C: CHEESE SLICE", (0.9, 0.8, 0.4)),
        ArenaMode.SANDWICH: ("🥪 Mode D: SANDWICH", (0.6, 0.4, 0.3)),
        ArenaMode.DYNAMIC_GALLERY: ("🎯 Mode E: DYNAMIC GALLERY", (0.9, 0.3, 0.3)),
    }

    # State
    current_mode_idx = 0
    difficulty = 0.5
    density = 0.5
    auto_cycle = True
    last_switch_time = time.time()
    cycle_interval = 5.0  # seconds
    current_result: ArenaResult = None
    sim_time = 0.0

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
    def generate_and_spawn(mode: ArenaMode):
        nonlocal current_result
        print(
            f"\n[GENERATE] {mode_info[mode][0]} | Difficulty: {difficulty:.2f} | Density: {density:.2f}")

        current_result = generator.generate(
            mode, density=density, difficulty=difficulty)
        spawner.spawn(current_result)

        print(
            f"  ├── Obstacles: {current_result.metrics.get('obstacle_count', 0)}")
        print(
            f"  ├── Gap Min: {current_result.metrics.get('gap_min', 'N/A'):.2f}m")
        print(f"  └── Solvable: {'✓' if current_result.solvable else '✗'}")

    def draw_debug_visualization():
        """Draw debug lines and markers"""
        debug_draw.clear_lines()
        debug_draw.clear_points()

        if current_result is None:
            return

        # Draw suggested path
        path = current_result.suggested_path
        if len(path) >= 2:
            for i in range(len(path) - 1):
                p1 = path[i]
                p2 = path[i + 1]
                # Yellow path line
                debug_draw.draw_lines(
                    [tuple(p1)], [tuple(p2)],
                    [(1.0, 1.0, 0.0, 1.0)],  # Yellow
                    [4.0]
                )
            # Path waypoints
            for p in path:
                debug_draw.draw_points(
                    [tuple(p)], [(1.0, 1.0, 0.0, 1.0)], [10.0])

        # Draw arena boundary
        bounds = [
            ((-5, -5, 0), (5, -5, 0)),
            ((5, -5, 0), (5, 5, 0)),
            ((5, 5, 0), (-5, 5, 0)),
            ((-5, 5, 0), (-5, -5, 0)),
            # Vertical edges
            ((-5, -5, 0), (-5, -5, 4)),
            ((5, -5, 0), (5, -5, 4)),
            ((5, 5, 0), (5, 5, 4)),
            ((-5, 5, 0), (-5, 5, 4)),
            # Top edges
            ((-5, -5, 4), (5, -5, 4)),
            ((5, -5, 4), (5, 5, 4)),
            ((5, 5, 4), (-5, 5, 4)),
            ((-5, 5, 4), (-5, -5, 4)),
        ]
        for p1, p2 in bounds:
            debug_draw.draw_lines([p1], [p2], [(0.5, 0.5, 0.5, 0.5)], [1.0])

        # Start/Goal labels
        debug_draw.draw_points([(-4, 0, 2.5)], [(0.0, 1.0, 0.0, 1.0)], [15.0])
        debug_draw.draw_points([(4, 0, 2.5)], [(0.0, 0.5, 1.0, 1.0)], [15.0])

    # =========================================================================
    # Step 9: Initial Generation
    # =========================================================================
    generate_and_spawn(modes[current_mode_idx])

    # =========================================================================
    # Step 10: Main Loop
    # =========================================================================
    print("\n" + "=" * 70)
    print("🎬 VISUALIZATION RUNNING")
    print("=" * 70)
    print("Controls:")
    print("  1-5    : Switch to Mode A-E")
    print("  R      : Regenerate current mode")
    print("  +/-    : Increase/decrease difficulty")
    print("  D      : Toggle density (0.3 / 0.5 / 0.7)")
    print("  Space  : Pause/resume auto-cycling")
    print("  Ctrl+C : Exit")
    print("=" * 70 + "\n")

    step = 0

    try:
        while sim_app.is_running():
            current_time = time.time()
            sim_time += dt

            # -----------------------------------------------------------------
            # Handle keyboard input
            # -----------------------------------------------------------------
            mode_changed = False

            # Number keys 1-5 for mode selection
            if key_pressed.get(carb.input.KeyboardInput.KEY_1, False):
                current_mode_idx = 0
                mode_changed = True
                key_pressed[carb.input.KeyboardInput.KEY_1] = False
            elif key_pressed.get(carb.input.KeyboardInput.KEY_2, False):
                current_mode_idx = 1
                mode_changed = True
                key_pressed[carb.input.KeyboardInput.KEY_2] = False
            elif key_pressed.get(carb.input.KeyboardInput.KEY_3, False):
                current_mode_idx = 2
                mode_changed = True
                key_pressed[carb.input.KeyboardInput.KEY_3] = False
            elif key_pressed.get(carb.input.KeyboardInput.KEY_4, False):
                current_mode_idx = 3
                mode_changed = True
                key_pressed[carb.input.KeyboardInput.KEY_4] = False
            elif key_pressed.get(carb.input.KeyboardInput.KEY_5, False):
                current_mode_idx = 4
                mode_changed = True
                key_pressed[carb.input.KeyboardInput.KEY_5] = False

            # R to regenerate
            if key_pressed.get(carb.input.KeyboardInput.R, False):
                mode_changed = True
                key_pressed[carb.input.KeyboardInput.R] = False

            # +/- for difficulty
            if key_pressed.get(carb.input.KeyboardInput.NUMPAD_ADD, False) or \
               key_pressed.get(carb.input.KeyboardInput.EQUAL, False):
                difficulty = min(1.0, difficulty + 0.1)
                mode_changed = True
                key_pressed[carb.input.KeyboardInput.NUMPAD_ADD] = False
                key_pressed[carb.input.KeyboardInput.EQUAL] = False

            if key_pressed.get(carb.input.KeyboardInput.NUMPAD_SUBTRACT, False) or \
               key_pressed.get(carb.input.KeyboardInput.MINUS, False):
                difficulty = max(0.0, difficulty - 0.1)
                mode_changed = True
                key_pressed[carb.input.KeyboardInput.NUMPAD_SUBTRACT] = False
                key_pressed[carb.input.KeyboardInput.MINUS] = False

            # D for density toggle
            if key_pressed.get(carb.input.KeyboardInput.D, False):
                densities = [0.3, 0.5, 0.7]
                current_idx = densities.index(
                    density) if density in densities else 1
                density = densities[(current_idx + 1) % len(densities)]
                mode_changed = True
                key_pressed[carb.input.KeyboardInput.D] = False

            # Space for pause/resume
            if key_pressed.get(carb.input.KeyboardInput.SPACE, False):
                auto_cycle = not auto_cycle
                status = "RESUMED" if auto_cycle else "PAUSED"
                print(f"[INFO] Auto-cycling {status}")
                key_pressed[carb.input.KeyboardInput.SPACE] = False

            # -----------------------------------------------------------------
            # Auto-cycling
            # -----------------------------------------------------------------
            if auto_cycle and (current_time - last_switch_time) >= cycle_interval:
                current_mode_idx = (current_mode_idx + 1) % len(modes)
                mode_changed = True
                last_switch_time = current_time

            # -----------------------------------------------------------------
            # Regenerate if mode changed
            # -----------------------------------------------------------------
            if mode_changed:
                generate_and_spawn(modes[current_mode_idx])
                last_switch_time = current_time

            # -----------------------------------------------------------------
            # Update dynamic obstacles (Mode E)
            # -----------------------------------------------------------------
            if current_result and current_result.mode == ArenaMode.DYNAMIC_GALLERY:
                new_positions = generator.update_dynamic_obstacles(
                    current_result.obstacles, sim_time
                )
                spawner.update_positions(new_positions)

            # -----------------------------------------------------------------
            # Draw debug visualization
            # -----------------------------------------------------------------
            draw_debug_visualization()

            # -----------------------------------------------------------------
            # Physics step
            # -----------------------------------------------------------------
            sim_context.step()

            # -----------------------------------------------------------------
            # Console status (every 2 seconds)
            # -----------------------------------------------------------------
            if step % 100 == 0:
                mode = modes[current_mode_idx]
                name, _ = mode_info[mode]
                cycle_status = "AUTO" if auto_cycle else "MANUAL"
                time_to_next = cycle_interval - \
                    (current_time - last_switch_time) if auto_cycle else 0

                print(f"  [{cycle_status}] {name} | Diff: {difficulty:.1f} | Dens: {density:.1f} | "
                      f"Next: {time_to_next:.1f}s | Obstacles: {len(current_result.obstacles) if current_result else 0}")

            step += 1

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user (Ctrl+C)")

    # =========================================================================
    # Cleanup
    # =========================================================================
    input_interface.unsubscribe_to_keyboard_events(keyboard, keyboard_sub)
    debug_draw.clear_lines()
    debug_draw.clear_points()
    spawner.clear()
    sim_app.close()

    print("\n" + "=" * 70)
    print("✅ Visualization finished!")
    print("=" * 70)


if __name__ == "__main__":
    main()
