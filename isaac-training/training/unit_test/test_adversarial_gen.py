import os
import math
import torch
import importlib.util
import sys


def load_module_from_path(path, name="command_generator"):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


HERE = os.path.dirname(__file__)
TRAINING_ROOT = os.path.abspath(os.path.join(HERE, ".."))
MODULE_PATH = os.path.join(TRAINING_ROOT, "scripts", "command_generator.py")


def test_persistence_and_timers():
    mod = load_module_from_path(MODULE_PATH)
    AdversarialCommandGenerator = mod.AdversarialCommandGenerator

    device = torch.device("cpu")
    num_envs = 8
    dt = 0.01
    max_vel = 2.0
    gen = AdversarialCommandGenerator(num_envs, device, max_vel, dt)

    # Prepare dummy inputs
    drone_pos = torch.zeros(num_envs, 3)
    drone_vel = torch.zeros(num_envs, 3)
    # random obstacle vectors (non-zero)
    obstacle_relative_vectors = (torch.rand(num_envs, 3) - 0.5) * 10.0

    # Force probabilities to pick only Mode 1 (Aggressive Step) so we can expect persistence
    probs_mode1 = [0.0, 1.0, 0.0, 0.0, 0.0]

    # First update: should set timers in [1,4] and produce a command
    out1 = gen.update_commands(
        drone_pos, drone_vel, obstacle_relative_vectors, probabilities=probs_mode1)
    timers_before = gen.timers.clone()
    modes_before = gen.current_modes.clone()

    # Step multiple small dt frames, commands should persist (mode 1 not oscillatory)
    steps = 10
    outs = [out1]
    for _ in range(steps):
        outn = gen.update_commands(
            drone_pos, drone_vel, obstacle_relative_vectors, probabilities=probs_mode1)
        outs.append(outn)

    # Check timers decremented
    assert torch.all(gen.timers <= timers_before)

    # For all steps before timer reaches zero, commands should remain equal (within float tol)
    # Find first index where timer <= 0
    # Since we used small number of steps < 1s and durations >=1s, they should all persist
    for i in range(1, len(outs)):
        assert torch.allclose(
            outs[0], outs[i], atol=1e-6), "Commands changed during persistence window"


def test_mode2_suicide_alignment():
    mod = load_module_from_path(MODULE_PATH)
    AdversarialCommandGenerator = mod.AdversarialCommandGenerator

    device = torch.device("cpu")
    num_envs = 16
    dt = 0.05
    max_vel = 3.0
    gen = AdversarialCommandGenerator(num_envs, device, max_vel, dt)

    drone_pos = torch.zeros(num_envs, 3)
    drone_vel = torch.zeros(num_envs, 3)

    # Create obstacle vectors pointing in +/- x directions for half/half
    obs = torch.zeros(num_envs, 3)
    obs[: num_envs // 2, 0] = 5.0  # obstacle ahead in +x
    obs[num_envs // 2:, 0] = -5.0  # obstacle behind in -x

    # Force sampling Mode 2 for all envs
    probs_mode2 = [0.0, 0.0, 1.0, 0.0, 0.0]
    out = gen.update_commands(drone_pos, drone_vel,
                              obs, probabilities=probs_mode2)

    # For mode 2 outputs, cosine similarity with obstacle vectors should be positive (dot>0)
    modes = gen.current_modes
    mode2_mask = modes == 2
    assert mode2_mask.any(), "No env sampled into mode 2"

    out2 = out[mode2_mask]
    obs2 = obs[mode2_mask]
    dot = (out2 * obs2).sum(dim=1)
    # Since out2 is pointing toward obstacle, dot product should be > 0 (cosine positive)
    assert torch.all(
        dot > 0), f"Mode2 outputs not pointing toward obstacles: dot={dot}"


if __name__ == "__main__":
    test_persistence_and_timers()
    test_mode2_suicide_alignment()
    print("All tests passed")


# =============================================================================
# Interactive Visual Demo (Isaac Sim GUI)
# =============================================================================
# Run with: python unit_test/test_adversarial_gen.py --demo
# Or:       python unit_test/test_adversarial_gen.py --demo headless=False

def run_visual_demo():
    """
    Interactive visual demo of AdversarialCommandGenerator in Isaac Sim.

    Spawns a drone and obstacle, then visualizes:
    - 🔴 RED Arrow: The adversarial command vector
    - 🔵 CYAN Line: Vector to obstacle (danger direction)
    - 🟢 GREEN Arrow: Actual drone velocity
    - Console: Mode name and timer
    """
    import hydra
    from omegaconf import DictConfig
    from omni.isaac.kit import SimulationApp

    # =========================================================================
    # Step 1: Launch Isaac Sim
    # =========================================================================
    print("=" * 70)
    print("🎮 AdversarialCommandGenerator - Interactive Visual Demo")
    print("=" * 70)
    print("[INFO] Launching Isaac Sim...")

    # Check for headless arg
    headless = False
    for arg in sys.argv:
        if "headless=True" in arg:
            headless = True
            break

    sim_app = SimulationApp({"headless": headless, "anti_aliasing": 1})

    # =========================================================================
    # Step 2: Import dependencies (after SimulationApp)
    # =========================================================================
    print("[INFO] Importing dependencies...")

    import omni.isaac.orbit.sim as sim_utils
    from omni.isaac.core.simulation_context import SimulationContext
    from omni.isaac.core.prims import RigidPrimView
    from omni.isaac.orbit.assets import AssetBaseCfg
    from omni_drones.robots.drone import MultirotorBase
    from omni_drones.controllers import LeePositionController
    from omni.isaac.debug_draw import _debug_draw
    import omni.isaac.core.utils.prims as prim_utils
    from pxr import UsdGeom, Gf

    # Load our AdversarialCommandGenerator
    mod = load_module_from_path(MODULE_PATH)
    AdversarialCommandGenerator = mod.AdversarialCommandGenerator

    # =========================================================================
    # Step 3: Create Simulation Context
    # =========================================================================
    print("[INFO] Creating simulation context...")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dt = 0.02  # 50 Hz physics

    sim_context = SimulationContext(
        stage_units_in_meters=1.0,
        physics_dt=dt,
        rendering_dt=dt,
        backend="torch",
        physics_prim_path="/physicsScene",
        device=device,
    )

    # =========================================================================
    # Step 4: Create Scene (Ground, Light, Obstacle Wall, Drone)
    # =========================================================================
    print("[INFO] Creating scene...")

    # Ground plane
    ground_cfg = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(50.0, 50.0)),
    )
    ground_cfg.spawn.func(ground_cfg.prim_path, ground_cfg.spawn)

    # Light
    light_cfg = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(
            color=(0.9, 0.9, 0.9), intensity=2500.0),
    )
    light_cfg.spawn.func(light_cfg.prim_path, light_cfg.spawn)

    # Create environment prim
    if not prim_utils.is_prim_path_valid("/World/envs/env_0"):
        prim_utils.define_prim("/World/envs/env_0")

    # Obstacle Wall (Red Cube) - 3 meters in front of drone spawn
    wall_pos = (3.0, 0.0, 1.0)  # x=3m ahead, centered, 1m high
    wall_size = (0.3, 2.0, 2.0)  # thin, wide, tall wall

    stage = sim_context.stage
    wall_path = "/World/envs/env_0/ObstacleWall"

    # Create cube with proper Xform setup
    from pxr import UsdPhysics
    wall_prim = UsdGeom.Cube.Define(stage, wall_path)

    # Add XformOp for transform operations (required before setting values)
    xform = UsdGeom.Xformable(wall_prim.GetPrim())
    xform.ClearXformOpOrder()
    translate_op = xform.AddTranslateOp()
    scale_op = xform.AddScaleOp()

    # Set position and scale
    translate_op.Set(Gf.Vec3d(*wall_pos))
    scale_op.Set(Gf.Vec3d(*wall_size))

    # Make it red
    wall_prim.GetDisplayColorAttr().Set([(0.9, 0.2, 0.2)])

    print(f"  ├── Obstacle Wall at: {wall_pos}")

    # Spawn Drone
    drone_spawn_pos = (0.0, 0.0, 1.5)
    model_name = "Hummingbird"
    drone_model = MultirotorBase.REGISTRY[model_name]
    drone_cfg = drone_model.cfg_cls(force_sensor=False)
    drone = drone_model(cfg=drone_cfg)
    drone.spawn(translations=[drone_spawn_pos])

    print(f"  ├── Drone spawned at: {drone_spawn_pos}")

    # Reset simulation
    sim_context.reset()
    drone.initialize()

    # =========================================================================
    # Step 5: Initialize Controller and Command Generator
    # =========================================================================
    print("[INFO] Initializing controllers...")

    # Position controller for drone
    controller = LeePositionController(
        g=9.81, uav_params=drone.params).to(device)

    # Our adversarial command generator (1 env for demo)
    num_envs = 1
    max_vel = 3.0
    generator = AdversarialCommandGenerator(num_envs, device, max_vel, dt)

    # Debug draw interface
    debug_draw = _debug_draw.acquire_debug_draw_interface()

    # =========================================================================
    # Step 6: Mode Name Mapping
    # =========================================================================
    MODE_NAMES = {
        0: ("🚀 NORMAL NAV", (0.2, 0.8, 0.2)),      # Green
        1: ("⚡ AGGRESSIVE STEP", (1.0, 0.6, 0.0)),  # Orange
        2: ("🛑 SUICIDE ATTACK", (1.0, 0.0, 0.0)),   # Red
        3: ("〰️ OSCILLATION", (0.8, 0.0, 0.8)),      # Purple
        4: ("🧘 RECOVERY HOVER", (0.0, 0.8, 0.8)),   # Cyan
    }

    # =========================================================================
    # Step 7: Main Simulation Loop
    # =========================================================================
    print("\n" + "=" * 70)
    print("🎬 DEMO RUNNING - Watch the Isaac Sim viewport!")
    print("=" * 70)
    print("  Legend:")
    print("    🔴 RED Arrow    = Adversarial Command (what generator wants)")
    print("    🔵 CYAN Line    = Vector to Obstacle (danger direction)")
    print("    🟢 GREEN Arrow  = Actual Drone Velocity")
    print("-" * 70)
    print("  Press Ctrl+C or close the Isaac Sim window to exit")
    print("=" * 70 + "\n")

    step = 0
    time_elapsed = 0.0
    arrow_scale = 0.5  # Scale factor for velocity arrows (m/s -> meters)

    # Target position for hover (will be modified by adversarial commands)
    base_target = torch.tensor(
        [0.0, 0.0, 1.5], device=device, dtype=torch.float32)
    current_target = base_target.clone()

    try:
        while sim_app.is_running():
            time_elapsed += dt

            # -----------------------------------------------------------------
            # Get drone state
            # -----------------------------------------------------------------
            raw_state = drone.get_state()
            if raw_state.dim() == 3:
                drone_state = raw_state[0, 0, :13]
            elif raw_state.dim() == 2:
                drone_state = raw_state[0, :13]
            else:
                drone_state = raw_state[:13]

            drone_pos = drone_state[:3]  # (3,)
            drone_vel = drone_state[7:10]  # (3,)

            # -----------------------------------------------------------------
            # Compute obstacle relative vector (drone -> wall center)
            # -----------------------------------------------------------------
            wall_center = torch.tensor(
                wall_pos, device=device, dtype=torch.float32)
            obstacle_rel = (wall_center - drone_pos).unsqueeze(0)  # (1, 3)

            # -----------------------------------------------------------------
            # Update adversarial command generator
            # -----------------------------------------------------------------
            drone_pos_batch = drone_pos.unsqueeze(0)  # (1, 3)
            drone_vel_batch = drone_vel.unsqueeze(0)  # (1, 3)

            adv_cmd = generator.update_commands(
                drone_pos_batch, drone_vel_batch, obstacle_rel
            )  # (1, 3)

            current_mode = generator.current_modes[0].item()
            current_timer = generator.timers[0].item()
            mode_name, mode_color = MODE_NAMES.get(
                current_mode, ("UNKNOWN", (0.5, 0.5, 0.5)))

            # -----------------------------------------------------------------
            # Apply command to target (blend with base to keep drone in view)
            # -----------------------------------------------------------------
            # For demo: we move the target based on adversarial command
            # but clamp to keep drone visible
            cmd_3d = adv_cmd[0]  # (3,)

            # Blend: target = base + 0.3 * cmd (scaled for visualization)
            # current_target = base_target + 0.3 * cmd_3d
            current_target = base_target + cmd_3d
            current_target[2] = torch.clamp(
                current_target[2], 0.8, 3.0)  # height limit

            target_yaw = torch.tensor([0.0], device=device)

            # Compute control action
            action = controller(
                drone_state, target_pos=current_target, target_yaw=target_yaw)
            drone.apply_action(action)

            # -----------------------------------------------------------------
            # Physics step
            # -----------------------------------------------------------------
            sim_context.step()

            # -----------------------------------------------------------------
            # Debug Visualization
            # -----------------------------------------------------------------
            debug_draw.clear_lines()
            debug_draw.clear_points()

            # Convert to numpy for drawing
            p0 = drone_pos.cpu().numpy()

            # 🔴 RED Arrow: Adversarial Command
            cmd_np = cmd_3d.cpu().numpy()
            cmd_end = p0 + cmd_np * arrow_scale
            debug_draw.draw_lines(
                [tuple(p0)], [tuple(cmd_end)],
                [(1.0, 0.0, 0.0, 1.0)],  # Red
                [5.0]  # Line width
            )
            # Arrow head (small point at end)
            debug_draw.draw_points(
                [tuple(cmd_end)], [(1.0, 0.0, 0.0, 1.0)], [15.0])

            # 🔵 CYAN Line: Vector to Obstacle
            wall_np = wall_center.cpu().numpy()
            debug_draw.draw_lines(
                [tuple(p0)], [tuple(wall_np)],
                [(0.0, 1.0, 1.0, 1.0)],  # Cyan
                [2.0]
            )
            # Obstacle marker
            debug_draw.draw_points(
                [tuple(wall_np)], [(0.0, 1.0, 1.0, 1.0)], [20.0])

            # 🟢 GREEN Arrow: Actual Velocity
            vel_np = drone_vel.cpu().numpy()
            vel_end = p0 + vel_np * arrow_scale
            debug_draw.draw_lines(
                [tuple(p0)], [tuple(vel_end)],
                [(0.0, 1.0, 0.0, 1.0)],  # Green
                [4.0]
            )
            debug_draw.draw_points(
                [tuple(vel_end)], [(0.0, 1.0, 0.0, 1.0)], [12.0])

            # -----------------------------------------------------------------
            # Console Status (every 10 steps = 0.2s)
            # -----------------------------------------------------------------
            if step % 10 == 0:
                cmd_mag = float(torch.norm(cmd_3d).cpu())
                vel_mag = float(torch.norm(drone_vel).cpu())
                obs_dist = float(torch.norm(obstacle_rel).cpu())

                # Color code for terminal (ANSI)
                if current_mode == 2:  # Suicide
                    color_code = "\033[91m"  # Red
                elif current_mode == 4:  # Recovery
                    color_code = "\033[96m"  # Cyan
                elif current_mode == 1:  # Aggressive
                    color_code = "\033[93m"  # Yellow
                else:
                    color_code = "\033[92m"  # Green
                reset_code = "\033[0m"

                print(f"  t={time_elapsed:6.2f}s | {color_code}{mode_name:20s}{reset_code} | "
                      f"Timer: {current_timer:4.2f}s | Cmd: {cmd_mag:4.2f} m/s | "
                      f"Vel: {vel_mag:4.2f} m/s | ObsDist: {obs_dist:4.2f}m")

            step += 1

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user (Ctrl+C)")

    # Cleanup
    debug_draw.clear_lines()
    debug_draw.clear_points()
    sim_app.close()

    print("\n" + "=" * 70)
    print("✅ Demo finished!")
    print("=" * 70)


# =============================================================================
# Entry Point
# =============================================================================
if __name__ == "__main__" and "--demo" in sys.argv:
    run_visual_demo()
elif __name__ == "__main__":
    test_persistence_and_timers()
    test_mode2_suicide_alignment()
    print("All tests passed")
    print("\n[TIP] Run with --demo flag for interactive Isaac Sim visualization:")
    print("      python unit_test/test_adversarial_gen.py --demo")
