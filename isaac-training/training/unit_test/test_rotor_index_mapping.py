#!/usr/bin/env python3
"""
Minimal rotor index mapping verification tool.

Purpose:
    Press 0/1/2/3 to spin only one rotor joint and verify
    index-to-physical-position correspondence.

Run:
    conda activate NavRL
    cd /home/mint/rl_dev/NavRL/isaac-training
    python training/unit_test/test_rotor_index_mapping.py
    python training/unit_test/test_rotor_index_mapping.py headless=False
    python training/unit_test/test_rotor_index_mapping.py drone.model_name=TaslabUAV

Controls:
    0/1/2/3 : Spin only rotor_{0..3}_joint
    Space   : Stop all rotors
    B       : Reset pose
    Ctrl+C  : Exit
"""

import os
import sys
import hydra
from omegaconf import DictConfig


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_ROOT = os.path.dirname(SCRIPT_DIR)
CFG_PATH = os.path.join(TRAINING_ROOT, "cfg")

if TRAINING_ROOT not in sys.path:
    sys.path.insert(0, TRAINING_ROOT)


@hydra.main(config_path=CFG_PATH, config_name="train", version_base=None)
def main(cfg: DictConfig):
    print("=" * 60)
    print("Rotor Index Mapping Test")
    print("=" * 60)
    print(f"[INFO] Headless mode: {cfg.headless}")
    print(f"[INFO] Device: {cfg.device}")
    print(f"[INFO] Drone model: {cfg.drone.model_name}")
    print("-" * 60)

    from omni.isaac.kit import SimulationApp
    simulation_app = SimulationApp({
        "headless": cfg.headless,
        "width": 1280,
        "height": 720,
        "anti_aliasing": 1,
    })

    import torch
    import carb
    import carb.input
    import omni.appwindow
    import omni.ui as ui
    import omni.isaac.core.utils.prims as prim_utils
    from omni.isaac.core.simulation_context import SimulationContext
    from omni_drones.robots.drone import MultirotorBase
    import omni_drones.utils.kit as kit_utils
    from pxr import UsdGeom, Gf, UsdLux, UsdPhysics

    def _try_key_enum(name: str):
        return getattr(carb.input.KeyboardInput, name, None)

    KEY_TO_INDEX = {
        _try_key_enum("KEY_0"): 0,
        _try_key_enum("NUMPAD_0"): 0,
        _try_key_enum("KEY_1"): 1,
        _try_key_enum("NUMPAD_1"): 1,
        _try_key_enum("KEY_2"): 2,
        _try_key_enum("NUMPAD_2"): 2,
        _try_key_enum("KEY_3"): 3,
        _try_key_enum("NUMPAD_3"): 3,
    }
    KEY_TO_INDEX = {k: v for k, v in KEY_TO_INDEX.items() if k is not None}

    def _fallback_digit_from_input(event_input) -> int:
        text = str(event_input)
        for digit in ("0", "1", "2", "3"):
            if f"KEY_{digit}" in text or f"NUMPAD_{digit}" in text:
                return int(digit)
        return -1

    sim_context = SimulationContext(
        stage_units_in_meters=1.0,
        physics_dt=0.02,
        rendering_dt=0.02,
        backend="torch",
        device=cfg.device,
    )
    stage = sim_context.stage
    device = cfg.device

    # Ground
    ground_path = "/World/GroundPlane"
    ground = UsdGeom.Mesh.Define(stage, ground_path)
    ground.CreatePointsAttr([(-20, -20, 0), (20, -20, 0), (20, 20, 0), (-20, 20, 0)])
    ground.CreateFaceVertexCountsAttr([4])
    ground.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    ground.CreateDisplayColorAttr([(0.15, 0.15, 0.18)])
    ground_prim = ground.GetPrim()
    if not ground_prim.HasAPI(UsdPhysics.CollisionAPI):
        UsdPhysics.CollisionAPI.Apply(ground_prim)
    if not ground_prim.HasAPI(UsdPhysics.MeshCollisionAPI):
        UsdPhysics.MeshCollisionAPI.Apply(ground_prim)

    # Light
    dome_light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome_light.CreateIntensityAttr(1000.0)
    dome_light.CreateColorAttr(Gf.Vec3f(0.9, 0.9, 1.0))

    # Environment prim
    if not prim_utils.is_prim_path_valid("/World/envs/env_0"):
        prim_utils.define_prim("/World/envs/env_0")

    # Drone model
    model_name = cfg.drone.model_name
    if model_name not in MultirotorBase.REGISTRY:
        print(f"[WARNING] Model '{model_name}' not found, using 'Hummingbird'")
        model_name = "Hummingbird"

    drone_model = MultirotorBase.REGISTRY[model_name]
    drone_cfg = drone_model.cfg_cls(force_sensor=False)
    drone = drone_model(cfg=drone_cfg)

    init_pos = (0.0, 0.0, 0.20)
    init_rot = (1.0, 0.0, 0.0, 0.0)
    drone.spawn(translations=[init_pos])

    root_path = f"/World/envs/env_0/{drone.name}_0"
    default_expr = f"/World/envs/.*/{drone.name}_*"
    init_expr = default_expr
    articulation_prim_path = root_path

    root_prim = stage.GetPrimAtPath(root_path)
    if root_prim.IsValid():
        articulation_paths = []
        for prim in stage.Traverse():
            path_str = prim.GetPath().pathString
            if path_str.startswith(root_path) and prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                articulation_paths.append(path_str)
        if articulation_paths:
            articulation_paths.sort(key=lambda p: p.count("/"), reverse=True)
            chosen = articulation_paths[0]
            articulation_prim_path = chosen
            if chosen != root_path:
                suffix = chosen[len(root_path):]
                init_expr = f"{default_expr}{suffix}"
            print(f"[INFO] Using articulation prim path: {chosen}")

    sim_context.reset()
    drone.initialize(prim_paths_expr=init_expr)

    try:
        kit_utils.set_articulation_properties(
            articulation_prim_path,
            sleep_threshold=0.0,
            stabilization_threshold=0.0,
        )
        print("[INFO] Anti-sleep enabled for articulation (sleep_threshold=0)")
    except Exception as exc:
        print(f"[WARNING] Failed to set anti-sleep articulation properties: {exc}")

    if getattr(drone, "rotor_joint_indices", None) is None:
        print("[ERROR] No rotor joints found. Check joint naming (expecting rotor*_joint style dof names).")
        simulation_app.close()
        return

    dof_names = list(drone._view._dof_names)
    rotor_indices = drone.rotor_joint_indices.detach().cpu().tolist()
    rotor_joint_names = [dof_names[i] for i in rotor_indices]
    rotor_directions = drone.directions.reshape(-1).detach().cpu().tolist()

    print("[INFO] Detected rotor joints:")
    print(f"       {rotor_joint_names}")
    print(f"       {dict(zip(rotor_joint_names, rotor_indices))}")
    print(f"[INFO] Rotor directions (from yaml): {rotor_directions}")
    print("[INFO] Controls: 0/1/2/3 latch-spin single rotor, Space stop, B reset")
    print("[INFO] Live monitor enabled (rad/s): CMD vs MEAS")

    # Keyboard
    appwindow = omni.appwindow.get_default_app_window()
    input_interface = carb.input.acquire_input_interface()
    keyboard = appwindow.get_keyboard()

    active_rotor = -1
    hud_window = None
    hud_label = None

    def _active_text() -> str:
        if 0 <= active_rotor < len(rotor_joint_names):
            sign = rotor_directions[active_rotor] if active_rotor < len(rotor_directions) else 1.0
            sign_text = "CCW(+1)" if sign > 0 else "CW(-1)"
            return f"Active Rotor: {active_rotor} ({rotor_joint_names[active_rotor]})  Dir={sign_text}"
        return "Active Rotor: STOP"

    if not cfg.headless:
        hud_window = ui.Window("Rotor Mapping HUD", width=500, height=90)
        hud_window.flags = ui.WINDOW_FLAGS_NO_RESIZE | ui.WINDOW_FLAGS_NO_COLLAPSE
        with hud_window.frame:
            with ui.VStack(spacing=6):
                ui.Label("Press 0/1/2/3 to select a rotor", height=24)
                hud_label = ui.Label(_active_text(), height=28)

    def _refresh_hud():
        if hud_label is not None:
            hud_label.text = _active_text()

    def on_keyboard_event(event):
        nonlocal active_rotor

        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input in KEY_TO_INDEX:
                idx = KEY_TO_INDEX[event.input]
                if idx < len(rotor_indices):
                    active_rotor = idx
                    print(f"[INFO] Active rotor index: {active_rotor} ({rotor_joint_names[active_rotor]})")
                    _refresh_hud()
                return True

            fallback_idx = _fallback_digit_from_input(event.input)
            if 0 <= fallback_idx < len(rotor_indices):
                active_rotor = fallback_idx
                print(f"[INFO] Active rotor index: {active_rotor} ({rotor_joint_names[active_rotor]})")
                _refresh_hud()
                return True

            if event.input == carb.input.KeyboardInput.SPACE:
                active_rotor = -1
                print("[INFO] Stop all rotors")
                _refresh_hud()
                return True

            if event.input == carb.input.KeyboardInput.B:
                drone.set_world_poses(
                    positions=torch.tensor([init_pos], device=device),
                    orientations=torch.tensor([init_rot], device=device),
                )
                drone.set_velocities(torch.zeros((1, 6), device=device))
                print("[INFO] Drone pose reset")
                _refresh_hud()
                return True

        return True

    keyboard_sub = input_interface.subscribe_to_keyboard_events(keyboard, on_keyboard_event)

    spin_speed = 120.0  # rad/s, low-speed persistent spin for mapping
    wake_pulse_yaw = 0.02  # rad/s, tiny alternating pulse to keep articulation awake
    step_count = 0
    monitor_initialized = False

    try:
        while simulation_app.is_running():
            # Keep base almost static; inject tiny alternating yaw pulse to avoid sleeping.
            pulse_sign = 1.0 if (step_count % 2 == 0) else -1.0
            wake_vel = torch.zeros((1, 6), device=device)
            wake_vel[0, 5] = wake_pulse_yaw * pulse_sign
            drone.set_velocities(wake_vel)

            # Only one rotor spins at a time
            rotor_vel = torch.zeros((1, len(rotor_indices)), device=device)
            if 0 <= active_rotor < len(rotor_indices):
                direction = rotor_directions[active_rotor] if active_rotor < len(rotor_directions) else 1.0
                rotor_vel[0, active_rotor] = spin_speed * float(direction)

            drone._view.set_joint_velocity_targets(
                rotor_vel,
                joint_indices=drone.rotor_joint_indices,
            )
            drone._view.set_joint_velocities(
                rotor_vel,
                joint_indices=drone.rotor_joint_indices,
            )

            joint_vel_all = drone._view.get_joint_velocities(clone=True)
            if joint_vel_all.dim() == 2:
                joint_vel_env0 = joint_vel_all[0]
            else:
                joint_vel_env0 = joint_vel_all.reshape(-1)
            rotor_vel_cmd = [float(v) for v in rotor_vel[0].detach().cpu().tolist()]
            rotor_vel_meas = [float(joint_vel_env0[idx].item()) for idx in rotor_indices]
            cmd_parts = [f"r{i}={val:8.2f}" for i, val in enumerate(rotor_vel_cmd)]
            meas_parts = [f"r{i}={val:8.2f}" for i, val in enumerate(rotor_vel_meas)]
            cmd_text = "[CMD ] " + "  ".join(cmd_parts)
            meas_text = "[MEAS] " + "  ".join(meas_parts)

            if not monitor_initialized:
                print(cmd_text)
                print(meas_text)
                monitor_initialized = True
            else:
                sys.stdout.write("\x1b[2F")
                sys.stdout.write("\x1b[2K" + cmd_text + "\n")
                sys.stdout.write("\x1b[2K" + meas_text + "\n")
                sys.stdout.flush()

            sim_context.step()
            step_count += 1

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user (Ctrl+C)")

    input_interface.unsubscribe_to_keyboard_events(keyboard, keyboard_sub)
    if hud_window is not None:
        hud_window.visible = False
    if monitor_initialized:
        print()
    simulation_app.close()


if __name__ == "__main__":
    main()
