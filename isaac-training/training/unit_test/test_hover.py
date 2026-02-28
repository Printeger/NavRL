#!/usr/bin/env python3
"""
UAV Hover Test (Empty Scene)
===========================
Loads drone config from training/cfg/drone.yaml, spawns UAV at origin, and
holds hover with keyboard control for position and yaw.

Run:
    conda activate NavRL
    cd /home/mint/rl_dev/NavRL/isaac-training
    python training/unit_test/test_hover.py
    python training/unit_test/test_hover.py headless=False
    python training/unit_test/test_hover.py drone.model_name=Firefly

Controls:
    W/S     : +X / -X
    A/D     : +Y / -Y
    Q/E     : +Z / -Z
    Z/X     : Yaw + / Yaw -
    B       : Reset target + drone state
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
    print("UAV Hover Test (Empty Scene)")
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

    import math
    import torch
    import carb
    import carb.input
    import omni.appwindow
    import omni.isaac.core.utils.prims as prim_utils
    from omni.isaac.core.simulation_context import SimulationContext
    from omni.isaac.debug_draw import _debug_draw
    from omni_drones.robots.drone import MultirotorBase
    from omni_drones.controllers import LeePositionController
    from pxr import UsdGeom, Gf, UsdLux, UsdPhysics, Usd

    def _get_reference_profile(reference_usd_path: str):
        try:
            stage_ref = Usd.Stage.Open(reference_usd_path)
        except Exception:
            return None
        if stage_ref is None:
            return None

        root = stage_ref.GetDefaultPrim()
        if not root or not root.IsValid():
            for child in stage_ref.GetPseudoRoot().GetChildren():
                if child.IsValid() and child.GetTypeName() in ("Xform", ""):
                    root = child
                    break
        if root is None or not root.IsValid():
            return None

        root_path = root.GetPath().pathString

        articulation_count = 0
        rigid_count = 0
        collision_count = 0
        joint_count = 0
        for prim in stage_ref.Traverse():
            if not prim.IsValid():
                continue
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                articulation_count += 1
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rigid_count += 1
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                collision_count += 1
            if "Joint" in (prim.GetTypeName() or ""):
                joint_count += 1

        rotor_body1_targets = set()
        for prim in stage_ref.Traverse():
            if not prim.IsValid() or "Joint" not in (prim.GetTypeName() or ""):
                continue
            body1_rel = prim.GetRelationship("physics:body1")
            if not body1_rel:
                continue
            targets = body1_rel.GetTargets()
            if not targets:
                continue
            rotor_body1_targets.add(str(targets[0]))

        base_candidates = set()
        for prim in stage_ref.Traverse():
            if not prim.IsValid() or "Joint" not in (prim.GetTypeName() or ""):
                continue
            body0_rel = prim.GetRelationship("physics:body0")
            if not body0_rel:
                continue
            targets = body0_rel.GetTargets()
            if not targets:
                continue
            base_candidates.add(str(targets[0]))

        base_name = None
        if len(base_candidates) == 1:
            base_name = next(iter(base_candidates)).split("/")[-1]

        return {
            "root_path": root_path,
            "base_name": base_name,
            "rotor_count": len(rotor_body1_targets),
            "joint_count": joint_count,
            "rigid_count": rigid_count,
            "collision_count": collision_count,
            "articulation_count": articulation_count,
        }

    def _is_multirotor_asset_compatible(usd_path: str, reference_profile: dict) -> bool:
        if not reference_profile:
            return True

        try:
            stage_asset = Usd.Stage.Open(usd_path)
        except Exception:
            return False
        if stage_asset is None:
            return False

        articulation_count = 0
        rigid_count = 0
        collision_count = 0
        joint_count = 0
        for prim in stage_asset.Traverse():
            if not prim.IsValid():
                continue
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                articulation_count += 1
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rigid_count += 1
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                collision_count += 1
            if "Joint" in (prim.GetTypeName() or ""):
                joint_count += 1

        rotor_body1_targets = set()
        base_candidates = set()
        for prim in stage_asset.Traverse():
            if not prim.IsValid() or "Joint" not in (prim.GetTypeName() or ""):
                continue

            body0_rel = prim.GetRelationship("physics:body0")
            if body0_rel:
                targets = body0_rel.GetTargets()
                if targets:
                    base_candidates.add(str(targets[0]))

            body1_rel = prim.GetRelationship("physics:body1")
            if body1_rel:
                targets = body1_rel.GetTargets()
                if targets:
                    rotor_body1_targets.add(str(targets[0]))

        if articulation_count < 1:
            return False
        if len(rotor_body1_targets) < max(4, reference_profile.get("rotor_count", 4)):
            return False
        if joint_count < max(4, reference_profile.get("joint_count", 4)):
            return False

        ref_base_name = reference_profile.get("base_name")
        if ref_base_name:
            base_names = {path.split("/")[-1] for path in base_candidates}
            if ref_base_name not in base_names:
                return False

        # 计数层面对齐，避免命名硬编码
        if rigid_count < reference_profile.get("rigid_count", 0):
            return False
        if collision_count < reference_profile.get("collision_count", 0):
            return False

        return True

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
    ground.CreatePointsAttr(
        [(-20, -20, 0), (20, -20, 0), (20, 20, 0), (-20, 20, 0)]
    )
    ground.CreateFaceVertexCountsAttr([4])
    ground.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    ground.CreateDisplayColorAttr([(0.15, 0.15, 0.18)])

    # Lights
    dome_light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome_light.CreateIntensityAttr(800.0)
    dome_light.CreateColorAttr(Gf.Vec3f(0.9, 0.9, 1.0))

    distant_light = UsdLux.DistantLight.Define(stage, "/World/DistantLight")
    distant_light.CreateIntensityAttr(2000.0)
    distant_light.CreateColorAttr(Gf.Vec3f(1.0, 0.98, 0.95))
    distant_light.CreateAngleAttr(1.0)
    xf = UsdGeom.Xformable(distant_light.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddRotateXYZOp().Set(Gf.Vec3d(-45, 30, 0))

    # Environment prim
    if not prim_utils.is_prim_path_valid("/World/envs/env_0"):
        prim_utils.define_prim("/World/envs/env_0")

    # Drone
    model_name = cfg.drone.model_name
    if model_name not in MultirotorBase.REGISTRY:
        print(f"[WARNING] Model '{model_name}' not found, using 'Hummingbird'")
        model_name = "Hummingbird"

    drone_model = MultirotorBase.REGISTRY[model_name]

    reference_model = MultirotorBase.REGISTRY.get("Hummingbird", drone_model)
    reference_profile = _get_reference_profile(reference_model.usd_path)
    if reference_profile:
        print(
            "[INFO] Reference profile "
            f"(model={reference_model.__name__}): "
            f"base={reference_profile.get('base_name')}, "
            f"rotors={reference_profile.get('rotor_count')}, "
            f"joints={reference_profile.get('joint_count')}"
        )
    else:
        print("[WARNING] Failed to load reference profile, compatibility check will be permissive.")

    if not _is_multirotor_asset_compatible(drone_model.usd_path, reference_profile):
        print(
            f"[WARNING] Asset '{drone_model.usd_path}' is not compatible with "
            f"reference profile ({reference_model.__name__}). Falling back to Hummingbird."
        )
        model_name = "Hummingbird"
        drone_model = MultirotorBase.REGISTRY[model_name]

    drone_cfg = drone_model.cfg_cls(force_sensor=False)
    drone = drone_model(cfg=drone_cfg)

    init_pos = (0.0, 0.0, 1.5)
    drone.spawn(translations=[init_pos])

    root_path = f"/World/envs/env_0/{drone.name}_0"
    default_expr = f"/World/envs/.*/{drone.name}_*"
    init_expr = default_expr

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
            if chosen != root_path:
                suffix = chosen[len(root_path):]
                init_expr = f"{default_expr}{suffix}"
            print(f"[INFO] Using articulation prim path: {chosen}")
        else:
            print(f"[WARNING] No ArticulationRootAPI found under {root_path}, fallback to {default_expr}")

    sim_context.reset()
    drone.initialize(prim_paths_expr=init_expr)

    controller = LeePositionController(g=9.81, uav_params=drone.params).to(device)
    debug_draw = _debug_draw.acquire_debug_draw_interface()

    # Keyboard
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
        keyboard, on_keyboard_event
    )

    target_pos = torch.tensor(
        [init_pos[0], init_pos[1], init_pos[2]],
        device=device,
        dtype=torch.float32,
    )
    target_yaw = torch.tensor([0.0], device=device)

    move_speed = 0.05
    yaw_speed = 0.02

    try:
        while simulation_app.is_running():
            # Input
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
            if key_pressed.get(carb.input.KeyboardInput.Z, False):
                target_yaw[0] += yaw_speed
            if key_pressed.get(carb.input.KeyboardInput.X, False):
                target_yaw[0] -= yaw_speed

            if key_pressed.get(carb.input.KeyboardInput.B, False):
                target_pos[0] = init_pos[0]
                target_pos[1] = init_pos[1]
                target_pos[2] = init_pos[2]
                target_yaw[0] = 0.0
                drone.set_world_poses(
                    positions=torch.tensor([init_pos], device=device),
                    orientations=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device),
                )
                drone.set_velocities(torch.zeros((1, 6), device=device))
                key_pressed[carb.input.KeyboardInput.B] = False

            # Clamp height
            target_pos[2] = torch.clamp(target_pos[2], 0.3, 5.0)

            # Control
            raw_state = drone.get_state()
            if raw_state.dim() == 3:
                drone_state = raw_state[0, 0, :13]
            elif raw_state.dim() == 2:
                drone_state = raw_state[0, :13]
            else:
                drone_state = raw_state[:13]

            action = controller(
                drone_state,
                target_pos=target_pos,
                target_yaw=target_yaw,
            )
            drone.apply_action(action)

            # Draw target point
            debug_draw.clear_points()
            debug_draw.draw_points([tuple(target_pos.tolist())], [(0.0, 1.0, 0.0, 1.0)], [8.0])

            sim_context.step()

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user (Ctrl+C)")

    input_interface.unsubscribe_to_keyboard_events(keyboard, keyboard_sub)
    debug_draw.clear_points()
    simulation_app.close()


if __name__ == "__main__":
    main()
