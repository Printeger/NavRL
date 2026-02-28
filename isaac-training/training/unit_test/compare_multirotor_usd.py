#!/usr/bin/env python3
"""
Compare two multirotor USD assets with focus on rotor-joint behavior.

Typical usage:
  python3 training/unit_test/compare_multirotor_usd.py \
    --reference /home/mint/rl_dev/NavRL/isaac-training/third_party/OmniDrones/omni_drones/robots/assets/usd/hummingbird.usd \
    --target /home/mint/rl_dev/NavRL/isaac-training/third_party/OmniDrones/omni_drones/robots/assets/usd/taslab_uav.usd
"""

import argparse
import math
import re
from typing import Dict, Optional, Tuple, List

from pxr import Usd, UsdGeom, UsdPhysics


def _load_yaml(path: str) -> Dict:
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError(
            "PyYAML is required to read --yaml-param. Please install with: pip install pyyaml"
        ) from exc

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid YAML content: {path}")
    return data


def _open_stage(path: str) -> Usd.Stage:
    stage = Usd.Stage.Open(path)
    if stage is None:
        raise RuntimeError(f"Failed to open USD: {path}")
    return stage


def _find_root(stage: Usd.Stage):
    root = stage.GetDefaultPrim()
    if root and root.IsValid():
        return root
    for child in stage.GetPseudoRoot().GetChildren():
        if child.IsValid():
            return child
    return None


def _parse_rotor_index_from_name(name: str) -> Optional[int]:
    m = re.match(r"^rotor_(\d+)$", name)
    return int(m.group(1)) if m else None


def _parse_joint_index_from_name(name: str) -> Optional[int]:
    m = re.match(r"^rotor_(\d+)_joint$", name)
    return int(m.group(1)) if m else None


def _get_rel_target(prim, rel_name: str) -> Optional[str]:
    rel = prim.GetRelationship(rel_name)
    if not rel:
        return None
    targets = rel.GetTargets()
    if not targets:
        return None
    return str(targets[0])


def _get_attr(prim, attr_name: str):
    attr = prim.GetAttribute(attr_name)
    if not attr or not attr.HasAuthoredValueOpinion():
        return None
    return attr.Get()


def _collect_drive_attrs(prim) -> Dict[str, object]:
    drive_attrs: Dict[str, object] = {}
    for attr in prim.GetAttributes():
        name = attr.GetName()
        if not name.startswith("drive:"):
            continue
        if not attr.HasAuthoredValueOpinion():
            continue
        drive_attrs[name] = attr.Get()
    return drive_attrs


def _collect_mesh_paths_under(prim) -> List[str]:
    paths: List[str] = []
    for sub_prim in Usd.PrimRange(prim):
        if not sub_prim.IsValid():
            continue
        if sub_prim.IsA(UsdGeom.Mesh):
            paths.append(sub_prim.GetPath().pathString)
    paths.sort()
    return paths


def _collect_rigid_props(prim) -> Dict[str, object]:
    return {
        "linear_damping": _get_attr(prim, "physics:linearDamping"),
        "angular_damping": _get_attr(prim, "physics:angularDamping"),
        "rigid_enabled": _get_attr(prim, "physics:rigidBodyEnabled"),
        "sleep_threshold": _get_attr(prim, "physxRigidBody:sleepThreshold"),
        "stabilization_threshold": _get_attr(prim, "physxRigidBody:stabilizationThreshold"),
        "disable_gravity": _get_attr(prim, "physxRigidBody:disableGravity"),
    }


def _find_base_prim(root) -> Optional[Usd.Prim]:
    if not root or not root.IsValid():
        return None
    direct = root.GetChild("base_link")
    if direct and direct.IsValid():
        return direct
    for child in root.GetChildren():
        if "base" in child.GetName().lower():
            return child
    return None


def _is_joint_prim(prim) -> bool:
    return prim.IsValid() and "Joint" in (prim.GetTypeName() or "")


def _vec3(v) -> Tuple[float, float, float]:
    if v is None:
        return (0.0, 0.0, 0.0)
    return (float(v[0]), float(v[1]), float(v[2]))


def _angle_deg(x: float, y: float) -> float:
    return math.degrees(math.atan2(y, x))


def _normalize_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def _angle_delta_deg(a: float, b: float) -> float:
    return abs(_normalize_deg(a - b))


def collect_profile(path: str) -> Dict:
    stage = _open_stage(path)
    root = _find_root(stage)
    if root is None or not root.IsValid():
        raise RuntimeError(f"No valid root prim in stage: {path}")

    base_prim = _find_base_prim(root)
    base_path = base_prim.GetPath().pathString if base_prim and base_prim.IsValid() else None

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    base_world_t = None
    if base_prim and base_prim.IsValid():
        try:
            base_world_t = xform_cache.GetLocalToWorldTransform(base_prim).ExtractTranslation()
        except Exception:
            base_world_t = None

    rotor_links: Dict[int, Dict] = {}
    rotor_joints: Dict[int, Dict] = {}

    for prim in stage.Traverse():
        if not prim.IsValid():
            continue

        name = prim.GetName()
        idx_link = _parse_rotor_index_from_name(name)
        idx_joint = _parse_joint_index_from_name(name)

        if idx_link is not None and not _is_joint_prim(prim):
            world_pos = None
            rel_xy = None
            rel_angle = None
            if base_world_t is not None:
                try:
                    p = xform_cache.GetLocalToWorldTransform(prim).ExtractTranslation()
                    world_pos = _vec3(p)
                    dx = float(p[0] - base_world_t[0])
                    dy = float(p[1] - base_world_t[1])
                    rel_xy = (dx, dy)
                    rel_angle = _angle_deg(dx, dy)
                except Exception:
                    pass

            rotor_links[idx_link] = {
                "name": name,
                "path": prim.GetPath().pathString,
                "world_pos": world_pos,
                "rel_xy": rel_xy,
                "rel_angle_deg": rel_angle,
            }

        if idx_joint is not None and _is_joint_prim(prim):
            joint_schema = UsdPhysics.Joint(prim)
            local_pos0 = _vec3(joint_schema.GetLocalPos0Attr().Get()) if joint_schema.GetLocalPos0Attr() else None
            local_pos1 = _vec3(joint_schema.GetLocalPos1Attr().Get()) if joint_schema.GetLocalPos1Attr() else None

            rotor_joints[idx_joint] = {
                "name": name,
                "path": prim.GetPath().pathString,
                "type": prim.GetTypeName(),
                "body0": _get_rel_target(prim, "physics:body0"),
                "body1": _get_rel_target(prim, "physics:body1"),
                "axis": _get_attr(prim, "physics:axis"),
                "lower": _get_attr(prim, "physics:lowerLimit"),
                "upper": _get_attr(prim, "physics:upperLimit"),
                "max_joint_velocity": _get_attr(prim, "physics:maxJointVelocity"),
                "joint_enabled": _get_attr(prim, "physics:jointEnabled"),
                "local_pos0": local_pos0,
                "local_pos1": local_pos1,
                "drive_attrs": _collect_drive_attrs(prim),
            }

    articulation_roots = []
    rigid_count = 0
    collision_count = 0
    for prim in stage.Traverse():
        if not prim.IsValid():
            continue
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulation_roots.append(prim.GetPath().pathString)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_count += 1
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collision_count += 1

    rotor_mesh_paths: Dict[int, List[str]] = {}
    mesh_owner_map: Dict[str, List[int]] = {}
    for idx, link in rotor_links.items():
        link_prim = stage.GetPrimAtPath(link["path"])
        if not link_prim or not link_prim.IsValid():
            rotor_mesh_paths[idx] = []
            continue
        mesh_paths = _collect_mesh_paths_under(link_prim)
        rotor_mesh_paths[idx] = mesh_paths
        for mesh_path in mesh_paths:
            mesh_owner_map.setdefault(mesh_path, []).append(idx)

    articulation_props: Dict[str, object] = {}
    if articulation_roots:
        art_prim = stage.GetPrimAtPath(articulation_roots[0])
        if art_prim and art_prim.IsValid():
            articulation_props = {
                "sleep_threshold": _get_attr(art_prim, "physxArticulation:sleepThreshold"),
                "stabilization_threshold": _get_attr(art_prim, "physxArticulation:stabilizationThreshold"),
                "solver_position_iterations": _get_attr(art_prim, "physxArticulation:solverPositionIterationCount"),
                "solver_velocity_iterations": _get_attr(art_prim, "physxArticulation:solverVelocityIterationCount"),
            }

    rigid_props: Dict[str, Dict[str, object]] = {}
    if base_prim and base_prim.IsValid():
        rigid_props["base_link"] = _collect_rigid_props(base_prim)
    for idx, link in rotor_links.items():
        link_prim = stage.GetPrimAtPath(link["path"])
        if link_prim and link_prim.IsValid():
            rigid_props[f"rotor_{idx}"] = _collect_rigid_props(link_prim)

    return {
        "path": path,
        "root": root.GetPath().pathString,
        "base_path": base_path,
        "articulation_roots": articulation_roots,
        "rigid_count": rigid_count,
        "collision_count": collision_count,
        "rotor_links": rotor_links,
        "rotor_joints": rotor_joints,
        "rotor_mesh_paths": rotor_mesh_paths,
        "mesh_owner_map": mesh_owner_map,
        "articulation_props": articulation_props,
        "rigid_props": rigid_props,
    }


def _fmt(v) -> str:
    if v is None:
        return "None"
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def _compare_joint_field(ref: Dict, tgt: Dict, idx: int, key: str, tol: float = 1e-6) -> Optional[str]:
    rv = ref.get(key)
    tv = tgt.get(key)
    if isinstance(rv, tuple) and isinstance(tv, tuple):
        if len(rv) != len(tv):
            return f"rotor_{idx}_joint {key}: tuple size mismatch ref={rv} tgt={tv}"
        for a, b in zip(rv, tv):
            if abs(float(a) - float(b)) > tol:
                return f"rotor_{idx}_joint {key}: ref={rv} tgt={tv}"
        return None
    if isinstance(rv, float) or isinstance(tv, float):
        if rv is None or tv is None:
            if rv != tv:
                return f"rotor_{idx}_joint {key}: ref={_fmt(rv)} tgt={_fmt(tv)}"
            return None
        if abs(float(rv) - float(tv)) > tol:
            return f"rotor_{idx}_joint {key}: ref={_fmt(rv)} tgt={_fmt(tv)}"
        return None
    if rv != tv:
        return f"rotor_{idx}_joint {key}: ref={_fmt(rv)} tgt={_fmt(tv)}"
    return None


def compare_profiles(reference: Dict, target: Dict) -> List[str]:
    diffs: List[str] = []

    if len(reference["articulation_roots"]) == 0:
        diffs.append("Reference has no articulation root (unexpected)")
    if len(target["articulation_roots"]) == 0:
        diffs.append("Target has no articulation root")

    ref_link_ids = sorted(reference["rotor_links"].keys())
    tgt_link_ids = sorted(target["rotor_links"].keys())
    ref_joint_ids = sorted(reference["rotor_joints"].keys())
    tgt_joint_ids = sorted(target["rotor_joints"].keys())

    if ref_link_ids != tgt_link_ids:
        diffs.append(f"Rotor link id set mismatch: ref={ref_link_ids}, tgt={tgt_link_ids}")
    if ref_joint_ids != tgt_joint_ids:
        diffs.append(f"Rotor joint id set mismatch: ref={ref_joint_ids}, tgt={tgt_joint_ids}")

    common_joint_ids = sorted(set(ref_joint_ids).intersection(tgt_joint_ids))
    for idx in common_joint_ids:
        rj = reference["rotor_joints"][idx]
        tj = target["rotor_joints"][idx]

        for field in [
            "type", "body0", "body1", "axis", "lower", "upper",
            "max_joint_velocity", "joint_enabled", "local_pos0", "local_pos1"
        ]:
            diff = _compare_joint_field(rj, tj, idx, field, tol=1e-5)
            if diff:
                diffs.append(diff)

        if rj.get("drive_attrs") != tj.get("drive_attrs"):
            diffs.append(
                f"rotor_{idx}_joint drive attrs differ: ref={rj.get('drive_attrs')} tgt={tj.get('drive_attrs')}"
            )

    common_link_ids = sorted(set(ref_link_ids).intersection(tgt_link_ids))
    for idx in common_link_ids:
        rl = reference["rotor_links"][idx]
        tl = target["rotor_links"][idx]
        ra = rl.get("rel_angle_deg")
        ta = tl.get("rel_angle_deg")
        if ra is not None and ta is not None:
            d = abs((ta - ra + 180.0) % 360.0 - 180.0)
            if d > 5.0:
                diffs.append(
                    f"rotor_{idx} relative angle differs: ref={ra:.2f}deg tgt={ta:.2f}deg (delta={d:.2f}deg)"
                )

        ref_mesh_count = len(reference.get("rotor_mesh_paths", {}).get(idx, []))
        tgt_mesh_count = len(target.get("rotor_mesh_paths", {}).get(idx, []))
        if ref_mesh_count != tgt_mesh_count:
            diffs.append(
                f"rotor_{idx} mesh count differs: ref={ref_mesh_count} tgt={tgt_mesh_count}"
            )

    for key in ["sleep_threshold", "stabilization_threshold"]:
        rv = reference.get("articulation_props", {}).get(key)
        tv = target.get("articulation_props", {}).get(key)
        if rv != tv:
            diffs.append(f"articulation {key} differs: ref={_fmt(rv)} tgt={_fmt(tv)}")

    for rigid_key in sorted(set(reference.get("rigid_props", {}).keys()) & set(target.get("rigid_props", {}).keys())):
        for prop_key in ["linear_damping", "angular_damping", "sleep_threshold", "stabilization_threshold", "disable_gravity"]:
            rv = reference["rigid_props"][rigid_key].get(prop_key)
            tv = target["rigid_props"][rigid_key].get(prop_key)
            if rv != tv:
                diffs.append(
                    f"{rigid_key} {prop_key} differs: ref={_fmt(rv)} tgt={_fmt(tv)}"
                )

    return diffs


def print_profile(profile: Dict, title: str):
    print("=" * 88)
    print(title)
    print("=" * 88)
    print(f"USD: {profile['path']}")
    print(f"root: {profile['root']}")
    print(f"base: {profile['base_path']}")
    print(f"articulation roots ({len(profile['articulation_roots'])}):")
    for p in profile["articulation_roots"]:
        print(f"  - {p}")
    print(f"rigid_count={profile['rigid_count']} collision_count={profile['collision_count']}")

    print("rotor links:")
    for idx in sorted(profile["rotor_links"].keys()):
        link = profile["rotor_links"][idx]
        print(
            f"  - rotor_{idx}: path={link['path']} rel_xy={_fmt(link['rel_xy'])} "
            f"angle_deg={_fmt(link['rel_angle_deg'])}"
        )

    print("rotor joints:")
    for idx in sorted(profile["rotor_joints"].keys()):
        joint = profile["rotor_joints"][idx]
        print(
            f"  - rotor_{idx}_joint: type={joint['type']} body0={joint['body0']} body1={joint['body1']} "
            f"axis={_fmt(joint['axis'])} lim=[{_fmt(joint['lower'])}, {_fmt(joint['upper'])}] "
            f"maxVel={_fmt(joint['max_joint_velocity'])} enabled={_fmt(joint['joint_enabled'])} "
            f"lp0={_fmt(joint['local_pos0'])} lp1={_fmt(joint['local_pos1'])}"
        )
        print(f"      drive={joint.get('drive_attrs', {})}")

    print("rotor mesh ownership:")
    for idx in sorted(profile.get("rotor_mesh_paths", {}).keys()):
        mesh_paths = profile["rotor_mesh_paths"][idx]
        print(f"  - rotor_{idx}: mesh_count={len(mesh_paths)}")
        for mesh_path in mesh_paths:
            print(f"      - {mesh_path}")

    non_unique = [
        (mesh_path, owners)
        for mesh_path, owners in profile.get("mesh_owner_map", {}).items()
        if len(owners) > 1
    ]
    if non_unique:
        print("  [WARN] Non-unique mesh ownership across rotors detected:")
        for mesh_path, owners in non_unique:
            print(f"      - {mesh_path} owned_by={owners}")
    else:
        print("  [PASS] Mesh ownership is unique across rotor links")

    print("articulation sleep/solver props:")
    print(f"  - {profile.get('articulation_props', {})}")

    print("rigid body damping/sleep props:")
    for key, value in profile.get("rigid_props", {}).items():
        print(f"  - {key}: {value}")


def print_yaml_consistency(target: Dict, yaml_path: str):
    print("=" * 88)
    print("YAML CONSISTENCY REPORT")
    print("=" * 88)
    print(f"YAML: {yaml_path}")

    data = _load_yaml(yaml_path)
    rotor_cfg = data.get("rotor_configuration", {}) if isinstance(data, dict) else {}

    yaml_angles = rotor_cfg.get("rotor_angles", [])
    yaml_dirs = rotor_cfg.get("directions", [])

    if not isinstance(yaml_angles, list):
        yaml_angles = []
    if not isinstance(yaml_dirs, list):
        yaml_dirs = []

    print(f"- yaml rotor_angles count: {len(yaml_angles)}")
    print(f"- yaml directions count: {len(yaml_dirs)}")

    usd_ids = sorted(target["rotor_links"].keys())
    if not usd_ids:
        print("- target USD has no rotor links; cannot compare angles")
        return

    print("- Angle consistency by rotor index:")
    for idx in usd_ids:
        usd_angle = target["rotor_links"][idx].get("rel_angle_deg")
        if usd_angle is None:
            print(f"  - rotor_{idx}: USD angle unavailable")
            continue

        if idx >= len(yaml_angles):
            print(f"  - rotor_{idx}: missing YAML rotor_angles[{idx}], USD={usd_angle:.3f} deg")
            continue

        yaml_angle_deg = math.degrees(float(yaml_angles[idx]))
        delta = _angle_delta_deg(usd_angle, yaml_angle_deg)
        status = "PASS" if delta <= 5.0 else "FAIL"
        print(
            f"  - rotor_{idx}: USD={usd_angle:8.3f} deg | YAML={yaml_angle_deg:8.3f} deg "
            f"| delta={delta:6.3f} deg -> {status}"
        )

    print("- Direction consistency (YAML internal checks):")
    if len(yaml_dirs) < len(usd_ids):
        print(f"  - FAIL: directions count too small, expected >= {len(usd_ids)}")
    else:
        valid_values = all(float(v) in (-1.0, 1.0) for v in yaml_dirs[:len(usd_ids)])
        if valid_values:
            print("  - PASS: directions values are in {-1, +1}")
        else:
            print("  - FAIL: directions has values outside {-1, +1}")

        alternating = True
        dir_sub = [float(v) for v in yaml_dirs[:len(usd_ids)]]
        for i in range(len(dir_sub)):
            if dir_sub[i] == dir_sub[(i + 1) % len(dir_sub)]:
                alternating = False
                break
        if alternating:
            print("  - PASS: adjacent rotor directions alternate sign")
        else:
            print("  - WARN: adjacent rotor directions are not strictly alternating")

        print(f"  - values by index: {dir_sub}")

    print(
        "- Note: USD static structure cannot fully prove CW/CCW physical spin semantics; "
        "use runtime single-rotor test to verify sign mapping."
    )


def main():
    parser = argparse.ArgumentParser(description="Compare two multirotor USD files for rotor behavior differences")
    parser.add_argument("--reference", required=True, help="Reference USD path (e.g., hummingbird.usd)")
    parser.add_argument("--target", required=True, help="Target USD path (e.g., taslab_uav.usd)")
    parser.add_argument("--yaml-param", default=None, help="Optional UAV YAML parameter file for rotor_angles/directions consistency report")
    args = parser.parse_args()

    ref = collect_profile(args.reference)
    tgt = collect_profile(args.target)

    print_profile(ref, "REFERENCE PROFILE")
    print_profile(tgt, "TARGET PROFILE")

    diffs = compare_profiles(ref, tgt)
    print("=" * 88)
    print("DIFF SUMMARY")
    print("=" * 88)
    if not diffs:
        print("No critical differences detected in rotor-link/joint fields.")
    else:
        for d in diffs:
            print(f"- {d}")

    if args.yaml_param:
        print_yaml_consistency(tgt, args.yaml_param)


if __name__ == "__main__":
    main()
