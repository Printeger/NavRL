#!/usr/bin/env python3
"""
USD Inspect + Align Tool (Two Functions Only)
==============================================

本脚本只保留两个功能：
1) inspect: 读取并打印指定 USD 的结构与关键信息
2) align  : 将指定 USD 向标准 USD 对齐，并清理 rotor joint drive 参数

align 会执行：
- 根 Prim / defaultPrim 对齐到标准 USD 风格
- base_link / rotor_0..3 / rotor_0_joint..3 命名对齐（按 RF, LF, LB, RB）
- joint 关系 physics:body0/body1 在重命名后自动重写
- 确保根 Prim 具备 ArticulationRootAPI，必要时补充最小 Physics API
- 清理 rotor_0~3_joint 的 drive 参数：
  - drive:angular:physics:stiffness = 0
  - drive:angular:physics:damping = 0
  - 移除 drive:angular:physics:targetPosition

用法示例：
  # 1) 读取信息
  python3 fix_usd.py inspect \
    --input /path/to/taslab_uav_normalized.usd

  # 2) 对齐到标准 USD（默认参考 Hummingbird）并输出新文件
  python3 fix_usd.py align \
    --input /path/to/taslab_uav_normalized.usd \
    --output /path/to/taslab_uav_aligned.usd

  # 3) 指定参考 USD
  python3 fix_usd.py align \
    --input /path/to/input.usd \
    --reference-usd /path/to/hummingbird.usd
"""

import argparse
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

from pxr import Sdf, Usd, UsdGeom, UsdPhysics


DEFAULT_REFERENCE_USD = "/home/mint/rl_dev/NavRL/isaac-training/third_party/OmniDrones/omni_drones/robots/assets/usd/hummingbird.usd"
ROTOR_ORDER = ("rf", "lf", "lb", "rb")


def _print_sep(char: str = "=", n: int = 80):
    print(char * n)


def _open_stage(path: str):
    stage = Usd.Stage.Open(path)
    if stage is None:
        raise RuntimeError(f"Cannot open USD: {path}")
    return stage


def _find_root(stage):
    root = stage.GetDefaultPrim()
    if root and root.IsValid():
        return root
    children = [c for c in stage.GetPseudoRoot().GetChildren() if c.IsValid()]
    if not children:
        return None
    for c in children:
        if c.GetTypeName() in ("Xform", ""):
            return c
    return children[0]


def _is_joint_prim(prim) -> bool:
    if not prim or not prim.IsValid():
        return False
    return "Joint" in (prim.GetTypeName() or "") or prim.HasAPI(UsdPhysics.Joint)


def _tokenize(name: str) -> List[str]:
    return [t for t in re.split(r"[^a-zA-Z0-9]+", name.lower()) if t]


def _extract_quad(name: str) -> Optional[str]:
    tokens = _tokenize(name)
    for key in ROTOR_ORDER:
        if key in tokens:
            return key
    compact = "".join(tokens)
    for key in ROTOR_ORDER:
        if key in compact:
            return key
    return None


def _parse_rotor_index(name: str) -> Optional[int]:
    m = re.match(r"^rotor_(\d+)$", name)
    return int(m.group(1)) if m else None


def _parse_rotor_joint_index(name: str) -> Optional[int]:
    m = re.match(r"^rotor_(\d+)_joint$", name)
    return int(m.group(1)) if m else None


def _relation_targets_str(prim, rel_name: str) -> str:
    rel = prim.GetRelationship(rel_name)
    if not rel:
        return "[]"
    targets = rel.GetTargets()
    if not targets:
        return "[]"
    return "[" + ", ".join([str(t) for t in targets]) + "]"


def collect_stage_info(stage) -> Dict[str, object]:
    root = _find_root(stage)
    default_prim = stage.GetDefaultPrim()
    default_path = default_prim.GetPath().pathString if default_prim and default_prim.IsValid() else None
    root_path = root.GetPath().pathString if root and root.IsValid() else None

    articulation_roots = []
    rigid_count = 0
    collision_count = 0
    joints = []
    rotor_links = []
    rotor_joints = []

    for prim in stage.Traverse():
        if not prim.IsValid():
            continue
        name = prim.GetName()
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulation_roots.append(prim.GetPath().pathString)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_count += 1
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collision_count += 1
        if _is_joint_prim(prim):
            joints.append(
                {
                    "path": prim.GetPath().pathString,
                    "type": prim.GetTypeName() or "Unknown",
                    "body0": _relation_targets_str(prim, "physics:body0"),
                    "body1": _relation_targets_str(prim, "physics:body1"),
                }
            )
        if re.match(r"^rotor_\d+$", name) and not _is_joint_prim(prim):
            rotor_links.append(prim.GetPath().pathString)
        if re.match(r"^rotor_\d+_joint$", name):
            rotor_joints.append(prim.GetPath().pathString)

    joints.sort(key=lambda item: item["path"])
    rotor_links.sort()
    rotor_joints.sort()

    return {
        "default_path": default_path,
        "root_path": root_path,
        "articulation_roots": sorted(articulation_roots),
        "rigid_count": rigid_count,
        "collision_count": collision_count,
        "joints": joints,
        "rotor_links": rotor_links,
        "rotor_joints": rotor_joints,
    }


def print_stage_info(info: Dict[str, object], label: str):
    _print_sep()
    print(f"USD INFO - {label}")
    _print_sep()
    print(f"Default Prim: {info['default_path']}")
    print(f"Root Prim   : {info['root_path']}")
    print(f"Articulation roots ({len(info['articulation_roots'])}):")
    for p in info["articulation_roots"]:
        print(f"  - {p}")
    print(f"RigidBody count : {info['rigid_count']}")
    print(f"Collision count : {info['collision_count']}")

    print(f"rotor links ({len(info['rotor_links'])}):")
    for p in info["rotor_links"]:
        print(f"  - {p}")

    print(f"rotor joints ({len(info['rotor_joints'])}):")
    for p in info["rotor_joints"]:
        print(f"  - {p}")

    print(f"joints ({len(info['joints'])}):")
    for item in info["joints"]:
        print(
            f"  - {item['path']} | type={item['type']} | "
            f"body0={item['body0']} | body1={item['body1']}"
        )


def _collect_reference_profile(reference_stage) -> Dict[str, object]:
    root = _find_root(reference_stage)
    if root is None or not root.IsValid():
        raise RuntimeError("Reference USD root prim not found")

    root_name = root.GetName()

    base_name = None
    rotor_link_names: List[str] = []
    rotor_joint_names: List[str] = []

    for child in root.GetChildren():
        name = child.GetName()
        if name == "base_link":
            base_name = name
        if re.match(r"^rotor_\d+$", name):
            rotor_link_names.append(name)

    if base_name is None:
        for child in root.GetChildren():
            if "base" in child.GetName().lower():
                base_name = child.GetName()
                break

    base_prim = root.GetChild(base_name) if base_name else None
    if base_prim and base_prim.IsValid():
        for child in base_prim.GetChildren():
            n = child.GetName()
            if re.match(r"^rotor_\d+_joint$", n):
                rotor_joint_names.append(n)

    rotor_link_names.sort(key=lambda n: int(n.split("_")[1]))
    rotor_joint_names.sort(key=lambda n: int(n.split("_")[1]))

    if base_name is None:
        raise RuntimeError("Reference USD has no base prim")
    if len(rotor_link_names) < 4 or len(rotor_joint_names) < 4:
        raise RuntimeError("Reference USD rotor naming profile is incomplete")

    return {
        "root_name": root_name,
        "base_name": base_name,
        "rotor_link_names": rotor_link_names[:4],
        "rotor_joint_names": rotor_joint_names[:4],
    }


def _build_naming_plan(stage, profile: Dict[str, object]) -> Tuple[List[Tuple[str, str, str]], List[str]]:
    root = _find_root(stage)
    if root is None or not root.IsValid():
        return [], ["target root prim not found"]

    issues: List[str] = []
    renames: List[Tuple[str, str, str]] = []

    target_root_path = f"/{profile['root_name']}"
    if root.GetPath().pathString != target_root_path:
        if stage.GetPrimAtPath(target_root_path).IsValid():
            issues.append(f"target root path already exists: {target_root_path}")
        else:
            renames.append((root.GetPath().pathString, target_root_path, "root"))

    # Resolve root path after potential rename
    root_path_after = target_root_path if any(k == "root" for _, _, k in renames) else root.GetPath().pathString

    # Base
    base_prim = root.GetChild(profile["base_name"])
    if not base_prim or not base_prim.IsValid():
        candidates = [c for c in root.GetChildren() if "base" in c.GetName().lower()]
        if len(candidates) == 1:
            old = candidates[0].GetPath().pathString
            new = f"{root_path_after}/{profile['base_name']}"
            if old != new:
                renames.append((old, new, "base"))
        elif len(candidates) == 0:
            issues.append("base prim candidate not found")
        else:
            issues.append("base prim candidates are ambiguous")

    canonical_links: Dict[int, str] = {}
    canonical_joints: Dict[int, str] = {}
    quad_link_candidates: Dict[str, List[str]] = {k: [] for k in ROTOR_ORDER}
    quad_joint_candidates: Dict[str, List[str]] = {k: [] for k in ROTOR_ORDER}

    for prim in stage.Traverse():
        if not prim.IsValid():
            continue
        name = prim.GetName()
        name_lower = name.lower()
        path = prim.GetPath().pathString

        link_idx = _parse_rotor_index(name)
        if link_idx is not None and not _is_joint_prim(prim):
            canonical_links.setdefault(link_idx, path)

        joint_idx = _parse_rotor_joint_index(name)
        if joint_idx is not None and _is_joint_prim(prim):
            canonical_joints.setdefault(joint_idx, path)

        quad = _extract_quad(name)
        if quad is None:
            continue

        if _is_joint_prim(prim):
            if any(k in name_lower for k in ("motor", "prop", "rotor", "paddle")):
                quad_joint_candidates[quad].append(path)
        else:
            if any(k in name_lower for k in ("motor", "prop", "rotor", "paddle")):
                quad_link_candidates[quad].append(path)

    for i, quad in enumerate(ROTOR_ORDER):
        target_link_name = profile["rotor_link_names"][i]
        target_joint_name = profile["rotor_joint_names"][i]

        src_link = canonical_links.get(i)
        if src_link is None:
            cands = sorted(quad_link_candidates[quad])
            src_link = cands[0] if cands else None
        if src_link is None:
            issues.append(f"missing rotor link candidate for {quad}")
        else:
            new_link = f"{root_path_after}/{target_link_name}"
            if src_link != new_link:
                renames.append((src_link, new_link, "link"))

        src_joint = canonical_joints.get(i)
        if src_joint is None:
            cands = sorted(quad_joint_candidates[quad])
            src_joint = cands[0] if cands else None
        if src_joint is None:
            issues.append(f"missing rotor joint candidate for {quad}")
        else:
            # joints usually under base
            new_joint = f"{root_path_after}/{profile['base_name']}/{target_joint_name}"
            if src_joint != new_joint:
                renames.append((src_joint, new_joint, "joint"))

    # Collision check
    target_paths = [new for _, new, _ in renames]
    if len(target_paths) != len(set(target_paths)):
        issues.append("rename collisions detected: multiple sources map to same target")

    # Sort deep first for safe rename
    renames.sort(key=lambda item: item[0].count("/"), reverse=True)
    return renames, issues


def _apply_renames(stage, renames: List[Tuple[str, str, str]]) -> bool:
    if not renames:
        print("  ✅ no renames needed")
        return True

    edits = Sdf.BatchNamespaceEdit()
    for old_path, new_path, _ in renames:
        edits.Add(Sdf.Path(old_path), Sdf.Path(new_path))

    ok = stage.GetRootLayer().Apply(edits)
    if not ok:
        print("  ❌ rename apply failed")
        return False

    print(f"  ✅ renamed prims: {len(renames)}")
    for old_path, new_path, kind in renames:
        print(f"    - [{kind}] {old_path} -> {new_path}")
    return True


def _rewrite_joint_rel_targets(stage, renames: List[Tuple[str, str, str]]) -> int:
    if not renames:
        return 0
    path_map = {old: new for old, new, _ in renames}
    rewritten = 0

    for prim in stage.Traverse():
        if not prim.IsValid() or not _is_joint_prim(prim):
            continue
        for rel_name in ("physics:body0", "physics:body1"):
            rel = prim.GetRelationship(rel_name)
            if not rel:
                continue
            old_targets = rel.GetTargets()
            if not old_targets:
                continue
            new_targets = []
            changed = False
            for t in old_targets:
                ts = str(t)
                mapped = path_map.get(ts)
                if mapped and mapped != ts:
                    new_targets.append(Sdf.Path(mapped))
                    changed = True
                else:
                    new_targets.append(t)
            if changed:
                rel.SetTargets(new_targets)
                rewritten += 1
    return rewritten


def _validate_joint_rel_targets(stage) -> List[str]:
    issues = []
    for prim in stage.Traverse():
        if not prim.IsValid() or not _is_joint_prim(prim):
            continue
        for rel_name in ("physics:body0", "physics:body1"):
            rel = prim.GetRelationship(rel_name)
            if not rel:
                issues.append(f"{prim.GetPath()} missing {rel_name}")
                continue
            targets = rel.GetTargets()
            if not targets:
                issues.append(f"{prim.GetPath()} empty {rel_name}")
                continue
            for t in targets:
                if not stage.GetPrimAtPath(t).IsValid():
                    issues.append(f"{prim.GetPath()} invalid {rel_name} target: {t}")
    return issues


def _ensure_minimum_physics(stage, root):
    has_articulation = any(p.HasAPI(UsdPhysics.ArticulationRootAPI) for p in stage.Traverse())
    if not has_articulation:
        UsdPhysics.ArticulationRootAPI.Apply(root)
        print(f"  ✅ applied ArticulationRootAPI on {root.GetPath()}")

    if not root.HasAPI(UsdPhysics.ArticulationRootAPI):
        UsdPhysics.ArticulationRootAPI.Apply(root)
        print(f"  ✅ ensured root articulation on {root.GetPath()}")

    has_rigid = any(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in stage.Traverse())
    if not has_rigid:
        UsdPhysics.RigidBodyAPI.Apply(root)
        print(f"  ✅ applied RigidBodyAPI on {root.GetPath()}")

    has_collision = any(p.HasAPI(UsdPhysics.CollisionAPI) for p in stage.Traverse())
    if not has_collision:
        meshes = [p for p in stage.Traverse() if p.IsValid() and p.GetTypeName() == "Mesh"]
        if meshes:
            for mesh in meshes:
                UsdPhysics.CollisionAPI.Apply(mesh)
                UsdPhysics.MeshCollisionAPI.Apply(mesh).CreateApproximationAttr("convexHull")
            print(f"  ✅ applied CollisionAPI/MeshCollisionAPI on {len(meshes)} meshes")
        else:
            UsdPhysics.CollisionAPI.Apply(root)
            print(f"  ✅ applied CollisionAPI on {root.GetPath()}")


def _clean_rotor_joint_drive(stage) -> Dict[str, int]:
    cleaned_joints = 0
    stiffness_updated = 0
    damping_updated = 0
    target_cleared = 0

    for prim in stage.Traverse():
        if not prim.IsValid() or not _is_joint_prim(prim):
            continue
        name = prim.GetName()
        if not re.match(r"^rotor_\d+_joint$", name):
            continue

        cleaned_joints += 1

        stiffness_attr = prim.GetAttribute("drive:angular:physics:stiffness")
        if stiffness_attr and stiffness_attr.IsValid():
            cur = stiffness_attr.Get() if stiffness_attr.HasAuthoredValueOpinion() else None
            if cur is None or float(cur) != 0.0:
                stiffness_attr.Set(0.0)
                stiffness_updated += 1

        damping_attr = prim.GetAttribute("drive:angular:physics:damping")
        if damping_attr and damping_attr.IsValid():
            cur = damping_attr.Get() if damping_attr.HasAuthoredValueOpinion() else None
            if cur is None or float(cur) != 0.0:
                damping_attr.Set(0.0)
                damping_updated += 1

        target_attr = prim.GetAttribute("drive:angular:physics:targetPosition")
        if target_attr and target_attr.IsValid() and target_attr.HasAuthoredValueOpinion():
            target_attr.Clear()
            target_cleared += 1

    print(
        "  ✅ rotor drive cleanup: "
        f"joints={cleaned_joints}, stiffness->0={stiffness_updated}, "
        f"damping->0={damping_updated}, targetPosition cleared={target_cleared}"
    )

    return {
        "cleaned_joints": cleaned_joints,
        "stiffness_updated": stiffness_updated,
        "damping_updated": damping_updated,
        "target_cleared": target_cleared,
    }


def run_inspect(input_path: str):
    stage = _open_stage(input_path)
    info = collect_stage_info(stage)
    print_stage_info(info, "Inspect")


def run_align(input_path: str, output_path: str, reference_usd: str):
    stage = _open_stage(input_path)
    reference_stage = _open_stage(reference_usd)

    _print_sep()
    print("Reference Profile")
    _print_sep()
    ref_profile = _collect_reference_profile(reference_stage)
    print(f"- root_name: {ref_profile['root_name']}")
    print(f"- base_name: {ref_profile['base_name']}")
    print(f"- rotor_link_names : {ref_profile['rotor_link_names']}")
    print(f"- rotor_joint_names: {ref_profile['rotor_joint_names']}")

    before = collect_stage_info(stage)
    print_stage_info(before, "Before Align")

    renames, issues = _build_naming_plan(stage, ref_profile)
    if issues:
        _print_sep("-")
        print("❌ alignment plan failed:")
        for issue in issues:
            print(f"- {issue}")
        sys.exit(2)

    _print_sep("-")
    print("Applying alignment...")

    if not _apply_renames(stage, renames):
        sys.exit(1)

    rewritten = _rewrite_joint_rel_targets(stage, renames)
    print(f"  ✅ rewritten joint relation entries: {rewritten}")

    rel_issues = _validate_joint_rel_targets(stage)
    if rel_issues:
        print("  ❌ invalid joint relation targets after rename:")
        for item in rel_issues:
            print(f"  - {item}")
        sys.exit(2)

    root = _find_root(stage)
    if root is None or not root.IsValid():
        print("❌ root prim missing after alignment")
        sys.exit(2)

    stage.SetDefaultPrim(root)
    print(f"  ✅ set defaultPrim: {root.GetPath()}")

    _ensure_minimum_physics(stage, root)
    _clean_rotor_joint_drive(stage)

    stage.Export(output_path)
    print(f"💾 exported: {output_path}")

    after_stage = _open_stage(output_path)
    after = collect_stage_info(after_stage)
    print_stage_info(after, "After Align")


def _default_aligned_output(input_path: str) -> str:
    stem, ext = os.path.splitext(input_path)
    if ext:
        return f"{stem}_aligned{ext}"
    return f"{input_path}_aligned"


def main():
    parser = argparse.ArgumentParser(
        description="USD tool with only two functions: inspect and align",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Read and print USD structure/info")
    inspect_parser.add_argument("--input", "-i", required=True, help="Input USD/USDC file")

    align_parser = subparsers.add_parser("align", help="Align USD to reference USD and clean rotor drive")
    align_parser.add_argument("--input", "-i", required=True, help="Input USD/USDC file")
    align_parser.add_argument("--reference-usd", default=DEFAULT_REFERENCE_USD, help="Reference standard USD")
    align_parser.add_argument("--output", "-o", default=None, help="Output USD path (default: *_aligned)")

    args = parser.parse_args()

    if args.command == "inspect":
        run_inspect(args.input)
        return

    if args.command == "align":
        output_path = args.output if args.output else _default_aligned_output(args.input)
        run_align(args.input, output_path, args.reference_usd)
        return


if __name__ == "__main__":
    main()
