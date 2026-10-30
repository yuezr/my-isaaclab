#!/usr/bin/env python3
"""Merge UR5 arm URDF with LEAP Hand right URDF into a single importable package.

Output layout:
  ~/pro/ur5_leaphand/source/ur5_leaphand_right.urdf
  ~/pro/ur5_leaphand/source/meshes/ur5/{visual,collision}/*
  ~/pro/ur5_leaphand/source/meshes/leap_hand/{visual,collision}/*
"""
from __future__ import annotations

import math
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # <ext>/data/ur5_leaphand
SRC = ROOT / "urdf"
MESH_OUT = SRC / "meshes"

UR5_URDF = Path(
    "/home/yzr/isaacsim-6.0.1/extsDeprecated/isaacsim.robot_motion.motion_generation/"
    "motion_policy_configs/universal_robots/ur5/ur5.urdf"
)
UR5_PKG_ROOT = Path("/home/yzr/pro/assets/urdf_src/ur_description")  # package://ur_description/
LEAP_URDF = Path("/home/yzr/pro/assets/urdf_src/dex-urdf/robots/hands/leap_hand/leap_hand_right.urdf")

MOUNT_PARENT = "tool0"
MOUNT_XYZ = (0.0, 0.0, 0.0)
MOUNT_RPY = (0.0, 0.0, 0.0)

DROP_TAGS = {"transmission", "gazebo", "ros2_control", "joint_properties", "safety_controller"}


def strip_tags(root: ET.Element) -> None:
    for j in root.findall("joint"):
        for child in list(j):
            if child.tag in DROP_TAGS:
                j.remove(child)
    for child in list(root):
        if child.tag in DROP_TAGS:
            root.remove(child)


def copy_mesh(src: Path, subdir: str, rel: str, missing: list[str]) -> str | None:
    """Copy `src` to meshes/<subdir>/<rel>, return the URDF-relative path."""
    if not src.is_file():
        missing.append(str(src))
        return None
    dst = MESH_OUT / subdir / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
        shutil.copy2(src, dst)
    return f"meshes/{subdir}/{rel}"


def main() -> int:
    for p in (UR5_URDF, LEAP_URDF):
        if not p.is_file():
            sys.exit(f"missing source urdf: {p}")
    SRC.mkdir(parents=True, exist_ok=True)

    arm = ET.parse(UR5_URDF).getroot()
    hand = ET.parse(LEAP_URDF).getroot()
    strip_tags(arm)
    strip_tags(hand)

    missing: list[str] = []

    # ---- 1. arm meshes: package://ur_description/meshes/... -> meshes/ur5/...
    for m in arm.iter("mesh"):
        fn = m.get("filename", "")
        if fn.startswith("package://ur_description/"):
            rel = fn[len("package://ur_description/") :]
            new = copy_mesh(UR5_PKG_ROOT / rel, "ur5", rel, missing)
            if new:
                m.set("filename", new)
        elif fn.startswith("package://"):
            rel = fn.split("package://", 1)[1].split("/", 1)[1]
            new = copy_mesh(UR5_PKG_ROOT / rel, "ur5", rel, missing)
            if new:
                m.set("filename", new)

    # ---- 1b. give the massless frame links a tiny inertial.
    # `flange` / `tool0` (arm) and `base` (hand) are pure frames in the upstream URDFs. The USD
    # importer drops massless links as rigid bodies but keeps the joints that reference them, so
    # PhysX ends up anchoring the hand to the world and the whole articulation tears apart.
    # A small mass keeps these frames as real (but negligible) bodies -- and keeps `tool0`
    # available as an end-effector frame.
    STUB_MASS = "0.01"
    for root, link_names in ((arm, ("base_link", "flange", "tool0")), (hand, ("base",))):
        for name in link_names:
            link = root.find(f"link[@name='{name}']")
            if link is None or link.find("inertial") is not None:
                continue
            inert = ET.SubElement(link, "inertial")
            ET.SubElement(inert, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
            ET.SubElement(inert, "mass", {"value": STUB_MASS})
            ET.SubElement(inert, "inertia", {
                "ixx": "1e-6", "ixy": "0", "ixz": "0", "iyy": "1e-6", "iyz": "0", "izz": "1e-6",
            })

    # ---- 2. hand: rename links/joints with leap_ prefix, relocate meshes
    rename: dict[str, str] = {}
    for link in hand.findall("link"):
        old = link.get("name")
        rename[old] = f"leap_{old}"
        link.set("name", rename[old])
    for joint in hand.findall("joint"):
        joint.set("name", f"leap_{joint.get('name')}")
        joint.find("parent").set("link", rename[joint.find("parent").get("link")])
        joint.find("child").set("link", rename[joint.find("child").get("link")])

    for m in hand.iter("mesh"):
        fn = m.get("filename", "")
        rel = fn.split("/", 1)[1] if fn.startswith("meshes/") else fn
        bucket = "visual" if rel.startswith("visual/") else "collision"
        new = copy_mesh(LEAP_URDF.parent / fn, f"leap_hand/{bucket}", rel.split("/", 1)[1], missing)
        if new:
            m.set("filename", new)

    # copy the .mtl / texture siblings that OBJ files reference
    for obj in (LEAP_URDF.parent / "meshes").rglob("*.obj"):
        for sib in obj.parent.glob(obj.stem + ".*"):
            if sib.suffix.lower() in {".mtl", ".png", ".jpg", ".jpeg", ".tga"}:
                bucket = "visual" if sib.parent.name == "visual" else "collision"
                copy_mesh(sib, f"leap_hand/{bucket}", sib.name, missing)

    # ---- 3. attach hand root to the arm flange
    mount = ET.SubElement(arm, "joint", {"name": "leap_mount_joint", "type": "fixed"})
    ET.SubElement(mount, "origin", {
        "xyz": " ".join(f"{v:g}" for v in MOUNT_XYZ),
        "rpy": " ".join(f"{v:g}" for v in MOUNT_RPY),
    })
    ET.SubElement(mount, "parent", {"link": MOUNT_PARENT})
    ET.SubElement(mount, "child", {"link": "leap_base"})

    # ---- 4. append hand links + joints
    for link in hand.findall("link"):
        arm.append(link)
    for joint in hand.findall("joint"):
        arm.append(joint)

    out = SRC / "ur5_leaphand_right.urdf"
    tree = ET.ElementTree(arm)
    ET.indent(tree, space="  ")
    tree.write(out, encoding="utf-8", xml_declaration=True)

    # ---- 5. report
    links = [l.get("name") for l in arm.findall("link")]
    joints = [j for j in arm.findall("joint")]
    movable = [j for j in joints if j.get("type") in ("revolute", "continuous", "prismatic")]
    dup = {n for n in links if links.count(n) > 1}
    print(f"wrote   : {out}")
    print(f"links   : {len(links)}  joints: {len(joints)}  movable: {len(movable)}")
    print(f"root    : {arm.find('link').get('name')}")
    print(f"mount   : {MOUNT_PARENT} -> leap_base  xyz={MOUNT_XYZ} rpy={MOUNT_RPY}")
    if dup:
        print(f"!! duplicate link names: {dup}")
    print("movable joints:")
    for j in movable:
        print(f"  {j.get('name'):16s} {j.get('type'):9s} "
              f"{j.find('parent').get('link'):18s} -> {j.find('child').get('link')}")
    if missing:
        print(f"!! missing meshes ({len(missing)}):")
        for m in missing:
            print("   ", m)
    else:
        print("meshes  : all resolved")
    n_mesh = len(list(MESH_OUT.rglob("*.*")))
    print(f"mesh files on disk: {n_mesh}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
