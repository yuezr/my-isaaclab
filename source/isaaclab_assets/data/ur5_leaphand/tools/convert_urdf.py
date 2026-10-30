#!/usr/bin/env python3
"""Convert URDF -> USD and leave the articulation root on the root rigid body.

Isaac Lab then welds it to the world itself (`ArticulationRootPropertiesCfg(fix_root_link=True)`),
which follows omni.physx's own recipe: joint on the body, articulation root API moved to the parent.
"""
from __future__ import annotations

import sys
from pathlib import Path

from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": True})
simulation_app = app_launcher.app

from pxr import Usd, UsdPhysics  # noqa: E402

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]  # <ext>/data/ur5_leaphand
URDF = ROOT / "urdf" / "ur5_leaphand_right.urdf"
USD_DIR = ROOT / "usd"


def main() -> int:
    cfg = UrdfConverterCfg(
        asset_path=str(URDF),
        usd_dir=str(USD_DIR),
        usd_file_name="ur5_leaphand_right.usd",
        fix_base=False,
        merge_fixed_joints=False,
        self_collision=True,
        collision_from_visuals=False,
        collision_type="Convex Hull",
        joint_drive=UrdfConverterCfg.JointDriveCfg(target_type="none"),
        run_asset_transformer=False,
        force_usd_conversion=True,
    )
    usd_path = Path(UrdfConverter(cfg).usd_path)
    print("USD_PATH:", usd_path, flush=True)

    stage = Usd.Stage.Open(str(usd_path))
    root = stage.GetDefaultPrim()
    body = None
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            body = prim
            break
    if body is None:
        print("ERROR: no rigid body", flush=True)
        simulation_app.close()
        return 1

    for prim in stage.Traverse():
        # importer stamps this on the root body; Isaac Lab would see two articulation roots
        if prim.HasAPI("NewtonArticulationRootAPI"):
            prim.RemoveAPI("NewtonArticulationRootAPI")
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI) and prim != body:
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    if not body.HasAPI(UsdPhysics.ArticulationRootAPI):
        UsdPhysics.ArticulationRootAPI.Apply(body)
    print("ARTICULATION_ROOT:", body.GetPath(), flush=True)

    stage.GetRootLayer().Save()
    print("SIZE_MB:", round(usd_path.stat().st_size / 1e6, 2), flush=True)
    simulation_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
