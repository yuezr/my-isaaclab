# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spawn the UR5 + LEAP Hand asset and exercise the joints.

.. code-block:: bash

    ./isaaclab.sh -p scripts/demos/ur5_leaphand.py --headless --enable_cameras

The base of this articulation is free (see :mod:`isaaclab_assets.robots.ur5_leaphand`), so the demo
writes the root pose back every step to hold the robot in place.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="UR5 + LEAP Hand demo")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--out_dir", type=str, default=str(Path.cwd() / "ur5_leaphand_shots"))
parser.add_argument("--steps", type=int, default=200, help="control steps per phase")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.sensors.camera import Camera, CameraCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab_assets.robots.ur5_leaphand import (  # noqa: E402
    ARM_READY_POSE,
    HAND_GRASP_POSE,
    HAND_OPEN_POSE,
    UR5_LEAPHAND_CFG,
)

try:
    from PIL import Image

    HAVE_PIL = True
except Exception:  # noqa: BLE001
    HAVE_PIL = False


def design_scene():
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/ground", cfg)

    cfg = sim_utils.DomeLightCfg(intensity=2200.0, color=(0.85, 0.85, 0.85))
    cfg.func("/World/Light", cfg)

    sim_utils.create_prim("/World/Robot", "Xform")

    robot_cfg = UR5_LEAPHAND_CFG.copy()
    robot_cfg.prim_path = "/World/Robot/ur5_leaphand"
    robot = Articulation(cfg=robot_cfg)

    camera_cfg = CameraCfg(
        prim_path="/World/Camera",
        update_period=0,
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
    )
    camera = Camera(cfg=camera_cfg)
    return robot, camera


def save_rgb(camera: Camera, path: Path) -> None:
    rgb = camera.data.output["rgb"]
    arr = rgb[0, ..., :3].detach().cpu().numpy().astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    if HAVE_PIL:
        Image.fromarray(arr).save(path)
        print(f"[shot] {path}")
    else:
        ppm = path.with_suffix(".ppm")
        with open(ppm, "wb") as f:
            f.write(f"P6\n{arr.shape[1]} {arr.shape[0]}\n255\n".encode())
            f.write(arr.tobytes())
        print(f"[shot] {ppm} (no PIL, wrote ppm)")


def main() -> int:
    sim = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args_cli.device or "cuda:0"))
    sim.set_camera_view([2.0, 1.9, 1.5], [0.35, 0.0, 0.55])

    robot, camera = design_scene()
    sim.reset()

    print("=" * 70)
    print(f"num_dof    : {robot.num_joints}")
    print(f"num_bodies : {len(robot.body_names)}")
    print(f"joint_names: {robot.joint_names}")
    print("=" * 70)

    dt = sim.get_physics_dt()
    name_to_idx = {n: i for i, n in enumerate(robot.joint_names)}

    def pose_tensor(pose: dict[str, float]) -> torch.Tensor:
        target = robot.data.default_joint_pos.torch.clone()
        for name, val in pose.items():
            target[0, name_to_idx[name]] = val
        return target

    root_pose = robot.data.default_root_pose.torch.clone()
    root_vel = robot.data.default_root_vel.torch.clone()

    def run(pose: dict[str, float], steps: int, tag: str) -> None:
        target = pose_tensor(pose)
        for _ in range(steps):
            # the asset is free-base: hold it in place
            robot.write_root_pose_to_sim_index(root_pose=root_pose)
            robot.write_root_velocity_to_sim_index(root_velocity=root_vel)
            robot.set_joint_position_target_index(target=target)
            robot.write_data_to_sim()
            sim.step()
            robot.update(dt)
            camera.update(dt)
        q = robot.data.joint_pos.torch[0]
        err = (q - target[0]).abs()
        print(f"[{tag}] max|q-target|={float(err.max()):.3f} rad  max|qd|={float(robot.data.joint_vel.torch[0].abs().max()):.3f}")
        for name, e in zip(robot.joint_names, err.tolist()):
            if e > 0.15:
                print(f"    {name:20s} err={e:.3f}")

    robot.write_joint_position_to_sim_index(position=robot.data.default_joint_pos.torch)

    run({**ARM_READY_POSE, **HAND_OPEN_POSE}, args_cli.steps, "open")

    # frame the whole robot
    body_pos = robot.data.body_pos_w.torch[0]
    center = body_pos.mean(dim=0)
    radius = float((body_pos - center).norm(dim=1).max())
    offset = torch.tensor([1.0, 1.0, 0.55], device=body_pos.device, dtype=body_pos.dtype)
    offset = offset / offset.norm()
    camera.set_world_poses_from_view(
        (center + offset * max(1.2, radius * 3.0)).unsqueeze(0), center.unsqueeze(0)
    )
    for _ in range(5):
        sim.step()
        camera.update(dt)
    save_rgb(camera, Path(args_cli.out_dir) / "01_open.png")

    run({**ARM_READY_POSE, **HAND_GRASP_POSE}, args_cli.steps, "grasp")
    for _ in range(5):
        sim.step()
        camera.update(dt)
    save_rgb(camera, Path(args_cli.out_dir) / "02_grasp.png")

    # close-up on the hand
    names = robot.body_names
    palm = body_pos[names.index("leap_palm_lower")] if "leap_palm_lower" in names else center
    camera.set_world_poses_from_view(
        (palm + torch.tensor([0.45, 0.45, 0.15], device=palm.device)).unsqueeze(0), palm.unsqueeze(0)
    )
    for _ in range(5):
        sim.step()
        camera.update(dt)
    save_rgb(camera, Path(args_cli.out_dir) / "03_hand_closeup.png")

    simulation_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
