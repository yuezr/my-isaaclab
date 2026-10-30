# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import FrameTransformer


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w.torch[:, 2] > minimal_height, 1.0, 0.0)

def j6_angle_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["panda_joint6"]),
    safe_low: float = 1.0,       # 下界：j6 < 1.0 开始罚
    safe_high: float = 3.0,      # 上界：j6 > 3.0 开始罚
    max_penalty: float = 2.0,
    minimal_height: float = 0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """惩罚 j6 偏离 [safe_low, safe_high] 区间。"""
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    j6 = robot.data.joint_pos.torch[:, robot_cfg.joint_ids[0]]
    is_lifted = obj.data.root_pos_w.torch[:, 2] > minimal_height
    
    dev_low = torch.clamp(safe_low - j6, min=0.0)
    dev_high = torch.clamp(j6 - safe_high, min=0.0)
    deviation = dev_low + dev_high
    return is_lifted.float() * deviation ** 2 * max_penalty
    
def object_pose_debug(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    print_every: int = 20,
    lifted_threshold: float = 0.05,
) -> torch.Tensor:
    """返回 0；每 print_every 次调用打印物体倾斜角 / 欧拉角 / lifted 统计。"""
    obj = env.scene[object_cfg.name]
    ee = env.scene[ee_frame_cfg.name]

    quat = obj.data.root_quat_w.torch          # (N, 4) w,x,y,z
    pos = obj.data.root_pos_w.torch            # (N, 3)
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    # --- 1) 倾斜角：z 轴与世界 z 轴夹角（0=竖直，90°=侧翻）---
    up_alignment = torch.clamp(1.0 - 2.0 * (x ** 2 + y ** 2), -1.0, 1.0)
    tilt_rad = torch.acos(up_alignment)
    tilt_deg = tilt_rad * 180.0 / torch.pi

    # --- 2) 欧拉角 ZYX（roll, pitch, yaw）---
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    # --- 3) EE 四元数（如果 FrameTransformer 不暴露 target_quat_w，把这段 try 去掉）---
    try:
        ee_quat = ee.data.target_quat_w.torch[..., 0, :]   # (N, 4)
        ew, ex, ey, ez = ee_quat[:, 0], ee_quat[:, 1], ee_quat[:, 2], ee_quat[:, 3]
        ee_up = torch.clamp(1.0 - 2.0 * (ex ** 2 + ey ** 2), -1.0, 1.0)
        ee_tilt_deg = torch.acos(ee_up) * 180.0 / torch.pi
    except Exception:
        ee_tilt_deg = None

    # --- 4) 打印 ---
    count = getattr(env, "_pose_debug_count", 0) + 1
    env._pose_debug_count = count

    if count % print_every == 0:
        lifted = pos[:, 2] > lifted_threshold
        n_lift = int(lifted.sum().item())

        if n_lift > 0:
            tilt_l = tilt_deg[lifted]
            roll_l = roll[lifted] * 180.0 / torch.pi
            pitch_l = pitch[lifted] * 180.0 / torch.pi
            yaw_l = yaw[lifted] * 180.0 / torch.pi
            stats = (
                f"  lifted_count = {n_lift}/{env.num_envs}\n"
                f"  tilt_deg  : mean={tilt_l.mean().item():6.2f}  "
                f"std={tilt_l.std().item():5.2f}  max={tilt_l.max().item():6.2f}\n"
                f"  roll_deg  : mean={roll_l.mean().item():+7.2f}  "
                f"std={roll_l.std().item():5.2f}\n"
                f"  pitch_deg : mean={pitch_l.mean().item():+7.2f}  "
                f"std={pitch_l.std().item():5.2f}\n"
                f"  yaw_deg   : mean={yaw_l.mean().item():+7.2f}  "
                f"std={yaw_l.std().item():5.2f}"
            )
        else:
            stats = f"  lifted_count = 0/{env.num_envs}  (无抬起样本)"

        # env 0 细节
        q0 = quat[0].detach().cpu().tolist()
        p0 = pos[0].detach().cpu().tolist()
        t0 = tilt_deg[0].item()
        r0 = roll[0].item() * 180.0 / torch.pi
        pch0 = pitch[0].item() * 180.0 / torch.pi
        y0 = yaw[0].item() * 180.0 / torch.pi
        ori_raw = (up_alignment[0] ** 2).item()   # object_orientation_reward 的原始值

        ee_line = ""
        if ee_tilt_deg is not None:
            ee_line = f"\n  ee_tilt_deg[0] = {ee_tilt_deg[0].item():6.2f}"

        print(
            f"\n[POSE_DEBUG step={count}]"
            f"\n  --- 抬起物体姿态统计 ---"
            f"\n{stats}"
            f"\n  --- env 0 细节 ---"
            f"\n  object_pos   = [{p0[0]:+.4f}, {p0[1]:+.4f}, {p0[2]:+.4f}]"
            f"\n  object_quat  = w={q0[0]:+.4f} x={q0[1]:+.4f} y={q0[2]:+.4f} z={q0[3]:+.4f}"
            f"\n  tilt_deg     = {t0:6.2f}"
            f"\n  roll_deg     = {r0:+7.2f}   pitch_deg = {pch0:+7.2f}   yaw_deg = {y0:+7.2f}"
            f"\n  orientation_reward_raw = {ori_raw:.4f}   (1=完全竖直, 0=侧翻)"
            f"{ee_line}"
        )

    return torch.zeros(env.num_envs, device=env.device)
    
def object_orientation_reward(
    env: ManagerBasedRLEnv,
    minimal_height: float = 0.075,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """奖励物体在被抬起时保持水平姿态（Z轴对齐）。"""
    obj = env.scene[object_cfg.name]
    is_lifted = obj.data.root_pos_w.torch[:, 2] > minimal_height
    quat = obj.data.root_quat_w.torch
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    # 倾斜角惩罚：完全水平为 1.0，翻转 90 度为 0.0
    up_alignment = torch.clamp(1.0 - 2.0 * (x**2 + y**2), min=0.0)
    return is_lifted.float() * (up_alignment ** 2)


def object_stability_penalty(
    env: ManagerBasedRLEnv,
    minimal_height: float = 0.075,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """惩罚物体的晃动（线速度和角速度）。"""
    obj = env.scene[object_cfg.name]
    is_lifted = obj.data.root_pos_w.torch[:, 2] > minimal_height

    # 获取物体的线速度和角速度
    lin_vel = torch.linalg.norm(obj.data.root_lin_vel_w.torch, dim=1)
    ang_vel = torch.linalg.norm(obj.data.root_ang_vel_w.torch, dim=1)

    # 速度越快，惩罚越大
    return is_lifted.float() * (lin_vel ** 2 + ang_vel ** 2)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w.torch
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w.torch[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.linalg.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


def joint_limit_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"]),
    safe_margin: float = 0.5,
    max_penalty: float = 1.0,
) -> torch.Tensor:
    """递进式关节限位惩罚。

    设计：
    - 距离限位 > safe_margin：惩罚 0
    - 距离从 safe_margin 到 0：二次递增
    - 距离 = 0（贴限位）：惩罚 = max_penalty
    - 距离 < 0（越界）：惩罚 > max_penalty
    """
    robot = env.scene[robot_cfg.name]

    # 只取手臂关节（排除手指），形状: (num_envs, 7)
    joint_pos = robot.data.joint_pos.torch[:, robot_cfg.joint_ids]

    # joint_limits 形状: (num_envs, num_joints, 2)
    # 需要在关节维上切片，而不是第一维
    joint_limits_all = robot.data.joint_limits.torch          # (num_envs, num_joints, 2)
    joint_limits = joint_limits_all[:, robot_cfg.joint_ids, :]  # (num_envs, 7, 2)

    lower = joint_limits[:, :, 0]   # (num_envs, 7)
    upper = joint_limits[:, :, 1]   # (num_envs, 7)

    # 每个关节到最近限位的距离
    dist_to_lower = joint_pos - lower          # (num_envs, 7)
    dist_to_upper = upper - joint_pos          # (num_envs, 7)
    dist_to_limit = torch.minimum(dist_to_lower, dist_to_upper)

    # ratio: 0（远离限位）→ 1（贴限位）→ >1（越界）
    ratio = torch.clamp(1.0 - dist_to_limit / safe_margin, min=0.0)

    # 二次递进
    penalty_per_joint = max_penalty * ratio ** 2  # (num_envs, 7)

    return penalty_per_joint.sum(dim=1)           # (num_envs,)


def grip_object(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    distance_threshold: float = 0.05,
    closed_target_tolerance: float = 1e-4,
    far_close_penalty: float = 0.01,
    require_physical_close: bool = False,
) -> torch.Tensor:
    """Reward gripper closing near the object.

    三态（阶段 4a 默认）：
    - 靠近且闭合 -> +1.0
    - 远离且闭合 -> -far_close_penalty（默认 -0.01，不是 -0.1）
    - 其他情况   ->  0.0

    require_physical_close=False 时只看指令闭合 + 距离；
    True 时额外要求 finger_pos < 0.035（物理闭合），用于后期收紧。
    """
    obj = env.scene[object_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    robot = env.scene[robot_cfg.name]

    cube_pos_w = obj.data.root_pos_w.torch
    ee_pos_w = ee_frame.data.target_pos_w.torch[..., 0, :]
    distance = torch.linalg.norm(cube_pos_w - ee_pos_w, dim=1)

    finger_targets = robot.data.joint_pos_target.torch[:, robot_cfg.joint_ids]
    finger_pos = robot.data.joint_pos.torch[:, robot_cfg.joint_ids]

    # 1. 指令判定：目标必须趋向于闭合
    close_command = torch.all(torch.abs(finger_targets) <= closed_target_tolerance, dim=1)
    # 2. 物理判定：手指实际位置必须合拢（正常张开 0.04，夹住物体后会退回到 0.02 左右）
    #    阶段 4a 默认不启用，保留给后期收紧用。
    finger_actually_closed = torch.all((finger_pos > 0.005) & (finger_pos < 0.035), dim=1)

    near_object = distance < distance_threshold

    if require_physical_close:
        valid_close = close_command & finger_actually_closed & near_object
    else:
        valid_close = close_command & near_object

    reward = torch.zeros_like(distance)

    # 有效闭合 + 靠近 -> +1
    reward = torch.where(
        valid_close,
        torch.ones_like(distance),
        reward,
    )

    # 指令闭合 + 远离物体 -> 轻微惩罚（默认 -0.01，不再把闭合期望打成负值）
    reward = torch.where(
        close_command & ~near_object,
        -far_close_penalty * torch.ones_like(distance),
        reward,
    )

    # 调试打印：每 20 次调用打印一次
    debug_count = getattr(env, "_grip_debug_count", 0) + 1
    env._grip_debug_count = debug_count

    if debug_count % 20 == 0:
        min_idx = torch.argmin(distance)

        near_count = int(near_object.sum().item())
        close_count = int(close_command.sum().item())
        valid_count = int(valid_close.sum().item())
        penalty_count = int((close_command & ~near_object).sum().item())

        # ===== 读取所有关节的位置和速度 =====
        all_joint_pos = robot.data.joint_pos.torch[0].detach().cpu().tolist()
        all_joint_vel = robot.data.joint_vel.torch[0].detach().cpu().tolist()

        # 关节名（顺序对应 Franka Panda）
        joint_names = [
            "j1_base_yaw",     # joint1
            "j2_shoulder",     # joint2  ← 肩部
            "j3_upper_arm",    # joint3
            "j4_elbow",        # joint4  ← 肘部
            "j5_forearm",      # joint5
            "j6_wrist_pitch",  # joint6  ← 手腕
            "j7_wrist_yaw",    # joint7
            "f1_left",         # finger1
            "f2_right",        # finger2
        ]

        # 物体高度（判断是否抬起）；阈值与 lifting_object 门槛保持一致（0.05）
        obj_z = obj.data.root_pos_w.torch[0, 2].item()
        is_lifted_env0 = obj_z > 0.05

        # 末端执行器世界坐标
        ee_pos = ee_frame.data.target_pos_w.torch[0, 0, :].detach().cpu().tolist()

        # 打印关节位置和速度
        joint_pos_str = "\n".join(
            f"    {name:18s} pos={pos:+.4f} rad   vel={vel:+.4f} rad/s"
            for name, pos, vel in zip(joint_names, all_joint_pos, all_joint_vel)
        )

        print(
            f"\n[GRIP_DEBUG step={debug_count}]"
            f"\n  --- 抓取指标 ---"
            f"\n  min_distance = {distance.min().item():.4f} m"
            f"\n  threshold    = {distance_threshold:.4f} m"
            f"\n  far_close_penalty = {far_close_penalty:.4f}"
            f"\n  require_physical_close = {require_physical_close}"
            f"\n  near_count   = {near_count}/{env.num_envs}"
            f"\n  close_count  = {close_count}/{env.num_envs}"
            f"\n  valid_close  = {valid_count}/{env.num_envs}"
            f"\n  penalty      = {penalty_count}/{env.num_envs}"
            f"\n  closest_env  = {min_idx.item()}"
            f"\n  closest_distance = {distance[min_idx].item():.4f} m"
            f"\n  closest_reward = {reward[min_idx].item():.1f}"
            f"\n  --- env 0 状态 ---"
            f"\n  object_z     = {obj_z:.4f} m   (lifted={is_lifted_env0})"
            f"\n  ee_pos_w     = [{ee_pos[0]:+.4f}, {ee_pos[1]:+.4f}, {ee_pos[2]:+.4f}]"
            f"\n  --- env 0 关节状态 ---"
            f"\n{joint_pos_str}"
            f"\n  --- 手指目标位置 ---"
            f"\n  finger_targets = {finger_targets[:4].detach().cpu().tolist()}"
            f"\n  finger_pos     = {finger_pos[:4].detach().cpu().tolist()}"
        )

    return reward


def ee_above_object_windowed(
    env,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    xy_sigma: float = 0.03,
    z_center: float = 0.05,
    z_sigma: float = 0.05,
    z_min: float = 0.0,
    z_max: float = 0.15,
) -> torch.Tensor:
    """只奖励“在物体正上方 5cm 附近”的接近，不奖励高空悬停。"""
    obj = env.scene[object_cfg.name]
    ee = env.scene[ee_frame_cfg.name]
    obj_pos = obj.data.root_pos_w.torch
    ee_pos = ee.data.target_pos_w.torch[..., 0, :]

    xy_dist = torch.linalg.norm(obj_pos[:, :2] - ee_pos[:, :2], dim=1)
    xy_score = torch.exp(-(xy_dist / xy_sigma) ** 2)

    z_above = ee_pos[:, 2] - (obj_pos[:, 2] + 0.03)
    z_score = torch.exp(-((z_above - z_center) / z_sigma) ** 2)

    in_window = ((z_above > z_min) & (z_above < z_max)).float()
    return xy_score * z_score * in_window


def object_is_lifted_stable(
    env: ManagerBasedRLEnv,
    minimal_height: float = 0.075,
    max_lin_vel: float = 0.8,
    max_ang_vel: float = 3.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """判断物体是否被"稳定"提起。

    与 object_is_lifted 的区别：
    - object_is_lifted: 只看物体高度 > minimal_height
    - object_is_lifted_stable: 还要求物体线速度和角速度足够小

    用于抑制"用 j5/j6/j7 旋转甩起物体"的偷懒行为：
    - 正常提起：物体被夹爪稳定抓住，速度缓慢 → 给 +1
    - 旋转甩起：物体在空中翻滚，速度很快 → 不给分

    参数:
        minimal_height: 物体被提起的最小高度阈值 (m)
        max_lin_vel:    物体线速度上限 (m/s)，超过则视为"甩起"
        max_ang_vel:    物体角速度上限 (rad/s)，超过则视为"翻滚"
        object_cfg:     物体配置
    """
    obj = env.scene[object_cfg.name]

    # 高度条件
    obj_z = obj.data.root_pos_w.torch[:, 2]
    is_lifted = obj_z > minimal_height

    # 速度条件（世界坐标系）
    obj_lin_vel = obj.data.root_lin_vel_w.torch
    obj_ang_vel = obj.data.root_ang_vel_w.torch
    lin_speed = torch.linalg.norm(obj_lin_vel, dim=1)
    ang_speed = torch.linalg.norm(obj_ang_vel, dim=1)

    # 稳定性判断
    is_stable = (lin_speed < max_lin_vel) & (ang_speed < max_ang_vel)

    return (is_lifted & is_stable).float()


class object_goal_distance(ManagerTermBase):
    """Reward the agent for tracking the object-to-goal pose using a tanh kernel.

    If ``success_threshold`` is provided in the term params, this also tracks per-episode
    success (sticky binary: object ever within ``success_threshold`` of the commanded goal
    while lifted above ``minimal_height``) and logs the mean across environments under
    ``Metrics/success_rate`` on reset.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._track_success = cfg.params.get("success_threshold") is not None
        if self._track_success:
            self._succeeded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor):
        if self._track_success:
            self._env.extras.setdefault("log", {})["Metrics/success_rate"] = (
                self._succeeded[env_ids].float().mean().item()
            )
            self._succeeded[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        minimal_height: float,
        command_name: str,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        success_threshold: float | None = None,
    ) -> torch.Tensor:
        robot: RigidObject = env.scene[robot_cfg.name]
        obj: RigidObject = env.scene[object_cfg.name]
        command = env.command_manager.get_command(command_name)
        des_pos_w, _ = combine_frame_transforms(
            robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, command[:, :3]
        )
        object_pos_w = obj.data.root_pos_w.torch
        distance = torch.linalg.norm(des_pos_w - object_pos_w, dim=1)
        is_lifted = object_pos_w[:, 2] > minimal_height
        if success_threshold is not None:
            self._succeeded |= is_lifted & (distance < success_threshold)
        return is_lifted.float() * (1 - torch.tanh(distance / std))
