# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab_physx.assets import DeformableObjectCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

from . import mdp

##
# Scene definition
##


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING
    # target object: will be populated by agent env cfg
    object: RigidObjectCfg | DeformableObjectCfg = MISSING

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0, 0, 0.707, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.6), pos_y=(-0.25, 0.25), pos_z=(0.25, 0.5), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP（阶段 4a：先只修“从上方接近并抓取”，姿态/稳定暂关）。"""

    # ===== 接近：恢复 1.0，这是把策略从高空拉下来的关键 =====
    reaching_object = RewTerm(
        func=mdp.object_ee_distance, params={"std": 0.1}, weight=1.0,
    )

    # ===== 上方接近引导：改用 windowed 版，权重 1.0，不再奖励高空悬停 =====
    # 注意：原 cfg 调的是 mdp.ee_above_object_reward，该函数在 rewards.py 中不存在，
    # 这里必须改成 mdp.ee_above_object_windowed。
    ee_above_object = RewTerm(
        func=mdp.ee_above_object_windowed,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "xy_sigma": 0.03,
            "z_center": 0.02,   # 0.05 → 0.02：把“理想高度”从 5cm 降到 2cm
            "z_sigma": 0.05,
            "z_min": 0.0,
            "z_max": 0.06,      # 0.15 → 0.06：窗口上限收紧，不再奖励高空悬停
        },
        weight=1.0,
    )

    # ===== 抓取：三态，放宽距离阈值 + 极小远离闭合惩罚；物理闭合开关默认关闭 =====
    grip_object = RewTerm(
        func=mdp.grip_object,
        params={
            "robot_cfg": SceneEntityCfg(
                "robot",
                joint_names=["panda_finger_joint1", "panda_finger_joint2"],
            ),
            "distance_threshold": 0.02,        # 0.05 → 0.02：真正接近才给 +1
            "closed_target_tolerance": 1e-4,
            "far_close_penalty": 0.01,
            "require_physical_close": True,    # False → True：必须物理夹住
        },
        weight=3.0,
    )

    object_pose_debug = RewTerm(
        func=mdp.object_pose_debug,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "print_every": 20,
            "lifted_threshold": 0.05,
        },
        weight=1e-10,
    )
    joint_limit_penalty = RewTerm(
        func=mdp.joint_limit_penalty,
        params={
            "robot_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"]),
            "safe_margin": 0.5,       # 宽缓冲区
            "max_penalty": 1.0,       # 贴限位时单关节惩罚 = 1.0
        },
        weight=-0.5,                  # 小权重起步
    )

    j6_angle_penalty = RewTerm(
        func=mdp.j6_angle_penalty,
        params={
            "robot_cfg": SceneEntityCfg("robot", joint_names=["panda_joint6"]),
            "safe_low": 1.0,
            "safe_high": 3.5,      # 3.0 → 3.5，放行 j6=3.36 的抓取姿态
            "max_penalty": 2.0,
            "minimal_height": 0.05,
        },
        weight=-1.0,
    )

    # ===== 抬起：门槛降到 0.05，降低早期探索难度 =====
    lifting_object = RewTerm(
        func=mdp.object_is_lifted,
        params={"minimal_height": 0.05},
        weight=12.0,
    )

    # ===== 姿态奖励：阶段 4a 先关，等抓取/抬起稳定后再开 =====
    object_orientation = RewTerm(
        func=mdp.object_orientation_reward,
        params={"minimal_height": 0.05},
        weight=6.0,
    )

    # ===== 防摇晃惩罚：阶段 4a 先关，等抓取/抬起稳定后再开 =====
    object_stability_penalty = RewTerm(
        func=mdp.object_stability_penalty,
        params={"minimal_height": 0.05},
        weight=-0.1,
    )

    # ===== 粗粒度对准 =====
    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={
            "std": 0.3,
            "minimal_height": 0.075,
            "command_name": "object_pose",
            "success_threshold": 0.05,
        },
        weight=16.0,
    )

    # ===== 细粒度对准 =====
    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={
            "std": 0.03,
            "minimal_height": 0.075,
            "command_name": "object_pose",
        },
        weight=8.0,
    )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-3, "num_steps": 10000}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-3, "num_steps": 10000}
    )


##
# Environment configuration
##


@configclass
class LiftEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physics = PhysxCfg(
            bounce_threshold_velocity=0.01,
            gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
            gpu_total_aggregate_pairs_capacity=16 * 1024,
            friction_correlation_distance=0.00625,
        )