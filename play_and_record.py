#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Franka Cube 策略执行与奖励分解调试。

环境：
- Isaac Lab 3.0.0-beta2.patch1
- Isaac Sim 6.0.1
- RSL-RL

功能：
1. 默认运行 250 步。
2. 每 10 步打印一次机器人状态和奖励组成。
3. 每步记录 7 个关节角度、EE/Object 高度及总奖励。
4. 结束时保存 PNG 曲线图。
5. 不生成 CSV，不逐步打印调试信息。
"""

import argparse
import os
import traceback
import faulthandler
import importlib.metadata as metadata
from datetime import datetime
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from isaaclab.app import AppLauncher
from isaaclab.utils.math import combine_frame_transforms

# ============================================================
# 1. 命令行参数
# ============================================================

# 根据脚本启动时间自动生成图片文件名
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

DEFAULT_OUTPUT = os.path.expanduser(
    f"~/pro/IsaacLab-3.0.0-beta2.patch1/"
    f"joint_trajectory_{RUN_TIMESTAMP}.png"
)

parser = argparse.ArgumentParser(
    description="Franka Cube 策略执行及奖励分解调试"
)

parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Lift-Cube-Franka-Play-v0",
    help="Gym 环境任务名称",
)

parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help="并行环境数量，只记录环境 0",
)

parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="训练得到的 checkpoint 文件路径",
)

parser.add_argument(
    "--steps",
    type=int,
    default=250,
    help="最多执行多少步，默认 250",
)

parser.add_argument(
    "--debug_interval",
    type=int,
    default=10,
    help="每多少步输出一次调试信息，默认 10",
)

parser.add_argument(
    "--output",
    type=str,
    default=DEFAULT_OUTPUT,
    help="输出曲线图片路径",
)

AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()


# ============================================================
# 2. 检查参数
# ============================================================

args_cli.checkpoint = os.path.abspath(
    os.path.expanduser(args_cli.checkpoint)
)

args_cli.output = os.path.abspath(
    os.path.expanduser(args_cli.output)
)

if not os.path.isfile(args_cli.checkpoint):
    parser.error(
        f"找不到 checkpoint 文件：{args_cli.checkpoint}"
    )

if args_cli.steps <= 0:
    parser.error("--steps 必须大于 0")

if args_cli.num_envs <= 0:
    parser.error("--num_envs 必须大于 0")

if args_cli.debug_interval <= 0:
    parser.error("--debug_interval 必须大于 0")

os.makedirs(
    os.path.dirname(args_cli.output),
    exist_ok=True,
)

print("=" * 90, flush=True)
print("[INFO] Franka Cube 策略与奖励分解调试", flush=True)
print(f"[INFO] 当前目录：{os.getcwd()}", flush=True)
print(f"[INFO] Checkpoint：{args_cli.checkpoint}", flush=True)
print(f"[INFO] 计划步数：{args_cli.steps}", flush=True)
print(f"[INFO] 调试输出间隔：{args_cli.debug_interval} 步", flush=True)
print(f"[INFO] 图片路径：{args_cli.output}", flush=True)
print("=" * 90, flush=True)


# ============================================================
# 3. 初始化数据容器
# ============================================================

joint_names = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
]

records = {
    name: [] for name in joint_names
}

object_z_list = []
ee_z_list = []
target_z_list = []
reward_list = []

env = None
simulation_app = None
reward_manager = None
reward_dt = None
plot_saved = False


# ============================================================
# 4. 绘图函数
# ============================================================

def save_trajectory_plot(
    output_path,
    joint_names,
    records,
    object_z_list,
    ee_z_list,
    target_z_list,
    reward_list,
):
    """保存关节角度、物体高度和总奖励曲线。"""

    if not reward_list:
        print(
            "[WARN] 没有已记录的奖励数据，跳过绘图。",
            flush=True,
        )
        return False

    output_path = os.path.abspath(
        os.path.expanduser(output_path)
    )

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True,
    )

    fig = None

    try:
        fig, axes = plt.subplots(
            3,
            1,
            figsize=(14, 12),
            sharex=True,
        )

        # ----------------------------------------------------
        # 子图 1：7 个关节角度
        # ----------------------------------------------------

        for name in joint_names:
            values = records[name]

            if values:
                axes[0].plot(
                    range(1, len(values) + 1),
                    values,
                    label=name,
                    linewidth=1.2,
                )

        axes[0].set_ylabel("Joint angle (rad)")
        axes[0].set_title(
            "Franka 7-Joint Angles During Policy Execution"
        )
        axes[0].legend(
            loc="upper right",
            ncol=4,
            fontsize=8,
        )
        axes[0].grid(True, alpha=0.3)

        # ----------------------------------------------------
        # 子图 2：Object 和 EE 高度
        # ----------------------------------------------------

        if object_z_list:
            axes[1].plot(
                range(1, len(object_z_list) + 1),
                object_z_list,
                label="Object Z",
                color="orange",
                linewidth=1.5,
            )
        if target_z_list:
            axes[1].plot(
                range(1, len(target_z_list) + 1),
                target_z_list,
                label="Command Target Z",
                color="purple",
                linestyle=":",
                linewidth=1.8,
            )
        if ee_z_list:
            axes[1].plot(
                range(1, len(ee_z_list) + 1),
                ee_z_list,
                label="End-Effector Target Z",
                color="blue",
                linewidth=1.5,
            )

        axes[1].axhline(
            y=0.075,
            color="red",
            linestyle="--",
            label="Lift threshold (0.075 m)",
        )

        axes[1].set_ylabel("Height (m)")
        axes[1].set_title(
            "Object and End-Effector Height"
        )
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # ----------------------------------------------------
        # 子图 3：总奖励
        # ----------------------------------------------------

        axes[2].plot(
            range(1, len(reward_list) + 1),
            reward_list,
            color="green",
            linewidth=1.2,
        )

        axes[2].set_ylabel("Environment reward")
        axes[2].set_xlabel("Simulation step")
        axes[2].set_title(
            "Total Reward During Policy Execution"
        )
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        print("[INFO] 正在保存曲线图...", flush=True)

        fig.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight",
            format="png",
        )

        if (
            os.path.isfile(output_path)
            and os.path.getsize(output_path) > 0
        ):
            print(
                f"[SUCCESS] 曲线图已保存：{output_path}",
                flush=True,
            )
            print(
                f"[SUCCESS] 文件大小："
                f"{os.path.getsize(output_path):,} bytes",
                flush=True,
            )
            return True

        print(
            f"[ERROR] 未找到有效图片：{output_path}",
            flush=True,
        )
        return False

    finally:
        if fig is not None:
            plt.close(fig)


# ============================================================
# 5. 奖励分解调试函数
# ============================================================

def print_reward_debug(
    step,
    total_reward,
    manager,
    step_dt,
):
    """
    打印指定环境当前步的逐项奖励。

    RewardManager 中的每项缓存值：
        weighted_value = raw_value * weight

    env.step() 返回的总奖励：
        total_reward = sum(weighted_value * step_dt)

    注意：
    权重为 0 的项无法根据缓存值还原原始函数值。
    """

    print("", flush=True)
    print("#" * 110, flush=True)

    print(
        f"[REWARD_DEBUG] Step {step}/{args_cli.steps}",
        flush=True,
    )

    print(
        f"[REWARD_DEBUG] env.step() 总奖励 = "
        f"{total_reward:+.8f}",
        flush=True,
    )

    print(
        f"[REWARD_DEBUG] 环境步长 step_dt = "
        f"{step_dt:.8f} s",
        flush=True,
    )

    # 获取环境 0 的当前奖励项值。
    # 这里的值已经乘过 weight，但尚未乘 step_dt。
    active_values = manager.get_active_iterable_terms(0)

    weighted_value_by_name = {
        name: float(values[0])
        for name, values in active_values
    }

    reconstructed_total = 0.0

    print("-" * 110, flush=True)
    print(
        f"{'奖励项':<30}"
        f"{'原始值':>14}"
        f"{'权重':>14}"
        f"{'加权值':>16}"
        f"{'实际贡献':>18}",
        flush=True,
    )
    print("-" * 110, flush=True)

    for term_name in manager.active_terms:

        term_cfg = manager.get_term_cfg(term_name)
        weight = float(term_cfg.weight)

        weighted_value = weighted_value_by_name.get(
            term_name,
            0.0,
        )

        # 还原奖励函数的原始返回值
        if abs(weight) > 1.0e-12:
            raw_value = weighted_value / weight
            raw_text = f"{raw_value:+.6f}"
        else:
            raw_text = "N/A"

        # 当前奖励项实际加到环境总奖励中的数值
        actual_contribution = weighted_value * step_dt

        reconstructed_total += actual_contribution

        print(
            f"{term_name:<30}"
            f"{raw_text:>14}"
            f"{weight:>+14.6f}"
            f"{weighted_value:>+16.8f}"
            f"{actual_contribution:>+18.8f}",
            flush=True,
        )

    difference = total_reward - reconstructed_total

    print("-" * 110, flush=True)

    print(
        f"{'各项实际贡献之和':<58}"
        f"{reconstructed_total:>+18.8f}",
        flush=True,
    )

    print(
        f"{'env.step() 返回总奖励':<58}"
        f"{total_reward:>+18.8f}",
        flush=True,
    )

    print(
        f"{'总奖励差值':<58}"
        f"{difference:>+18.8e}",
        flush=True,
    )

    if abs(difference) > 1.0e-5:
        print(
            "[WARN] 分项求和与环境总奖励存在差异，"
            "请检查环境步长、包装器和重置行为。",
            flush=True,
        )

    print("#" * 110, flush=True)


# ============================================================
# 6. 启动 Isaac Sim
# ============================================================

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


try:

    # ========================================================
    # 7. 导入 Isaac Lab 和 RSL-RL 模块
    # ========================================================

    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401

    from isaaclab_tasks.utils.parse_cfg import (
        load_cfg_from_registry,
    )

    from isaaclab_rl.rsl_rl import (
        RslRlVecEnvWrapper,
        handle_deprecated_rsl_rl_cfg,
    )

    from rsl_rl.runners import OnPolicyRunner

    print("[INFO] Isaac Sim 启动成功", flush=True)

    # ========================================================
    # 8. 加载配置
    # ========================================================

    print("[INFO] 加载环境配置...", flush=True)

    env_cfg = load_cfg_from_registry(
        args_cli.task,
        "env_cfg_entry_point",
    )

    env_cfg.scene.num_envs = args_cli.num_envs

    agent_cfg = load_cfg_from_registry(
        args_cli.task,
        "rsl_rl_cfg_entry_point",
    )

    installed_version = metadata.version("rsl-rl-lib")

    print(
        f"[INFO] rsl-rl-lib 版本：{installed_version}",
        flush=True,
    )

    agent_cfg = handle_deprecated_rsl_rl_cfg(
        agent_cfg,
        installed_version,
    )

    print("[INFO] RSL-RL 配置处理完成", flush=True)

    # ========================================================
    # 9. 创建环境并初始化奖励管理器
    # ========================================================

    print("[INFO] 正在创建环境...", flush=True)

    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
    )

    env = RslRlVecEnvWrapper(
        env,
        clip_actions=getattr(
            agent_cfg,
            "clip_actions",
            None,
        ),
    )

    unwrapped_env = env.unwrapped

    reward_manager = unwrapped_env.reward_manager

    # 这是环境一次完整 RL step 的时间间隔，
    # 通常等于 sim.dt * decimation。
    reward_dt = float(unwrapped_env.step_dt)

    print("[INFO] 环境创建成功", flush=True)
    print(
        f"[INFO] RL 环境步长：{reward_dt:.8f} s",
        flush=True,
    )

    print("[INFO] 当前任务启用的奖励项和权重：", flush=True)

    for term_name in reward_manager.active_terms:
        term_cfg = reward_manager.get_term_cfg(term_name)

        print(
            f"    {term_name:<35}"
            f"weight={float(term_cfg.weight):+.8f}",
            flush=True,
        )

    # ========================================================
    # 10. 初始化 Runner 并加载 checkpoint
    # ========================================================

    print("[INFO] 初始化 RSL-RL Runner...", flush=True)

    agent_dict = agent_cfg.to_dict()

    agent_dict["obs_groups"] = {
        "actor": ["policy"],
        "critic": ["policy"],
    }

    print(
        f"[DEBUG] Runner 类：{agent_cfg.class_name}",
        flush=True,
    )

    print(
        f"[DEBUG] 使用设备：{agent_cfg.device}",
        flush=True,
    )

    print(
        f"[DEBUG] actor 配置键："
        f"{list(agent_dict.get('actor', {}).keys())}",
        flush=True,
    )

    print(
        f"[DEBUG] critic 配置键："
        f"{list(agent_dict.get('critic', {}).keys())}",
        flush=True,
    )

    faulthandler.dump_traceback_later(
        30,
        repeat=False,
    )

    try:
        runner = OnPolicyRunner(
            env,
            agent_dict,
            log_dir=None,
            device=agent_cfg.device,
        )
    finally:
        faulthandler.cancel_dump_traceback_later()

    print("[INFO] Runner 初始化成功", flush=True)

    print(
        f"[INFO] 加载 checkpoint：{args_cli.checkpoint}",
        flush=True,
    )

    runner.load(args_cli.checkpoint)

    print("[INFO] Checkpoint 加载成功", flush=True)

    policy = runner.get_inference_policy(
        device=unwrapped_env.device,
    )

    print("[INFO] 策略加载成功", flush=True)

    # ========================================================
    # 11. 获取机器人、方块和末端执行器
    # ========================================================

    robot = unwrapped_env.scene["robot"]
    obj = unwrapped_env.scene["object"]
    ee = unwrapped_env.scene["ee_frame"]

    joint_ids, found_joint_names = robot.find_joints(
        joint_names,
        preserve_order=True,
    )

    if len(joint_ids) != 7:
        raise RuntimeError(
            "没有找到全部 7 个机械臂关节。\n"
            f"期望：{joint_names}\n"
            f"实际：{found_joint_names}"
        )

    print("[INFO] 找到的 7 个关节：", flush=True)

    for name in found_joint_names:
        print(f"    {name}", flush=True)

    # ========================================================
    # 12. 重置环境
    # ========================================================

    print("[INFO] 重置环境...", flush=True)

    obs, _ = env.reset()

    print(
        f"[INFO] 开始执行，共 {args_cli.steps} 步；"
        f"每 {args_cli.debug_interval} 步输出一次调试信息。",
        flush=True,
    )

    # ========================================================
    # 13. 执行策略并记录数据
    # ========================================================

    with torch.inference_mode():

        for step in range(args_cli.steps):

            # ------------------------------------------------
            # 13.1 策略计算及仿真
            # ------------------------------------------------

            actions = policy(obs)

            obs, rew, dones, extras = env.step(actions)

            current_step = step + 1

            # ------------------------------------------------
            # 13.2 记录 7 个关节角度
            # ------------------------------------------------

            joint_pos = (
                robot.data.joint_pos.torch[
                    0, joint_ids
                ]
                .detach()
                .cpu()
                .numpy()
            )

            for i, name in enumerate(joint_names):
                records[name].append(
                    float(joint_pos[i])
                )

            # ------------------------------------------------
            # 13.3 记录 Object 世界坐标 XYZ
            # ------------------------------------------------

            object_xyz = (
                obj.data.root_pos_w.torch[0, :3]
                .detach()
                .cpu()
                .numpy()
            )

            object_x = float(object_xyz[0])
            object_y = float(object_xyz[1])
            object_z = float(object_xyz[2])

            object_z_list.append(object_z)

            # ------------------------------------------------
            # 13.4 记录 EE 目标位置世界坐标 XYZ
            # ------------------------------------------------

            ee_xyz = (
                ee.data.target_pos_w.torch[0, 0, :3]
                .detach()
                .cpu()
                .numpy()
            )

            ee_x = float(ee_xyz[0])
            ee_y = float(ee_xyz[1])
            ee_z = float(ee_xyz[2])

            ee_z_list.append(ee_z)
            target_command = (
                unwrapped_env.command_manager.get_command("object_pose")
            )

            target_pos_w, _ = combine_frame_transforms(
                robot.data.root_pos_w.torch[0:1, :3],
                robot.data.root_quat_w.torch[0:1, :4],
                target_command[0:1, :3],
                target_command[0:1, 3:7],
            )

            target_z = float(
                target_pos_w[0, 2].detach().cpu().item()
            )

            target_z_list.append(target_z)
            # ------------------------------------------------
            # 13.5 记录当前步总奖励
            # ------------------------------------------------

            step_reward = float(
                rew[0].detach().cpu().item()
            )

            reward_list.append(step_reward)

            # ------------------------------------------------
            # 13.6 每隔指定步数输出一次调试信息
            # ------------------------------------------------

            if current_step % args_cli.debug_interval == 0:

                # A. 当前步的奖励组成
                print_reward_debug(
                    step=current_step,
                    total_reward=step_reward,
                    manager=reward_manager,
                    step_dt=reward_dt,
                )

                # B. 机器人状态
                print(
                    f"\n[STATE_DEBUG] Step "
                    f"{current_step}/{args_cli.steps}",
                    flush=True,
                )

                print(
                    "  --- 7 个机械臂关节角度 (rad) ---",
                    flush=True,
                )

                for i, name in enumerate(joint_names):
                    print(
                        f"    {name:<18} = "
                        f"{float(joint_pos[i]):+.6f}",
                        flush=True,
                    )

                print(
                    "  --- EE 末端目标位置 (world, m) ---",
                    flush=True,
                )

                print(
                    f"    X = {ee_x:+.6f} m",
                    flush=True,
                )
                print(
                    f"    Y = {ee_y:+.6f} m",
                    flush=True,
                )
                print(
                    f"    Z = {ee_z:+.6f} m",
                    flush=True,
                )

                print(
                    "  --- Object 方块世界坐标 (m) ---",
                    flush=True,
                )

                print(
                    f"    X = {object_x:+.6f} m",
                    flush=True,
                )
                print(
                    f"    Y = {object_y:+.6f} m",
                    flush=True,
                )
                print(
                    f"    Z = {object_z:+.6f} m",
                    flush=True,
                )

                print(
                    "  --- 当前步摘要 ---",
                    flush=True,
                )

                print(
                    f"    Environment reward = "
                    f"{step_reward:+.8f}",
                    flush=True,
                )

                print(
                    f"    Object Z >= 0.075 m: "
                    f"{object_z >= 0.075}",
                    flush=True,
                )

                print("=" * 110, flush=True)

    print(
        f"[INFO] 轨迹记录完成：{len(reward_list)} 步",
        flush=True,
    )


finally:

    # ========================================================
    # 14. 优先保存曲线，然后关闭仿真
    # ========================================================

    print(
        "[INFO] 正在保存已经记录的数据...",
        flush=True,
    )

    if reward_list:

        try:
            plot_saved = save_trajectory_plot(
                output_path=args_cli.output,
                joint_names=joint_names,
                records=records,
                object_z_list=object_z_list,
                ee_z_list=ee_z_list,
                target_z_list=target_z_list,
                reward_list=reward_list,
            )

        except Exception as plot_error:
            print(
                f"[ERROR] 绘图失败：{plot_error!r}",
                flush=True,
            )
            traceback.print_exc()

    else:
        print(
            "[WARN] 没有记录到奖励数据，未生成图片。",
            flush=True,
        )

    # --------------------------------------------------------
    # 关闭环境
    # --------------------------------------------------------

    if env is not None:
        try:
            print("[INFO] 关闭 Gym 环境...", flush=True)
            env.close()
            print("[INFO] Gym 环境已关闭", flush=True)

        except Exception as close_error:
            print(
                f"[WARN] 关闭 Gym 环境失败：{close_error!r}",
                flush=True,
            )

    # --------------------------------------------------------
    # 关闭 Isaac Sim
    # --------------------------------------------------------

    if simulation_app is not None:
        try:
            print("[INFO] 关闭 Isaac Sim...", flush=True)
            simulation_app.close()
            print("[INFO] Isaac Sim 已关闭", flush=True)

        except Exception as close_error:
            print(
                f"[WARN] 关闭 Isaac Sim 失败：{close_error!r}",
                flush=True,
            )


# ============================================================
# 15. 最终统计
# ============================================================

print("", flush=True)
print("=" * 90, flush=True)
print("[INFO] 程序执行结束", flush=True)

print(
    f"[INFO] 实际记录步数：{len(reward_list)}",
    flush=True,
)

if reward_list:

    print(
        f"[INFO] 平均每步奖励："
        f"{sum(reward_list) / len(reward_list):+.8f}",
        flush=True,
    )

    print(
        f"[INFO] 记录期间方块最高高度："
        f"{max(object_z_list):.6f} m",
        flush=True,
    )

    print(
        f"[INFO] 记录期间 EE 最高目标高度："
        f"{max(ee_z_list):.6f} m",
        flush=True,
    )

if plot_saved:
    print(
        f"[SUCCESS] 图片路径：{args_cli.output}",
        flush=True,
    )
else:
    print(
        "[WARN] 图片未确认保存成功，请检查上方日志。",
        flush=True,
    )

print("=" * 90, flush=True)