# UR5 + LEAP Hand（Isaac Lab 内嵌资产）

> 环境：Ubuntu 22.04 · 双 RTX 4090 · Isaac Sim 6.0.1 (`~/isaacsim-6.0.1`) · Isaac Lab 3.0.0-beta2.patch1
> 位置：`~/pro/IsaacLab-3.0.0-beta2.patch1/source/isaaclab_assets/data/ur5_leaphand/`

这份资产已经**装进 Isaac Lab 仓库本身**，可以直接当官方机器人配置用：

```python
from isaaclab_assets.robots import UR5_LEAPHAND_CFG          # 包级懒加载导出
from isaaclab_assets.robots.ur5_leaphand import HAND_JOINTS  # 子模块直取
```

---

## 1. 目录结构

```
source/isaaclab_assets/
├── data/ur5_leaphand/                  # ← 资产（45 MB）
│   ├── urdf/ur5_leaphand_right.urdf    # 合并后的 URDF：33 link / 22 可动关节
│   ├── urdf/meshes/{ur5,leap_hand}/    # 网格，URDF 里是相对路径
│   ├── usd/ur5_leaphand_right/         # USD 产物（27.9 MB 单层 usda）
│   ├── shots/                          # 验证渲染图
│   ├── tools/build_urdf.py             # 合成 URDF（幂等）
│   ├── tools/convert_urdf.py           # URDF → USD
│   └── README.md                       # 本文件
└── isaaclab_assets/robots/ur5_leaphand.py   # ← ArticulationCfg

scripts/demos/ur5_leaphand.py           # ← 演示/验证脚本
```

`robots/__init__.pyi` 里已登记，所以 `from isaaclab_assets.robots import UR5_LEAPHAND_CFG` 生效（本版本用 `lazy_export()` 自动导出，靠的就是这个 stub）。

## 2. 复现 / 重新生成

```bash
cd ~/pro/IsaacLab-3.0.0-beta2.patch1

# 重新合成 URDF（改装配关系、换左手、改法兰位姿都在这里）
python3 source/isaaclab_assets/data/ur5_leaphand/tools/build_urdf.py

# 重新转 USD（约 40 s）
./isaaclab.sh -p source/isaaclab_assets/data/ur5_leaphand/tools/convert_urdf.py

# 验证 + 出图
./isaaclab.sh -p scripts/demos/ur5_leaphand.py --headless --enable_cameras \
    --out_dir /tmp/ur5_leaphand_demo
```

实测输出：`num_dof 22 / num_bodies 27`；开手阶段机械臂跟踪误差 `0.000 rad`，握拳阶段只有 `leap_0`/`leap_4` 差约 `0.19 rad`（手指互顶，正常）。

## 3. 在任务里怎么用

```python
from isaaclab_assets.robots import UR5_LEAPHAND_CFG, ARM_READY_POSE, HAND_OPEN_POSE

robot_cfg = UR5_LEAPHAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
# 或者
robot_cfg = UR5_LEAPHAND_CFG.copy()
robot_cfg.prim_path = "{ENV_REGEX_NS}/Robot"
```

| 组 | 关节 | stiffness / damping / effort |
|---|---|---|
| `arm` | `shoulder_pan_joint` `shoulder_lift_joint` `elbow_joint` | 1200 / 60 / 150 N·m |
| `wrist` | `wrist_1/2/3_joint` | 600 / 30 / 28 N·m |
| `hand` | `leap_0` … `leap_15` | 3.0 / 0.15 / 0.95 N·m |

**注意**：`Articulation.joint_names` 是按运动树排序的（`leap_1, leap_5, leap_9, leap_12, leap_0, …`），不是按名字/编号，因此一律用"关节名 → 索引"查表，别写死下标。

## 4. 关节编号（LEAP Hand，右手）

| 编号 | 手指 | 每个手指的 4 个 DoF 顺序 |
|---|---|---|
| 0–3 | index | MCP 屈曲 / MCP 侧摆 / PIP / DIP |
| 4–7 | middle | 同上 |
| 8–11 | ring | 同上 |
| 12–15 | thumb | 基部旋转 + 3 段屈曲 |

与真机 SDK 的 16 路电机编号一致，`leap_i` 直接对应电机 i，方便 sim2real。

## 5. 资产从哪来

| 部件 | 来源 |
|---|---|
| UR5 URDF | **Isaac Sim 6.0.1 自带**：`~/isaacsim-6.0.1/extsDeprecated/isaacsim.robot_motion.motion_generation/motion_policy_configs/universal_robots/ur5/ur5.urdf`（xacro 已展开） |
| UR5 网格 | [Universal_Robots_ROS2_Description](https://github.com/UniversalRobots/Universal_Robots_ROS2_Description) |
| LEAP Hand | [dex-urdf](https://github.com/dexsuite/dex-urdf) → `robots/hands/leap_hand/leap_hand_right.urdf`（上游为 [LEAP_Hand_Sim](https://github.com/leap-hand/LEAP_Hand_Sim)） |

`build_urdf.py` 里的两个输入路径正是上面两处，重新生成时需要它们还在（`~/pro/assets/urdf_src/`）。

## 6. 踩过的坑

### 6.1 两个 URDF 的根链接重名
UR5 有 `base`，LEAP 也有 `base`。→ 手部所有 link/joint 加 `leap_` 前缀。

### 6.2 无质量 frame 会让整只手被焊到世界
`tool0`/`flange`/`leap_base` 这类纯坐标系链接没有 `<inertial>`，导入器不建刚体却保留指向它们的 fixed joint，PhysX 把父体当"世界"，手掌被钉在世界原点 → 整条链在 t=0 发散（`q ≈ 4e5 rad`）。
→ `build_urdf.py` 自动给 `base_link` / `flange` / `tool0` / `leap_base` 补 0.01 kg + 1e-6 惯量。

### 6.3 导入器的 `fix_base=True` 在本版本不可用
只要把该 articulation 锚定到世界，PhysX 返回的关节状态就是垃圾（t=0 时 `q ≈ 4.4e5`）。已排除：

| 尝试 | 结果 |
|---|---|
| 导入器 `fix_base=True` | 垃圾 |
| 手写 `UsdPhysics.FixedJoint`（body0 留空=世界） | 同上，数值完全一致 |
| `omni.physx.utils.createJoint(from_prim=None)` | 同上 |
| 根刚体 `kinematicEnabled=True` | PhysX 直接建不出 articulation |
| Isaac Lab `fix_root_link=True`（官方推荐路径） | 同上 |
| `merge_fixed_joints=True` / `self_collision=False` | 同上 |

**自由基座完全正常**（Isaac Lab 自带 UR10 资产做对照组也正常，排除框架用法问题）。可能与 Isaac Sim 6.0 的 Newton 后端有关：导入器会额外在根刚体打 `NewtonArticulationRootAPI`，Isaac Lab 会当成第二个 articulation root 直接报错，必须删掉。

**当前方案**：USD 保持自由基座，控制循环里每步写回根位姿（`scripts/demos/ur5_leaphand.py` 已实现）：

```python
root_pose = robot.data.default_root_pose.torch.clone()
root_vel = robot.data.default_root_vel.torch.clone()
robot.write_root_pose_to_sim_index(root_pose=root_pose)
robot.write_root_velocity_to_sim_index(root_velocity=root_vel)
```

### 6.4 基座必须抬高
UR5 零位是平躺的（tool0 约在 `(0.82, 0.19, 0.0)`）。基座放在 z=0 时关节被地面卡住，PD 抬不起来（实测 5 s 后仍差 1.5 rad）。`cfg.init_state.pos` 用 `(0, 0, 0.75)`，相当于装在 0.75 m 台面上。

## 7. 现状

- ✅ 合并 URDF、网格解析、USD 转换
- ✅ 包内加载：22 DoF / 27 bodies，关节名、限位、惯量正确
- ✅ 位置控制（臂 0.000 rad 跟踪）、自碰撞、RTX 渲染
- ⚠️ 硬固定基座不可用（6.3），用"每步钉根位姿"绕行

## 8. 说明

这是**对 Isaac Lab 检出目录的本地修改**（未提交），`git status` 会看到：

```
?? source/isaaclab_assets/data/
 M source/isaaclab_assets/isaaclab_assets/robots/__init__.pyi
?? source/isaaclab_assets/isaaclab_assets/robots/ur5_leaphand.py
?? scripts/demos/ur5_leaphand.py
```

`git clean -fdx`、重新 clone 或升级 Isaac Lab 都会丢掉它们。要长期保留，建议把 `data/ur5_leaphand` + 这两个 py 文件另存一份，或做成独立 package 用 `pip install -e` 装。

下一步建议：拿 Isaac Sim 资产库里的 stock UR5 USD 当手臂、手部 USD 挂上去，彻底绕开 URDF 导入器这条路径，以恢复硬固定基座。
