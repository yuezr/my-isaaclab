# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for a UR5 arm equipped with a LEAP Hand (right).

The asset was built from:

* the UR5 URDF shipped with Isaac Sim (``ur_description``, already xacro-flattened)
* the LEAP Hand right URDF from `dex-urdf <https://github.com/dexsuite/dex-urdf>`_ (16 DoF, no mimic joints)

and merged with ``data/ur5_leaphand/tools/build_urdf.py``.  The USD is produced by
``data/ur5_leaphand/tools/convert_urdf.py`` and lives in ``data/ur5_leaphand/usd``.

Joint names
-----------
* arm: ``shoulder_pan_joint``, ``shoulder_lift_joint``, ``elbow_joint``, ``wrist_1/2/3_joint``
* hand: ``leap_0`` .. ``leap_15`` (index 0-3, middle 4-7, ring 8-11, thumb 12-15; the four DoF of
  a finger are MCP flexion, MCP abduction, PIP, DIP).  The numbering matches the LEAP hand SDK
  motor order, so ``leap_i`` maps directly onto motor ``i``.

.. note::
    Isaac Lab sorts ``Articulation.joint_names`` by kinematic tree order, not by name
    (it comes out as ``leap_1, leap_5, leap_9, leap_12, leap_0, ...``).  Always look joints up by
    name instead of hard-coding indices.

.. note::
    The base is *free* in the USD.  Anchoring this articulation to the world makes PhysX report a
    corrupted joint state at t=0, so ``fix_root_link`` must stay ``False`` and the base is held by
    writing the root pose back every step (see ``scripts/demos/ur5_leaphand.py``).
"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

##
# Configuration
##

UR5_LEAPHAND_USD_PATH = str(
    Path(ISAACLAB_ASSETS_DATA_DIR) / "ur5_leaphand" / "usd" / "ur5_leaphand_right" / "ur5_leaphand_right.usda"
)
"""Path to the UR5 + LEAP Hand USD asset."""

ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
"""Names of the six arm joints, in kinematic order."""

HAND_JOINTS = [f"leap_{i}" for i in range(16)]
"""Names of the sixteen hand joints (LEAP hand SDK motor order)."""

#: Demo-ready arm pose: reaches forward with the wrist rotated so the hand faces the workspace.
#: The UR5 zero pose lies flat on the ground, and at z=0 the arm cannot lift itself off the floor,
#: hence the raised base below.
ARM_READY_POSE = {
    "shoulder_pan_joint": 0.0,
    "shoulder_lift_joint": -1.2,
    "elbow_joint": 1.4,
    "wrist_1_joint": -1.8,
    "wrist_2_joint": -1.5708,
    "wrist_3_joint": 0.0,
}

HAND_OPEN_POSE = {name: 0.0 for name in HAND_JOINTS}
"""LEAP hand rest pose (fully open)."""

HAND_GRASP_POSE = {name: 0.0 for name in HAND_JOINTS}
"""Rough power grasp: flexes the three flexion DoF of every finger, keeps abduction at zero."""
for _base in (0, 4, 8):  # index, middle, ring
    HAND_GRASP_POSE[f"leap_{_base + 0}"] = 1.05  # MCP flexion
    HAND_GRASP_POSE[f"leap_{_base + 1}"] = 0.0  # MCP abduction
    HAND_GRASP_POSE[f"leap_{_base + 2}"] = 1.20  # PIP
    HAND_GRASP_POSE[f"leap_{_base + 3}"] = 1.00  # DIP
HAND_GRASP_POSE["leap_12"] = 0.60  # thumb base
HAND_GRASP_POSE["leap_13"] = 1.20
HAND_GRASP_POSE["leap_14"] = 1.10
HAND_GRASP_POSE["leap_15"] = 0.90

UR5_LEAPHAND_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=UR5_LEAPHAND_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
            # must stay False: see the module docstring
            fix_root_link=False,
        ),
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={**{j: 0.0 for j in ARM_JOINTS}, **HAND_OPEN_POSE},
        # stand the arm on a 0.75 m "table"; at z=0 the UR5 cannot lift itself off the floor
        pos=(0.0, 0.0, 0.75),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint"],
            effort_limit_sim=150.0,
            velocity_limit_sim=3.14,
            stiffness=1200.0,
            damping=60.0,
        ),
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=["wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
            effort_limit_sim=28.0,
            velocity_limit_sim=3.14,
            stiffness=600.0,
            damping=30.0,
        ),
        "hand": ImplicitActuatorCfg(
            joint_names_expr=["leap_.*"],
            effort_limit_sim=0.95,
            velocity_limit_sim=8.48,
            stiffness=3.0,
            damping=0.15,
        ),
    },
)
"""UR5 arm with a LEAP Hand, position controlled, free base (hold the root pose in the task)."""
