# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka Panda configuration for the tactile box-lift task (no-tactile baseline)."""

from __future__ import annotations

import math

from isaaclab.sim import SimulationCfg
from isaaclab.sim._impl.newton_manager_cfg import NewtonCfg
from isaaclab.sim._impl.solvers_cfg import MJWarpSolverCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.tactile_lift.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.tactile_lift.tactile_lift_env_cfg import TactileLiftEnvCfg

from isaaclab_assets import FRANKA_PANDA_CFG  # isort: skip


@configclass
class FrankaBoxLiftEnvCfg(TactileLiftEnvCfg):
    """Franka Panda box-lift without tactile observations (baseline).

    Observations: joint positions, joint velocities, box pose, last action.
    Rewards: box lifted (sparse), EE-to-box distance (dense), action / joint-vel penalties.
    """

    sim: SimulationCfg = SimulationCfg(
        newton_cfg=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                njmax=30,
                nconmax=30,
                ls_iterations=20,
                cone="pyramidal",
                impratio=1,
                ls_parallel=True,
                integrator="implicit",
            ),
            num_substeps=1,
        )
    )

    def __post_init__(self):
        super().__post_init__()

        # Robot
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Actions: joint position control for arm + gripper
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            scale=0.5,
            use_default_offset=True,
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger_joint.*"],
            open_command_expr={"panda_finger_joint.*": 0.04},
            close_command_expr={"panda_finger_joint.*": 0.0},
        )

        # Reward: end-effector body
        self.rewards.ee_to_box.params["robot_cfg"] = mdp.SceneEntityCfg(
            "robot", body_names=["panda_hand"]
        )


@configclass
class FrankaBoxLiftEnvCfg_PLAY(FrankaBoxLiftEnvCfg):
    """Play (evaluation) variant — fewer envs, no observation noise."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
