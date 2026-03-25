# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom reward functions for the tactile lift task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def box_lifted(
    env: ManagerBasedRLEnv,
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    height_threshold: float = 0.15,
) -> torch.Tensor:
    """Sparse reward: +1 when the box rises above ``height_threshold`` metres.

    Args:
        env: The RL environment.
        box_cfg: Scene entity config for the box.
        height_threshold: Minimum z height (relative to the initial table surface)
            that counts as a successful lift.  Defaults to 0.15 m.

    Returns:
        Float tensor of shape ``(num_envs,)``.
    """
    box: RigidObject = env.scene[box_cfg.name]
    return (box.data.root_pos_w[:, 2] > height_threshold).float()


def ee_to_box_distance(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["panda_hand"]),
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """Dense reward: negative distance from the end-effector to the box centre.

    Encourages the robot to approach the box before attempting to lift.

    Args:
        env: The RL environment.
        robot_cfg: Scene entity config for the robot; ``body_names`` should specify
            the end-effector link (e.g. ``"panda_hand"``).
        box_cfg: Scene entity config for the box.

    Returns:
        Float tensor of shape ``(num_envs,)``.
    """
    from isaaclab.assets import Articulation

    robot: Articulation = env.scene[robot_cfg.name]
    box: RigidObject = env.scene[box_cfg.name]

    # End-effector world position
    ee_pos = robot.data.body_pos_w[:, robot_cfg.body_ids[0], :]  # (N, 3)
    box_pos = box.data.root_pos_w  # (N, 3)
    dist = torch.norm(ee_pos - box_pos, dim=-1)  # (N,)
    return -dist


def force_efficiency(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tactile"),
) -> torch.Tensor:
    """Dense reward: penalise mean contact force across all hexagons.

    This term incentivises the robot to use the minimum force necessary to
    grasp and lift the box.  The reward is always ``<= 0``.

    A well-tuned weight (e.g. ``-0.005``) keeps this term small relative to the
    lifting reward while still steering the policy towards lighter contact.

    Args:
        env: The RL environment.
        sensor_cfg: Scene entity config identifying the tactile sensor.

    Returns:
        Float tensor of shape ``(num_envs,)``.
    """
    from isaaclab_newton.sensors import TactileSensor

    sensor: TactileSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w  # (N, H, 3)
    mean_force_norm = torch.norm(forces, dim=-1).mean(dim=-1)  # (N,)
    return -mean_force_norm
