# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom observation functions for the tactile lift task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def tactile_force_norms(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tactile"),
) -> torch.Tensor:
    """Per-hexagon contact force norms from the tactile sensor.

    Args:
        env: The RL environment.
        sensor_cfg: Scene entity config identifying the tactile sensor.

    Returns:
        Tensor of shape ``(num_envs, num_hexagons)`` containing the L2 norm
        of the contact force on each hexagon.
    """
    from isaaclab_newton.sensors import TactileSensor

    sensor: TactileSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w  # (N, H, 3)
    return torch.norm(forces, dim=-1)  # (N, H)


def tactile_force_vectors(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tactile"),
) -> torch.Tensor:
    """Flattened per-hexagon contact force vectors from the tactile sensor.

    Args:
        env: The RL environment.
        sensor_cfg: Scene entity config identifying the tactile sensor.

    Returns:
        Tensor of shape ``(num_envs, num_hexagons * 3)`` containing the contact
        force vector of each hexagon, concatenated.
    """
    from isaaclab_newton.sensors import TactileSensor

    sensor: TactileSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w  # (N, H, 3)
    return forces.reshape(forces.shape[0], -1)  # (N, H*3)
