# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka Panda configuration for the tactile box-lift task (with tactile sensor)."""

from __future__ import annotations

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_newton.sensors import TactileSensorCfg

import isaaclab_tasks.manager_based.manipulation.tactile_lift.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.tactile_lift.config.franka.joint_pos_env_cfg import (
    FrankaBoxLiftEnvCfg,
    FrankaBoxLiftEnvCfg_PLAY,
)


@configclass
class FrankaBoxLiftTactileEnvCfg(FrankaBoxLiftEnvCfg):
    """Franka Panda box-lift **with** tactile sensor observations and force-efficiency reward.

    Compared to the no-tactile baseline (:class:`FrankaBoxLiftEnvCfg`), this variant:

    1. Adds a :class:`~isaaclab_newton.sensors.TactileSensor` on ``panda_hand``
       (19 hexagons in a 2-ring layout, each 6 mm radius × 2 mm thick).
    2. Extends the policy observation with per-hexagon contact force norms.
    3. Adds a force-efficiency reward term that penalises high mean contact force.

    The comparison experiment trains both variants and plots mean return + mean contact
    force to demonstrate whether tactile feedback leads to a lighter-touch grasp.
    """

    def __post_init__(self):
        super().__post_init__()

        # ------------------------------------------------------------------
        # Tactile sensor on panda_hand palm
        # ------------------------------------------------------------------
        self.scene.tactile = TactileSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_hand/TactileSensor",
            attach_prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
            num_rings=2,       # 19 hexagons
            hex_radius=0.006,  # 6 mm circumradius
            hex_height=0.002,  # 2 mm thickness
            # Offset the sensor patch slightly away from the palm surface
            sensor_pos=(0.0, 0.0, 0.04),
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Box"],
            track_air_time=False,
            history_length=0,
        )

        # ------------------------------------------------------------------
        # Add tactile force norms to policy observations
        # ------------------------------------------------------------------
        self.observations.policy.tactile_forces = ObsTerm(
            func=mdp.tactile_force_norms,
            params={"sensor_cfg": SceneEntityCfg("tactile")},
            scale=0.1,  # scale down large force values
        )

        # ------------------------------------------------------------------
        # Add force-efficiency reward (encourages lighter-touch grasps)
        # ------------------------------------------------------------------
        self.rewards.force_efficiency = RewTerm(
            func=mdp.force_efficiency,
            weight=-0.005,
            params={"sensor_cfg": SceneEntityCfg("tactile")},
        )


@configclass
class FrankaBoxLiftTactileEnvCfg_PLAY(FrankaBoxLiftTactileEnvCfg):
    """Play (evaluation) variant for the tactile environment."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
