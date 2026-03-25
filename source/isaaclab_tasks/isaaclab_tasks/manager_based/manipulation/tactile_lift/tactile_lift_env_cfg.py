# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base configuration for the tactile box-lift environment."""

from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.manipulation.tactile_lift.mdp as mdp

##
# Scene definition
##


@configclass
class TactileLiftSceneCfg(InteractiveSceneCfg):
    """Scene configuration for the tactile box-lift task.

    The scene contains:

    * A ground plane.
    * A flat table (modelled as a static cuboid).
    * A Franka Panda robot arm (configured in the robot-specific subclass).
    * A rigid box that the robot must lift.
    * Optionally a :class:`~isaaclab_newton.sensors.TactileSensor` mounted on the robot
      palm (configured in the tactile variant).
    """

    # world
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # table (static cuboid acting as a support surface)
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.6, 0.6, 0.05),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.5, 0.4)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8, dynamic_friction=0.6, restitution=0.0
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0)),
    )

    # robot arm — to be overridden in the robot-specific config
    robot: ArticulationCfg = MISSING

    # rigid box to be lifted (6 cm cube)
    box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Box",
        spawn=sim_utils.CuboidCfg(
            size=(0.06, 0.06, 0.06),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.6, 0.9)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8, dynamic_friction=0.6, restitution=0.0
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.055)),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: ActionTerm = MISSING
    gripper_action: ActionTerm | None = None


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP.

    The base class does not include tactile observations.  The tactile variant
    adds them by extending this class in ``joint_pos_tactile_env_cfg.py``.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations (proprioception + task state)."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        box_pos = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("box")})
        box_quat = ObsTerm(func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("box")})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Event (reset / randomisation) configuration."""

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.5, 1.5), "velocity_range": (0.0, 0.0)},
    )

    reset_box_pose = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("box"),
            "pose_range": {
                "x": (0.4, 0.6),
                "y": (-0.1, 0.1),
                "z": (0.055, 0.055),
            },
            "velocity_range": {},
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the base (no-tactile) variant."""

    # Sparse: reward for actually lifting the box
    box_lifted = RewTerm(
        func=mdp.box_lifted,
        weight=10.0,
        params={"box_cfg": SceneEntityCfg("box"), "height_threshold": 0.15},
    )

    # Dense: approach reward to guide exploration
    ee_to_box = RewTerm(
        func=mdp.ee_to_box_distance,
        weight=1.0,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=["panda_hand"]),
            "box_cfg": SceneEntityCfg("box"),
        },
    )

    # Action regularisation
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0001)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination conditions."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


##
# Environment configuration
##


@configclass
class TactileLiftEnvCfg(ManagerBasedRLEnvCfg):
    """Base configuration for the tactile box-lift environment."""

    scene: TactileLiftSceneCfg = TactileLiftSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 10.0
        self.viewer.eye = (3.5, 3.5, 3.5)
        self.sim.dt = 1.0 / 60.0
