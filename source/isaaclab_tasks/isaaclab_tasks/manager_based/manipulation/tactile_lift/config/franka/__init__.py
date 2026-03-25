# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# ------------------------------------------------------------------
# No-tactile baseline
# ------------------------------------------------------------------

gym.register(
    id="Isaac-TactileLift-Franka-NoTactile-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FrankaBoxLiftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaBoxLiftPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-TactileLift-Franka-NoTactile-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FrankaBoxLiftEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaBoxLiftPPORunnerCfg",
    },
)

# ------------------------------------------------------------------
# Tactile variant
# ------------------------------------------------------------------

gym.register(
    id="Isaac-TactileLift-Franka-Tactile-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_tactile_env_cfg:FrankaBoxLiftTactileEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaBoxLiftTactilePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-TactileLift-Franka-Tactile-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_tactile_env_cfg:FrankaBoxLiftTactileEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaBoxLiftTactilePPORunnerCfg",
    },
)
