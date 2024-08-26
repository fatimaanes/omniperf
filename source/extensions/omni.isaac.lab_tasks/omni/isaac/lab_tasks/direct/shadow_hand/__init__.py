# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Shadow Hand environment.
"""

import gymnasium as gym

from . import agents
from .shadow_hand_camera_env import ShadowHandCameraEnv, ShadowHandRGBCameraEnvCfg, ShadowHandDepthCameraEnvCfg, ShadowHandRGBDCameraEnvCfg, ShadowHandRGBCameraAsymmetricEnvCfg, ShadowHandDepthCameraAsymmetricEnvCfg
from .shadow_hand_env_cfg import ShadowHandEnvCfg, ShadowHandOpenAIEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Repose-Cube-Shadow-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.inhand_manipulation:InHandManipulationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ShadowHandEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShadowHandPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.inhand_manipulation:InHandManipulationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ShadowHandOpenAIEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_ff_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShadowHandAsymFFPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Repose-Cube-Shadow-OpenAI-LSTM-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.inhand_manipulation:InHandManipulationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ShadowHandOpenAIEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_lstm_cfg.yaml",
    },
)

### Camera

gym.register(
    id="Isaac-Shadow-Hand-RGB-Camera-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.shadow_hand:ShadowHandCameraEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ShadowHandRGBCameraEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_camera_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Shadow-Hand-RGB-Camera-Resnet-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.shadow_hand:ShadowHandCameraEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ShadowHandRGBCameraEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_camera_ppo_resnet_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Shadow-Hand-Depth-Camera-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.shadow_hand:ShadowHandCameraEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ShadowHandDepthCameraEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_camera_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Shadow-Hand-RGBD-Camera-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.shadow_hand:ShadowHandCameraEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ShadowHandRGBDCameraEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_camera_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Shadow-Hand-RGBD-Camera-Resnet-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.shadow_hand:ShadowHandCameraEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ShadowHandRGBDCameraEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_camera_ppo_resnet_cfg.yaml",
    },
)



# gym.register(
#     id="Isaac-Shadow-Hand-RGB-Camera-Asymmetric-Direct-v0",
#     entry_point="omni.isaac.lab_tasks.direct.shadow_hand:ShadowHandCameraEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": ShadowHandRGBCameraAsymmetricEnvCfg,
#         "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_camera_asym_cfg.yaml",
#     },
# )

# gym.register(
#     id="Isaac-Shadow-Hand-Depth-Camera-Asymmetric-Direct-v0",
#     entry_point="omni.isaac.lab_tasks.direct.shadow_hand:ShadowHandCameraEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": ShadowHandDepthCameraAsymmetricEnvCfg,
#         "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_camera_asym_cfg.yaml",
#     },
# )