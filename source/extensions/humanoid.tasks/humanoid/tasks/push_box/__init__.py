"""
Humanoid push box environment.
"""

import gymnasium as gym
from . import agents
from .push_box_env import PushBoxEnv, PushBoxEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Humanoid-Push-Box-v0",
    entry_point="humanoid.tasks.push_box:PushBoxEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PushBoxEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
