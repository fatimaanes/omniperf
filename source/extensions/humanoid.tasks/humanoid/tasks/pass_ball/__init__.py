"""
Humanoid pass a ball from one hand to another environment.
"""

import gymnasium as gym

from .pass_ball_env import PassBallEnv, PassBallEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Humanoid-Pass-Ball-v0",
    entry_point="humanoid.tasks.pass_ball:PassBallEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PassBallEnvCfg
    },
)
