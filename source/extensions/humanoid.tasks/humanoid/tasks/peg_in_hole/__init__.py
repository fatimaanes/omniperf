"""
Humanoid peg in hole environment.
"""

import gymnasium as gym

from .peg_in_hole_env import PegInHoleEnv, PegInHoleEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Humanoid-Peg-In-Hole-v0",
    entry_point="humanoid.tasks.peg_in_hole:PegInHoleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PegInHoleEnvCfg
    },
)
