"""
Humanoid push box environment.
"""

import gymnasium as gym

from .orient_cubes_env import OrientCubesEnv, OrientCubesEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Humanoid-Orient-Cubes-v0",
    entry_point="humanoid.tasks.orient_cubes:OrientCubesEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": OrientCubesEnvCfg
    },
)
