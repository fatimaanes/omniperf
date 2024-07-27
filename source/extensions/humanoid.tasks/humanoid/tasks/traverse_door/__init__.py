"""
Humanoid open cabinet environment.
"""

import gymnasium as gym

from .traverse_door_env import TraverseDoorEnv, TraverseDoorEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Humanoid-Traverse-Door-v0",
    entry_point="humanoid.tasks.traverse_door:TraverseDoorEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": TraverseDoorEnvCfg,
    },
)
