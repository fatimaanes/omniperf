"""
Humanoid stack items environment.
"""

import gymnasium as gym

from .walk_env import WalkEnv, WalkEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Humanoid-Walk-v0",
    entry_point="humanoid.tasks.walk:WalkEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": WalkEnvCfg
    },
)
