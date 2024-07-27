"""
Humanoid stack items environment.
"""

import gymnasium as gym

from .stack_items_env import StackItemsEnv, StackItemsEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Humanoid-Stack-Items-v0",
    entry_point="humanoid.tasks.stack_items:StackItemsEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": StackItemsEnvCfg
    },
)
