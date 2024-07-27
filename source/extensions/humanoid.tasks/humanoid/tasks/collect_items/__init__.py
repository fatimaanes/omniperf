"""
Humanoid stack items environment.
"""

import gymnasium as gym

from .collect_items_env import CollectItemsEnv, CollectItemsEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Humanoid-Collect-Items-v0",
    entry_point="humanoid.tasks.collect_items:CollectItemsEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CollectItemsEnvCfg
    },
)
