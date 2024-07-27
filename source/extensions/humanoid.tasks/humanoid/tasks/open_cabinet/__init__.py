"""
Humanoid open cabinet environment.
"""

import gymnasium as gym

from .open_cabinet_env import OpenCabinetEnv, OpenCabinetEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Humanoid-Open-Cabinet-v0",
    entry_point="humanoid.tasks.open_cabinet:OpenCabinetEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": OpenCabinetEnvCfg,
    },
)
