"""
Humanoid mvoe a large package environment.
"""

import gymnasium as gym

from .move_package_env import MovePackageEnv, MovePackageEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Humanoid-Move-Package-v0",
    entry_point="humanoid.tasks.move_package:MovePackageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": MovePackageEnvCfg
    },
)
