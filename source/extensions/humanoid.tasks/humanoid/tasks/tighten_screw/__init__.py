"""
Humanoid stack items environment.
"""

import gymnasium as gym

from .tighten_screw_env import TightenScrewEnv, TightenScrewEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Humanoid-Tighten-Screw-v0",
    entry_point="humanoid.tasks.tighten_screw:TightenScrewEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": TightenScrewEnvCfg
    },
)
