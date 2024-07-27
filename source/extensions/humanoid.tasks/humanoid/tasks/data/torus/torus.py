import os
from omni.isaac.lab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg
import omni.isaac.lab.sim as sim_utils

current_file_path = os.path.abspath(__file__)
parent_dir_path = os.path.dirname(current_file_path)

TORUS_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Torus",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{parent_dir_path}/torus.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            kinematic_enabled=False,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(1, 0.0, 0.0), rot=(1, 0, 0, 0)),
)
"""Configuration for the Unitree H1 Humanoid robot with Torso's rigid body property disabled."""
