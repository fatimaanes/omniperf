import os
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets.articulation.articulation_cfg import ArticulationCfg
from omni.isaac.lab.actuators import ImplicitActuatorCfg

current_file_path = os.path.abspath(__file__)
parent_dir_path = os.path.dirname(current_file_path)

DOOR_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Door",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{parent_dir_path}/door.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.8, 0.0, 1.05),
        rot = (0.7071068, 0.0, 0.0, 0.7071068),
        joint_pos={
            ".*door_joint": -1.7,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "door": ImplicitActuatorCfg(
            joint_names_expr=["door_joint"],
            effort_limit=87.0,
            velocity_limit=100.0,
            stiffness=10.0,
            damping=1.0,
        ),
    },
)
