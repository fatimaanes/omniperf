import os
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets.articulation.articulation_cfg import ArticulationCfg
from omni.isaac.lab.actuators import ImplicitActuatorCfg

current_file_path = os.path.abspath(__file__)
parent_dir_path = os.path.dirname(current_file_path)

H1_WITH_HAND_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{parent_dir_path}/h1_with_hand.usd",
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
        pos=(0.0, 0.0, 1.05),
        joint_pos={
            ".*_hip_yaw_joint": 0.0,
            ".*_hip_roll_joint": 0.0,
            ".*_hip_pitch_joint": -0.28,  # -16 degrees
            ".*_knee_joint": 0.79,  # 45 degrees
            ".*_ankle_joint": -0.52,  # -30 degrees
            "torso_joint": 0.0,
            ".*_shoulder_pitch_joint": 0.28,
            ".*_shoulder_roll_joint": 0.0,
            ".*_shoulder_yaw_joint": 0.0,
            ".*_elbow_joint": 0.52,
            ".*_index_proximal_joint": 0.0,
            ".*_middle_proximal_joint": 0.0,
            ".*_pinky_proximal_joint": 0.0,
            ".*_ring_proximal_joint": 0.0,
            ".*_index_intermediate_joint": 0.0,
            ".*_middle_intermediate_joint": 0.0,
            ".*_pinky_intermediate_joint": 0.0,
            ".*_ring_intermediate_joint": 0.0,
            ".*_thumb_intermediate_joint": 0.0,
            ".*_thumb_proximal_yaw_joint": 0.0,
            ".*_thumb_proximal_pitch_joint": 0.0,
            ".*_thumb_distal_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_hip_pitch_joint", ".*_knee_joint", "torso_joint"],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                ".*_hip_yaw_joint": 150.0,
                ".*_hip_roll_joint": 150.0,
                ".*_hip_pitch_joint": 200.0,
                ".*_knee_joint": 200.0,
                "torso_joint": 200.0,
            },
            damping={
                ".*_hip_yaw_joint": 5.0,
                ".*_hip_roll_joint": 5.0,
                ".*_hip_pitch_joint": 5.0,
                ".*_knee_joint": 5.0,
                "torso_joint": 5.0,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_joint"],
            effort_limit=100,
            velocity_limit=100.0,
            stiffness={".*_ankle_joint": 20.0},
            damping={".*_ankle_joint": 4.0},
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint", ".*_elbow_joint"],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                ".*_shoulder_pitch_joint": 40.0,
                ".*_shoulder_roll_joint": 40.0,
                ".*_shoulder_yaw_joint": 40.0,
                ".*_elbow_joint": 40.0,
            },
            damping={
                ".*_shoulder_pitch_joint": 10.0,
                ".*_shoulder_roll_joint": 10.0,
                ".*_shoulder_yaw_joint": 10.0,
                ".*_elbow_joint": 10.0,
            },
        ),
        "hands": ImplicitActuatorCfg(
            joint_names_expr=[".*_hand_joint", ".*_index_proximal_joint", ".*_middle_proximal_joint", ".*_pinky_proximal_joint",
                              ".*_ring_proximal_joint", ".*_index_intermediate_joint", ".*_middle_intermediate_joint", 
                              ".*_pinky_intermediate_joint", ".*_ring_intermediate_joint", ".*_thumb_intermediate_joint",
                              ".*_thumb_proximal_yaw_joint", ".*_thumb_proximal_pitch_joint"  ,".*_thumb_distal_joint"],
            stiffness=None,
            damping=None
        ),
    }
)
"""Configuration for the Unitree H1 Humanoid robot with hand."""
