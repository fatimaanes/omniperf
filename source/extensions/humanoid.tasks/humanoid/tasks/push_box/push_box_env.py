from __future__ import annotations
from collections.abc import Sequence
import os

from humanoid.tasks.data import H1_WITH_HAND_CFG
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.assets.articulation.articulation import Articulation
from omni.isaac.lab.assets.asset_base_cfg import AssetBaseCfg
from omni.isaac.lab.assets.rigid_object.rigid_object import RigidObject
from omni.isaac.lab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg
from omni.isaac.lab.envs import DirectRLEnvCfg
from omni.isaac.lab.envs.direct_rl_env import DirectRLEnv
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG
from omni.isaac.lab.markers.visualization_markers import VisualizationMarkers, VisualizationMarkersCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg
from omni.isaac.lab.sensors.camera.camera import Camera
from omni.isaac.lab.sensors.camera.camera_cfg import CameraCfg
from omni.isaac.lab.sensors import TiledCamera, TiledCameraCfg, save_images_to_file
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.sim.spawners.from_files.from_files import spawn_ground_plane
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import sample_uniform
from omni.isaac.lab.utils.math import quat_from_angle_axis

from PIL import Image
import torch
import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.envs.mdp as mdp


@configclass
class EventCfg:
    """Configuration for randomization."""

    # -- object
    object_pos = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cracker_box"), # SceneEntityCfg("box")
            "pose_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (-3.14, 3.14)},
            "velocity_range": {},
        },
    )
    
@configclass
class PushBoxEnvCfg(DirectRLEnvCfg):
    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=16, env_spacing=8.0, replicate_physics=False)
    # robot
    robot: ArticulationCfg = H1_WITH_HAND_CFG.replace(prim_path="/World/envs/env_.*/Robot") # type: ignore
    # table
    table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.9, 0, 1.04), rot=(1, 0, 0, 0)),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )
    
    current_path = os.path.abspath(__file__)
    humanoid_task_dir = os.path.dirname(os.path.dirname(current_path))
    # room
    room = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Room",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, 0), rot=(1, 0, 0, 0)),
        spawn=UsdFileCfg(usd_path=f"{humanoid_task_dir}/data/scene/room_1.usd"),
    )
    # box
    box: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Box",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.7, 0, 1.06), rot=(1, 0, 0, 0)),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd",
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
    )
    # goal object
    goal_object_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/goal")
    goal_object_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)

    # tomato soup can
    tomato_soup_can: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/TomatoSoupCan",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.8, -0.25, 1.06), rot=(0.7071068, -0.7071068, 0, 0)),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd",
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
    )
    # sugar box (yellow box)
    sugar_box: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/SugarBox",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.8, 0.0, 1.06), rot=(1, 0, 0, 0)),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/004_sugar_box.usd",
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
    )
    # cracker box (red box)
    cracker_box: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/CrackerBox",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.8, 0.35, 1.06), rot=(1, 0, 0, 0)),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd",
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
    )
    # cabinet
    cabinet = ArticulationCfg(
        prim_path="/World/envs/env_.*/Cabinet",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd",
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-1.6, 2.2, 0.6),
            rot=(1, 0, 0, 0),
            joint_pos={
                "door_left_joint": 0.0,
                "door_right_joint": 0.0,
                "drawer_bottom_joint": 0.0,
                "drawer_top_joint": 0.0,
            },
        ),
        actuators={
            "drawers": ImplicitActuatorCfg(
                joint_names_expr=["drawer_top_joint", "drawer_bottom_joint"],
                effort_limit=87.0,
                velocity_limit=100.0,
                stiffness=10.0,
                damping=1.0,
            ),
            "doors": ImplicitActuatorCfg(
                joint_names_expr=["door_left_joint", "door_right_joint"],
                effort_limit=87.0,
                velocity_limit=100.0,
                stiffness=10.0,
                damping=2.5,
            ),
        },
    )

    # cameras
    width = 320
    height = 240
    camera = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/d435_rgb_module_link/camera",
        # update_period=0.1,
        width = width,
        height = height,
        #data_types=["rgb", "distance_to_image_plane"],
        data_types=["rgb"],
        spawn=sim_utils.FisheyeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset = CameraCfg.OffsetCfg(pos=(0.05, 0.0, 0.0), rot=(1, 0, 0, 0), convention="opengl"), # "ros"
    )
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/d435_rgb_module_link/camera",
        width = width,
        height = height,
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, -0.27, 1.5), rot=(0.0, 0.0, 0.0, -1.0), convention="opengl"),
        # offset=TiledCameraCfg.OffsetCfg(pos=(-2.0, 0.0, 0.75), rot=(-0.5, -0.5, 0.5, 0.5), convention="opengl"),
        data_types=["rgba"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
    )
    third_person_camera = CameraCfg(
        prim_path="/World/envs/env_.*/camera",
        # update_period=0.1,
        width = width,
        height = height,
        #data_types=["rgb", "distance_to_image_plane"],
        data_types=["rgb"],
        spawn=sim_utils.FisheyeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset = CameraCfg.OffsetCfg(pos=(-0.36, 1.1, 1.7), rot=(0.28176, 0.22504, -0.52485, -0.77104), convention="opengl"),
    )
    tiled_third_person_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/tiled_camera",
        width = width,
        height = height,
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, -0.27, 1.5), rot=(0.0, 0.0, 0.0, -1.0), convention="opengl"),
        # offset=TiledCameraCfg.OffsetCfg(pos=(-2.0, 0.0, 0.75), rot=(-0.5, -0.5, 0.5, 0.5), convention="opengl"),
        data_types=["rgba"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
    )
    # domain randomization config
    events: EventCfg = EventCfg()
    # env
    episode_length_s = 2000
    decimation = 2
    action_scale = 1.0
    num_actions = 45
    num_observations = 99
    pos_done_threshold = 0.01
    quat_done_threshold = 0.01


class PushBoxEnv(DirectRLEnv):
    cfg: PushBoxEnvCfg

    def __init__(self, cfg: PushBoxEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.robot_dof_targets = torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)
        self.robot_dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.dt = self.cfg.sim.dt * self.cfg.decimation
        # self.object_default_pos = torch.tensor(self.cfg.box.init_state.pos, dtype=torch.float, device=self.device)
        self.object_default_pos = torch.tensor(self.cfg.cracker_box.init_state.pos, dtype=torch.float, device=self.device)
        # unit tensors
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

    def _setup_scene(self):
        # add robot
        self.robot = Articulation(self.cfg.robot)

        self.state_only = True

        if not self.state_only:
            self.camera = Camera(self.cfg.third_person_camera)
        # self.camera = Camera(self.cfg.camera)
        # self.camera = Camera(self.cfg.tiled_third_person_camera)
        # self.camera = Camera(self.cfg.tiled_camera)

        self.cabinet = Articulation(self.cfg.cabinet)
        self.tomato_soup_can = RigidObject(self.cfg.tomato_soup_can)
        self.sugar_box = RigidObject(self.cfg.sugar_box) # yellow box
        self.cracker_box = RigidObject(self.cfg.cracker_box) # red box

        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate (no need to filter for this environment)
        self.scene.clone_environments(copy_from_source=False)
        # register humanoid and box
        self.scene.articulations["robot"] = self.robot
        self.scene.articulations["cabinet"] = self.cabinet
        self.scene.rigid_objects["tomato_soup_can"] = self.tomato_soup_can
        self.scene.rigid_objects["sugar_box"] = self.sugar_box
        self.scene.rigid_objects["cracker_box"] = self.cracker_box

        # after Hydra config support is available, we can set reolutions and camera type from the command line
        if not self.state_only:
            self.scene.sensors["camera"] = self.camera

        # add table
        self.cfg.table.spawn.func(
            self.cfg.table.prim_path,
            self.cfg.table.spawn, 
            translation=self.cfg.table.init_state.pos, 
            orientation=self.cfg.table.init_state.rot,
        )
        self.cfg.room.spawn.func(
            self.cfg.room.prim_path,
            self.cfg.room.spawn, 
            translation=self.cfg.room.init_state.pos, 
            orientation=self.cfg.room.init_state.rot,
        )
        # initialize goal marker and buffer
        self.goal_markers = VisualizationMarkers(self.cfg.goal_object_cfg)
        self.goal_pos_w = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.goal_quat = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
    
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        self.actions[:] = torch.clamp(self.actions, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        self.robot.set_joint_position_target(self.actions)
        self.robot.write_data_to_sim()
            
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        goal_dist_pos = torch.norm(self.goal_pos_w - self.cracker_box.data.root_state_w[:, :3], p=2, dim=-1)
        goal_dist_quat = torch.norm(self.goal_quat - self.cracker_box.data.root_state_w[:, 3:7], p=2, dim=-1)
        # goal_dist_pos = torch.norm(self.goal_pos_w - self.box.data.root_state_w[:, :3], p=2, dim=-1)
        # goal_dist_quat = torch.norm(self.goal_quat - self.box.data.root_state_w[:, 3:7], p=2, dim=-1)
        success_buf = (goal_dist_pos < self.cfg.pos_done_threshold) & (goal_dist_quat < self.cfg.quat_done_threshold)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return success_buf, time_out
    
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES # type: ignore
        super()._reset_idx(env_ids) # type: ignore
        self.robot.reset(env_ids=env_ids)
        # robot state
        root_state = self.robot.data.default_root_state[env_ids, :7]
        root_state[:, 0:3] += self.scene.env_origins[env_ids, :]
        root_vel = torch.zeros([len(env_ids), 6]).to(self.device) # type: ignore
        self.robot.write_root_pose_to_sim(root_pose=root_state, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(root_velocity=root_vel, env_ids=env_ids)
        # robot joint state
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        # reset buffers
        self.robot_dof_targets[env_ids, :] = 0
        self._reset_target_pose(env_ids)
        
    def _reset_target_pose(self, env_ids):
        # reset goal position and rotation
        rand_floats = sample_uniform(-0.2, 0.2, (len(env_ids), 2), device=self.device)
        self.goal_pos_w[env_ids, :] = self.object_default_pos + self.scene.env_origins[env_ids]
        self.goal_pos_w[env_ids, :2] += rand_floats
        rand_floats = sample_uniform(-1.0, 1.0, (len(env_ids)), device=self.device)
        self.goal_quat[env_ids, :] = quat_from_angle_axis(rand_floats * torch.pi, self.z_unit_tensor[env_ids])
        # update goal pose and markers
        self.goal_markers.visualize(self.goal_pos_w, self.goal_quat)

    def _get_observations(self) -> dict:
        data_type = "rgb" #  distance_to_image_plane
        if not self.state_only:
            observations = self.camera.data.output[data_type].clone()[:, :, :, :3]
        else:
            observations = torch.zeros((self.num_envs, self.cfg.height, self.cfg.width, 3), device=self.device)
        
        save_to_file = False
        if save_to_file:
            obs_cpu = observations.to('cpu')
            img = obs_cpu[0].numpy()
            img_pil = Image.fromarray(img)
            file_path = "humanoid-push-box-rgb.png"

        states = torch.zeros(self.num_envs, self.cfg.num_observations, device=self.device)

        # states = torch.cat(
        #     (
        #         # humanoid
        #         self.robot.data.root_pos_w - self.scene.env_origins,
        #         self.robot.data.joint_pos,
        #         self.robot.data.joint_vel,
        #         # object
        #         self.bolt.data.root_state_w, - self.scene.env_origins,
        #         self.bolt.data.root_quat_w,
        #         self.nut.data.root_state_w, - self.scene.env_origins,
        #         self.nut.data.root_quat_w,
        #         # actions
        #         self.actions,
        #     ),
        #     dim=-1,
        # )

        return {"camera": observations, "policy": states}
    
    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)