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
from omni.isaac.lab.sensors.camera.camera import Camera
from omni.isaac.lab.sensors.camera.camera_cfg import CameraCfg
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
            "asset_cfg": SceneEntityCfg("box"),
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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=16, env_spacing=6.0, replicate_physics=False)
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
    # camera
    camera = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/d435_rgb_module_link/camera",
        # update_period=0.1,
        height=1280,
        width=720,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.FisheyeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset = CameraCfg.OffsetCfg(pos=(0.05, 0.0, 0.0), rot=(1, 0, 0, 0), convention="ros"),
    )
    third_person_camera = CameraCfg(
        prim_path="/World/envs/env_.*/camera",
        # update_period=0.1,
        height=384,
        width=384,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.FisheyeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset = CameraCfg.OffsetCfg(pos=(-0.36, 1.1, 1.7), rot=(0.28176, 0.22504, -0.52485, -0.77104), convention="opengl"),
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
        self.object_default_pos = torch.tensor(self.cfg.box.init_state.pos, dtype=torch.float, device=self.device)
        # unit tensors
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

    def _setup_scene(self):
        # add robot
        self.robot = Articulation(self.cfg.robot)
        # add box
        self.box = RigidObject(self.cfg.box)
        # add camera
        self.camera = Camera(self.cfg.camera)
        self.third_person_camera = Camera(self.cfg.third_person_camera)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate (no need to filter for this environment)
        self.scene.clone_environments(copy_from_source=False)
        # register humanoid and box
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["box"] = self.box
        self.scene.sensors["camera"] = self.camera
        self.scene.sensors["third_person_camera"] = self.third_person_camera
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
            
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        goal_dist_pos = torch.norm(self.goal_pos_w - self.box.data.root_state_w[:, :3], p=2, dim=-1)
        goal_dist_quat = torch.norm(self.goal_quat - self.box.data.root_state_w[:, 3:7], p=2, dim=-1)
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
        observation = self.third_person_camera.data.output[data_type].clone()[:, :, :, :3]
        
        save_to_file = False
        if save_to_file:
            obs_cpu = observation.to('cpu')
            img = obs_cpu[0].numpy()
            img_pil = Image.fromarray(img)
            file_path = "humanoid-push-box-rgb.png"

        return observation
    
    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros_like(self.goal_pos_w)