from __future__ import annotations
from collections.abc import Sequence


from humanoid.tasks.data import H1_WITH_HAND_CFG
from humanoid.tasks.data.card_box.card_box import CARD_BOX_CFG
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.assets.articulation.articulation import Articulation
from omni.isaac.lab.assets.rigid_object.rigid_object import RigidObject
from omni.isaac.lab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg
from omni.isaac.lab.envs import DirectRLEnvCfg
from omni.isaac.lab.envs.direct_rl_env import DirectRLEnv
from omni.isaac.lab.markers.visualization_markers import VisualizationMarkers, VisualizationMarkersCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.sim.spawners.from_files.from_files import spawn_ground_plane
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import sample_uniform
from omni.isaac.lab.utils.math import quat_from_angle_axis

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
class MovePackageEnvCfg(DirectRLEnvCfg):
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
    # box
    box: RigidObjectCfg = CARD_BOX_CFG.replace(prim_path="/World/envs/env_.*/CardBox") # type: ignore
    # goal object
    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "goal": sim_utils.UsdFileCfg(
                usd_path=CARD_BOX_CFG.spawn.usd_path, # type: ignore
            )
        },
    )
    # domain randomization config
    events: EventCfg = EventCfg()
    # env
    episode_length_s = 2
    decimation = 2
    action_scale = 1.0
    num_actions = 45
    num_observations = 99
    pos_done_threshold = 0.01
    quat_done_threshold = 0.01


class MovePackageEnv(DirectRLEnv):
    cfg: MovePackageEnvCfg

    def __init__(self, cfg: MovePackageEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.robot_dof_targets = torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)
        self.robot_dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.dt = self.cfg.sim.dt * self.cfg.decimation
        self.object_default_pos = torch.tensor(self.cfg.box.init_state.pos, dtype=torch.float, device=self.device)
        # unit tensors
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.box = RigidObject(self.cfg.box)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate (no need to filter for this environment)
        self.scene.clone_environments(copy_from_source=False)
        # register humanoid and box
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["box"] = self.box
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
        targets = self.robot_dof_targets + self.dt * self.actions * self.cfg.action_scale
        self.robot_dof_targets[:] = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        self.robot.set_joint_position_target(targets)

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
        self.goal_pos_w[env_ids, 0] += 3.0
        self.goal_pos_w[env_ids, :2] += rand_floats
        rand_floats = sample_uniform(-1.0, 1.0, (len(env_ids)), device=self.device)
        self.goal_quat[env_ids, :] = quat_from_angle_axis(rand_floats * torch.pi, self.z_unit_tensor[env_ids])
        # update goal pose and markers
        self.goal_markers.visualize(self.goal_pos_w, self.goal_quat)

    def _get_observations(self) -> dict:
        states = torch.cat(
            (
                # humanoid
                self.robot.data.root_pos_w - self.scene.env_origins,
                self.robot.data.joint_pos,
                self.robot.data.joint_vel,
                # box
                self.box.data.root_state_w, - self.scene.env_origins,
                self.box.data.root_quat_w,
                self.box.data.root_vel_w,
                self.box.data.root_lin_vel_w,
                self.box.data.root_ang_vel_w,
                # goal
                self.goal_pos_w - self.scene.env_origins,
                self.goal_quat,
                # actions
                self.actions,
            ),
            dim=-1,
        )
        return {"policy": states}