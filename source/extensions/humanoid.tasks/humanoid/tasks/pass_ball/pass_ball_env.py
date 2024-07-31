from __future__ import annotations
from collections.abc import Sequence


from humanoid.tasks.data import H1_WITH_HAND_CFG
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

import torch
import omni.isaac.lab.sim as sim_utils


@configclass
class PassBallEnvCfg(DirectRLEnvCfg):
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
    # ball
    ball: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Ball",
        spawn=sim_utils.SphereCfg(
            radius=0.06,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.0, 0.0)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.2, 0.35, 1.1)),
    )
    # goal object
    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "goal": sim_utils.SphereCfg(
                radius=0.06,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.3)),
            )
        },
    )
    # env
    episode_length_s = 2
    decimation = 2
    action_scale = 1.0
    num_actions = 45
    num_observations = 99
    pos_done_threshold = 0.01


class PassBallEnv(DirectRLEnv):
    cfg: PassBallEnvCfg

    def __init__(self, cfg: PassBallEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.robot_dof_targets = torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)
        self.robot_dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.dt = self.cfg.sim.dt * self.cfg.decimation
        
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.ball = RigidObject(self.cfg.ball)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate (no need to filter for this environment)
        self.scene.clone_environments(copy_from_source=False)
        # register humanoid and box
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["ball"] = self.ball
        # initialize goal marker and buffer
        self.goal_markers = VisualizationMarkers(self.cfg.goal_object_cfg)
        self.goal_pos_w = self.scene.env_origins.clone()
        self.goal_pos_w[:] += torch.tensor((0.2, -0.35, 1.1), device=self.device)
        self.goal_markers.visualize(self.goal_pos_w)
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
        goal_dist_pos = torch.norm(self.goal_pos_w - self.ball.data.root_state_w[:, :3], p=2, dim=-1)
        success_buf = goal_dist_pos < self.cfg.pos_done_threshold
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

    def _get_observations(self) -> dict:
        states = torch.cat(
            (
                # humanoid
                self.robot.data.root_pos_w - self.scene.env_origins,
                self.robot.data.joint_pos,
                self.robot.data.joint_vel,
                # ball
                self.ball.data.root_state_w, - self.scene.env_origins,
                self.ball.data.root_vel_w,
                self.ball.data.root_lin_vel_w,
                self.ball.data.root_ang_vel_w,
                # goal
                self.goal_pos_w - self.scene.env_origins,
                # actions
                self.actions,
            ),
            dim=-1,
        )
        return {"policy": states}
    
    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)