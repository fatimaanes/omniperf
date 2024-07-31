# run this envionemnt with command: 


from __future__ import annotations
from collections.abc import Sequence


from humanoid.tasks.data import H1_WITHOUT_HAND_CFG, H1_WITH_HAND_CFG
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.assets.articulation.articulation import Articulation
from omni.isaac.lab.assets.asset_base_cfg import AssetBaseCfg
from omni.isaac.lab.assets.rigid_object.rigid_object import RigidObject
from omni.isaac.lab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg
from omni.isaac.lab.envs import DirectRLEnvCfg, mdp
from omni.isaac.lab.envs.direct_rl_env import DirectRLEnv
from omni.isaac.lab.managers.scene_entity_cfg import SceneEntityCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.sim.spawners.from_files.from_files import spawn_ground_plane
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.managers import EventTermCfg as EventTerm

import torch
import omni.isaac.lab.sim as sim_utils
    
@configclass
class WalkEnvCfg(DirectRLEnvCfg):
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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=6.0, replicate_physics=False)
    # robot
    robot: ArticulationCfg = H1_WITHOUT_HAND_CFG.replace(prim_path="/World/envs/env_.*/Robot") # type: ignore

    # env
    episode_length_s = 500
    decimation = 4
    action_scale = 0.25
    num_actions = 19
    num_observations = 99
    pos_done_threshold = 0.05


class WalkEnv(DirectRLEnv):
    cfg: WalkEnvCfg

    def __init__(self, cfg: WalkEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.robot_dof_targets = torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)
        self.robot_dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.dt = self.cfg.sim.dt * self.cfg.decimation
        # unit tensors
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
    
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        # targets = self.robot_dof_targets + self.dt * self.actions * self.cfg.action_scale
        control_type = "torque"
        if control_type == "position":
            targets = self.actions * self.cfg.action_scale
            self.robot_dof_targets[:] = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
            self.robot.set_joint_position_target(targets)
        elif control_type == "torque":
            targets = self.actions
            self.robot.set_joint_effort_target(targets)
        self.robot.write_data_to_sim()

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return time_out, time_out
    
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
                # actions
                self.actions,
            ),
            dim=-1,
        )
        return {"policy": states}
    
    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)