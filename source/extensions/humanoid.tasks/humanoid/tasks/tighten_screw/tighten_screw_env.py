from __future__ import annotations
from collections.abc import Sequence


from humanoid.tasks.data import H1_WITH_HAND_CFG
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.assets.articulation.articulation import Articulation
from omni.isaac.lab.assets.asset_base_cfg import AssetBaseCfg
from omni.isaac.lab.assets.rigid_object.rigid_object import RigidObject
from omni.isaac.lab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg
from omni.isaac.lab.envs import DirectRLEnvCfg
from omni.isaac.lab.envs.direct_rl_env import DirectRLEnv
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.sim.spawners.from_files.from_files import spawn_ground_plane
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

import torch
import omni.isaac.lab.sim as sim_utils

    
@configclass
class TightenScrewEnvCfg(DirectRLEnvCfg):
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
    # bolt
    bolt: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Bolt",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.8, 0.0, 1.04), rot=(1, 0, 0, 0)),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Factory/factory_bolt_m12_loose/factory_bolt_m12_loose.usd",
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
    # nut
    nut: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Nut",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.8, 0.1, 1.06), rot=(1, 0, 0, 0)),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Factory/factory_nut_m12_loose/factory_nut_m12_loose.usd",
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
    # env
    episode_length_s = 2
    decimation = 2
    action_scale = 1.0
    num_actions = 45
    num_observations = 99
    pos_done_threshold = 0.01
    height_done_threshold = 0.03


class TightenScrewEnv(DirectRLEnv):
    cfg: TightenScrewEnvCfg

    def __init__(self, cfg: TightenScrewEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.robot_dof_targets = torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)
        self.robot_dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.dt = self.cfg.sim.dt * self.cfg.decimation
        # unit tensors
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.bolt = RigidObject(self.cfg.bolt)
        self.nut = RigidObject(self.cfg.nut) # yellow box
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["bolt"] = self.bolt
        self.scene.rigid_objects["nut"] = self.nut
        # add table
        self.cfg.table.spawn.func(
            self.cfg.table.prim_path,
            self.cfg.table.spawn, 
            translation=self.cfg.table.init_state.pos, 
            orientation=self.cfg.table.init_state.rot,
        )
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
        nut_bolt_horizontal_dist = torch.norm(self.nut.data.root_state_w[:, :2] - self.bolt.data.root_state_w[:, :2], p=2, dim=-1)
        success_buf = (nut_bolt_horizontal_dist < self.cfg.pos_done_threshold) & (self.nut.data.root_state_w[:, 2] - self.bolt.data.root_state_w[:, 2] < self.cfg.height_done_threshold)
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
                # object
                self.bolt.data.root_state_w, - self.scene.env_origins,
                self.bolt.data.root_quat_w,
                self.nut.data.root_state_w, - self.scene.env_origins,
                self.nut.data.root_quat_w,
                # actions
                self.actions,
            ),
            dim=-1,
        )
        return {"policy": states}
    
    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)