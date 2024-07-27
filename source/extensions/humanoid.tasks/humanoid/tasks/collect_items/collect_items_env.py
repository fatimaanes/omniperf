from __future__ import annotations
from collections.abc import Sequence


from humanoid.tasks.data import H1_WITH_HAND_CFG
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
class EventCfg:
    """Configuration for randomization."""

    # -- tomato_soup_can
    tomato_soup_can_pos = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("tomato_soup_can"),
            "pose_range": {"x": (-2, 2), "y": (-2, 2), "yaw": (-3.14, 3.14)},
            "velocity_range": {},
        },
    )  
    # -- sugar_box
    sugar_box_pos = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("sugar_box"),
            "pose_range": {"x": (-2, 2), "y": (-2, 2), "yaw": (-3.14, 3.14)},
            "velocity_range": {},
        },
    )  
    # -- cracker_box
    cracker_box_pos = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cracker_box"),
            "pose_range": {"x": (-2, 2), "y": (-2, 2), "yaw": (-3.14, 3.14)},
            "velocity_range": {},
        },
    )  
    
@configclass
class CollectItemsEnvCfg(DirectRLEnvCfg):
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
    # bucket
    bucket = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Bucket",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.9, 0, 0.0), rot=(1, 0, 0, 0)),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/Props/SM_BucketPlastic_B.usd"),
    )
    # tomato soup can
    tomato_soup_can: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/TomatoSoupCan",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.8, -0.25, 0.1), rot=(0.7071068, -0.7071068, 0, 0)),
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.8, 0.0, 0.1), rot=(1, 0, 0, 0)),
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.8, 0.35, 0.1), rot=(1, 0, 0, 0)),
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
    # domain randomization config
    events: EventCfg = EventCfg()
    # env
    episode_length_s = 2
    decimation = 2
    action_scale = 1.0
    num_actions = 45
    num_observations = 99
    pos_done_threshold = 0.05


class CollectItemsEnv(DirectRLEnv):
    cfg: CollectItemsEnvCfg

    def __init__(self, cfg: CollectItemsEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.robot_dof_targets = torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)
        self.robot_dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.dt = self.cfg.sim.dt * self.cfg.decimation
        # unit tensors
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.tomato_soup_can = RigidObject(self.cfg.tomato_soup_can)
        self.sugar_box = RigidObject(self.cfg.sugar_box) # yellow box
        self.cracker_box = RigidObject(self.cfg.cracker_box) # red box
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["tomato_soup_can"] = self.tomato_soup_can
        self.scene.rigid_objects["sugar_box"] = self.sugar_box
        self.scene.rigid_objects["cracker_box"] = self.cracker_box
        # add bucket
        self.cfg.bucket.spawn.func(
            self.cfg.bucket.prim_path,
            self.cfg.bucket.spawn, 
            translation=self.cfg.bucket.init_state.pos, 
            orientation=self.cfg.bucket.init_state.rot,
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
        bucket_pos_w = self.scene.env_origins.clone() + torch.tensor(self.cfg.bucket.init_state.pos, device=self.device)
        cracker_box_dist = torch.norm(self.cracker_box.data.root_state_w[:, :3] - bucket_pos_w, p=2, dim=-1)
        sugar_box_dist = torch.norm(self.sugar_box.data.root_state_w[:, :3] - bucket_pos_w, p=2, dim=-1)
        tomato_soup_can_dist = torch.norm(self.tomato_soup_can.data.root_state_w[:, :3] - bucket_pos_w, p=2, dim=-1)
        success_buf = (cracker_box_dist < self.cfg.pos_done_threshold) & (sugar_box_dist < self.cfg.pos_done_threshold) & (tomato_soup_can_dist < self.cfg.pos_done_threshold)
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
                self.tomato_soup_can.data.root_state_w, - self.scene.env_origins,
                self.tomato_soup_can.data.root_quat_w,
                self.sugar_box.data.root_state_w, - self.scene.env_origins,
                self.sugar_box.data.root_quat_w,
                self.cracker_box.data.root_state_w, - self.scene.env_origins,
                self.cracker_box.data.root_quat_w,
                # actions
                self.actions,
            ),
            dim=-1,
        )
        return {"policy": states}