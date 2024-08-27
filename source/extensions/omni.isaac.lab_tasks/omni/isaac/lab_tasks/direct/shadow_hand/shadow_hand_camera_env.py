# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import os
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from collections.abc import Sequence

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import TiledCamera, TiledCameraCfg, save_images_to_file
from omni.isaac.lab.sim import PhysxCfg, SimulationCfg
from omni.isaac.lab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.direct.inhand_manipulation.inhand_manipulation_env import InHandManipulationEnv, unscale
from .shadow_hand_env_cfg import ShadowHandEnvCfg

from omni.isaac.lab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul


@configclass
class ShadowHandRGBCameraEnvCfg(ShadowHandEnvCfg):
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=256, env_spacing=5.0, replicate_physics=True)

    # camera
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0, -0.35, 1.0), rot=(0.7071, 0., 0.7071, 0.), convention="world"),
        # offset=TiledCameraCfg.OffsetCfg(pos=(0.0, -0.27, 1.5), rot=(0.0, 0.0, 0.0, -1.0), convention="opengl"),
        # offset=TiledCameraCfg.OffsetCfg(pos=(-0.1, -0.9, 0.92), rot=(0.866, 0.5, 0.0, 0.0), convention="opengl"),
        # offset=TiledCameraCfg.OffsetCfg(pos=(-0.9, -0.3, 0.6), rot=(-0.5, -0.5, 0.5, 0.5), convention="opengl"),
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width = int(os.environ.get("ISAAC_LAB_CAMERA_WIDTH", 320)),
        height = int(os.environ.get("ISAAC_LAB_CAMERA_HEIGHT", 240)),
    )
    write_image_to_file = True

    # env
    num_channels = 3
    #num_observations = 157-17+27+24+24#649#536 #num_channels * tiled_camera.height * tiled_camera.width #+ 157
    num_observations = num_channels * tiled_camera.height * tiled_camera.width


@configclass
class ShadowHandRGBDCameraEnvCfg(ShadowHandEnvCfg):
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=32, env_spacing=5.0, replicate_physics=True)

    # camera
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(-0.2, -0.2, 2.0), rot=(0.0, 0.0, 0.0, -1.0), convention="opengl"),
        # offset=TiledCameraCfg.OffsetCfg(pos=(-2.0, 0.0, 0.75), rot=(-0.5, -0.5, 0.5, 0.5), convention="opengl"),
        data_types=["rgba", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width = int(os.environ.get("ISAAC_LAB_CAMERA_WIDTH", 320)),
        height = int(os.environ.get("ISAAC_LAB_CAMERA_HEIGHT", 240)),
    )
    write_image_to_file = False

    # env
    num_channels = 4
    num_observations = num_channels * tiled_camera.height * tiled_camera.width


@configclass
class ShadowHandDepthCameraEnvCfg(ShadowHandEnvCfg):
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=32, env_spacing=5.0, replicate_physics=True)

    # camera
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, -0.2, 2.0), rot=(0.0, 0.0, 0.0, -1.0), convention="opengl"),
        # offset=TiledCameraCfg.OffsetCfg(pos=(-2.0, 0.0, 0.75), rot=(-0.5, -0.5, 0.5, 0.5), convention="opengl"),
        data_types=["depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width = int(os.environ.get("ISAAC_LAB_CAMERA_WIDTH", 320)),
        height = int(os.environ.get("ISAAC_LAB_CAMERA_HEIGHT", 240)),
    )
    write_image_to_file = False

    # env
    num_channels = 1
    num_observations = num_channels * tiled_camera.height * tiled_camera.width #+ 157


@configclass
class ShadowHandRGBCameraAsymmetricEnvCfg(ShadowHandRGBCameraEnvCfg):
    # env
    asymmetric_obs = True
    num_states = 187


@configclass
class ShadowHandDepthCameraAsymmetricEnvCfg(ShadowHandDepthCameraEnvCfg):
    # env
    asymmetric_obs = True
    num_states = 187

    

class ShadowHandCameraEnv(InHandManipulationEnv):
    cfg: ShadowHandEnvCfg

    def __init__(self, cfg: ShadowHandEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # self.goal_pos[:, :] = torch.tensor([-0.15, -0.15, 0.5], device=self.device)
        #self._get_embeddings_model()
        # hide goal cubes
        self.goal_pos[:, :] = torch.tensor([-0.2, -0.45, 10.0], device=self.device)

    def _configure_gym_env_spaces(self):
        """Configure the action and observation spaces for the Gym environment."""
        # observation space (unbounded since we don't impose any limits)
        self.num_actions = self.cfg.num_actions
        self.num_observations = self.cfg.num_observations
        self.num_states = self.cfg.num_states

        # set up spaces
        self.single_observation_space = gym.spaces.Dict()
        self.single_observation_space["policy"] = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.cfg.tiled_camera.height, self.cfg.tiled_camera.width, self.cfg.num_channels),
        )
        if self.num_states > 0:
            self.single_observation_space["critic"] = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.cfg.tiled_camera.height, self.cfg.tiled_camera.width, self.cfg.num_channels),
            )
        self.single_action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_actions,))

        # batch the spaces for vectorized environments
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)

        # RL specifics
        self.actions = torch.zeros(self.num_envs, self.num_actions, device=self.sim.device)

    def _setup_scene(self):
        # add hand, in-hand object, and goal object
        self.hand = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
        # clone and replicate (no need to filter for this environment)
        self.scene.clone_environments(copy_from_source=False)
        # add articultion to scene - we must register to scene to randomize with EventManager
        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["object"] = self.object
        self.scene.sensors["tiled_camera"] = self._tiled_camera
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        
    def compute_camera_observations(self):
        rgb_img = self._tiled_camera.data.output["rgb"].clone()
        mean_tensor = torch.mean(rgb_img, dim=(1, 2), keepdim=True)
        rgb_img -= mean_tensor
        # depth_img = self._tiled_camera.data.output["depth"].clone()
        # depth_img[depth_img==float("inf")] = 0
        # depth_img /= 5.0
        # depth_img /= torch.max(depth_img)
        # rgb_embeddings = self.rgb_model(rgb_img).squeeze()
        # # # relative_pose = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))
        # # # rgb_embeddings[:, 3:] = relative_pose.clone()
        # data_type = "rgb" if "rgb" in self.cfg.tiled_camera.data_types else "rgbd"
        # if "rgb" in self.cfg.tiled_camera.data_types:
        #     camera_data = self._tiled_camera.data.output[data_type]
        #     # normalize the camera data for better training results
        #     mean_tensor = torch.mean(camera_data, dim=(1, 2), keepdim=True)
        #     camera_data -= mean_tensor
        # elif "depth" in self.cfg.tiled_camera.data_types:
        #     camera_data = self._tiled_camera.data.output[data_type]
        #     camera_data[camera_data == float("inf")] = 0
        # observations = {"policy": camera_data.clone()}

        # if self.cfg.write_image_to_file:
        #     save_images_to_file(rgb_img, f"shadow_hand_{data_type}.png")

        return rgb_img

    # def compute_embeddings_observations(self):
    #     rgb_img = self._tiled_camera.data.output["rgb"].clone()
    #     mean_tensor = torch.mean(rgb_img, dim=(1, 2), keepdim=True)
    #     rgb_img -= mean_tensor
    #     # depth_img = self._tiled_camera.data.output["depth"].clone()
    #     # depth_img[depth_img==float("inf")] = 0
    #     # depth_img /= 5.0
    #     # depth_img /= torch.max(depth_img)
    #     # rgb_embeddings = self.rgb_model(rgb_img).squeeze()
    #     # # relative_pose = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))
    #     # # rgb_embeddings[:, 3:] = relative_pose.clone()
    #     gt_keypoints = gen_keypoints(pose=torch.cat((self.object_pos, self.object_rot), dim=1))
    #     for i in range(3):
    #         gt_keypoints.view(-1, 24)[:, i] -= self.object.data.default_root_state[:, i]
    #     object_pose = torch.cat([self.object_pos, gt_keypoints.view(-1, 24)], dim=-1)

    #     pose_loss, rgb_embeddings = self.trainer.step(rgb_img, object_pose)

    #     if "log" not in self.extras:
    #         self.extras["log"] = dict()
    #     self.extras["log"]["pose_loss"] = pose_loss

    #     rgb_embeddings_clone = rgb_embeddings.clone().detach()
    #     # rgb_embeddings_clone[:, 3:] = gt_keypoints.view(-1, 24).clone()

    #     goal_keypoints = gen_keypoints(pose=torch.cat((torch.zeros_like(self.goal_pos), self.goal_rot), dim=-1))
    #     # zero_pos_obj_keypoints = gen_keypoints(pose=torch.cat((torch.zeros_like(self.goal_pos), self.object_rot), dim=-1))
    #     predicted_cube_pos = rgb_embeddings_clone[:, :3]
    #     zero_pos_obj_keypoints = rgb_embeddings_clone[:, 3:]
    #     for i in range(3):
    #         zero_pos_obj_keypoints[:, i] -= predicted_cube_pos[:, i]

    #     obs = torch.cat(
    #         (
    #             rgb_embeddings_clone,
    #             goal_keypoints.view(-1, 24),
    #             goal_keypoints.view(-1, 24)-zero_pos_obj_keypoints,
    #             # depth_embeddings
    #         ),
    #         dim=-1
    #     )

    #     return obs

    def compute_state_observations(self):
        obs = torch.cat(
            (
                # hand
                unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),  # 0:24
                self.cfg.vel_obs_scale * self.hand_dof_vel,  # 24:48
                # goal
                self.in_hand_pos,  # 61:64
                self.goal_rot,  # 64:68
                # fingertips
                self.fingertip_pos.view(self.num_envs, self.num_fingertips * 3),  # 72:87
                self.fingertip_rot.view(self.num_envs, self.num_fingertips * 4),  # 87:107
                self.fingertip_velocities.view(self.num_envs, self.num_fingertips * 6),  # 107:137
                # actions
                self.actions,  # 137:157
            ),
            dim=-1,
        )
        return obs

    def _get_observations(self) -> dict:
        # state_obs = self.compute_state_observations()
        # embedding_obs = self.compute_embeddings_observations()
        # obs = torch.cat(
        #     (state_obs, embedding_obs), dim=-1
        # )

        obs = self.compute_camera_observations()
        observations = {"policy": obs}
        return observations
        # if self.cfg.asymmetric_obs:
        #     self.fingertip_force_sensors = self.hand.root_physx_view.get_link_incoming_joint_force()[
        #         :, self.finger_bodies
        #     ]
        #     states = self.compute_full_state()

        # if self.cfg.num_channels == 1:
        #     # depth
        #     data_type = "depth"
        #     camera_data = self._tiled_camera.data.output[data_type].clone()
        #     camera_data[camera_data==float("inf")] = 0
        #     camera_data /= 5.0
        #     self._save_images("depth", camera_data)
        #     observations = {"policy": camera_data}
        #     if self.cfg.asymmetric_obs:
        #         observations = {"policy": camera_data, "critic": states}
        # elif self.cfg.num_channels == 3:
        #     # RGB
        #     data_type = "rgba"
        #     rgb_data = 1 - self._tiled_camera.data.output[data_type][..., :3].clone()
        #     self._save_images("rgb", rgb_data)
        #     observations = {"policy": rgb_data}
        #     if self.cfg.asymmetric_obs:
        #         observations = {"policy": rgb_data, "critic": states}
        # elif self.cfg.num_channels == 4:
        #     # RGB+D
        #     depth_data = self._tiled_camera.data.output["depth"].clone()
        #     depth_data[depth_data==float("inf")] = 0
        #     depth_data /= 2.0
        #     rgb_data = 1 - self._tiled_camera.data.output["rgba"][..., :3].clone()
        #     self._save_images("rgb", rgb_data)
        #     self._save_images("depth", depth_data)
        #     camera_data = torch.cat((rgb_data, depth_data), dim=-1)
        #     observations = {"policy": camera_data}
        #     if self.cfg.asymmetric_obs:
        #         observations = {"policy": camera_data, "critic": states}

        # return observations

    def _save_images(self, data_type, camera_data):
        if self.cfg.write_image_to_file:
            save_images_to_file(camera_data, f"shadow_hand_{data_type}.png")


@torch.jit.script
def local_to_world_space(pos_offset_local: torch.Tensor, pose_global: torch.Tensor):
    """ Convert a point from the local frame to the global frame
    Args:
        pos_offset_local: Point in local frame. Shape: [N, 3]
        pose_global: The spatial pose of this point. Shape: [N, 7]
    Returns:
        Position in the global frame. Shape: [N, 3]
    """
    quat_pos_local = torch.cat(
        [
            torch.zeros(pos_offset_local.shape[0], 1, dtype=torch.float32, device=pos_offset_local.device), 
            pos_offset_local
        ], dim=-1
    )
    quat_trans = torch.cat(
        [
            torch.zeros(pos_offset_local.shape[0], 1, dtype=torch.float32, device=pos_offset_local.device), 
            pose_global[:, 0:3]
        ], dim=-1
    )
    quat_global = pose_global[:, 3:7]
    quat_global_conj = quat_conjugate(quat_global)
    pos_offset_global = quat_mul(quat_global, quat_mul(quat_pos_local, quat_global_conj))[:, 1:4]

    result_pos_gloal = pos_offset_global + pose_global[:, 0:3]

    return result_pos_gloal


    size = [2*0.03, 2*0.03, 2*0.03]

@torch.jit.script
def gen_keypoints(
    pose: torch.Tensor, 
    num_keypoints: int = 8, 
    size: Tuple[float, float, float] = (2*0.03, 2*0.03, 2*0.03)
):

    num_envs = pose.shape[0]
    keypoints_buf = torch.ones(num_envs, num_keypoints, 3, dtype=torch.float32, device=pose.device)
    for i in range(num_keypoints):
        # which dimensions to negate
        n = [((i >> k) & 1) == 0 for k in range(3)]
        corner_loc = [(1 if n[k] else -1) * s / 2 for k, s in enumerate(size)],
        corner = torch.tensor(corner_loc, dtype=torch.float32, device=pose.device) * keypoints_buf[:, i, :]
        keypoints_buf[:, i, :] = local_to_world_space(corner, pose)
    return keypoints_buf