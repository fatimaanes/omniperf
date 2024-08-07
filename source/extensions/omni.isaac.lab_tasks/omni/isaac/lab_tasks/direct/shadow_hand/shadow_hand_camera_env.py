# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from collections.abc import Sequence

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import Camera, CameraCfg, TiledCamera, TiledCameraCfg, save_images_to_file
#from omni.isaac.lab.sensors.camera.tiled_camera import TiledCameraOld
from omni.isaac.lab.sim import PhysxCfg, SimulationCfg
from omni.isaac.lab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.utils import configclass

from .shadow_hand_env import ShadowHandEnv
from .shadow_hand_env_cfg import ShadowHandEnvCfg


@configclass
class ShadowHandRGBCameraEnvCfg(ShadowHandEnvCfg):
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=128, env_spacing=5.0, replicate_physics=True)

    # camera
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, -0.27, 1.5), rot=(0.0, 0.0, 0.0, -1.0), convention="opengl"),
        # offset=TiledCameraCfg.OffsetCfg(pos=(-2.0, 0.0, 0.75), rot=(-0.5, -0.5, 0.5, 0.5), convention="opengl"),
        data_types=["rgb"],
        # class_type=TiledCameraOld,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width=320,
        height=240,
    )
    # tiled_camera: CameraCfg = CameraCfg(
    #     prim_path="/World/envs/env_.*/Camera",
    #     offset=TiledCameraCfg.OffsetCfg(pos=(0.0, -0.27, 1.5), rot=(0.0, 0.0, 0.0, -1.0), convention="opengl"),
    #     # offset=TiledCameraCfg.OffsetCfg(pos=(-2.0, 0.0, 0.75), rot=(-0.5, -0.5, 0.5, 0.5), convention="opengl"),
    #     data_types=["rgb"],
    #     # class_type=TiledCameraOld,
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
    #     ),
    #     width=320,
    #     height=240,
    # )

    write_image_to_file = False

    # env
    num_channels = 3
    num_observations = num_channels * tiled_camera.height * tiled_camera.width #+ 157


@configclass
class ShadowHandRGBDCameraEnvCfg(ShadowHandEnvCfg):
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=32, env_spacing=5.0, replicate_physics=True)

    # camera
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, -0.2, 2.0), rot=(0.0, 0.0, 0.0, -1.0), convention="opengl"),
        # offset=TiledCameraCfg.OffsetCfg(pos=(-2.0, 0.0, 0.75), rot=(-0.5, -0.5, 0.5, 0.5), convention="opengl"),
        data_types=["rgba", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width=160,
        height=120,
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
        width=320,
        height=240,
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

    

class ShadowHandCameraEnv(ShadowHandEnv):
    cfg: ShadowHandEnvCfg

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
        self._tiled_camera = self.cfg.tiled_camera.class_type(self.cfg.tiled_camera)
        # clone and replicate (no need to filter for this environment)
        self.scene.clone_environments(copy_from_source=False)
        # add articultion to scene - we must register to scene to randomize with EventManager
        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["object"] = self.object
        self.scene.sensors["tiled_camera"] = self._tiled_camera
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _get_observations(self) -> dict:
        if self.cfg.asymmetric_obs:
            self.fingertip_force_sensors = self.hand.root_physx_view.get_link_incoming_joint_force()[
                :, self.finger_bodies
            ]
            states = self.compute_full_state()

        if self.cfg.num_channels == 1:
            # depth
            data_type = "depth"
            camera_data = self._tiled_camera.data.output[data_type].clone()
            camera_data[camera_data==float("inf")] = 0
            camera_data /= 2.0
            self._save_images("depth", camera_data)
            observations = {"policy": camera_data}
            if self.cfg.asymmetric_obs:
                observations = {"policy": camera_data, "critic": states}
        elif self.cfg.num_channels == 3:
            # RGB
            data_type = "rgb"
            rgb_data = self._tiled_camera.data.output[data_type].clone()
            self._save_images("rgb", rgb_data)
            observations = {"policy": rgb_data}
            if self.cfg.asymmetric_obs:
                observations = {"policy": rgb_data, "critic": states}
        elif self.cfg.num_channels == 4:
            # RGB+D
            depth_data = self._tiled_camera.data.output["depth"].clone()
            depth_data[depth_data==float("inf")] = 0
            depth_data /= 2.0
            rgb_data = self._tiled_camera.data.output["rgba"][..., :3].clone()
            self._save_images("rgb", rgb_data)
            self._save_images("depth", depth_data)
            camera_data = torch.cat((rgb_data, depth_data), dim=-1)
            observations = {"policy": camera_data}
            if self.cfg.asymmetric_obs:
                observations = {"policy": camera_data, "critic": states}

        return observations

    def _save_images(self, data_type, camera_data):
        if self.cfg.write_image_to_file:
            save_images_to_file(camera_data, f"shadow_hand_{data_type}_32.png")