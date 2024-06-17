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
from omni.isaac.lab.sensors import TiledCamera, TiledCameraCfg, save_images_to_file
from omni.isaac.lab.sim import PhysxCfg, SimulationCfg
from omni.isaac.lab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.utils import configclass

from .shadow_hand_env import ShadowHandEnv
from .shadow_hand_env_cfg import ShadowHandEnvCfg


@configclass
class ShadowHandRGBCameraEnvCfg(ShadowHandEnvCfg):
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=32, env_spacing=5.0, replicate_physics=True)

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
            gpu_found_lost_pairs_capacity=2**18,
            gpu_found_lost_aggregate_pairs_capacity=2**10,
            gpu_total_aggregate_pairs_capacity=2**10,
        ),
    )

    # camera
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, -0.2, 2.0), rot=(0.0, 0.0, 0.0, -1.0), convention="opengl"),
        # offset=TiledCameraCfg.OffsetCfg(pos=(-2.0, 0.0, 0.75), rot=(-0.5, -0.5, 0.5, 0.5), convention="opengl"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width=320,
        height=240,
    )
    write_image_to_file = False

    # env
    num_channels = 3
    num_observations = num_channels * tiled_camera.height * tiled_camera.width #+ 157


@configclass
class ShadowHandDepthCameraEnvCfg(ShadowHandEnvCfg):
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=32, env_spacing=5.0, replicate_physics=True)

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
            gpu_found_lost_pairs_capacity=2**18,
            gpu_found_lost_aggregate_pairs_capacity=2**10,
            gpu_total_aggregate_pairs_capacity=2**10,
        ),
    )

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
        self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
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

        # if self.cfg.obs_type == "openai":
        #     obs = self.compute_reduced_observations()
        # elif self.cfg.obs_type == "full":
        #     obs = self.compute_full_observations()
        # else:
        #     print("Unknown observations type!")

        data_type = "rgb" if "rgb" in self.cfg.tiled_camera.data_types else "depth"
        camera_data = self._tiled_camera.data.output[data_type].clone()
        observations = {"policy": camera_data}
        if self.cfg.asymmetric_obs:
            observations = {"policy": camera_data, "critic": states}

        if self.cfg.write_image_to_file:
            data_scale = 1 if "rgb" in self.cfg.tiled_camera.data_types else 2.5
            img = camera_data / data_scale
            save_images_to_file(img, f"shadow_hand_{data_type}.png")

        return observations
