# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to benchmark RL agent with RL-Games."""

"""Launch Isaac Sim Simulator first."""

import argparse
import time

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--max_iterations", type=int, default=10, help="RL Policy training iterations.")
parser.add_argument(
    "--benchmark_backend",
    type=str,
    default="OsmoKPIFile",
    choices=["LocalLogMetrics", "JSONFileMetrics", "OsmoKPIFile"],
    help="Benchmarking backend options, defaults OsmoKPIFile",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, _ = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

app_start_time_begin = time.perf_counter_ns()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

app_start_time_end = time.perf_counter_ns()

"""Rest everything follows."""

# enable benchmarking extension
from omni.isaac.core.utils.extensions import enable_extension

enable_extension("omni.isaac.benchmark.services")
from omni.isaac.benchmark.services import BaseIsaacBenchmark

imports_time_begin = time.perf_counter_ns()

import gymnasium as gym
import math
import os
import torch
from datetime import datetime

from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner

from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml

import omni.isaac.lab_tasks  # noqa: F401
import humanoid.tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

imports_time_end = time.perf_counter_ns()

from omni.isaac.lab.utils.timer import Timer
from utils import (
    log_app_start_time,
    log_python_imports_time,
    log_rl_policy_episode_lengths,
    log_rl_policy_rewards,
    log_runtime_step_times,
    log_scene_creation_time,
    log_simulation_start_time,
    log_task_start_time,
    log_total_start_time,
    parse_tf_logs,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


# Create the benchmark
benchmark = BaseIsaacBenchmark(
    benchmark_name="benchmark_rlgames_train",
    workflow_metadata={
        "metadata": [
            {"name": "task", "data": args_cli.task},
            {"name": "seed", "data": args_cli.seed},
            {"name": "num_envs", "data": args_cli.num_envs},
            {"name": "max_iterations", "data": args_cli.max_iterations},
        ]
    },
    backend_type=args_cli.benchmark_backend,
)


def main():
    """Train with RL-Games agent."""
    # parse seed from command line
    args_cli_seed = args_cli.seed

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )

    agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")
    # override from command line
    if args_cli_seed is not None:
        agent_cfg["params"]["seed"] = args_cli_seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs
    log_dir = agent_cfg["params"]["config"].get("full_experiment_name", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    # set directory into agent config
    # logging directory path: <train_dir>/<full_experiment_name>
    agent_cfg["params"]["config"]["train_dir"] = log_root_path
    agent_cfg["params"]["config"]["full_experiment_name"] = log_dir

    # multi-gpu training config
    if args_cli.distributed:
        agent_cfg["params"]["seed"] += app_launcher.global_rank
        agent_cfg["params"]["config"]["device"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["device_name"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["multi_gpu"] = True
        # update env config device
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"

    # max iterations
    if args_cli.max_iterations:
        agent_cfg["params"]["config"]["max_epochs"] = args_cli.max_iterations

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_root_path, log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_root_path, log_dir, "params", "agent.pkl"), agent_cfg)

    # read configurations about the agent-training
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    task_startup_time_begin = time.perf_counter_ns()

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rl-games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

    task_startup_time_end = time.perf_counter_ns()

    # register the environment to rl-games registry
    # note: in agents configuration: environment name must be "rlgpu"
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    # create runner from rl-games
    runner = Runner(IsaacAlgoObserver())
    runner.load(agent_cfg)

    # set seed of the env
    env.seed(agent_cfg["params"]["seed"])
    # reset the agent and env
    runner.reset()

    benchmark.set_phase("sim_runtime")

    # train the agent
    runner.run({"train": True, "play": False, "sigma": None})

    benchmark.store_measurements()

    # parse tensorboard file stats
    tensorboard_log_dir = os.path.join(log_root_path, log_dir, "summaries")
    log_data = parse_tf_logs(tensorboard_log_dir)

    # prepare RL timing dict
    rl_training_times = {
        "Environment only step time": log_data["performance/step_time"],
        "Environment + Inference step time": log_data["performance/step_inference_time"],
        "Environment + Inference + Policy update time": log_data["performance/rl_update_time"],
        "Environment only FPS": log_data["performance/step_fps"],
        "Environment + Inference FPS": log_data["performance/step_inference_fps"],
        "Environment + Inference + Policy update FPS": log_data["performance/step_inference_rl_update_fps"],
    }

    # log additional metrics to benchmark services
    log_app_start_time(benchmark, (app_start_time_end - app_start_time_begin) / 1e6)
    log_python_imports_time(benchmark, (imports_time_end - imports_time_begin) / 1e6)
    log_task_start_time(benchmark, (task_startup_time_end - task_startup_time_begin) / 1e6)
    log_scene_creation_time(benchmark, Timer.get_timer_info("scene_creation") * 1000)
    log_simulation_start_time(benchmark, Timer.get_timer_info("simulation_start") * 1000)
    log_total_start_time(benchmark, (task_startup_time_end - app_start_time_begin) / 1e6)
    log_runtime_step_times(benchmark, rl_training_times, compute_stats=True)
    log_rl_policy_rewards(benchmark, log_data["rewards/iter"])
    log_rl_policy_episode_lengths(benchmark, log_data["episode_lengths/iter"])

    benchmark.stop()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
