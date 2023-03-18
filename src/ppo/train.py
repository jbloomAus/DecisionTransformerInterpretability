import os
from typing import Optional

import torch as t
from gymnasium.vector import SyncVectorEnv
from tqdm.autonotebook import tqdm

import wandb
from src.config import (EnvironmentConfig, OnlineTrainConfig, RunConfig,
                        TransformerModelConfig)

from .agent import FCAgent, TrajPPOAgent
from .memory import Memory
from .utils import get_printable_output_for_probe_envs

device = t.device("cuda" if t.cuda.is_available() else "cpu")


def train_ppo(
        run_config: RunConfig,
        online_config: OnlineTrainConfig,
        environment_config: EnvironmentConfig,
        transformer_model_config: Optional[TransformerModelConfig],
        envs: SyncVectorEnv,
        trajectory_writer=None,
        probe_idx=None):
    """
    Trains a PPO agent on a given environment.

    Args:
    - args: an instance of PPOArgs containing the hyperparameters for training
    - envs: the environment to train on
    - trajectory_writer: an optional object to write trajectories to a file
    - probe_idx: index of probe environment, if training on probe environment

    Returns:
    None
    """

    memory = Memory(envs, online_config, device)

    if transformer_model_config is None:
        agent = FCAgent(
            envs,
            device=device,
            hidden_dim=online_config.hidden_size
        )
    else:
        agent = TrajPPOAgent(
            envs=envs,
            transformer_model_config=transformer_model_config,
            environment_config=environment_config,
            device=device,
        )

    num_updates = online_config.total_timesteps // online_config.batch_size

    optimizer, scheduler = agent.make_optimizer(
        num_updates,
        initial_lr=online_config.learning_rate,
        end_lr=online_config.learning_rate if not online_config.decay_lr else 0.0)

    # out = wg.Output(layout={"padding": "15px"})
    # display(out)
    progress_bar = tqdm(range(num_updates), position=0, leave=True)

    if run_config.track:
        video_path = os.path.join("videos", run_config.run_name)
        if not os.path.exists(video_path):
            os.makedirs(video_path)
        videos = [i for i in os.listdir(video_path) if i.endswith(".mp4")]
        for video in videos:
            os.remove(os.path.join(video_path, video))
        videos = [i for i in os.listdir(video_path) if i.endswith(".mp4")]

    for update in progress_bar:

        agent.rollout(memory, online_config.num_steps, envs, trajectory_writer)
        agent.learn(memory, online_config, optimizer,
                    scheduler, run_config.track)

        if run_config.track:
            memory.log()
            videos = check_and_upload_new_video(
                video_path=video_path, videos=videos, step=memory.global_step)

        output = memory.get_printable_output()
        progress_bar.set_description(output)

        memory.reset()

    if trajectory_writer is not None:
        trajectory_writer.tag_terminated_trajectories()
        trajectory_writer.write(upload_to_wandb=run_config.track)

    envs.close()

    return agent


def check_and_upload_new_video(video_path, videos, step=None):
    """
    Checks if new videos have been generated in the video path directory since the last check, and if so,
    uploads them to the current WandB run.

    Args:
    - video_path: The path to the directory where the videos are being saved.
    - videos: A list of the names of the videos that have already been uploaded to WandB.
    - step: The current step in the training loop, used to associate the video with the correct timestep.

    Returns:
    - A list of the names of all the videos currently present in the video path directory.
    """

    current_videos = [i for i in os.listdir(video_path) if i.endswith(".mp4")]
    new_videos = [i for i in current_videos if i not in videos]
    if new_videos:
        for new_video in new_videos:
            path_to_video = os.path.join(video_path, new_video)
            wandb.log({"video": wandb.Video(
                path_to_video,
                fps=4,
                caption=new_video,
                format="mp4",
            )}, step=step)
    return current_videos
