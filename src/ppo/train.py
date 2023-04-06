import os
from typing import Optional, Union

from gymnasium.vector import SyncVectorEnv
from tqdm.autonotebook import tqdm

import wandb
from src.config import (EnvironmentConfig, OnlineTrainConfig, RunConfig,
                        TransformerModelConfig, LSTMModelConfig)

from .memory import Memory
from .utils import store_model_checkpoint
from .agent import get_agent, PPOAgent


def train_ppo(
        run_config: RunConfig,
        online_config: OnlineTrainConfig,
        environment_config: EnvironmentConfig,
        model_config: Optional[Union[TransformerModelConfig, LSTMModelConfig]],
        envs: SyncVectorEnv,
        trajectory_writer=None) -> PPOAgent:
    """
    Trains a PPO agent on a given environment.

    Args:
    - run_config (RunConfig): An object containing general run configuration details.
    - online_config (OnlineTrainConfig): An object containing online training configuration details.
    - environment_config (EnvironmentConfig): An object containing environment-specific configuration details.
    - model_config (Optional[Union[TransformerModelConfig, LSTMModelConfig]]): An optional object containing either Transformer or LSTM model configuration details.
    - envs (SyncVectorEnv): The environment in which to perform training.
    - trajectory_writer (optional): An optional object for writing trajectories to a file.

    Returns:
    - agent (PPOAgent): The trained PPO agent.
    """

    memory = Memory(envs, online_config, run_config.device)
    agent = get_agent(model_config, envs, environment_config, online_config)
    num_updates = online_config.total_timesteps // online_config.batch_size

    optimizer, scheduler = agent.make_optimizer(
        num_updates=num_updates,
        initial_lr=online_config.learning_rate,
        end_lr=online_config.learning_rate if not online_config.decay_lr else 0.0)

    checkpoint_num = 1
    if run_config.track:
        video_path = os.path.join("videos", run_config.run_name)
        prepare_video_dir(video_path)
        videos = []
        checkpoint_artifact = wandb.Artifact(
            f"{run_config.exp_name}_checkpoints", type="model")
        checkpoint_interval = num_updates // online_config.num_checkpoints + 1
        checkpoint_num = store_model_checkpoint(
            agent, online_config, run_config, checkpoint_num, checkpoint_artifact)

    progress_bar = tqdm(range(num_updates), position=0, leave=True)
    for n in progress_bar:

        agent.rollout(memory, online_config.num_steps, envs, trajectory_writer)
        agent.learn(memory, online_config, optimizer,
                    scheduler, run_config.track)

        if run_config.track:
            memory.log()
            videos = check_and_upload_new_video(
                video_path=video_path, videos=videos, step=memory.global_step)
            if (n+1) % checkpoint_interval == 0:
                checkpoint_num = store_model_checkpoint(
                    agent, online_config, run_config, checkpoint_num, checkpoint_artifact)

        output = memory.get_printable_output()
        progress_bar.set_description(output)

        memory.reset()

    if run_config.track:
        checkpoint_num = store_model_checkpoint(
            agent, online_config, run_config, checkpoint_num, checkpoint_artifact)
        wandb.log_artifact(checkpoint_artifact)  # Upload checkpoints to wandb

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


def prepare_video_dir(video_path):
    if not os.path.exists(video_path):
        os.makedirs(video_path)
    videos = [i for i in os.listdir(video_path) if i.endswith(".mp4")]
    for video in videos:
        os.remove(os.path.join(video_path, video))
    videos = []
