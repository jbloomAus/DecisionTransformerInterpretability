import json
import os
from typing import Optional, Union

import torch as t
from gymnasium.vector import SyncVectorEnv
from tqdm.autonotebook import tqdm
from dataclasses import dataclass

import wandb
from src.config import (EnvironmentConfig, OnlineTrainConfig, RunConfig,
                        TransformerModelConfig, ConfigJsonEncoder, LSTMModelConfig)

from .agent import PPOAgent, FCAgent, TransformerPPOAgent, LSTMPPOAgent
from .memory import Memory

device = t.device("cuda" if t.cuda.is_available() else "cpu")


def train_ppo(
        run_config: RunConfig,
        online_config: OnlineTrainConfig,
        environment_config: EnvironmentConfig,
        model_config: Optional[Union[TransformerModelConfig, LSTMModelConfig]],
        envs: SyncVectorEnv,
        trajectory_writer=None):
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
    agent = get_agent(model_config, envs, environment_config, online_config)
    num_updates = online_config.total_timesteps // online_config.batch_size

    optimizer, scheduler = agent.make_optimizer(
        num_updates=num_updates,
        initial_lr=online_config.learning_rate,
        end_lr=online_config.learning_rate if not online_config.decay_lr else 0.0)

    if run_config.track:
        video_path = os.path.join("videos", run_config.run_name)
        prepare_video_dir(video_path)
        videos = []
        checkpoint_artifact = wandb.Artifact(f"{run_config.exp_name}_checkpoints", type="model")
        checkpoint_interval = num_updates // online_config.num_checkpoints + 1
        checkpoint_num = 1

    def store_model_checkpoint():
        nonlocal checkpoint_num
        checkpoint_name = f"{run_config.exp_name}_{checkpoint_num:0>2}"
        checkpoint_path = f"models/{checkpoint_name}.pt"
        t.save({
            "model_state_dict": agent.state_dict(),
            "online_config": json.dumps(online_config, cls=ConfigJsonEncoder),
        }, checkpoint_path)
        checkpoint_artifact.add_file(local_path=checkpoint_path, name=f"{checkpoint_name}.pt")
        checkpoint_num += 1

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
                store_model_checkpoint()

        output = memory.get_printable_output()
        progress_bar.set_description(output)

        memory.reset()

    if run_config.track:
        store_model_checkpoint()
        wandb.log_artifact(checkpoint_artifact)

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


def get_agent(
        model_config: dataclass,
        envs: SyncVectorEnv,
        environment_config: EnvironmentConfig,
        online_config) -> PPOAgent:
    """
    Returns an agent based on the given configuration.

    Args:
    - transformer_model_config: The configuration for the transformer model.
    - envs: The environment to train on.
    - environment_config: The configuration for the environment.
    - online_config: The configuration for online training.

    Returns:
    - An agent.
    """
    if model_config is not None:
        if isinstance(model_config, TransformerModelConfig):
            agent = TransformerPPOAgent(
                envs=envs,
                transformer_model_config=model_config,
                environment_config=environment_config,
                device=device,
            )
        elif isinstance(model_config, LSTMModelConfig):
            agent = LSTMPPOAgent(
                envs=envs,
                environment_config=environment_config,
                lstm_config=model_config,
                device=device,
            )
    else:
        agent = FCAgent(
            envs,
            device=device,
            hidden_dim=online_config.hidden_size
        )
    return agent
