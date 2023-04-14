import os
from argparse import Namespace

import gymnasium.vector
import numpy as np
import torch as t
from einops import rearrange
from tqdm import tqdm

import wandb
from src.models.trajectory_transformer import (
    CloneTransformer,
    DecisionTransformer,
    TrajectoryTransformer,
)

from .utils import get_max_len_from_model_type


def evaluate_dt_agent(
    env_id: str,
    model: TrajectoryTransformer,
    env_func,
    trajectories=300,
    track=False,
    batch_number=0,
    initial_rtg=0.98,
    use_tqdm=True,
    device="cpu",
    num_envs=8,
):
    model.eval()

    env = gymnasium.vector.SyncVectorEnv([env_func for _ in range(num_envs)])
    video_path = os.path.join("videos", env.envs[0].run_name)

    if not hasattr(model, "transformer_config"):
        model.transformer_config = Namespace(
            n_ctx=model.n_ctx,
            time_embedding_type=model.time_embedding_type,
        )

    max_len = get_max_len_from_model_type(
        model.model_type,
        model.transformer_config.n_ctx,
    )

    traj_lengths = []
    rewards = []
    n_terminated = 0
    n_truncated = 0
    reward_total = 0
    n_positive = 0

    if not os.path.exists(video_path):
        os.makedirs(video_path)

    videos = [i for i in os.listdir(video_path) if i.endswith(".mp4")]
    for video in videos:
        os.remove(os.path.join(video_path, video))
    videos = [i for i in os.listdir(video_path) if i.endswith(".mp4")]

    if use_tqdm:
        pbar = tqdm(range(trajectories), desc="Evaluating DT")
        pbar_it = iter(pbar)
    else:
        pbar = range(trajectories)

    # each env will get its own seed by incrementing on the given seed
    obs, _ = env.reset(seed=0)
    obs = t.tensor(obs["image"]).unsqueeze(1)
    rtg = rearrange(t.ones(num_envs, dtype=t.int) * initial_rtg, "e -> e 1 1")
    a = rearrange(t.zeros(num_envs, dtype=t.int), "e -> e 1 1")
    timesteps = rearrange(t.zeros(num_envs, dtype=t.int), "e -> e 1 1")

    obs = obs.to(device)
    rtg = rtg.to(device)
    a = a.to(device)
    timesteps = timesteps.to(device)

    if model.transformer_config.time_embedding_type == "linear":
        timesteps = timesteps.to(t.float32)

    # get first action
    if isinstance(model, DecisionTransformer):
        state_preds, action_preds, reward_preds = model.forward(
            states=obs, actions=None, rtgs=rtg, timesteps=timesteps
        )
    elif isinstance(model, CloneTransformer):
        state_preds, action_preds = model.forward(
            states=obs, actions=None, timesteps=timesteps
        )
    else:
        raise ValueError("Model type not supported")

    new_action = t.argmax(action_preds, dim=-1).squeeze(-1)
    new_obs, new_reward, terminated, truncated, info = env.step(new_action)

    current_trajectory_length = t.ones(num_envs, dtype=t.int)
    while n_terminated + n_truncated < trajectories:
        # concat init obs to new obs
        obs = t.cat(
            [obs, t.tensor(new_obs["image"]).unsqueeze(1).to(device)], dim=1
        )

        # add new reward to init reward
        rtg = t.cat(
            [
                rtg,
                rtg[:, -1:, :]
                - rearrange(t.tensor(new_reward).to(device), "e -> e 1 1"),
            ],
            dim=1,
        )

        # add new timesteps
        timesteps = t.cat(
            [
                timesteps,
                rearrange(current_trajectory_length.to(device), "e -> e 1 1"),
            ],
            dim=1,
        )

        if model.transformer_config.time_embedding_type == "linear":
            timesteps = timesteps.to(t.float32)

        a = t.cat([a, rearrange(new_action, "e -> e 1 1")], dim=1)

        # truncations:
        obs = obs[:, -max_len:] if obs.shape[1] > max_len else obs
        actions = (
            a[:, -(obs.shape[1] - 1) :]
            if (a.shape[1] > 1 and max_len > 1)
            else None
        )
        timesteps = (
            timesteps[:, -max_len:]
            if timesteps.shape[1] > max_len
            else timesteps
        )
        rtg = rtg[:, -max_len:] if rtg.shape[1] > max_len else rtg

        if isinstance(model, DecisionTransformer):
            state_preds, action_preds, reward_preds = model.forward(
                states=obs, actions=actions, rtgs=rtg, timesteps=timesteps
            )
        elif isinstance(model, CloneTransformer):
            state_preds, action_preds = model.forward(
                states=obs, actions=actions, timesteps=timesteps
            )
        else:
            raise NotImplementedError(
                "Model type not supported for evaluation."
            )

        new_action = t.argmax(action_preds, dim=-1).squeeze(-1)
        if new_action.dim() > 1:
            new_action = new_action[:, -1]
        # convert to numpy
        new_obs, new_reward, terminated, truncated, info = env.step(new_action)
        # print(f"took action  {action} at timestep {i} for reward {new_reward}")

        n_positive = n_positive + sum(new_reward > 0)
        reward_total += sum(new_reward)
        n_terminated += sum(terminated)
        n_truncated += sum(truncated)

        if use_tqdm:
            pbar.set_description(
                f"Evaluating DT: Finished running {n_terminated + n_truncated} episodes."
                f"Current episodes are at timestep {current_trajectory_length.tolist()} for reward {new_reward}"
            )

        dones = np.logical_or(terminated, truncated)
        current_trajectory_length += np.invert(dones)
        traj_lengths.extend(current_trajectory_length[dones].tolist())
        rewards.extend(new_reward[dones])
        current_trajectory_length[dones] = 0

        if np.any(dones):
            if use_tqdm:
                [next(pbar_it, None) for _ in range(sum(dones))]

            current_videos = [
                i for i in os.listdir(video_path) if i.endswith(".mp4")
            ]
            if track and (
                len(current_videos) > len(videos)
            ):  # we have a new video
                new_videos = [i for i in current_videos if i not in videos]
                for new_video in new_videos:
                    path_to_video = os.path.join(video_path, new_video)
                    wandb.log(
                        {
                            f"media/video/{initial_rtg}/": wandb.Video(
                                path_to_video,
                                fps=4,
                                format="mp4",
                                caption=f"{env_id}, after {n_terminated + n_truncated} episodes, reward {new_reward}, rtg {initial_rtg}",
                            )
                        },
                        step=batch_number,
                    )
            videos = current_videos  # update videos

    collected_trajectories = n_terminated + n_truncated

    statistics = {
        "initial_rtg": initial_rtg,
        "prop_completed": n_terminated / collected_trajectories,
        "prop_truncated": n_truncated / collected_trajectories,
        "mean_reward": reward_total / collected_trajectories,
        "prop_positive_reward": n_positive / collected_trajectories,
        "mean_traj_length": sum(traj_lengths) / collected_trajectories,
        "traj_lengths": traj_lengths,
        "rewards": rewards,
    }

    env.close()
    if track:
        # log statistics at batch number but prefix with eval
        for key, value in statistics.items():
            if key == "initial_rtg":
                continue
            if key == "traj_lengths":
                wandb.log(
                    {
                        f"eval/{str(initial_rtg)}/traj_lengths": wandb.Histogram(
                            value
                        )
                    },
                    step=batch_number,
                )
            elif key == "rewards":
                wandb.log(
                    {
                        f"eval/{str(initial_rtg)}/rewards": wandb.Histogram(
                            value
                        )
                    },
                    step=batch_number,
                )
            wandb.log(
                {f"eval/{str(initial_rtg)}/" + key: value}, step=batch_number
            )

    return statistics
