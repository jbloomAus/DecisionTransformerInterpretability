import os

import gymnasium.vector
import torch as t
import torch.nn as nn
from einops import rearrange
from tqdm import tqdm
import wandb
from argparse import Namespace
from src.models.trajectory_model import TrajectoryTransformer, DecisionTransformer, CloneTransformer
from .offline_dataset import TrajectoryDataset
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import random_split, DataLoader
import numpy as np
from .utils import get_max_len_from_model_type


def train(
        model: TrajectoryTransformer,
        trajectory_data_set: TrajectoryDataset,
        env,
        make_env,
        batch_size=128,
        lr=0.0001,
        weight_decay=0.0,
        device="cpu",
        track=False,
        train_epochs=100,
        test_epochs=10,
        test_frequency=10,
        eval_frequency=10,
        eval_episodes=10,
        initial_rtg=[0.0, 1.0],
        eval_max_time_steps=100):
    loss_fn = nn.CrossEntropyLoss()
    model = model.to(device)
    optimizer = t.optim.Adam(model.parameters(), lr=lr,
                             weight_decay=weight_decay)

    train_dataset, test_dataset = random_split(
        trajectory_data_set, [0.90, 0.10])

    # Create the train DataLoader
    train_sampler = WeightedRandomSampler(
        weights=trajectory_data_set.sampling_probabilities[train_dataset.indices],
        num_samples=len(train_dataset),
        replacement=True,
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler)

    # Create the test DataLoader
    test_sampler = WeightedRandomSampler(
        weights=trajectory_data_set.sampling_probabilities[test_dataset.indices],
        num_samples=len(test_dataset),
        replacement=True,
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, sampler=test_sampler)

    train_batches_per_epoch = len(train_dataloader)
    pbar = tqdm(range(train_epochs))
    for epoch in pbar:
        for batch, (s, a, r, d, rtg, ti, m) in (enumerate(train_dataloader)):
            total_batches = epoch * train_batches_per_epoch + batch

            model.train()

            if model.transformer_config.time_embedding_type == "linear":
                ti = ti.to(t.float32)

            a[a == -10] = env.action_space.n  # dummy action for padding

            optimizer.zero_grad()

            if isinstance(model, DecisionTransformer):
                action = a[:, :-1].unsqueeze(-1) if a.shape[1] > 1 else None
                _, action_preds, _ = model.forward(
                    states=s,
                    # remove last action
                    actions=action,
                    rtgs=rtg[:, :-1],  # remove last rtg
                    timesteps=ti.unsqueeze(-1)
                )
            elif isinstance(model, CloneTransformer):
                _, action_preds = model.forward(
                    states=s,
                    # remove last action
                    actions=a[:, :- \
                              1].unsqueeze(-1) if a.shape[1] > 1 else None,
                    timesteps=ti.unsqueeze(-1)
                )

            action_preds = rearrange(action_preds, 'b t a -> (b t) a')
            a_exp = rearrange(a, 'b t -> (b t)').to(t.int64)

            # ignore dummy action
            loss = loss_fn(
                action_preds[a_exp != env.action_space.n],
                a_exp[a_exp != env.action_space.n]
            )

            loss.backward()
            optimizer.step()

            pbar.set_description(f"Training DT: {loss.item():.4f}")

            if track:
                wandb.log({"train/loss": loss.item()}, step=total_batches)
                tokens_seen = (total_batches + 1) * \
                    batch_size * (model.transformer_config.n_ctx // 3)
                wandb.log({"metrics/tokens_seen": tokens_seen},
                          step=total_batches)

        # # at test frequency
        if epoch % test_frequency == 0:
            test(
                model=model,
                dataloader=test_dataloader,
                env=env,
                epochs=test_epochs,
                track=track,
                batch_number=total_batches)

        eval_env_func = make_env(
            env_id=env.spec.id,
            seed=batch,
            idx=0,
            capture_video=True,
            max_steps=min(model.environment_config.max_steps,
                          eval_max_time_steps),
            run_name=f"dt_eval_videos_{batch}",
            fully_observed=False,
            flat_one_hot=(
                trajectory_data_set.observation_type == "one_hot"),
            # defensive coding, fix later.
            agent_view_size=env.observation_space['image'].shape[0] if "image" in list(
                env.observation_space.keys()) else 7,
        )

        if epoch % eval_frequency == 0:
            for rtg in initial_rtg:
                evaluate_dt_agent(
                    env_id=env.spec.id,
                    model=model,
                    env_func=eval_env_func,
                    trajectories=eval_episodes,
                    track=track,
                    batch_number=total_batches,
                    initial_rtg=float(rtg),
                    device=device)

    return model


def test(
        model: TrajectoryTransformer,
        dataloader: DataLoader,
        env,
        epochs=10,
        track=False,
        batch_number=0):
    model.eval()

    loss_fn = nn.CrossEntropyLoss()

    loss = 0
    n_correct = 0
    n_actions = 0

    pbar = tqdm(range(epochs))
    test_batches_per_epoch = len(dataloader)

    for epoch in pbar:
        for batch, (s, a, r, d, rtg, ti, m) in (enumerate(dataloader)):
            if model.transformer_config.time_embedding_type == "linear":
                ti = ti.to(t.float32)

            a[a == -10] = env.action_space.n

            if isinstance(model, DecisionTransformer):
                _, action_preds, _ = model.forward(
                    states=s,
                    actions=a[:, :-
                              1].unsqueeze(-1) if a.shape[1] > 1 else None,
                    rtgs=rtg[:, :-1],
                    timesteps=ti.unsqueeze(-1)
                )
            elif isinstance(model, CloneTransformer):
                _, action_preds = model.forward(
                    states=s,
                    # remove last action
                    actions=a[:, :- \
                              1].unsqueeze(-1) if a.shape[1] > 1 else None,
                    timesteps=ti.unsqueeze(-1)
                )

            action_preds = rearrange(action_preds, 'b t a -> (b t) a')
            a_exp = rearrange(a, 'b t -> (b t)').to(t.int64)

            a_hat = t.argmax(action_preds, dim=-1)
            a_exp = rearrange(a, 'b t -> (b t)').to(t.int64)

            action_preds = action_preds[a_exp != env.action_space.n]
            a_hat = a_hat[a_exp != env.action_space.n]
            a_exp = a_exp[a_exp != env.action_space.n]

            n_actions += a_exp.shape[0]
            n_correct += (a_hat == a_exp).sum()
            loss += loss_fn(action_preds, a_exp)

            accuracy = n_correct.item() / n_actions
            pbar.set_description(f"Testing DT: Accuracy so far {accuracy:.4f}")

    mean_loss = loss.item() / epochs * test_batches_per_epoch

    if track:
        wandb.log({"test/loss": mean_loss}, step=batch_number)
        wandb.log({"test/accuracy": accuracy}, step=batch_number)

    return mean_loss, accuracy


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
        num_envs=8):
    model.eval()

    env = gymnasium.vector.SyncVectorEnv(
        [env_func for _ in range(num_envs)]
    )
    video_path = os.path.join("videos", env.envs[0].run_name)

    if not hasattr(model, "transformer_config"):
        model.transformer_config = Namespace(
            n_ctx=model.n_ctx,
            time_embedding_type=model.time_embedding_type,
        )

    max_len = get_max_len_from_model_type(
        model_type="decision_transformer" if isinstance(
            model, DecisionTransformer) else "clone_transformer",
        n_ctx=model.transformer_config.n_ctx,
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
    obs = t.tensor(obs['image']).unsqueeze(1)
    rtg = rearrange(t.ones(num_envs, dtype=t.int) * initial_rtg, 'e -> e 1 1')
    a = rearrange(t.zeros(num_envs, dtype=t.int), 'e -> e 1 1')
    timesteps = rearrange(t.zeros(num_envs, dtype=t.int), 'e -> e 1 1')

    obs = obs.to(device)
    rtg = rtg.to(device)
    a = a.to(device)
    timesteps = timesteps.to(device)

    if model.transformer_config.time_embedding_type == "linear":
        timesteps = timesteps.to(t.float32)

    # get first action
    if isinstance(model, DecisionTransformer):
        state_preds, action_preds, reward_preds = model.forward(
            states=obs, actions=None, rtgs=rtg, timesteps=timesteps)
    elif isinstance(model, CloneTransformer):
        state_preds, action_preds = model.forward(
            states=obs, actions=None, timesteps=timesteps)
    else:  # it's probably a legacy model in which case the interface is:
        state_preds, action_preds, reward_preds = model.forward(
            states=obs, actions=a, rtgs=rtg, timesteps=timesteps)

    new_action = t.argmax(action_preds, dim=-1).squeeze(-1)
    new_obs, new_reward, terminated, truncated, info = env.step(new_action)

    current_trajectory_length = t.ones(num_envs, dtype=t.int)
    while n_terminated + n_truncated < trajectories:

        # concat init obs to new obs
        obs = t.cat(
            [obs, t.tensor(new_obs['image']).unsqueeze(1).to(device)], dim=1)

        # add new reward to init reward
        rtg = t.cat([rtg, rtg[:, -1:, :] -
                     rearrange(t.tensor(new_reward).to(device), 'e -> e 1 1')], dim=1)

        # add new timesteps
        timesteps = t.cat([timesteps, rearrange(
            current_trajectory_length.to(device), 'e -> e 1 1')], dim=1)

        if model.transformer_config.time_embedding_type == "linear":
            timesteps = timesteps.to(t.float32)

        a = t.cat([a, rearrange(new_action, 'e -> e 1 1')], dim=1)

        # truncations:
        obs = obs[:, -max_len:] if obs.shape[1] > max_len else obs
        actions = a[:, -(obs.shape[1] - 1):] if (a.shape[1]
                                                 > 1 and max_len > 1) else None
        timesteps = timesteps[:, -
                              max_len:] if timesteps.shape[1] > max_len else timesteps

        if isinstance(model, DecisionTransformer):
            state_preds, action_preds, reward_preds = model.forward(
                states=obs, actions=actions, rtgs=rtg, timesteps=timesteps)
        elif isinstance(model, CloneTransformer):
            state_preds, action_preds = model.forward(
                states=obs, actions=actions, timesteps=timesteps)
        else:  # it's probably a legacy model in which case the interface is:
            steps = model.transformer_config.n_ctx // 3
            state_preds, action_preds, reward_preds = model.forward(
                states=obs[:, -steps:], actions=a[:, -steps:], rtgs=rtg[:, -steps:], timesteps=timesteps[:, -steps:])

        action = t.argmax(action_preds, dim=-1).squeeze(-1)
        new_obs, new_reward, terminated, truncated, info = env.step(action)
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

            current_videos = [i for i in os.listdir(
                video_path) if i.endswith(".mp4")]
            if track and (len(current_videos) > len(videos)):  # we have a new video
                new_videos = [i for i in current_videos if i not in videos]
                for new_video in new_videos:
                    path_to_video = os.path.join(video_path, new_video)
                    wandb.log({f"media/video/{initial_rtg}/": wandb.Video(
                        path_to_video,
                        fps=4,
                        format="mp4",
                        caption=f"{env_id}, after {n_terminated + n_truncated} episodes, reward {new_reward}, rtg {initial_rtg}"
                    )}, step=batch_number)
            videos = current_videos  # update videos

    collected_trajectories = (n_terminated + n_truncated)

    statistics = {
        "initial_rtg": initial_rtg,
        "prop_completed": n_terminated / collected_trajectories,
        "prop_truncated": n_truncated / collected_trajectories,
        "mean_reward": reward_total / collected_trajectories,
        "prop_positive_reward": n_positive / collected_trajectories,
        "mean_traj_length": sum(traj_lengths) / collected_trajectories,
        "traj_lengths": traj_lengths,
        "rewards": rewards
    }

    env.close()
    if track:
        # log statistics at batch number but prefix with eval
        for key, value in statistics.items():
            if key == "initial_rtg":
                continue
            if key == "traj_lengths":
                wandb.log({f"eval/{str(initial_rtg)}/traj_lengths": wandb.Histogram(
                    value)}, step=batch_number)
            elif key == "rewards":
                wandb.log({f"eval/{str(initial_rtg)}/rewards": wandb.Histogram(
                    value)}, step=batch_number)
            wandb.log({f"eval/{str(initial_rtg)}/" +
                       key: value}, step=batch_number)

    return statistics
