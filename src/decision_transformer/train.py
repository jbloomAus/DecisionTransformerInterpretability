import os
import torch as t
import torch.nn as nn
from einops import rearrange
from tqdm import tqdm
import wandb

from .decision_transformer import DecisionTransformer
from .offline_dataset import TrajectoryLoader


def train(
    dt: DecisionTransformer, 
    trajectory_data_set: TrajectoryLoader, 
    env, 
    make_env,
    batch_size=128, 
    max_len=20, 
    batches=1000, 
    lr=0.0001,
    weight_decay=0.0,
    device="cpu",
    track=False,
    test_frequency=10,
    test_batches=10,
    eval_frequency=10,
    eval_episodes=10):

    loss_fn = nn.CrossEntropyLoss()

    dt = dt.to(device)

    optimizer = t.optim.Adam(dt.parameters(), lr=lr, weight_decay=weight_decay)
    pbar = tqdm(range(batches))
    for batch in pbar:

        dt.train()
        
        s, a, r, d, rtg, timesteps, mask = trajectory_data_set.get_batch(
            batch_size, max_len=max_len)

        s.to(device)
        a.to(device)
        r.to(device)
        d.to(device)
        rtg.to(device)
        timesteps.to(device)
        mask.to(device)

        a[a == -10] = env.action_space.n  # dummy action for padding

        optimizer.zero_grad()

        logits, _ = dt.forward(
            states=s,
            actions=a.to(t.int32).unsqueeze(-1),
            rtgs=rtg[:, :-1, :],
            timesteps=timesteps.unsqueeze(-1)
        )

        logits = rearrange(logits, 'b t a -> (b t) a')
        a_exp = rearrange(a, 'b t -> (b t)').to(t.int64)

        # ignore dummy action
        loss = loss_fn(
            logits[a_exp != env.action_space.n],
            a_exp[a_exp != env.action_space.n]
        )

        loss.backward()
        optimizer.step()

        pbar.set_description(f"Training DT: {loss.item():.4f}")

        if track:
            wandb.log({"train/loss": loss.item()}, step=batch)
            tokens_seen = (batch + 1) * batch_size * max_len
            wandb.log({"metrics/tokens_seen": tokens_seen}, step=batch)

        # at save frequency
        if batch % 100 == 0:
            t.save(dt.state_dict(), f"decision_transformer_{batch}.pt")

        # # at test frequency
        if batch % test_frequency == 0:
            test(
                dt = dt, 
                trajectory_data_set = trajectory_data_set,
                env = env, 
                batch_size = batch_size, 
                max_len = max_len, 
                batches = test_batches, 
                device = device, 
                track = track,
                batch_number = batch)

        if batch % eval_frequency == 0:
            evaluate_dt_agent(
                trajectory_data_set = trajectory_data_set, 
                dt = dt, 
                make_env = make_env, 
                trajectories = eval_episodes,
                max_len = max_len, 
                track=track,
                batch_number = batch)
    return dt

def test(
    dt: DecisionTransformer, 
    trajectory_data_set: TrajectoryLoader, 
    env, 
    batch_size=128, 
    max_len=20, 
    batches=10, 
    device="cpu",
    track=False,
    batch_number=0):

    dt.eval()

    loss_fn = nn.CrossEntropyLoss()

    loss = 0
    n_correct = 0
    n_actions = 0

    pbar = tqdm(range(batches), desc="Testing DT")
    for i in pbar:

        s, a, r, d, rtg, timesteps, mask = trajectory_data_set.get_batch(
            batch_size, max_len=max_len)

        s.to(device)
        a.to(device)
        r.to(device)
        d.to(device)
        rtg.to(device)
        timesteps.to(device)
        mask.to(device)

        a[a == -10] = env.action_space.n  # dummy action for padding

        logits, _ = dt.forward(
            states=s,
            actions=a.to(t.int32).unsqueeze(-1),
            rtgs=rtg[:, :-1, :],
            timesteps=timesteps.unsqueeze(-1)
        )

        logits = rearrange(logits, 'b t a -> (b t) a')
        a_hat = t.argmax(logits, dim=-1)
        a_exp = rearrange(a, 'b t -> (b t)').to(t.int64)

        logits = logits[a_exp != env.action_space.n]
        a_hat = a_hat[a_exp != env.action_space.n]
        a_exp = a_exp[a_exp != env.action_space.n]

        n_actions += a_exp.shape[0]
        n_correct += (a_hat == a_exp).sum()
        loss += loss_fn(logits, a_exp)

        accuracy = n_correct.item() / n_actions
        pbar.set_description(f"Testing DT: Accuracy so far {accuracy:.4f}")

    mean_loss = loss.item() / batches

    if track:
        wandb.log({"test/loss": mean_loss}, step=batch_number)
        wandb.log({"test/accuracy": accuracy}, step=batch_number)

    return mean_loss, accuracy

def evaluate_dt_agent(
    trajectory_data_set: TrajectoryLoader, 
    dt: DecisionTransformer, 
    make_env, 
    max_len=30, 
    trajectories=300,
    track=False,
    batch_number=0,
    initial_rtg=0.98):

    dt.eval()
    run_name = f"dt_eval_videos_{batch_number}"
    video_path = os.path.join("videos", run_name)
    env_id = trajectory_data_set.metadata['args']['env_id']
    env = make_env(env_id, seed=batch_number, idx=0, capture_video=True,
                    max_steps = dt.max_timestep,
                   run_name=run_name, fully_observed=False)
    env = env()

    traj_lengths = []
    n_terminated = 0
    n_truncated = 0
    reward_total = 0
    n_positive = 0

    videos = [i for i in os.listdir(video_path) if i.endswith(".mp4")]
    for video in videos:
        os.remove(os.path.join(video_path, video))

    for seed in range(trajectories):

        obs, _ = env.reset(seed = seed)
        obs = t.tensor(obs['image']).unsqueeze(0).unsqueeze(0)
        rtg = t.tensor([initial_rtg]).unsqueeze(0).unsqueeze(0)
        a = t.tensor([0]).unsqueeze(0).unsqueeze(0)
        timesteps = t.tensor([0]).unsqueeze(0).unsqueeze(0)

        # get first action
        logits, loss = dt.forward(
            states=obs, actions=a, rtgs=rtg, timesteps=timesteps)
        new_action = t.argmax(logits, dim=-1)[0].item()
        new_obs, new_reward, terminated, truncated, info = env.step(new_action)

        i = 0
        while not (terminated or truncated):

            # concat init obs to new obs
            obs = t.cat(
                [obs, t.tensor(new_obs['image']).unsqueeze(0).unsqueeze(0)], dim=1)

            # add new reward to init reward
            rtg = t.cat([rtg, t.tensor(
                [rtg[-1][-1].item() - new_reward]).unsqueeze(0).unsqueeze(0)], dim=1)

            # add new timesteps
            timesteps = t.cat([timesteps, t.tensor(
                [timesteps[-1][-1].item()+1]).unsqueeze(0).unsqueeze(0)], dim=1)
            a = t.cat(
                [a, t.tensor([new_action]).unsqueeze(0).unsqueeze(0)], dim=1)

            logits, loss = dt.forward(
                states=obs[:, -max_len:] if obs.shape[1] > max_len else obs,
                actions=a[:, -max_len:] if a.shape[1] > max_len else a,
                rtgs=rtg[:, -max_len:] if rtg.shape[1] > max_len else rtg,
                timesteps=timesteps[:, -
                                    max_len:] if timesteps.shape[1] > max_len else timesteps
            )
            action = t.argmax(logits, dim=-1)[0][-1].item()
            new_obs, new_reward, terminated, truncated, info = env.step(action)

            # print(f"took action  {action} at timestep {i} for reward {new_reward}")
            i = i + 1
        
        n_positive = n_positive + (new_reward > 0)
        reward_total = reward_total + new_reward
        n_terminated = n_terminated + terminated
        n_truncated = n_truncated + truncated

        current_videos = [i for i in os.listdir(video_path) if i.endswith(".mp4")]
        if track and (len(current_videos) > len(videos)): # we have a new video
            new_videos = [i for i in current_videos if i not in videos]
            assert len(new_videos) == 1, "more than one new video found, new videos: {}".format(new_videos)
            path_to_video = os.path.join(video_path, new_videos[0])
            wandb.log({"media/video": wandb.Video(
                path_to_video, 
                fps=4, 
                format="mp4",
                caption=f"video of agent playing {env_id} at batch {batch_number}, episode {i}, reward {new_reward}"
            )}, step=batch_number)
        videos = current_videos # update videos

    env.close()
    if track:
        wandb.log({"eval/prop_completed": n_terminated / trajectories}, step=batch_number)
        wandb.log({"eval/prop_truncated": n_truncated / trajectories}, step=batch_number)
        wandb.log({"eval/mean_reward": reward_total / trajectories}, step=batch_number)
        wandb.log({"eval/prop_positive_reward": n_positive / trajectories}, step=batch_number)
        wandb.log({"eval/mean_traj_length": sum(traj_lengths) / trajectories}, step=batch_number)

    return n_terminated / trajectories