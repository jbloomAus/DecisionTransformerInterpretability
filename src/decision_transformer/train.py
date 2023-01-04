import numpy as np
import torch as t
import torch.nn as nn
from einops import rearrange
from tqdm.notebook import tqdm

from .decision_transformer import DecisionTransformer
from .offline_dataset import TrajectoryLoader
from src.ppo.utils import make_env


def train(trajectory_data_set: TrajectoryLoader, batch_size = 128, max_len = 20, batches = 1000, lr = 0.0001, device = "cpu"):

    loss_fn = nn.CrossEntropyLoss()

    env_id = trajectory_data_set.metadata['args']['env_id']
    env = make_env(env_id, seed = 0, idx = 0, capture_video=False, run_name = "dev", fully_observed=False)
    env = env()

    dt = DecisionTransformer(
        env = env, 
        state_embedding_type="grid", # hard-coded for now to minigrid.
        max_game_length=100)

    dt  = dt.to(device)

    dt.train()

    optimizer = t.optim.Adam(dt.parameters(), lr=lr)

    pbar = tqdm(range(batches))
    for i in pbar:

        s, a, r, d, rtg, timesteps, mask = trajectory_data_set.get_batch(batch_size, max_len=max_len)

        s.to(device)
        a.to(device)
        r.to(device)
        d.to(device)
        rtg.to(device)
        timesteps.to(device)
        mask.to(device)

        a[a==-10] = 6
    
        optimizer.zero_grad()

        logits, _ = dt.forward(
            states = s,
            actions = a.to(t.int32).unsqueeze(-1),
            rtgs = rtg[:,:-1,:],
            timesteps = timesteps.unsqueeze(-1)
        )

        logits = rearrange(logits, 'b t a -> (b t) a')
        a_exp = rearrange(a, 'b t -> (b t)').to(t.int64)
        
        loss = loss_fn(logits, a_exp)

        loss.backward()
        optimizer.step()

        pbar.set_description(f"Training DT: {loss.item():.4f}")

        # at save frequency
        if i % 100 == 0:
            t.save(dt.state_dict(), f"decision_transformer_{i}.pt")

        # # at test frequency
        # if i % 100 == 0:
        #     test(dt, trajectory_data_set, batch_size, max_len, batches, device)

    return dt

def test(dt, trajectory_data_set: TrajectoryLoader, batch_size = 128, max_len = 20, batches = 10, device = "cpu"):

    env_id = trajectory_data_set.metadata['args']['env_id']
    env = make_env(env_id, seed = 0, idx = 0, capture_video=False, run_name = "dev", fully_observed=False)
    env = env()

    dt.eval()

    loss_fn = nn.CrossEntropyLoss()

    loss = 0
    n_correct = 0
    n_actions = 0


    pbar = tqdm(range(batches), desc="Testing DT")
    for i in pbar:

        s, a, r, d, rtg, timesteps, mask = trajectory_data_set.get_batch(batch_size, max_len=max_len)

        s.to(device)
        a.to(device)
        r.to(device)
        d.to(device)
        rtg.to(device)
        timesteps.to(device)
        mask.to(device)

        a[a==-10] = 6
    
        logits, _ = dt.forward(
            states = s,
            actions = a.to(t.int32).unsqueeze(-1),
            rtgs = rtg[:,:-1,:],
            timesteps = timesteps.unsqueeze(-1)
        )

        logits = rearrange(logits, 'b t a -> (b t) a')
        a_hat = t.argmax(logits, dim=-1)
        a_exp = rearrange(a, 'b t -> (b t)').to(t.int64)


        n_actions  += a_exp.shape[0]
        n_correct += (a_hat == a_exp).sum() 
        loss += loss_fn(logits, a_exp)
        

        accuracy = n_correct.item() / n_actions
        pbar.set_description(f"Testing DT: Accuracy so far {accuracy:.4f}")
    
    mean_loss = loss.item() / batches
    return mean_loss, accuracy

def evaluate_dt_agent(trajectory_data_set: TrajectoryLoader, dt: DecisionTransformer, device = "cpu", max_len = 30, trajectories = 300):

    env_id = trajectory_data_set.metadata['args']['env_id']
    env = make_env(env_id, seed = 15, idx = 0, capture_video=True, run_name = "dev", fully_observed=False)
    env = env()


    all_frames = []
    n_completed = 0
    for seed in range(trajectories):

        frames = []
        obs, _ = env.reset()
        frames.append(env.render())

        obs = t.tensor(obs['image']).unsqueeze(0).unsqueeze(0)
        rtg = t.tensor([1]).unsqueeze(0).unsqueeze(0)
        a = t.tensor([6]).unsqueeze(0).unsqueeze(0)
        timesteps = t.tensor([0]).unsqueeze(0).unsqueeze(0)

        # get first action
        logits, loss = dt.forward(states = obs, actions = a, rtgs = rtg, timesteps = timesteps)
        new_action = t.argmax(logits, dim=-1)[0].item()
        new_obs, new_reward, terminated, truncated, info = env.step(new_action)
        frames.append(env.render())

        i = 1
        while not terminated:

            # concat init obs to new obs
            obs = t.cat([obs, t.tensor(new_obs['image']).unsqueeze(0).unsqueeze(0)], dim=1)

            # add new reward to init reward
            rtg = t.cat([rtg, t.tensor([rtg[-1][-1].item() - new_reward]).unsqueeze(0).unsqueeze(0)], dim=1)

            # add new timesteps
            timesteps = t.cat([timesteps, t.tensor([timesteps[-1][-1].item()+1]).unsqueeze(0).unsqueeze(0)], dim=1)
            a = t.cat([a, t.tensor([new_action]).unsqueeze(0).unsqueeze(0)], dim=1)

            logits, loss = dt.forward(
                states = obs[:, -max_len:] if obs.shape[1] > max_len else obs,
                actions = a[:, -max_len:] if a.shape[1] > max_len else a,
                rtgs = rtg[:, -max_len:] if rtg.shape[1] > max_len else rtg,
                timesteps = timesteps[:, -max_len:] if timesteps.shape[1] > max_len else timesteps
            )
            action = t.argmax(logits, dim=-1)[0][-1].item()
            new_obs, new_reward, terminated, truncated, info = env.step(action)
            frames.append(env.render())

            # print(f"took action  {action} at timestep {i} for reward {new_reward}")
            i = i + 1

            if i > max_len:
                # print("exceeded context window breaking")
                break

            if terminated:
                n_completed = n_completed + 1
                

        frames = np.array(frames)
        all_frames.append(frames)

    print(f"completed {n_completed} trajectories out of {trajectories}")

    return n_completed/ trajectories, all_frames