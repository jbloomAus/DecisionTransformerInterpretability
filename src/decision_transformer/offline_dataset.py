import pickle
import numpy as np
import torch as t 
from einops import rearrange
import random

# not technically a data loader, rework later to work as one.
class TrajectoryLoader():

    def __init__(self, trajectory_path, pct_traj=1.0, rtg_scale = 1, normalize_state = False, device = 'cpu'):
        self.trajectory_path = trajectory_path
        self.pct_traj = pct_traj
        self.load_trajectories()
        self.device = device
        self.normalize_state = normalize_state
        self.rtg_scale = rtg_scale

    def load_trajectories(self) -> None:
        
        traj_reader = TrajectoryReader(self.trajectory_path)
        data = traj_reader.read()

        print(data['metadata'])

        observations = data['data'].get('observations')
        actions = data['data'].get('actions')
        rewards = data['data'].get('rewards')
        dones = data['data'].get('dones')
        truncated = data['data'].get('truncated')
        infos = data['data'].get('infos')

        observations = np.array(observations)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)
        infos = np.array(infos, dtype=np.ndarray)

        t_observations = rearrange(t.tensor(observations), "t b h w c -> (b t) h w c")
        t_actions = rearrange(t.tensor(actions), "t b -> (b t)")
        t_rewards = rearrange(t.tensor(rewards), "t b -> (b t)")
        t_dones = rearrange(t.tensor(dones), "t b -> (b t)")
        t_truncated = rearrange(t.tensor(truncated), "t b -> (b t)")

        t_done_or_truncated = t.logical_or(t_dones, t_truncated)
        done_indices = t.where(t_done_or_truncated )[0]

        self.actions = t.tensor_split(t_actions, done_indices+1)
        self.rewards = t.tensor_split(t_rewards, done_indices+1)
        self.dones = t.tensor_split(t_dones, done_indices+1)
        self.truncated = t.tensor_split(t_truncated, done_indices+1)
        self.states = t.tensor_split(t_observations, done_indices+1)
        self.returns = [r.sum() for r in self.rewards]
        self.timesteps = [t.arange(len(i)) for i in self.states]
        self.traj_lens = np.array([len(i) for i in self.states])
        self.num_timesteps = sum(self.traj_lens)
        self.num_trajectories = len(self.states)

        self.state_dim = list(self.states[0][0].shape)
        self.act_dim = list(self.actions[0][0].shape)
        self.max_ep_len = max([len(i) for i in self.states])
        self.metadata = data['metadata']

    def get_indices_of_top_p_trajectories(self, pct_traj):
        num_timesteps = max(int(pct_traj*self.num_timesteps), 1)
        sorted_inds = np.argsort(self.returns) 

        num_trajectories = 1
        timesteps = self.traj_lens[sorted_inds[-1]]
        ind = self.num_trajectories - 2

        # this while statement checks two things:
        # 1. that we haven't gone past the end of the array
        # 2. that the number of timesteps we've added is less than the number of timesteps we want

        # changing this line had a huge impact on performance
        while ind >= 0 and timesteps + self.traj_lens[sorted_inds[ind]] < num_timesteps:
            timesteps += self.traj_lens[sorted_inds[ind]]
            ind -= 1
            num_trajectories += 1

        sorted_inds = sorted_inds[-num_trajectories:]

        return sorted_inds

    def get_sampling_probabilities(self):
        sorted_inds = self.get_indices_of_top_p_trajectories(self.pct_traj)
        p_sample = self.traj_lens[sorted_inds] / sum(self.traj_lens[sorted_inds])
        return p_sample

    def discount_cumsum(self, x, gamma):
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0]-1)):
            discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
        return discount_cumsum

    def get_state_mean_std(self):
        # used for input normalization
        all_states = np.concatenate(self.states, axis=0)
        state_mean, state_std = np.mean(all_states, axis=0), np.std(all_states, axis=0) + 1e-6
        return state_mean, state_std

    def get_batch(self, batch_size=256, max_len=100):

        rewards = self.rewards
        states = self.states
        actions = self.actions
        dones = self.dones

        # asset first dim is same for all inputs
        assert len(rewards) == len(states) == len(actions) == len(dones), f"shapes are not the same: {len(rewards)} {len(states)} {len(actions)} {len(dones)}"
        p_sample = self.get_sampling_probabilities()
        sorted_inds = self.get_indices_of_top_p_trajectories(self.pct_traj)
        state_mean, state_std = self.get_state_mean_std()

        batch_inds = np.random.choice(
            np.arange(len(sorted_inds)),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )

        # initialize lists
        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):

            # get the trajectory
            traj_rewards = rewards[sorted_inds[batch_inds[i]]]
            traj_states = states[sorted_inds[batch_inds[i]]]
            traj_actions = actions[sorted_inds[batch_inds[i]]]
            traj_dones = dones[sorted_inds[batch_inds[i]]]

            # start index
            si = random.randint(0, traj_rewards.shape[0] - 1)

            # get sequences from dataset
            s.append(traj_states[si:si + max_len].reshape(1, -1, *self.state_dim))
            a.append(traj_actions[si:si + max_len].reshape(1, -1, *self.act_dim))
            r.append(traj_rewards[si:si + max_len].reshape(1, -1, 1))
            d.append(traj_dones[si:si + max_len].reshape(1, -1))
            
            # get timesteps
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= self.max_ep_len] = self.max_ep_len-1  # padding cutoff

            # get rewards to go
            rtg.append(self.discount_cumsum(traj_rewards[si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))

            # if the trajectory is shorter than max_len, pad it
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1] # sometime the trajectory is shorter than max_len (due to random start index or end of episode)
            if a[-1].shape[1] != tlen:
                a[-1] = np.concatenate([a[-1], np.ones((1, 1, *self.act_dim)) * -10.], axis=1)

            assert tlen <= max_len, f"tlen: {tlen} max_len: {max_len}" # sanity check

            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, *self.state_dim)), s[-1]], axis=1)
            if self.normalize_state:
                s[-1] = (s[-1] - state_mean) / state_std

            a[-1] = np.concatenate([np.ones((1, max_len - tlen, *self.act_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / self.rtg_scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = t.from_numpy(np.concatenate(s, axis=0)).to(dtype=t.float32, device=self.device)
        a = t.from_numpy(np.concatenate(a, axis=0)).to(dtype=t.float32, device=self.device)
        r = t.from_numpy(np.concatenate(r, axis=0)).to(dtype=t.float32, device=self.device)
        d = t.from_numpy(np.concatenate(d, axis=0)).to(dtype=t.long,    device=self.device)
        rtg = t.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=t.float32, device= self.device)
        timesteps = t.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=t.long, device=self.device)
        mask = t.from_numpy(np.concatenate(mask, axis=0)).to(device=self.device)

        return s, a, r, d, rtg, timesteps, mask

class TrajectoryReader():
    '''
    The trajectory reader is responsible for reading trajectories from a file.
    '''
    def __init__(self, path):
        self.path = path

    def read(self):
        with open(self.path, 'rb') as f:
            data = pickle.load(f)

        return data

        