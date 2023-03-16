import gzip
import lzma
import pickle
import random

import numpy as np
import plotly.express as px
import torch as t
from einops import rearrange

from torch.utils.data import Dataset


class TrajectoryReader():
    '''
    The trajectory reader is responsible for reading trajectories from a file.
    '''

    def __init__(self, path):
        self.path = path.strip()

    def read(self):
        # if path ends in .pkl, read as pickle
        if self.path.endswith('.pkl'):
            with open(self.path, 'rb') as f:
                data = pickle.load(f)
        # if path ends in .xz, read as lzma
        elif self.path.endswith('.xz'):
            with lzma.open(self.path, 'rb') as f:
                data = pickle.load(f)
        elif self.path.endswith('.gz'):
            with gzip.open(self.path, 'rb') as f:
                data = pickle.load(f)
        else:
            raise ValueError(
                f"Path {self.path} is not a valid trajectory file")

        return data


class TrajectoryDataset(Dataset):

    def __init__(self,
                 trajectory_path,
                 max_len=1,
                 prob_go_from_end=0,
                 pct_traj=1.0,
                 rtg_scale=1,
                 normalize_state=False,
                 device='cpu'):
        self.trajectory_path = trajectory_path
        self.max_len = max_len
        self.prob_go_from_end = prob_go_from_end
        self.pct_traj = pct_traj
        self.device = device
        self.normalize_state = normalize_state
        self.rtg_scale = rtg_scale
        self.load_trajectories()

    def load_trajectories(self) -> None:

        traj_reader = TrajectoryReader(self.trajectory_path)
        data = traj_reader.read()

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

        # check whether observations are flat or an image
        if observations.shape[-1] == 3:
            self.observation_type = 'index'
        elif observations.shape[-1] == 20:
            self.observation_type = 'one_hot'
        else:
            raise ValueError(
                "Observations are not flat or images, check the shape of the observations: ", observations.shape)

        if self.observation_type != 'flat':
            t_observations = rearrange(
                t.tensor(observations), "t b h w c -> (b t) h w c")
        else:
            t_observations = rearrange(
                t.tensor(observations), "t b f -> (b t) f")

        t_actions = rearrange(t.tensor(actions), "t b -> (b t)")
        t_rewards = rearrange(t.tensor(rewards), "t b -> (b t)")
        t_dones = rearrange(t.tensor(dones), "t b -> (b t)")
        t_truncated = rearrange(t.tensor(truncated), "t b -> (b t)")

        t_done_or_truncated = t.logical_or(t_dones, t_truncated)
        done_indices = t.where(t_done_or_truncated)[0]

        self.actions = t.tensor_split(t_actions, done_indices + 1)
        self.rewards = t.tensor_split(t_rewards, done_indices + 1)
        self.dones = t.tensor_split(t_dones, done_indices + 1)
        self.truncated = t.tensor_split(t_truncated, done_indices + 1)
        self.states = t.tensor_split(t_observations, done_indices + 1)
        self.returns = [r.sum() for r in self.rewards]
        self.timesteps = [t.arange(len(i)) for i in self.states]
        self.traj_lens = np.array([len(i) for i in self.states])

        # remove trajs with length 0
        traj_len_mask = self.traj_lens > 0
        self.actions = [i for i, m in zip(self.actions, traj_len_mask) if m]
        self.rewards = [i for i, m in zip(self.rewards, traj_len_mask) if m]
        self.dones = [i for i, m in zip(self.dones, traj_len_mask) if m]
        self.truncated = [i for i, m in zip(
            self.truncated, traj_len_mask) if m]
        self.states = [i for i, m in zip(self.states, traj_len_mask) if m]
        self.returns = [i for i, m in zip(self.returns, traj_len_mask) if m]
        self.timesteps = [i for i, m in zip(
            self.timesteps, traj_len_mask) if m]
        self.traj_lens = self.traj_lens[traj_len_mask]

        self.num_timesteps = sum(self.traj_lens)
        self.num_trajectories = len(self.states)

        self.state_dim = list(self.states[0][0].shape)
        self.act_dim = list(self.actions[0][0].shape)
        self.max_ep_len = max([len(i) for i in self.states])
        self.metadata = data['metadata']

        self.indices = self.get_indices_of_top_p_trajectories(self.pct_traj)
        self.sampling_probabilities = self.get_sampling_probabilities()

        if self.normalize_state:
            self.state_mean, self.state_std = self.get_state_mean_std()
        else:
            self.state_mean = 0
            self.state_std = 1

    def get_indices_of_top_p_trajectories(self, pct_traj):
        num_timesteps = max(int(pct_traj * self.num_timesteps), 1)
        sorted_inds = np.argsort(self.returns)

        num_trajectories = 1
        timesteps = self.traj_lens[sorted_inds[-1]]
        ind = self.num_trajectories - 1

        while ind >= 0 and timesteps + self.traj_lens[sorted_inds[ind]] < num_timesteps:
            timesteps += self.traj_lens[sorted_inds[ind]]
            ind -= 1
            num_trajectories += 1

        sorted_inds = sorted_inds[-num_trajectories:]

        return sorted_inds

    def get_sampling_probabilities(self):
        p_sample = self.traj_lens[self.indices] / \
            sum(self.traj_lens[self.indices])
        return p_sample

    def discount_cumsum(self, x, gamma):
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for time in reversed(range(x.shape[0] - 1)):
            discount_cumsum[time] = x[time] + gamma * discount_cumsum[time + 1]
        return discount_cumsum

    def get_state_mean_std(self):
        # used for input normalization
        all_states = np.concatenate(self.states, axis=0)
        state_mean, state_std = np.mean(
            all_states, axis=0), np.std(all_states, axis=0) + 1e-6
        return state_mean, state_std

    def get_batch(self, batch_size=256, max_len=100, prob_go_from_end=None):

        sorted_inds = self.indices

        batch_inds = np.random.choice(
            np.arange(len(sorted_inds)),
            size=batch_size,
            replace=True,
            p=self.sampling_probabilities,  # reweights so we sample according to timesteps
        )

        # initialize np arrays not lists
        states, actions, rewards, dones, rewards_to_gos, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):

            # get the trajectory
            traj_index = sorted_inds[batch_inds[i]]

            s, a, r, d, rtg, ti, m = self.get_traj(
                traj_index, max_len, prob_go_from_end=prob_go_from_end)

            rewards.append(r)
            actions.append(a)
            states.append(s)
            dones.append(d)
            rewards_to_gos.append(rtg)
            mask.append(m)
            timesteps.append(ti)

        return self.return_tensors(states, actions, rewards, rewards_to_gos, dones, timesteps, mask)

    def get_traj(self, traj_index, max_len=100, prob_go_from_end=None):

        traj_rewards = self.rewards[traj_index]
        traj_states = self.states[traj_index]
        traj_actions = self.actions[traj_index]
        traj_dones = self.dones[traj_index]

        # start index
        si = random.randint(0, traj_rewards.shape[0] - 1)
        if prob_go_from_end is not None:
            if random.random() < prob_go_from_end:
                si = traj_rewards.shape[0] - max_len
                si = max(0, si)  # make sure it's not negative

        # get sequences from dataset
        s = traj_states[si:si + max_len].reshape(1, -1, *self.state_dim)
        a = traj_actions[si:si + max_len].reshape(1, -1, *self.act_dim)
        r = traj_rewards[si:si + max_len].reshape(1, -1, 1)
        d = traj_dones[si:si + max_len].reshape(1, -1)
        rtg = self.discount_cumsum(traj_rewards[si:], gamma=1.)[
            :s.shape[1] + 1].reshape(1, -1, 1)
        ti = np.arange(si, si + s.shape[1]).reshape(1, -1)

        # if the trajectory is shorter than max_len, pad it
        if rtg.shape[1] <= s.shape[1]:
            rtg = np.concatenate([rtg, np.zeros((1, 1, 1))], axis=1)

        # sometime the trajectory is shorter than max_len (due to random start index or end of episode)
        tlen = s.shape[1]

        # sanity check
        assert tlen <= max_len, f"tlen: {tlen} max_len: {max_len}"

        padding_required = max_len - tlen
        s = self.add_padding(s, 0, padding_required)
        a = self.add_padding(a, -10, padding_required)
        r = self.add_padding(r, 0, padding_required)
        d = self.add_padding(d, 2, padding_required)
        rtg = self.add_padding(rtg, 0, padding_required)
        ti = self.add_padding(ti, 0, padding_required)
        m = self.add_padding(np.ones((1, tlen)), 0, padding_required)

        # padding and state + reward normalization
        s = (s - self.state_mean) / self.state_std
        rtg = rtg / self.rtg_scale

        return self.return_tensors(s, a, r, rtg, d, ti, m)

    def add_padding(self, tokens, padding_token, padding_required):
        if padding_required > 0:
            return np.concatenate([np.ones((1, padding_required, *tokens.shape[2:])) * padding_token, tokens], axis=1)
        return tokens

    def return_tensors(self, s, a, r, rtg, d, timesteps, mask):

        if isinstance(s, t.Tensor):
            s = s.to(dtype=t.float32, device=self.device)
        else:
            s = t.from_numpy(s).to(dtype=t.float32, device=self.device)

        if isinstance(a, t.Tensor):
            a = a.to(dtype=t.long, device=self.device)
        else:
            a = t.from_numpy(a).to(dtype=t.long, device=self.device)

        if isinstance(r, t.Tensor):
            r = r.to(dtype=t.float32, device=self.device)
        else:
            r = t.from_numpy(r).to(dtype=t.float32, device=self.device)

        if isinstance(rtg, t.Tensor):
            rtg = rtg.to(dtype=t.float32, device=self.device)
        else:
            rtg = t.from_numpy(rtg).to(dtype=t.float32, device=self.device)

        if isinstance(d, t.Tensor):
            d = d.to(dtype=t.bool, device=self.device)
        else:
            d = t.from_numpy(d).to(dtype=t.bool, device=self.device)
        timesteps = t.from_numpy(timesteps).to(
            dtype=t.long, device=self.device)
        mask = t.from_numpy(mask).to(dtype=t.bool, device=self.device)

        # squeeze out the batch dimension
        s = s.squeeze(0)
        a = a.squeeze(0)
        r = r.squeeze(0)
        rtg = rtg.squeeze(0)
        d = d.squeeze(0)
        timesteps = timesteps.squeeze(0)
        mask = mask.squeeze(0)

        return s, a, r, d, rtg, timesteps, mask

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        traj_index = self.indices[idx]
        s, a, r, d, rtg, ti, m = self.get_traj(
            traj_index,
            max_len=self.max_len,
            prob_go_from_end=self.prob_go_from_end
        )
        return s, a, r, d, rtg, ti, m


class TrajectoryVisualizer:

    def __init__(self, trajectory_dataset: TrajectoryDataset):

        self.trajectory_loader = trajectory_dataset

    def plot_reward_over_time(self):

        reward = [i[-1] for i in self.trajectory_loader.rewards if len(i) > 0]
        timesteps = [i.max()
                     for i in self.trajectory_loader.timesteps if len(i) > 0]

        # create a categorical color array for reward <0, 0, >0
        colors = np.zeros(len(reward))
        colors[np.array(reward) < 0] = -1
        colors[np.array(reward) > 0] = 1

        color_map = {-1: "Negative", 0: "Zero", 1: "Positive"}

        fig = px.scatter(
            y=reward,
            x=timesteps,
            color=[color_map[i] for i in colors],
            title="Reward vs Timesteps",
            template="plotly_white",
            labels={
                "x": "Timesteps",
                "y": "Reward",
            },
            marginal_x="histogram",
            marginal_y="histogram",
        )

        return fig

    def plot_base_action_frequencies(self):

        fig = px.bar(
            y=t.concat(self.trajectory_loader.actions).bincount()
            # x=[IDX_TO_ACTION[i] for i in range(7)],
            # color=[IDX_TO_ACTION[i] for i in range(7)],
        )

        fig.update_layout(
            title="Base Action Frequencies",
            xaxis_title="Action",
            yaxis_title="Frequency",
        )

        return fig
