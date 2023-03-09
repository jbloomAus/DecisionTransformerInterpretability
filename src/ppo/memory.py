from collections import defaultdict
from dataclasses import dataclass
from typing import List

import gymnasium as gym
import numpy as np
import torch as t
from einops import rearrange
from torchtyping import TensorType as TT

import wandb

from src.config import OnlineTrainConfig
from .utils import PPOArgs, get_obs_preprocessor


@dataclass
class Minibatch:
    '''
    A dataclass containing the tensors of a minibatch of experiences.
    '''
    obs: TT["batch", "obs_shape"]  # noqa: F821
    actions: TT["batch"]  # noqa: F821
    logprobs: TT["batch"]  # noqa: F821
    advantages: TT["batch"]  # noqa: F821
    values: TT["batch"]  # noqa: F821
    returns: TT["batch"]  # noqa: F821


class Memory():
    '''
    A memory buffer for storing experiences during the rollout phase.
    '''

    def __init__(self, envs: gym.vector.SyncVectorEnv, args: OnlineTrainConfig, device: t.device):
        """Initializes the memory buffer.

        envs: A SyncVectorEnv object.
        args: A PPOArgs object containing the PPO training hyperparameters.
        device: The device to store the tensors on, either "cpu" or "cuda".
        """
        self.envs = envs
        self.args = args
        self.next_obs = None
        self.next_done = None
        self.next_value = None
        self.device = device
        self.global_step = 0
        self.obs_preprocessor = get_obs_preprocessor(envs.observation_space)
        self.reset()

    def add(self, *data: t.Tensor):
        """
        Adds an experience to storage. Called during the rollout phase.

        *data: A tuple containing the tensors of (obs, done, action, logprob, value, reward) for an agent.
        """
        info, *experiences = data
        self.experiences.append(experiences)
        if info and isinstance(info, dict):
            if "final_info" in info.keys():

                for item in info["final_info"]:
                    if isinstance(item, dict):
                        if "episode" in item.keys():
                            self.episode_lengths.append(item["episode"]["l"])
                            self.episode_returns.append(item["episode"]["r"])
                            self.add_vars_to_log(
                                episode_length=item["episode"]["l"],
                                episode_return=item["episode"]["r"],
                            )

                    self.global_step += 1

    def sample_experiences(self):
        '''Prints out a randomly selected experience as a sanity check.

        Each experience consists of a tuple containing:
        - obs: observations of the environment
        - done: whether the episode has terminated
        - action: the action taken by the agent
        - logprob: the log probability of taking that action
        - value: the estimated value of the current state
        - reward: the reward received from taking that action

        The output will be a sample from the stored experiences, in the format:
            Sample X/Y:
            obs    : [...]
            done   : [...]
            action : [...]
            logprob: [...]
            value  : [...]
            reward : [...]
        '''
        idx = np.random.randint(0, len(self.experiences))
        print(f"Sample {idx+1}/{len(self.experiences)}:")
        for i, n in enumerate(["obs", "done", "action", "logprob", "value", "reward"]):
            print(f"{n:8}: {self.experiences[idx][i].cpu().numpy().tolist()}")

    def get_minibatch_indexes(self, batch_size: int, minibatch_size: int) -> List[np.ndarray]:
        '''Return a list of length (batch_size // minibatch_size) where each element
        is an array of indexes into the batch.

        Each index should appear exactly once.
        '''
        assert batch_size % minibatch_size == 0
        indices = np.random.permutation(batch_size)
        indices = rearrange(
            indices, "(mb_num mb_size) -> mb_num mb_size", mb_size=minibatch_size)
        return list(indices)

    def compute_advantages(
        self,
        next_value: TT["env"],  # noqa: F821
        next_done: TT["env"],  # noqa: F821
        rewards: TT["T", "env"],  # noqa: F821
        values: TT["T", "env"],  # noqa: F821
        dones: TT["T", "env"],  # noqa: F821
        device: t.device,
        gamma: float,
        gae_lambda: float
    ) -> TT["T", "env"]:  # noqa: F821
        '''Compute advantages using Generalized Advantage Estimation.
        '''
        T = values.shape[0]
        next_values = t.concat([values[1:], next_value.unsqueeze(0)])
        next_dones = t.concat([dones[1:], next_done.unsqueeze(0)])
        deltas = rewards + gamma * next_values * (1.0 - next_dones) - values
        advantages = t.zeros_like(deltas).to(device)
        advantages[-1] = deltas[-1]
        for t_ in reversed(range(1, T)):
            advantages[t_ - 1] = deltas[t_ - 1] + gamma * \
                gae_lambda * (1.0 - dones[t_]) * advantages[t_]
        return advantages

    def get_minibatches(self) -> List[Minibatch]:
        '''Return a list of length (batch_size // minibatch_size)
          where each element is an array of indexes into the batch.

        Args:
        - batch_size (int): total size of the batch.
        - minibatch_size (int): size of each minibatch.

        Returns:
        - List[MiniBatch]: a list of minibatches.
        '''
        obs, dones, actions, logprobs, values, rewards = [
            t.stack(arr) for arr in zip(*self.experiences)]
        advantages = self.compute_advantages(
            self.next_value,
            self.next_done,
            rewards,
            values,
            dones,
            self.device,
            self.args.gamma,
            self.args.gae_lambda
        )
        returns = advantages + values
        indexes = self.get_minibatch_indexes(
            self.args.batch_size, self.args.minibatch_size)

        minibatches = []
        for ind in indexes:
            batch = []
            for arr in [obs, actions, logprobs, advantages, values, returns]:
                flat_arr = arr.flatten(0, 1)
                batch.append(flat_arr[ind])
            minibatches.append(Minibatch(*batch))

        return minibatches

    def get_printable_output(self) -> str:
        '''Sets a new progress bar description, if any episodes have terminated.
        If not, then the bar's description won't change.
        '''
        if self.episode_lengths:
            global_step = self.global_step
            avg_episode_length = np.mean(self.episode_lengths)
            avg_episode_return = np.mean(self.episode_returns)
            return f"{global_step=:<06}\n{avg_episode_length=:<3.2f}\n{avg_episode_return=:<3.2f}"

    def reset(self) -> None:
        '''Function to be called at the end of each rollout period, to make
        space for new experiences to be generated.
        '''
        self.experiences = []
        self.vars_to_log = defaultdict(dict)
        self.episode_lengths = []
        self.episode_returns = []
        if self.next_obs is None:
            (obs, info) = self.envs.reset()
            obs = self.obs_preprocessor(obs)
            self.next_obs = t.tensor(obs).to(self.device)
            self.next_done = t.zeros(self.envs.num_envs).to(
                self.device, dtype=t.float)

    def add_vars_to_log(self, **kwargs):
        '''Add variables to storage, for eventual logging (if args.track=True).
        '''
        self.vars_to_log[self.global_step] |= kwargs

    def log(self) -> None:
        '''Logs variables to wandb.
        '''
        for step, vars_to_log in self.vars_to_log.items():
            wandb.log(vars_to_log, step=step)

    # TODO work out how to TT with obs shape at end
    def get_obs_traj(self, steps: int, pad_to_length: int) -> TT["env", "T", "obs"]:
        '''Returns a tensor of shape (steps, envs, obs_shape) containing the observations from the last steps.

        Args:
        - steps (int): number of steps to return.
        - pad_to_length (int): if the number of steps is less than this, then the tensor will be padded with zeros.

        Returns:
        - TT["T", "env", "obs"]: a tensor of shape (steps, envs, obs_shape) containing
        the observations from the last steps.
        '''
        # if self.experiences[0][1][0].shape == ():  # if obs is a scalar
        #     obs_shape = 1
        # else:
        #     obs_shape = self.experiences[0][1][0].shape

        # obs_shape = self.experiences[0][1].shape

        # total bullshit because of how these experiences are stored.
        obs = [exp[0] for exp in self.experiences]
        obs = t.stack(obs)  # obs now has shape (steps, envs, obs_shape)

        # then get the last steps
        obs_traj = obs[-steps:]

        # then pad with zeros if needed
        # TODO check if 0 padding is the right way to pad obs in this codebase
        if steps < pad_to_length:
            if len(self.experiences[0][1].shape) == 1:
                pad = t.zeros((pad_to_length - steps, self.envs.num_envs))
            else:
                pad = t.zeros((pad_to_length - steps, self.envs.num_envs,
                               self.envs.observation_space.shape[0]))

            obs_traj = t.cat([pad, obs_traj], dim=0)

        else:
            obs_traj = obs_traj[-pad_to_length:]

        obs_traj = rearrange(obs_traj, 't e ... -> e t ...')
        return obs_traj

    def get_act_traj(self, steps: int, pad_to_length: int) -> TT["env", "T", "act"]:
        '''Returns a tensor of shape (steps, envs, obs_shape) containing the observations from the last steps.

        Args:
        - steps (int): number of steps to return.
        - pad_to_length (int): if the number of steps is less than this, then the tensor will be padded with zeros.

        Returns:
        - TT["T", "env", "obs"]: a tensor of shape (steps, envs, obs_shape) containing
        the observations from the last steps.
        '''

        # total bullshit because of how these experiences are stored.
        act = [exp[2] for exp in self.experiences]
        act = t.stack(act)  # obs now has shape (steps, envs, obs_shape)

        # then get the last steps
        act_traj = act[-steps:]

        # then pad with zeros if needed
        if steps < pad_to_length:
            pad = t.ones((pad_to_length - steps, self.envs.num_envs)
                         ) * self.envs.single_action_space.n

            act_traj = t.cat([pad, act_traj], dim=0)

        else:
            act_traj = act_traj[-pad_to_length:]

        act_traj = rearrange(act_traj, 't e ... -> e t ...')
        return act_traj

    def get_timestep_traj(self, steps: int, pad_to_length: int) -> TT["env", "T", "1"]:
        '''
        Returns a tensor of shape (steps, envs, 1) containing the time steps from the last steps.

        Args:
        - steps (int): number of steps to return.
        - pad_to_length (int): if the number of steps is less than this, then the tensor will be padded with zeros.

        Returns:
        - a tensor of shape (steps, envs, 1) containing
        the time steps from the last steps.
        '''
        n_envs = self.envs.num_envs
        timesteps = t.arange(0, steps).repeat(n_envs, 1)[:, -pad_to_length:]
        return timesteps
