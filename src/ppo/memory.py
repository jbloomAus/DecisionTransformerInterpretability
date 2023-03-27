import random
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Any

import gymnasium as gym
import numpy as np
import torch as t
from einops import rearrange
from torchtyping import TensorType as TT

import wandb
from src.config import OnlineTrainConfig
from src.utils import pad_tensor

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


@dataclass
class TrajectoryMinibatch:
    '''
    A dataclass containing the tensors of a minibatch of experiences,
    including trajectory information leading up to each step.
    '''
    obs: TT["batch", "T", "obs_shape"]  # noqa: F821
    actions: TT["batch", "T"]  # noqa: F821
    logprobs: TT["batch"]  # noqa: F821
    advantages: TT["batch"]  # noqa: F821
    values: TT["batch"]  # noqa: F821
    returns: TT["batch"]  # noqa: F821
    timesteps: TT["batch", "T"]  # noqa: F821
    rewards: TT["batch", "T"]  # noqa: F821


class Memory():
    '''
    A memory buffer for storing experiences during the rollout phase.
    '''

    def __init__(self, envs: gym.vector.SyncVectorEnv, args: OnlineTrainConfig, device: t.device = t.device("cpu")):
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
        info = data[0]
        experiences = data[1:]
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
        '''
        Compute advantages using Generalized Advantage Estimation.

        Generalized Advantage Estimation (GAE) is a technique used in
        Proximal Policy Optimization (PPO) to estimate the advantage
        function, which is the difference between the expected value
        of the cumulative reward and the estimated value of the current state.

        Args:
        - next_value (Tensor): the estimated value of the next state.
        - next_done (Tensor): whether the next state is terminal.
        - rewards (Tensor): the rewards received from taking actions.
        - values (Tensor): the estimated values of the states.
        - dones (Tensor): whether the states are terminal.
        - device (torch.device): the device to store the tensors on.
        - gamma (float): the discount factor.
        - gae_lambda (float): the GAE lambda parameter.

        Returns:
        - advantages (Tensor): the advantages of the states.
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

    def get_trajectory_minibatches(self, timesteps: int, prob_go_from_end: float = 0.1) -> List[TrajectoryMinibatch]:
        '''Return a list of trajectory minibatches, where each minibatch contains
        experiences from a single trajectory.

        Args:
        - timesteps (int): the number of timesteps to include in each minibatch.

        Returns:
        - List[TrajectoryMinibatch]: a list of minibatches.
        '''
        obs, dones, actions, logprobs, values, rewards = [
            t.stack(arr) for arr in zip(*self.experiences)]

        next_values = t.cat([values[1:], self.next_value.unsqueeze(0)])
        next_dones = t.cat([dones[1:], self.next_done.unsqueeze(0)])

        # px.imshow(obs[:,1,:,:,0].transpose(-1,-2), animation_frame = 0, range_color = [0,10]).show()
        # set last value of dones to 1
        dones[-1] = t.ones(dones.shape[-1])

        # hack for now.
        # will cause problems if you only have one environment
        if logprobs.shape[-1] == 1:
            logprobs = logprobs.squeeze(-1)

        # rearrange to flatten out the env dimension (2nd dimension)
        obs = rearrange(obs, "T E ... -> (E T) ...")
        dones = rearrange(dones, "T E -> (E T)")
        next_dones = rearrange(next_dones, "T E -> (E T)")
        actions = rearrange(actions, "T E ... -> (E T) ...")
        logprobs = rearrange(logprobs, "T E -> (E T)")
        values = rearrange(values, "T E -> (E T)")
        next_values = rearrange(next_values, "T E -> (E T)")
        rewards = rearrange(rewards, "T E -> (E T)")

        # find the indices of the end of each trajectory
        traj_end_idxs = (t.where(dones)[0]).tolist()

        # split these trajectories on the dones
        traj_obs = t.tensor_split(obs, traj_end_idxs)
        traj_actions = t.tensor_split(actions, traj_end_idxs)
        traj_logprobs = t.tensor_split(logprobs, traj_end_idxs)
        traj_values = t.tensor_split(values, traj_end_idxs)
        traj_rewards = t.tensor_split(rewards, traj_end_idxs)
        traj_dones = t.tensor_split(dones, traj_end_idxs)
        traj_next_values = t.tensor_split(next_values, traj_end_idxs)
        traj_next_dones = t.tensor_split(next_dones, traj_end_idxs)
        print([i for i, (next_val, obs) in enumerate(
            zip(traj_next_values, traj_obs)) if len(next_val) != len(obs)])
        # px.imshow(traj_obs[0][:,:,:,0].transpose(-1,-2), animation_frame = 0, range_color = [0,10]).show()

        # so now we have lists of trajectories, what we want is to split each trajectory
        # so for each trajectory, sample an index and go n_steps back from that.
        # since we're encoding  states and actions, we want to go context_length//2 back
        # if that happens to go off the end, then we
        minibatches = []

        # remove trajectories of length 0
        traj_obs = [traj for traj in traj_obs if len(traj) > 0]
        traj_actions = [traj for traj in traj_actions if len(traj) > 0]
        traj_logprobs = [traj for traj in traj_logprobs if len(traj) > 0]
        traj_values = [traj for traj in traj_values if len(traj) > 0]
        traj_rewards = [traj for traj in traj_rewards if len(traj) > 0]
        traj_dones = [traj for traj in traj_dones if len(traj) > 0]
        traj_next_values = [traj for traj in traj_next_values if len(traj) > 0]
        traj_next_dones = [traj for traj in traj_next_dones if len(traj) > 0]

        n_trajectories = len(traj_obs)
        trajectory_lengths = [len(traj) for traj in traj_obs]

        for _ in range(self.args.num_minibatches):

            minibatch_obs = []
            minibatch_actions = []
            minibatch_logprobs = []
            minibatch_advantages = []
            minibatch_values = []
            minibatch_returns = []
            minibatch_timesteps = []
            minibatch_rewards = []

            for _ in range(self.args.minibatch_size):

                # randomly select a trajectory
                traj_idx = np.random.randint(n_trajectories)

                # randomly select an end index from the trajectory
                # TODO later add a hyperparameter to oversample last step
                traj_len = trajectory_lengths[traj_idx]

                if traj_len <= timesteps:
                    end_idx = traj_len
                    start_idx = 0
                else:
                    if prob_go_from_end is not None:
                        if random.random() < prob_go_from_end:
                            end_idx = traj_len
                            start_idx = end_idx - timesteps
                        else:
                            end_idx = np.random.randint(timesteps, traj_len)
                            start_idx = end_idx - timesteps
                    else:
                        end_idx = np.random.randint(timesteps, traj_len)
                        start_idx = end_idx - timesteps

                # print('end_idx:', end_idx)
                # print('traj_len:', traj_len)
                # print('len(traj_next_values[traj_idx]):', len(traj_next_values[traj_idx]))
                # if len(traj_next_values[traj_idx]) != traj_len:
                #     print("uh oh")

                # get the trajectory
                current_traj_obs = traj_obs[traj_idx][start_idx:end_idx]
                current_traj_actions = traj_actions[traj_idx][start_idx:end_idx]
                current_traj_logprobs = traj_logprobs[traj_idx][start_idx:end_idx]
                current_traj_values = traj_values[traj_idx][start_idx:end_idx]
                current_traj_dones = traj_dones[traj_idx][start_idx:end_idx]
                current_traj_rewards = traj_rewards[traj_idx][start_idx:end_idx]
                current_traj_next_value = traj_next_values[traj_idx][end_idx - 1]
                current_traj_next_done = traj_next_dones[traj_idx][end_idx - 1]

                # make timesteps
                current_traj_timesteps = t.arange(start_idx, end_idx)

                # Compute the advantages and returns for this trajectory.
                current_traj_advantages = self.compute_advantages(
                    current_traj_next_value,
                    current_traj_next_done,
                    current_traj_rewards,
                    current_traj_values,
                    current_traj_dones,
                    self.device,
                    self.args.gamma,
                    self.args.gae_lambda
                )

                current_traj_returns = current_traj_advantages + current_traj_values

                # we need to pad current_traj_obs and current_traj_actions
                current_traj_obs = pad_tensor(
                    current_traj_obs,
                    timesteps,
                    ignore_first_dim=False,
                    pad_token=0,
                    pad_left=True
                )

                current_traj_actions = pad_tensor(
                    current_traj_actions,
                    timesteps,
                    ignore_first_dim=False,
                    pad_token=0,
                    pad_left=True
                )

                current_traj_timesteps = pad_tensor(
                    current_traj_timesteps,
                    timesteps,
                    ignore_first_dim=False,
                    pad_token=0,
                    pad_left=True
                )
                # print([i for i, (next_val, obs) in enumerate(zip(traj_next_values, traj_obs)) if len(next_val) != len(obs)])
                # add to minibatch
                minibatch_obs.append(current_traj_obs)
                minibatch_actions.append(current_traj_actions)
                minibatch_logprobs.append(current_traj_logprobs[-1])
                minibatch_advantages.append(current_traj_advantages[-1])
                minibatch_values.append(current_traj_values[-1])
                minibatch_returns.append(current_traj_returns[-1])
                minibatch_rewards.append(current_traj_rewards[-1])
                minibatch_timesteps.append(current_traj_timesteps)

            # stack the minibatch
            minibatch_obs = t.stack(minibatch_obs)
            minibatch_actions = t.stack(minibatch_actions)

            # only take the last values of the logprob, advantage,
            # value and return (relevant to the last step of each trajectory)
            minibatch_logprobs = t.stack(minibatch_logprobs)
            minibatch_advantages = t.stack(minibatch_advantages)
            minibatch_values = t.stack(minibatch_values)
            minibatch_returns = t.stack(minibatch_returns)
            minibatch_timesteps = t.stack(minibatch_timesteps)
            minibatch_rewards = t.stack(minibatch_rewards)

            minibatches.append(TrajectoryMinibatch(
                obs=minibatch_obs,
                actions=minibatch_actions,
                logprobs=minibatch_logprobs,
                advantages=minibatch_advantages,
                values=minibatch_values,
                returns=minibatch_returns,
                timesteps=minibatch_timesteps,
                rewards=minibatch_rewards
            ))

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
