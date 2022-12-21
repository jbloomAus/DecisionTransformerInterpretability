from distutils.util import strtobool
import gym
import numpy as np
from typing import List
import argparse
import os
import random
import torch as t
Arr = np.ndarray


def make_env(env_id: str, seed: int, idx: int, capture_video: bool, run_name: str):
    """Return a function that returns an environment after setting up boilerplate."""
    
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        obs = env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    
    return thunk

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)

def window_avg(arr: Arr, window: int):
    """
    Computes sliding window average
    """
    return np.convolve(arr, np.ones(window), mode="valid") / window

def cummean(arr: Arr):
    """
    Computes the cumulative mean
    """
    return np.cumsum(arr) / np.arange(1, len(arr) + 1)

# Taken from https://stackoverflow.com/questions/42869495/numpy-version-of-exponential-weighted-moving-average-equivalent-to-pandas-ewm
# See https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average
def ewma(arr : Arr, alpha : float):
    '''
    Returns the exponentially weighted moving average of x.
    Parameters:
    -----------
    x : array-like
    alpha : float {0 <= alpha <= 1}
    Returns:
    --------
    ewma: numpy array
          the exponentially weighted moving average
    '''
    # Coerce x to an array
    s = np.zeros_like(arr)
    s[0] = arr[0]
    for i in range(1,len(arr)):
        s[i] = alpha * arr[i] + (1-alpha)*s[i-1]
    return s


def sum_rewards(rewards : List[int], gamma : float = 1):
    """
    Computes the total discounted sum of rewards for an episode.
    By default, assume no discount
    Input:
        rewards [r1, r2, r3, ...] The rewards obtained during an episode
        gamma: Discount factor
    Output:
        The sum of discounted rewards 
        r1 + gamma*r2 + gamma^2 r3 + ...
    """
    total_reward = 0
    for r in rewards[:0:-1]: #reverse, excluding first
        total_reward += r
        total_reward *= gamma
    total_reward += rewards[0]
    return total_reward

arg_help_strings = {
    "exp_name": "the name of this experiment",
    "seed": "seed of the experiment",
    "torch_deterministic": "if toggled, " "`torch.backends.cudnn.deterministic=False`",
    "cuda": "if toggled, cuda will be enabled by default",
    "track": "if toggled, this experiment will be tracked with Weights and Biases",
    "wandb_project_name": "the wandb's project name",
    "wandb_entity": "the entity (team) of wandb's project",
    "capture_video": "whether to capture videos of the agent performances (check " "out `videos` folder)",
    "env_id": "the id of the environment",
    "total_timesteps": "total timesteps of the experiments",
    "learning_rate": "the learning rate of the optimizer",
    "buffer_size": "the replay memory buffer size",
    "gamma": "the discount factor gamma",
    "target_network_frequency": "the timesteps it takes to update the target " "network",
    "batch_size": "the batch size of samples from the replay memory",
    "start_e": "the starting epsilon for exploration",
    "end_e": "the ending epsilon for exploration",
    "exploration_fraction": "the fraction of `total-timesteps` it takes from " "start-e to go end-e",
    "learning_starts": "timestep to start learning",
    "train_frequency": "the frequency of training",
    "use_target_network": "If True, use a target network.",
}
toggles = ["torch_deterministic", "cuda", "track", "capture_video"]