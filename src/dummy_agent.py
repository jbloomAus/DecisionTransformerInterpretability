from src.utils import TrajectoryWriter
import torch.nn as nn
import torch as t
import gymnasium as gym
from typing import Optional
from src.ppo.utils import get_obs_preprocessor
import numpy as np
from dataclasses import dataclass
import tqdm._tqdm_notebook as tqdm


@dataclass
class RandomAgentArgs:
    def __init__(self,
                 run_name,
                 view_size: int = 7,
                 num_envs: int = 2,
                 one_hot_obs: bool = False,
                 prob_go_from_end: float = 0.0,
                 env_id: str = "MiniGrid-Dynamic-Obstacles-8x8-v0",
                 ):
        self.agent_type = "random"
        self.run_name = run_name
        self.capture_video = False
        self.fully_observed = False
        self.max_steps = 30
        self.one_hot = False
        self.view_size = view_size
        self.num_envs = num_envs
        self.one_hot_obs = one_hot_obs
        self.env_id = env_id
        self.seed = 1
        self.device = t.device('cuda' if t.cuda.is_available() else 'cpu')
        self.prob_go_from_end = prob_go_from_end


class RandomAgent(nn.Module):
    '''
    A random agent that selects a valid action at each step in a given minigrid environment.
    '''

    def __init__(
        self,
        env,
        seed: int = 1,
        num_steps: int = 128,
        device=t.device('cuda' if t.cuda.is_available() else 'cpu')
    ):
        '''
        Initializes the random agent.

        Args:
            env (gym-minigrid): The minigrid environment to act in.
            seed (int): The random seed to use. Default is 1.
        '''
        super().__init__()

        self.env = env
        self.action_space = env.action_space
        self.rng = t.Generator(device='cuda') if t.cuda.is_available(
        ) else t.Generator(device='cpu')
        self.rng.manual_seed(seed)
        self.obs_preprocessor = get_obs_preprocessor(env.observation_space)
        self.device = device
        self.num_steps = num_steps

    def get_action(self, obs):
        '''
        Randomly selects a valid action in the current minigrid environment.

        Args:
            obs (numpy array): The observation of the current state.

        Returns:
            action (int): The integer representing the selected action.
        '''
        # get a random action for each env
        action = t.randint(low=0, high=self.action_space.n, size=(
            obs.shape[0],), device=self.device, generator=self.rng)

        return action

    def collect_trajectories(self, env, num_trajectories: int, trajectory_writer: Optional['TrajectoryWriter'] = None) -> None:
        '''
        Collects trajectories from the environment.

        Args:
            env (gym-minigrid): The minigrid environment to act in.
            num_trajectories (int): The number of trajectories to collect.
            trajectory_writer (TrajectoryWriter, optional): The trajectory writer to accumulate the trajectories. Defaults to None.
        '''
        pbar = tqdm.tqdm(total=num_trajectories)
        for _ in range(num_trajectories):
            self.rollout(env, trajectory_writer)

    def rollout(self, env, trajectory_writer: Optional['TrajectoryWriter'] = None) -> None:
        '''
        Perform the rollout phase and accumulate trajectories in the trajectory writer.

        Args:
            env (gym-minigrid): The minigrid environment to act in.
            trajectory_writer (TrajectoryWriter, optional): The trajectory writer to accumulate the trajectories. Defaults to None.
        '''
        obs, info = env.reset()
        obs = self.obs_preprocessor(obs)
        done = False

        for _ in range(self.num_steps):
            action = self.get_action(obs)
            next_obs, reward, done, truncated, info = env.step(action)
            next_obs = self.obs_preprocessor(next_obs)

            if trajectory_writer is not None:
                trajectory_writer.accumulate_trajectory(
                    next_obs=obs.astype(np.float32),
                    reward=reward.astype(np.float32),
                    action=action.detach().numpy().astype(np.int64),
                    done=done.astype(bool),
                    truncated=truncated.astype(bool),
                    info=info
                )

            obs = next_obs

        env.close()
