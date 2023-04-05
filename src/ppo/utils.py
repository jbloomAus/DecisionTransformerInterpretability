import argparse
import json
import os
import random
import uuid
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from IPython.display import display

import wandb
from src.config import ConfigJsonEncoder

# import syncvectorenv

MAIN = __name__ == "__main__"

Arr = np.ndarray
ObsType = np.ndarray
ActType = int


def parse_args():
    parser = argparse.ArgumentParser(
        prog='PPO',
        description='Proximal Policy Optimization',
        epilog="'You are personally responsible for becoming more ethical than the society you grew up in.'― Eliezer Yudkowsky")
    parser.add_argument('--exp_name', type=str, default='MiniGrid-Dynamic-Obstacles-8x8-v0',
                        help='the name of this experiment')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed of the experiment')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='if toggled, cuda will be enabled by default')
    parser.add_argument('--track', action='store_true', default=False,
                        help='if toggled, this experiment will be tracked with Weights and Biases')
    parser.add_argument('--wandb_project_name', type=str, default="PPO-MiniGrid",
                        help="the wandb's project name")
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help="the entity (team) of wandb's project")
    parser.add_argument('--capture_video', action='store_true', default=True,
                        help='if toggled, a video will be captured during evaluation')
    parser.add_argument('--env_id', type=str, default='MiniGrid-Dynamic-Obstacles-8x8-v0',
                        help='the environment id')
    parser.add_argument('--hidden_size', type=int, default=64,
                        help='the size of the hidden layers')
    parser.add_argument('--view_size', type=int, default=7,
                        help='the size of the view')
    parser.add_argument('--total_timesteps', type=int, default=5000000,
                        help='the total number of timesteps to train for')
    parser.add_argument('--learning_rate', type=float, default=0.00025,
                        help='the learning rate of the optimizer')
    parser.add_argument('--decay_lr', action='store_true', default=False,
                        help='if toggled, the learning rate will decay linearly')
    parser.add_argument('--num_envs', type=int, default=10,
                        help='the number of parallel environments')
    parser.add_argument('--num_steps', type=int, default=128,
                        help='the number of steps to run in each environment per policy rollout')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='the discount factor gamma')
    parser.add_argument('--gae_lambda', type=float, default=0.95,
                        help='the lambda for the general advantage estimation')
    parser.add_argument('--num_minibatches', type=int, default=4,
                        help='the number of mini batches')
    parser.add_argument('--update_epochs', type=int, default=4,
                        help='the K epochs to update the policy')
    parser.add_argument('--clip_coef', type=float, default=0.2,
                        help='the surrogate clipping coefficient')
    parser.add_argument('--vf_coef', type=float, default=0.5,
                        help='value loss coefficient')
    parser.add_argument('--ent_coef', type=float, default=0.01,
                        help='entropy term coefficient')
    parser.add_argument('--max_grad_norm', type=float, default=0.5,
                        help='the maximum norm for the gradient clipping')
    parser.add_argument('--max_steps', type=int, default=1000,
                        help='the maximum number of steps total')
    parser.add_argument('--trajectory_path', type=str, default=None,
                        help='the path to the trajectory file')
    parser.add_argument('--fully_observed', action='store_true', default=False,
                        help='if toggled, the environment will be fully observed')
    parser.add_argument('--one_hot_obs', action='store_true', default=False,
                        help='if toggled, the environment will be partially observed one hot encoded')
    parser.add_argument('--num_checkpoints', type=int, default=10,
                        help="how many checkpoints are stored and uploaded to wandb during training")

    args = parser.parse_args()
    return args


@dataclass
class PPOArgs:
    exp_name: str = 'MiniGrid-Dynamic-Obstacles-8x8-v0'
    seed: int = 1
    cuda: bool = True
    track: bool = True
    wandb_project_name: str = "PPO-MiniGrid"
    wandb_entity: str = None
    capture_video: bool = True
    env_id: str = 'MiniGrid-Dynamic-Obstacles-8x8-v0'
    view_size: int = 7
    hidden_size: int = 64
    total_timesteps: int = 1800000
    learning_rate: float = 0.00025
    decay_lr: bool = False,
    num_envs: int = 4
    num_steps: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    clip_coef: float = 0.4
    ent_coef: float = 0.2
    vf_coef: float = 0.5
    max_grad_norm: float = 2
    max_steps: int = 1000
    one_hot_obs: bool = False
    trajectory_path: str = None
    fully_observed: bool = False

    def __post_init__(self):
        self.batch_size = int(self.num_envs * self.num_steps)
        self.minibatch_size = self.batch_size // self.num_minibatches
        if self.trajectory_path is None:
            self.trajectory_path = os.path.join(
                "trajectories", self.env_id + str(uuid.uuid4()) + ".gz")


arg_help_strings = dict(
    exp_name="the name of this experiment",
    seed="seed of the experiment",
    cuda="if toggled, cuda will be enabled by default",
    track="if toggled, this experiment will be tracked with Weights and Biases",
    wandb_project_name="the wandb's project name",
    wandb_entity="the entity (team) of wandb's project",
    capture_video="whether to capture videos of the agent performances (check out `videos` folder)",
    env_id="the id of the environment",
    total_timesteps="total timesteps of the experiments",
    learning_rate="the learning rate of the optimizer",
    num_envs="number of synchronized vector environments in our `envs` object",
    num_steps="number of steps taken in the rollout phase",
    gamma="the discount factor gamma",
    gae_lambda="the discount factor used in our GAE estimation",
    update_epochs="how many times you loop through the data generated in rollout",
    clip_coef="the epsilon term used in the policy loss function",
    ent_coef="coefficient of entropy bonus term",
    vf_coef="cofficient of value loss function",
    max_grad_norm="value used in gradient clipping",
    # batch_size = "number of random samples we take from the rollout data",
    # minibatch_size = "size of each minibatch we perform a gradient step on",
    max_steps="maximum number of steps in an episode",
    trajectory_path="path to save the trajectories",
)


def arg_help(args: Optional[PPOArgs], print_df=False):
    """Prints out a nicely displayed list of arguments, their default values, and what they mean."""
    if args is None:
        args = PPOArgs()
        changed_args = []
    else:
        default_args = PPOArgs()
        changed_args = [key for key in default_args.__dict__ if getattr(
            default_args, key) != getattr(args, key)]
    df = pd.DataFrame([arg_help_strings]).T
    df.columns = ["description"]
    df["default value"] = [repr(getattr(args, name)) for name in df.index]
    df.index.name = "arg"
    df = df[["default value", "description"]]
    if print_df:
        df.insert(1, "changed?", [
                  "yes" if i in changed_args else "" for i in df.index])
        with pd.option_context(
            'max_colwidth', 0,
            'display.width', 150,
            'display.colheader_justify', 'left'
        ):
            print(df)
    else:
        s = (
            df.style
            .set_table_styles([
                {'selector': 'td', 'props': 'text-align: left;'},
                {'selector': 'th', 'props': 'text-align: left;'}
            ])
            .apply(lambda row: [
                'background-color: red' if row.name in changed_args else None
            ] + [None, ] * (len(row) - 1), axis=1)
        )
        with pd.option_context("max_colwidth", 0):
            display(s)


def set_global_seeds(seed):
    '''Sets random seeds in several different ways (to guarantee reproducibility)
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_obs_preprocessor(obs_space):

    # handle cases where obs space is instance of gym.spaces.Box, gym.spaces.Dict, gym.spaces

    if isinstance(obs_space, gym.spaces.Box):
        return lambda x: np.array(x).astype(np.float32)

    elif isinstance(obs_space, gym.spaces.Dict):
        obs_space = obs_space.spaces
        if 'image' in obs_space:
            return lambda x: preprocess_images(x['image'])

    elif (isinstance(obs_space, gym.spaces.Discrete) or isinstance(obs_space, gym.spaces.MultiDiscrete)):
        return lambda x: np.array(x).astype(np.float32)

    else:
        raise NotImplementedError(
            "Observation space not supported: {}".format(obs_space))


def preprocess_images(images, device=None):
    # Bug of Pytorch: very slow if not first converted to numpy array
    images = np.array(images)
    images = images.astype(np.float32)
    return images


def get_obs_shape(single_observation_space) -> tuple:
    '''
    Returns the shape of a single observation.

    Args:
        single_observation_space (gym.spaces.Box, gym.spaces.Discrete, gym.spaces.Dict): The observation space of a single agent.

    Returns:
        tuple: The shape of a single observation.
    '''
    if isinstance(single_observation_space, gym.spaces.Box):
        obs_shape = single_observation_space.shape
    elif isinstance(single_observation_space, gym.spaces.Discrete):
        obs_shape = (single_observation_space.n,)
    elif isinstance(single_observation_space, gym.spaces.Dict):
        obs_shape = single_observation_space.spaces["image"].shape
    else:
        raise ValueError("Unsupported observation space")
    return obs_shape


def store_model_checkpoint(agent, online_config, run_config, checkpoint_num, checkpoint_artifact) -> int:

    checkpoint_name = f"{run_config.exp_name}_{checkpoint_num:0>2}"
    checkpoint_path = f"models/{checkpoint_name}.pt"

    torch.save({
        "model_state_dict": agent.state_dict(),
        "online_config": json.dumps(online_config, cls=ConfigJsonEncoder),
        "environment_config": json.dumps(agent.environment_config, cls=ConfigJsonEncoder),
        "model_config": json.dumps(agent.model_config, cls=ConfigJsonEncoder),
    }, checkpoint_path)

    checkpoint_artifact.add_file(
        local_path=checkpoint_path, name=f"{checkpoint_name}.pt")

    return checkpoint_num + 1
