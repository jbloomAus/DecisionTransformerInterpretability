import os
import uuid
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
from torch import device


@dataclass
class TransformerModelConfig():
    d_model: int = 128
    n_heads: int = 4
    d_mlp: int = 256
    n_layers: int = 2
    n_ctx: int = 3
    layer_norm: bool = False
    state_embedding_type: str = 'grid'
    time_embedding_type: str = 'learned'
    seed: int = 1
    device: str = 'cpu'

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0
        self.d_head = self.d_model // self.n_heads
        assert self.time_embedding_type in ['learned', 'linear']


@dataclass
class EnvironmentConfig():
    env_id: str = 'MiniGrid-Dynamic-Obstacles-8x8-v0'
    one_hot_obs: bool = False
    fully_observed: bool = False
    max_steps: int = 1000
    seed: int = 1
    view_size: int = 7
    capture_video: bool = False
    video_dir: str = 'videos'
    render_mode: str = 'rgb_array'
    action_space: None = None
    observation_space: None = None
    device: str = 'cpu'

    def __post_init__(self):

        env = gym.make(self.env_id)
        self.action_space = env.action_space or env.action_space
        self.observation_space = env.observation_space or env.observation_space


@dataclass
class OfflineTrainConfig:
    batch_size: int = 128
    lr: float = 0.0001
    weight_decay: float = 0.0
    pct_traj: float = 1.0
    prob_go_from_end: float = 0.0
    device: device = device("cpu")
    track: bool = False
    train_epochs: int = 100
    test_epochs: int = 10
    test_frequency: int = 10
    eval_frequency: int = 10
    eval_episodes: int = 10
    initial_rtg: list[float] = (0.0, 1.0)
    eval_max_time_steps: int = 100
    trajectory_path: Optional[str] = None


@dataclass
class OnlineTrainConfig:
    hidden_size: int = 64
    total_timesteps: int = 180000
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
    trajectory_path: str = None
    fully_observed: bool = False

    def __post_init__(self):
        self.batch_size = int(self.num_envs * self.num_steps)
        self.minibatch_size = self.batch_size // self.num_minibatches


@dataclass
class RunConfig:
    exp_name: str = 'MiniGrid-Dynamic-Obstacles-8x8-v0'
    seed: int = 1
    cuda: bool = True
    track: bool = True
    wandb_project_name: str = "PPO-MiniGrid"
    wandb_entity: str = None
    trajectory_path: Optional[str] = None

    def __post_init__(self):
        if self.trajectory_path is None:
            self.trajectory_path = os.path.join(
                "trajectories", self.exp_name + str(uuid.uuid4()) + ".gz")
