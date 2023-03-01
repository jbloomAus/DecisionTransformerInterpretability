from dataclasses import dataclass
import gymnasium as gym
from torch import device
from typing import Optional


@dataclass
class TransformerModelConfig():
    d_model: int = 128
    n_heads: int = 4
    d_mlp: int = 256
    n_layers: int = 2
    n_ctx: int = 3
    layer_norm: bool = False
    linear_time_embedding: bool = False
    state_embedding_type: str = 'grid'
    time_embedding_type: str = 'learned'
    seed: int = 1
    device: str = 'cpu'

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0
        self.d_head = self.d_model // self.n_heads


@dataclass
class EnvironmentConfig():
    env = None
    env_id: str = 'MiniGrid-Empty-8x8-v0'
    one_hot: bool = False
    fully_observed: bool = False
    max_steps: int = 1000
    seed: int = 1
    view_size: int = 7
    capture_video: bool = False
    video_dir: str = 'videos'
    render_mode: str = 'rgb_array'
    num_parralel_envs: int = 1
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
class WandbConfig:
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_entity: Optional[str] = None
