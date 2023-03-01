from dataclasses import dataclass
import gymnasium as gym

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

