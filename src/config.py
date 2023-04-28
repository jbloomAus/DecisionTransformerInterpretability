"""
This module contains the configuration classes for the project.
"""
import copy
import dataclasses
import json
import os
import uuid
from dataclasses import dataclass

import gymnasium as gym
import torch
from minigrid.wrappers import (
    FullyObsWrapper,
    OneHotPartialObsWrapper,
    RGBImgPartialObsWrapper,
)

from src.environments.wrappers import ViewSizeWrapper


@dataclass
class EnvironmentConfig:
    """
    Configuration class for the environment.
    """

    env_id: str = "MiniGrid-Dynamic-Obstacles-8x8-v0"
    one_hot_obs: bool = False
    img_obs: bool = False
    fully_observed: bool = False
    max_steps: int = 1000
    seed: int = 1
    view_size: int = 7
    capture_video: bool = False
    video_dir: str = "videos"
    video_frequency: int = 50
    render_mode: str = "rgb_array"
    action_space: None = None
    observation_space: None = None
    device: str = "cpu"

    def __post_init__(self):
        env = gym.make(self.env_id)

        if self.env_id.startswith("MiniGrid"):
            if self.fully_observed:
                env = FullyObsWrapper(env)
            elif self.one_hot_obs:
                env = OneHotPartialObsWrapper(env)
            elif self.img_obs:
                env = RGBImgPartialObsWrapper(env)

            if self.view_size != 7:
                env = ViewSizeWrapper(env, self.view_size)

        self.action_space = self.action_space or env.action_space
        self.observation_space = (
            self.observation_space or env.observation_space
        )
        if isinstance(self.device, str):
            self.device = torch.device(self.device)


@dataclass
class TransformerModelConfig:
    """
    Configuration class for the transformer model.
    """

    d_model: int = 128
    n_heads: int = 4
    d_mlp: int = 256
    n_layers: int = 2
    n_ctx: int = 2
    layer_norm: bool = False
    state_embedding_type: str = "grid"
    time_embedding_type: str = "embedding"
    seed: int = 1
    device: str = "cpu"

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0
        self.d_head = self.d_model // self.n_heads
        assert self.time_embedding_type in ["embedding", "linear"]
        if isinstance(self.device, str):
            self.device = torch.device(self.device)


@dataclass
class LSTMModelConfig:
    """
    Configuration class for the LSTM model.
    """

    environment_config: EnvironmentConfig
    image_dim: int = 128
    memory_dim: int = 128
    instr_dim: int = 128
    use_instr: bool = False
    lang_model: str = "gru"
    use_memory: bool = False
    recurrence: int = 4
    arch: str = "bow_endpool_res"
    aux_info: bool = False
    device: str = "cpu"

    def __post_init__(self):
        for part in self.arch.split("_"):
            if part not in [
                "original",
                "bow",
                "pixels",
                "endpool",
                "res",
                "simple",
            ]:
                raise ValueError(
                    "Incorrect architecture name: {}".format(self.arch)
                )

        self.endpool = "endpool" in self.arch
        self.bow = "bow" in self.arch
        self.pixel = "pixel" in self.arch
        self.res = "res" in self.arch

        if self.res and self.image_dim != 128:
            raise ValueError(
                f"image_dim is {self.model_config.image_dim}, expected 128"
            )

        assert self.lang_model in ["gru", "bigru", "attgru"]
        # self.observation_space = self.environment_config.observation_space
        # self.action_space = self.environment_config.action_space
        if isinstance(self.device, str):
            self.device = torch.device(self.device)


@dataclass
class OfflineTrainConfig:
    """
    Configuration class for offline training.
    """

    trajectory_path: str
    batch_size: int = 128
    lr: float = 0.0001
    weight_decay: float = 0.0
    pct_traj: float = 1.0
    prob_go_from_end: float = 0.0
    device: str = "cpu"
    track: bool = False
    train_epochs: int = 100
    test_epochs: int = 10
    test_frequency: int = 10
    eval_frequency: int = 10
    eval_episodes: int = 10
    model_type: str = "decision_transformer"
    convert_to_one_hot: bool = False
    initial_rtg: list[float] = (0.0, 1.0)
    eval_max_time_steps: int = 100
    eval_num_envs: int = 8
    num_checkpoints: int = 10

    def __post_init__(self):
        assert self.model_type in ["decision_transformer", "clone_transformer"]
        if isinstance(self.device, str):
            self.device = torch.device(self.device)


@dataclass
class OnlineTrainConfig:
    """
    Configuration class for online training.
    """

    use_trajectory_model: bool = False
    hidden_size: int = 64
    total_timesteps: int = 180000
    learning_rate: float = 0.00025
    decay_lr: bool = (False,)
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
    prob_go_from_end: float = 0.0
    num_checkpoints: int = 10
    device: str = "cpu"

    def __post_init__(self):
        self.batch_size = int(self.num_envs * self.num_steps)
        self.minibatch_size = self.batch_size // self.num_minibatches

        if self.trajectory_path is None:
            self.trajectory_path = os.path.join(
                "trajectories", str(uuid.uuid4()) + ".gz"
            )

        if isinstance(self.device, str):
            self.device = torch.device(self.device)


@dataclass
class RunConfig:
    """
    Configuration class for running the model.
    """

    exp_name: str = "MiniGrid-Dynamic-Obstacles-8x8-v0"
    seed: int = 1
    device: str = "cpu"
    track: bool = True
    wandb_project_name: str = "PPO-MiniGrid"
    wandb_entity: str = None

    def __post_init__(self):
        if isinstance(self.device, str):
            self.device = torch.device(self.device)


class ConfigJsonEncoder(json.JSONEncoder):
    def default(self, config: dataclasses.dataclass):
        new_config = copy.deepcopy(config)

        if hasattr(new_config, "device") and new_config.device is not None:
            new_config.device = str(new_config.device)

        # remove observation space and action space
        if (
            hasattr(new_config, "observation_space")
            and new_config.observation_space is not None
        ):
            # will be instantiated from env_id
            delattr(new_config, "observation_space")
        if (
            hasattr(new_config, "action_space")
            and new_config.action_space is not None
        ):
            # will be instantiated from env_id
            delattr(new_config, "action_space")

        # check if new config is a dataclass
        if dataclasses.is_dataclass(new_config):
            return dataclasses.asdict(new_config)
        elif isinstance(new_config, torch.device):
            return str(new_config)
        elif isinstance(new_config, gym.spaces.Space):
            return None  # don't save observation space and action space if they are named other stuff
        else:
            return super().default(config)


def parse_metadata_to_environment_config(metadata: dict):
    """
    Parses the metadata dictionary from a loaded trajectory to an EnvironmentConfig object.
    """

    env_id = metadata["env_id"]
    one_hot_obs = metadata["one_hot_obs"]
    img_obs = metadata["img_obs"]
    fully_observed = metadata["fully_observed"]
    max_steps = metadata["max_steps"]
    seed = metadata["seed"]
    view_size = metadata["view_size"]
    capture_video = metadata["capture_video"]
    video_dir = metadata["video_dir"]
    render_mode = metadata["render_mode"]

    return EnvironmentConfig(
        env_id=env_id,
        one_hot_obs=one_hot_obs,
        img_obs=img_obs,
        fully_observed=fully_observed,
        max_steps=max_steps,
        seed=seed,
        view_size=view_size,
        capture_video=capture_video,
        video_dir=video_dir,
        render_mode=render_mode,
    )
