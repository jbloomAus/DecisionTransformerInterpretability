import os
import pickle
from dataclasses import dataclass

import numpy as np
import pytest
import torch

from src.config import EnvironmentConfig
from src.decision_transformer.utils import load_decision_transformer
from src.environments.environments import make_env
from src.utils import TrajectoryWriter


@pytest.fixture
def run_config():
    @dataclass
    class DummyRunConfig:
        exp_name: str = "test"
        seed: int = 1
        track: bool = False
        wandb_project_name: str = "test"
        wandb_entity: str = "test"

    return DummyRunConfig()


@pytest.fixture
def environment_config():
    @dataclass
    class DummyEnvironmentConfig:
        env_id: str = "MiniGrid-Dynamic-Obstacles-8x8-v0"
        one_hot_obs: bool = False
        img_obs: bool = False
        fully_observed: bool = False
        max_steps: int = 1000
        seed: int = 1
        view_size: int = 7
        capture_video: bool = False
        video_dir: str = "videos"
        render_mode: str = "rgb_array"
        action_space: None = None
        observation_space: None = None
        device: str = torch.device("cpu")

    return DummyEnvironmentConfig()


@pytest.fixture
def online_config():
    @dataclass
    class DummyOnlineConfig:
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
        device: str = torch.device("cpu")

    return DummyOnlineConfig()


def test_trajectory_writer_numpy(
    environment_config, run_config, online_config
):
    trajectory_writer = TrajectoryWriter(
        path="tmp/test_trajectory_writer_writer.pkl",
        run_config=run_config,
        environment_config=environment_config,
        online_config=online_config,
        model_config=None,
    )

    # test accumulate trajectory when all the objects are initialized as np arrays

    trajectory_writer.accumulate_trajectory(
        next_obs=np.array([1, 2, 3]),
        reward=np.array([1, 2, 3]),
        done=np.array([1, 0, 0]),
        truncated=np.array([1, 0, 0]),
        action=np.array([1, 2, 3]),
        info={"a": 1, "b": 2, "c": 3},
    )

    trajectory_writer.write()

    # get the size of the file in bytes
    assert os.path.getsize("tmp/test_trajectory_writer_writer.pkl") > 0
    # make sure it's less than 1500 bytes (got larger with more hyperparameters)
    assert os.path.getsize("tmp/test_trajectory_writer_writer.pkl") < 1500

    with open("tmp/test_trajectory_writer_writer.pkl", "rb") as f:
        data = pickle.load(f)

        obs = data["data"]["observations"]
        assert type(obs) == np.ndarray
        assert obs.dtype == np.float64

        assert obs[0][0] == 1
        assert obs[0][1] == 2
        assert obs[0][2] == 3

        rewards = data["data"]["rewards"]
        assert type(rewards) == np.ndarray
        assert rewards.dtype == np.float64

        assert rewards[0][0] == 1
        assert rewards[0][1] == 2
        assert rewards[0][2] == 3

        dones = data["data"]["dones"]
        assert type(dones) == np.ndarray
        assert dones.dtype == bool

        assert dones[0][0]
        assert ~dones[0][1]
        assert ~dones[0][2]

        actions = data["data"]["actions"]
        assert type(actions) == np.ndarray
        assert actions.dtype == np.int64

        assert actions[0][0] == 1
        assert actions[0][1] == 2
        assert actions[0][2] == 3

        infos = data["data"]["infos"]
        assert type(infos) == np.ndarray
        assert infos.dtype == np.object

        assert infos[0]["a"] == 1
        assert infos[0]["b"] == 2
        assert infos[0]["c"] == 3


def test_trajectory_writer_torch(
    environment_config, run_config, online_config
):
    trajectory_writer = TrajectoryWriter(
        path="tmp/test_trajectory_writer_writer.pkl",
        run_config=run_config,
        environment_config=environment_config,
        online_config=online_config,
        model_config=None,
    )

    # test accumulate trajectory when all the objects are initialized as pytorch tensors

    # assert raises type error
    with pytest.raises(TypeError):
        trajectory_writer.accumulate_trajectory(
            next_obs=torch.tensor([1, 2, 3], dtype=torch.float64),
            reward=torch.tensor([1, 2, 3], dtype=torch.float64),
            done=torch.tensor([1, 0, 0], dtype=torch.bool),
            action=torch.tensor([1, 2, 3], dtype=torch.int64),
            info=[{"a": 1, "b": 2, "c": 3}],
        )


def test_trajectory_writer_lzma(environment_config, run_config, online_config):
    trajectory_writer = TrajectoryWriter(
        path="tmp/test_trajectory_writer_writer.xz",
        run_config=run_config,
        environment_config=environment_config,
        online_config=online_config,
        model_config=None,
    )

    # test accumulate trajectory when all the objects are initialized as np arrays

    trajectory_writer.accumulate_trajectory(
        next_obs=np.array([1, 2, 3]),
        reward=np.array([1, 2, 3]),
        done=np.array([1, 0, 0]),
        truncated=np.array([1, 0, 0]),
        action=np.array([1, 2, 3]),
        info={"a": 1, "b": 2, "c": 3},
    )

    trajectory_writer.write()

    # get the size of the file in bytes
    assert os.path.getsize("tmp/test_trajectory_writer_writer.xz") > 0
    # make sure it's less than 200 bytes
    assert os.path.getsize("tmp/test_trajectory_writer_writer.xz") < 1000
