import numpy as np
import pytest

from src.config import EnvironmentConfig
from src.environments.environments import make_env


def test_make_env():
    env_config = EnvironmentConfig(
        env_id="MiniGrid-Empty-5x5-v0",
        capture_video=False,
        render_mode="rgb_array",
        max_steps=100,
        fully_observed=False,
        video_frequency=50,
    )

    env_func = make_env(env_config, seed=0, idx=0, run_name="test")

    obs, info = env_func().reset()

    assert isinstance(obs, dict)
    assert obs["image"].shape == (7, 7, 3)
    assert env_func is not None


def test_make_env_change_view_size():
    env_config = EnvironmentConfig(
        env_id="MiniGrid-Empty-5x5-v0",
        capture_video=False,
        render_mode="rgb_array",
        max_steps=100,
        fully_observed=False,
        video_frequency=50,
        view_size=5,
    )

    env_func = make_env(env_config, seed=0, idx=0, run_name="test")

    obs, info = env_func().reset()

    assert isinstance(obs, dict)
    assert obs["image"].shape == (5, 5, 3)
    assert env_func is not None


def test_make_env_fully_observed():
    env_config = EnvironmentConfig(
        env_id="MiniGrid-Empty-5x5-v0",
        capture_video=False,
        render_mode="rgb_array",
        max_steps=100,
        fully_observed=True,
        video_frequency=50,
    )

    env_func = make_env(env_config, seed=0, idx=0, run_name="test")
    obs, info = env_func().reset()

    assert isinstance(obs, dict)
    assert obs["image"].shape == (5, 5, 3)
    assert env_func is not None


def test_make_env_flat_one_hot():
    env_config = EnvironmentConfig(
        env_id="MiniGrid-Empty-5x5-v0",
        capture_video=False,
        render_mode="rgb_array",
        max_steps=100,
        fully_observed=False,
        video_frequency=50,
        one_hot_obs=True,
    )

    env_func = make_env(env_config, seed=0, idx=0, run_name="test")

    obs, info = env_func().reset()

    assert isinstance(obs, dict)
    assert obs["image"].shape == (
        7,
        7,
        20,
    )
    assert obs["image"].max() == 1
    assert env_func is not None


def test_make_env_flat_one_hot_view_size_change():
    env_config = EnvironmentConfig(
        env_id="MiniGrid-Empty-5x5-v0",
        capture_video=False,
        render_mode="rgb_array",
        max_steps=100,
        fully_observed=False,
        video_frequency=50,
        one_hot_obs=True,
        view_size=5,
    )

    env_func = make_env(env_config, seed=0, idx=0, run_name="test")

    obs, info = env_func().reset()

    assert isinstance(obs, dict)
    assert obs["image"].shape == (
        5,
        5,
        20,
    )
    assert obs["image"].max() == 1
    assert env_func is not None
