import pytest

import gymnasium as gym
from src.environments import make_env
import numpy as np


def test_make_env():
    env_id = "MiniGrid-Empty-5x5-v0"
    seed = 0
    idx = 0
    capture_video = False
    run_name = "test"
    render_mode = "rgb_array"
    max_steps = 100
    fully_observed = False
    video_frequency = 50

    env_func = make_env(
        env_id=env_id,
        seed=seed,
        idx=idx,
        capture_video=capture_video,
        run_name=run_name,
        render_mode=render_mode,
        max_steps=max_steps,
        fully_observed=fully_observed,
        video_frequency=video_frequency
    )

    obs, info = env_func().reset()

    assert isinstance(obs, dict)
    assert obs["image"].shape == (7, 7, 3)
    assert env_func is not None


def test_make_env_mission_space():
    env_id = "MiniGrid-ObstructedMaze-1Dl-v0"
    seed = 0
    idx = 0
    capture_video = False
    run_name = "test"
    render_mode = "rgb_array"
    max_steps = 100
    fully_observed = False
    video_frequency = 50

    env_func = make_env(
        env_id=env_id,
        seed=seed,
        idx=idx,
        capture_video=capture_video,
        run_name=run_name,
        render_mode=render_mode,
        max_steps=max_steps,
        fully_observed=fully_observed,
        video_frequency=video_frequency
    )

    env = env_func()
    obs, info = env.reset()

    assert isinstance(obs, dict)
    assert obs["image"].shape == (7, 7, 3)
    assert env_func is not None


def test_make_env_mission_space_sync_vector():
    env_id = "MiniGrid-ObstructedMaze-1Dl-v0"
    seed = 0
    idx = 0
    capture_video = False
    run_name = "test"
    render_mode = "rgb_array"
    max_steps = 100
    fully_observed = False
    video_frequency = 50
    num_envs = 10

    env_func = make_env(
        env_id=env_id,
        seed=seed,
        idx=idx,
        capture_video=capture_video,
        run_name=run_name,
        render_mode=render_mode,
        max_steps=max_steps,
        fully_observed=fully_observed,
        video_frequency=video_frequency
    )

    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, seed + i, i, capture_video, run_name,
                  max_steps=max_steps,
                  fully_observed=fully_observed,
                  ) for i in range(num_envs)]
    )

    # obs, info = env_func().reset()

    # assert isinstance(obs, dict)
    # assert obs["image"].shape == (7, 7, 3)
    # assert env_func is not None


def test_make_env_fully_observed():
    env_id = "MiniGrid-Empty-5x5-v0"
    seed = 0
    idx = 0
    capture_video = False
    run_name = "test"
    render_mode = "rgb_array"
    max_steps = 100
    fully_observed = True
    video_frequency = 50

    env_func = make_env(
        env_id=env_id,
        seed=seed,
        idx=idx,
        capture_video=capture_video,
        run_name=run_name,
        render_mode=render_mode,
        max_steps=max_steps,
        fully_observed=fully_observed,
        video_frequency=video_frequency
    )
    obs, info = env_func().reset()

    assert isinstance(obs, dict)
    assert obs["image"].shape == (5, 5, 3)
    assert env_func is not None


def test_make_env_flat_one_hot():
    env_id = "MiniGrid-Empty-5x5-v0"
    seed = 0
    idx = 0
    capture_video = False
    run_name = "test"
    render_mode = "rgb_array"
    max_steps = 100
    fully_observed = False
    flat_one_hot = True
    video_frequency = 50

    env_func = make_env(
        env_id=env_id,
        seed=seed,
        idx=idx,
        capture_video=capture_video,
        run_name=run_name,
        render_mode=render_mode,
        max_steps=max_steps,
        fully_observed=fully_observed,
        flat_one_hot=flat_one_hot,
        video_frequency=video_frequency
    )

    obs, info = env_func().reset()

    assert isinstance(obs, dict)
    assert obs["image"].shape == (7, 7, 20,)
    assert obs["image"].max() == 1
    assert env_func is not None
