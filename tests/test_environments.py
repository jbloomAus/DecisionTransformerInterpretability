import pytest
from collections import Counter
from pytest import approx
from src.environments import make_env
import gymnasium as gym
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
        env_ids=env_id,
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


def test_make_env_change_view_size():
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
        env_ids=env_id,
        seed=seed,
        idx=idx,
        capture_video=capture_video,
        run_name=run_name,
        render_mode=render_mode,
        max_steps=max_steps,
        agent_view_size=5,
        fully_observed=fully_observed,
        video_frequency=video_frequency
    )

    obs, info = env_func().reset()

    assert isinstance(obs, dict)
    assert obs["image"].shape == (5, 5, 3)
    assert env_func is not None


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
        env_ids=env_id,
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
        env_ids=env_id,
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


def test_make_env_flat_one_hot_view_size_change():
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
        env_ids=env_id,
        seed=seed,
        idx=idx,
        capture_video=capture_video,
        run_name=run_name,
        render_mode=render_mode,
        max_steps=max_steps,
        fully_observed=fully_observed,
        flat_one_hot=flat_one_hot,
        agent_view_size=5,
        video_frequency=video_frequency
    )

    obs, info = env_func().reset()

    assert isinstance(obs, dict)
    assert obs["image"].shape == (5, 5, 20,)
    assert obs["image"].max() == 1
    assert env_func is not None


def test_multi_env_sampling():

    env_ids = ["MiniGrid-Dynamic-Obstacles-8x8-v0",
               "MiniGrid-Dynamic-Obstacles-5x5-v0"]
    seed = 1
    idx = 0
    capture_video = False
    run_name = "dev"
    fully_observed = False
    max_steps = 30
    num_envs = 1000

    envs = gym.vector.SyncVectorEnv([
        make_env(
            env_ids, seed=seed, idx=idx,
            capture_video=capture_video, run_name=run_name,
            fully_observed=fully_observed, max_steps=max_steps
        ) for i in range(num_envs)
    ])

    counts = Counter([env.env.spec.id for env in envs.envs])
    ratio = counts["MiniGrid-Dynamic-Obstacles-8x8-v0"] / \
        counts["MiniGrid-Dynamic-Obstacles-5x5-v0"]
    assert ratio == approx(
        1, rel=0.2), "The ratio of envs is not 50:50, it is {}".format(ratio)
