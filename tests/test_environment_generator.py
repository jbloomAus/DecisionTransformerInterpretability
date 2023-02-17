import pytest
from collections import Counter
from pytest import approx
from src.environment_generator import EnvGenerator, EnvironmentArgs, MissionSpaceEqualRedefinitionWrapper
import gymnasium as gym
import numpy as np


@pytest.fixture
def env_args():
    return EnvironmentArgs(
        env_ids=["MiniGrid-Empty-5x5-v0"],
        seed=0,
        capture_video=False,
        run_name="test",
        render_mode="rgb_array",
        max_steps=100,
        fully_observed=False,
        flat_one_hot=False,
        agent_view_size=7,
        video_frequency=50
    )


def test_env_generator(env_args):
    env_generator = EnvGenerator(env_args)
    env = env_generator.generate_envs(1)
    obs, info = env.reset()

    assert isinstance(obs, dict)
    assert obs["image"].shape == (1, 7, 7, 3)
    assert env is not None


def test_env_generator_change_view_size(env_args):
    env_args.agent_view_size = 5
    env_generator = EnvGenerator(env_args)
    env = env_generator.generate_envs(1)
    obs, info = env.reset()

    assert isinstance(obs, dict)
    assert obs["image"].shape == (1, 5, 5, 3)
    assert env is not None


def test_env_generator_fully_observed(env_args):
    env_args.fully_observed = True
    env_generator = EnvGenerator(env_args)
    env = env_generator.generate_envs(1)
    obs, info = env.reset()

    assert isinstance(obs, dict)
    assert obs["image"].shape == (1, 5, 5, 3)
    assert env is not None


def test_env_generator_flat_one_hot(env_args):
    env_args.flat_one_hot = True
    env_generator = EnvGenerator(env_args)


def test_env_generator_flat_one_hot_view_size_change(env_args):
    env_args.agent_view_size = 5
    env_args.flat_one_hot = True
    env_generator = EnvGenerator(env_args)
    env = env_generator.generate_envs(1)
    obs, info = env.reset()

    assert isinstance(obs, dict)
    assert obs["image"].shape == (1, 5, 5, 20,)
    assert obs["image"].max() == 1
    assert env is not None


def test_env_generator_multi_sampling(env_args):
    env_args.env_ids = ["MiniGrid-Dynamic-Obstacles-8x8-v0",
                        "MiniGrid-Dynamic-Obstacles-5x5-v0"]
    env_args.seed = 1
    env_args.capture_video = False
    env_args.run_name = "dev"
    env_args.fully_observed = False
    env_args.max_steps = 30
    num_envs = 1000

    env_generator = EnvGenerator(env_args)

    envs = env_generator.generate_envs(num_envs)

    counts = Counter([env.env.spec.id for env in envs.envs])
    ratio = counts["MiniGrid-Dynamic-Obstacles-8x8-v0"] / \
        counts["MiniGrid-Dynamic-Obstacles-5x5-v0"]
    assert ratio == approx(
        1, rel=0.1), "The ratio of envs is not 50:50, it is {}".format(ratio)


def test_env_generator_multi_sampling_different_envs(env_args):
    env_args.env_ids = ["MiniGrid-SimpleCrossingS9N3-v0",
                        "MiniGrid-LavaCrossingS11N5-v0"]
    env_args.seed = 1
    env_args.capture_video = False
    env_args.run_name = "dev"
    env_args.fully_observed = False
    env_args.max_steps = 30
    num_envs = 1000

    env_generator = EnvGenerator(env_args)

    envs = env_generator.generate_envs(num_envs)

    counts = Counter([env.env.spec.id for env in envs.envs])
    ratio = counts["MiniGrid-SimpleCrossingS9N3-v0"] / \
        counts["MiniGrid-LavaCrossingS11N5-v0"]
    assert ratio == approx(
        1, rel=0.1), "The ratio of envs is not 50:50, it is {}".format(ratio)


def test_env_generator_multi_sampling_many_different_envs(env_args):
    env_args.env_ids = [
        "MiniGrid-SimpleCrossingS9N1-v0",
        "MiniGrid-SimpleCrossingS9N3-v0",
        "MiniGrid-SimpleCrossingS9N2-v0",
        "MiniGrid-LavaCrossingS9N1-v0",
        "MiniGrid-LavaCrossingS9N2-v0",
        "MiniGrid-LavaCrossingS9N3-v0",
        "MiniGrid-LavaCrossingS11N5-v0"]
    env_args.seed = 1
    env_args.capture_video = False
    env_args.run_name = "dev"
    env_args.fully_observed = False
    env_args.max_steps = 30
    num_envs = 1000

    env_generator = EnvGenerator(env_args)

    envs = env_generator.generate_envs(num_envs)

    assert set([env.env.spec.id for env in envs.envs]) == set(env_args.env_ids)


def test_mission_space_equal_redefinition_wrapper():

    env1 = gym.make("MiniGrid-DoorKey-5x5-v0")
    env1 = MissionSpaceEqualRedefinitionWrapper(env1)

    env2 = gym.make("MiniGrid-LavaCrossingS11N5-v0")
    env2 = MissionSpaceEqualRedefinitionWrapper(env2)
    assert env1.observation_space == env2.observation_space

    env1.reset()
    env2.reset()
    assert env1.observation_space == env2.observation_space
