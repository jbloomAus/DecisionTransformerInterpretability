import pytest
import numpy as np
import gymnasium as gym
from src.ppo.utils import get_obs_preprocessor
from src.ppo.my_probe_envs import Probe1
from src.environments import make_env
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper, FlatObsWrapper, FullyObsWrapper


def test_get_obs_preprocessor_probe1():

    gym.envs.registration.register(id=f"Probe1-v0", entry_point=Probe1)

    env = make_env(
        env_ids='Probe1',
        seed=0,
        idx=0,
        capture_video=False,
        run_name="Test",
        render_mode=None, max_steps=None)
    env = env()
    (obs, info) = env.reset()  # has shape (obs, info), where obs is an ordered dict
    obs.shape
    assert isinstance(get_obs_preprocessor(
        env.observation_space)(obs), np.ndarray)
    assert (get_obs_preprocessor(env.observation_space)
            (obs)).dtype == np.dtype('float32')


def test_get_obs_preprocessor_MiniGrid():

    env = make_env(
        env_ids='MiniGrid-DoorKey-5x5-v0',
        seed=0,
        idx=0,
        capture_video=False,
        run_name="Test")
    env = env()
    (obs, info) = env.reset()  # has shape (obs, info), where obs is an ordered dict
    obs['image'].shape
    assert isinstance(get_obs_preprocessor(
        env.observation_space)(obs), np.ndarray)
    assert (get_obs_preprocessor(env.observation_space)
            (obs)).dtype == np.dtype('float32')


def test_get_obs_preprocessor_MiniGrid_flat():

    env = make_env(
        env_ids='MiniGrid-DoorKey-5x5-v0',
        seed=0,
        idx=0,
        capture_video=False,
        run_name="Test")
    env = FlatObsWrapper(env())
    (obs, info) = env.reset()  # has shape (obs, info), where obs is an ordered dict
    print(obs.shape)
    assert isinstance(get_obs_preprocessor(
        env.observation_space)(obs), np.ndarray)
    assert (get_obs_preprocessor(env.observation_space)
            (obs)).dtype == np.dtype('float32')


def test_get_obs_preprocessor_MiniGrid_full_obs():

    env = make_env(
        env_ids='MiniGrid-DoorKey-5x5-v0',
        seed=0,
        idx=0,
        capture_video=False,
        run_name="Test")
    env = FullyObsWrapper(env())
    (obs, info) = env.reset()  # has shape (obs, info), where obs is an ordered dict
    assert isinstance(get_obs_preprocessor(
        env.observation_space)(obs), np.ndarray)
    assert (get_obs_preprocessor(env.observation_space)
            (obs)).dtype == np.dtype('float32')


def test_get_obs_preprocessor_MiniGrid_full_img_obs():

    env = make_env(
        env_ids='MiniGrid-DoorKey-5x5-v0',
        seed=0,
        idx=0,
        capture_video=False,
        run_name="Test")
    env = ImgObsWrapper(env())
    (obs, info) = env.reset()  # has shape (obs, info), where obs is an ordered dict
    assert isinstance(get_obs_preprocessor(
        env.observation_space)(obs), np.ndarray)
    assert get_obs_preprocessor(env.observation_space)(
        obs).dtype == np.dtype('float32')


def test_get_obs_preprocessor_MiniGrid_full_img_partial():

    env = make_env(
        env_ids='MiniGrid-DoorKey-5x5-v0',
        seed=0,
        idx=0,
        capture_video=False,
        run_name="Test")
    env = RGBImgPartialObsWrapper(env())
    (obs, info) = env.reset()  # has shape (obs, info), where obs is an ordered dict
    assert isinstance(get_obs_preprocessor(
        env.observation_space)(obs), np.ndarray)
    assert (get_obs_preprocessor(env.observation_space)
            (obs)).dtype == np.dtype('float32')
