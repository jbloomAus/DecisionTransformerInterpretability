import gymnasium as gym
import pytest

from src.environments.registration import register_envs


# test that the envs are registered
@pytest.mark.parametrize(
    "env_id",
    [
        "DynamicObstaclesMultiEnv-v0",
        "CrossingMultiEnv-v0",
        "MultiRoomMultiEnv-v0",
    ],
)
def test_register_envs(env_id):
    register_envs()

    env_ids = gym.envs.registry.keys()
    assert env_id in env_ids
