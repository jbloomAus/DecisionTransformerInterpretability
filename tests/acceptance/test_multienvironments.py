import pytest
import gymnasium as gym
from minigrid.envs import DynamicObstaclesEnv, CrossingEnv, MultiRoomEnv
from src.environments.multienvironments import MultiEnvSampler

# test that the envs are registered


def test_multienv_sampler():
    max_steps = 1000
    render_mode = "rgb_array"
    envs = []
    env = DynamicObstaclesEnv(
        size=6,
        n_obstacles=0,
        agent_start_pos=None,
        render_mode="rgb_array",
        max_steps=max_steps,
    )
    envs.append(env)
    for size in range(6, 10):
        for num_obstacles in range(5, 7):
            env = DynamicObstaclesEnv(
                size=size,
                n_obstacles=num_obstacles,
                agent_start_pos=None,
                render_mode=render_mode,
                max_steps=max_steps,
            )
            envs.append(env)

    multi_env = MultiEnvSampler(envs)
    assert multi_env.n_envs == len(envs)
    # assert obs space is the same
    assert multi_env.observation_space == envs[0].observation_space
    # assert action space is the same
    assert multi_env.action_space == envs[0].action_space
    # assert reward range is the same
    # assert equal probs
    assert min(multi_env.p) == 1 / len(envs)
    assert max(multi_env.p) == 1 / len(envs)
