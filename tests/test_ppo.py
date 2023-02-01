import pytest

import torch as t
import gymnasium as gym
from gymnasium.spaces import Discrete

from src.ppo.train import train_ppo
from src.ppo.agent import Agent
from src.ppo.my_probe_envs import  Probe1, Probe2, Probe3, Probe4, Probe5
from src.ppo.utils import PPOArgs
from src.environments import make_env

for i in range(5):
    probes = [Probe1, Probe2, Probe3, Probe4, Probe5]
    gym.envs.registration.register(id=f"Probe{i+1}-v0", entry_point=probes[i])


@pytest.mark.parametrize("env_name", ["Probe1-v0", "Probe2-v0", "Probe3-v0", "Probe4-v0", "Probe5-v0"])
def test_probe_envs(env_name):

    for i in range(5):
        probes = [Probe1, Probe2, Probe3, Probe4, Probe5]
        gym.envs.registration.register(id=f"Probe{i+1}-v0", entry_point=probes[i])

    envs = gym.vector.SyncVectorEnv(
        [make_env(env_name, i, i, False, "test", 
        render_mode=None, max_steps = None, fully_observed = False) for i in range(4)]
    )

    args = PPOArgs(
        exp_name = 'Test',
        env_id = env_name,
        num_envs = 4, # batch size is derived from num environments * minibatch size
        num_minibatches=4,
        num_steps=128,
        track = False,
        capture_video=False,
        cuda = False,
        total_timesteps=10000,
        max_steps=None)

    # currently, ppo has tests which run inside main if it 
    # detects "Probe" in the env name. We will fix this 
    # eventually.
    ppo = train_ppo(args, envs = envs)

def test_empty_env():

    env_name = "MiniGrid-Empty-5x5-v0"

    envs = gym.vector.SyncVectorEnv(
        [make_env(env_name, i, i, False, "test", 
        render_mode=None, max_steps = None, fully_observed = False, flat_one_hot=False) for i in range(4)]
    )

    args = PPOArgs(
        exp_name = 'Test',
        env_id = env_name,
        num_envs = 4, # batch size is derived from num environments * minibatch size
        num_minibatches=4,
        num_steps=128,
        track = False,
        capture_video=False,
        cuda = False,
        total_timesteps=10000,
        max_steps=None)

    # currently, ppo has tests which run inside main if it 
    # detects "Probe" in the env name. We will fix this 
    # eventually.
    ppo = train_ppo(args, envs = envs)

def test_empty_env_flat_one_hot():

    env_name = "MiniGrid-Empty-5x5-v0"

    envs = gym.vector.SyncVectorEnv(
        [make_env(env_name, i, i, False, "test", 
        render_mode=None, max_steps = None, fully_observed = False, flat_one_hot=True) for i in range(4)]
    )

    args = PPOArgs(
        exp_name = 'Test',
        env_id = env_name,
        num_envs = 4, # batch size is derived from num environments * minibatch size
        num_minibatches=4,
        num_steps=128,
        track = False,
        capture_video=False,
        cuda = False,
        total_timesteps=10000,
        max_steps=None)

    # currently, ppo has tests which run inside main if it 
    # detects "Probe" in the env name. We will fix this 
    # eventually.
    ppo = train_ppo(args, envs = envs)

def test_ppo_agent_gym():
    
    envs = gym.vector.SyncVectorEnv(
        [make_env('CartPole-v1', 1, i+1, False, "test", max_steps=None) for i in range(2)]
    )
    assert envs.single_action_space.shape is not None
    assert isinstance(envs.single_action_space, Discrete), "only discrete action space is supported"

    # memory = Memory(envs, args, "cpu")
    agent = Agent(envs, "cpu")

    assert agent.num_obs == 4
    assert agent.num_actions == 2


def test_ppo_agent_minigrid():
    
    envs = gym.vector.SyncVectorEnv(
        [make_env('MiniGrid-Empty-8x8-v0', 1, i+1, False, "test") for i in range(2)]
    )
    assert envs.single_action_space.shape is not None
    assert isinstance(envs.single_action_space, Discrete), "only discrete action space is supported"

    # memory = Memory(envs, args, "cpu")
    agent = Agent(envs, "cpu")

    
    assert agent.num_obs == 7*7*3 # depends on whether you wrapped in Fully observed or not
    assert agent.num_actions == 7
