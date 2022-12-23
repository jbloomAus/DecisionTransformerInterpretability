import pytest

import gymnasium as gym
from gymnasium.spaces import Discrete

from src.ppo.train import train_ppo
from src.ppo.agent import Agent
from src.ppo.my_probe_envs import  Probe1, Probe2, Probe3, Probe4, Probe5
from src.ppo.utils import make_env, PPOArgs

for i in range(5):
    probes = [Probe1, Probe2, Probe3, Probe4, Probe5]
    gym.envs.registration.register(id=f"Probe{i+1}-v0", entry_point=probes[i])


@pytest.mark.parametrize("env_name", ["Probe1-v0", "Probe2-v0", "Probe3-v0", "Probe4-v0", "Probe5-v0"])
def test_probe_envs(env_name):

    for i in range(5):
        probes = [Probe1, Probe2, Probe3, Probe4, Probe5]
        gym.envs.registration.register(id=f"Probe{i+1}-v0", entry_point=probes[i])

    args = PPOArgs(
        exp_name = 'Test',
        env_id = env_name,
        num_envs = 4, # batch size is derived from num environments * minibatch size
        track = False,
        capture_video=False,
        cuda = False,
        total_timesteps=10000,
        max_steps=None)

    # currently, ppo has tests which run inside main if it 
    # detects "Probe" in the env name. We will fix this 
    # eventually.
    ppo = train_ppo(args)


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

    
    assert agent.num_obs == 8*8*3 # depends on whether you wrapped in Fully observed or not
    assert agent.num_actions == 7
