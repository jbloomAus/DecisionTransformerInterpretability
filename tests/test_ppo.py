import pytest

import gymnasium as gym

from src.ppo.ppo import train_ppo, PPOArgs
from src.ppo.my_probe_envs import  Probe1, Probe2, Probe3, Probe4, Probe5

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
        total_timesteps=10000)

    # currently, ppo has tests which run inside main if it 
    # detects "Probe" in the env name. We will fix this 
    # eventually.
    ppo = train_ppo(args)
    