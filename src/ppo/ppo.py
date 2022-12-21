
import warnings
import gymnasium as gym
import torch as t

from .agent import Agent
from .my_probe_envs import Probe1, Probe2, Probe3, Probe4, Probe5
from .utils import PPOArgs, arg_help, make_env, set_global_seeds
from .train import train_ppo

warnings.filterwarnings("ignore", category= DeprecationWarning)
device = t.device("cuda" if t.cuda.is_available() else "cpu")

if __name__ == "__main__":
    args = PPOArgs()
    args.track = False

    for i in range(5):
        probes = [Probe1, Probe2, Probe3, Probe4, Probe5]
        gym.envs.registration.register(id=f"Probe{i+1}-v0", entry_point=probes[i])

    arg_help(args)
    train_ppo(args)

