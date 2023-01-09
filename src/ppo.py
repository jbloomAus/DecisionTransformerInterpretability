
import warnings
import gymnasium as gym
import torch as t
import re 
import wandb
import time

from ppo.my_probe_envs import Probe1, Probe2, Probe3, Probe4, Probe5
from ppo.utils import PPOArgs, arg_help, set_global_seeds
from ppo.train import train_ppo
from utils import TrajectoryWriter
from environments import make_env
from gymnasium.spaces import Discrete

warnings.filterwarnings("ignore", category= DeprecationWarning)
device = t.device("cuda" if t.cuda.is_available() else "cpu")

if __name__ == "__main__":
    args = PPOArgs()

    for i in range(5):
        probes = [Probe1, Probe2, Probe3, Probe4, Probe5]
        gym.envs.registration.register(id=f"Probe{i+1}-v0", entry_point=probes[i])

    if args.trajectory_path:
        trajectory_writer = TrajectoryWriter(args.trajectory_path, args)
    else:
        trajectory_writer = None

    arg_help(args)

    # Check if running one of the probe envs
    probe_match = re.match(r"Probe(\d)-v0", args.env_id)
    probe_idx = int(probe_match.group(1)) - 1 if probe_match else None

    # Verify environment is registered
    all_envs = [env_spec for env_spec in gym.envs.registry]
    assert args.env_id in all_envs, f"Environment {args.env_id} not registered."

    # wandb initialisation, 
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args), # vars is equivalent to args.__dict__
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    # make envs
    set_global_seeds(args.seed)
    device = t.device("cuda" if t.cuda.is_available() and args.cuda else "cpu")

    if args.env_id in ["Probe1-v0", "Probe2-v0", "Probe3-v0", "Probe4-v0", "Probe5-v0"]:
        envs = gym.vector.SyncVectorEnv(
            [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name, 
            render_mode=None, max_steps = args.max_steps, fully_observed = args.fully_observed) for i in range(args.num_envs)]
        )
    else:
        envs = gym.vector.SyncVectorEnv(
            [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name,
            max_steps = args.max_steps, fully_observed = args.fully_observed) for i in range(args.num_envs)]
        )
    assert envs.single_action_space.shape is not None
    assert isinstance(envs.single_action_space, Discrete), "only discrete action space is supported"

    train_ppo(
        args, 
        envs,
        trajectory_writer=trajectory_writer, 
        probe_idx=probe_idx
    )
    if args.track:
        run.finish()

