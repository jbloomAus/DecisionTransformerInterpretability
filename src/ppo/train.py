
import re
import time

import gymnasium as gym
import ipywidgets as wg
import torch as t
from gymnasium.spaces import Discrete
from IPython.display import display
from tqdm import tqdm

import wandb

from .agent import Agent
from .memory import Memory
from .utils import PPOArgs, make_env, set_global_seeds

device = t.device("cuda" if t.cuda.is_available() else "cpu")

def get_printable_output_for_probe_envs(args: PPOArgs, agent: Agent, probe_idx: int, update: int, num_updates: int):
    """Tests a probe environment, by printing output in the form of a widget.
    We should see rapid convergence in both actions and observations.
    """
    obs_for_probes = [[[0.0]], [[-1.0], [+1.0]], [[0.0], [1.0]], [[0.0]], [[0.0], [1.0]]]
    expected_value_for_probes = [1.0, [-1.0, +1.0], [args.gamma, 1.0], 1.0, [1.0, 1.0]]
    expected_actions_for_probs = [None, None, None, 1, [0, 1]]
    
    obs = t.tensor(obs_for_probes[probe_idx]).to(device)
    output = ""

    # Check if the value is what you expect
    value = agent.critic(obs).detach().cpu().numpy().squeeze()
    expected_value = expected_value_for_probes[probe_idx]
    output += f"Obs: {update+1}/{num_updates}\n\nActual value: {value}\nExpected value: {expected_value}"
    # Check if the action is what you expect
    expected_action = expected_actions_for_probs[probe_idx]
    if expected_action is not None:
        logits = agent.actor(obs)
        probs = logits.softmax(-1).detach().cpu().numpy().squeeze()
        probs = str(probs).replace('\n', '')
        output += f"\n\nActual prob: {probs}\nExpected action: {expected_action}"

    return output

def train_ppo(args: PPOArgs, trajectory_writer = None):

    # Check if running one of the probe envs
    probe_match = re.match(r"Probe(\d)-v0", args.env_id)
    probe_idx = int(probe_match.group(1)) - 1 if probe_match else None

    # Verify environment is registered
    all_envs = [env_spec for env_spec in gym.envs.registry]
    assert args.env_id in all_envs, f"Environment {args.env_id} not registered."

    # wandb initialisation, 
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args), # vars is equivalent to args.__dict__
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    set_global_seeds(args.seed)
    device = t.device("cuda" if t.cuda.is_available() and args.cuda else "cpu")

    if args.env_id in ["Probe1-v0", "Probe2-v0", "Probe3-v0", "Probe4-v0", "Probe5-v0"]:
        envs = gym.vector.SyncVectorEnv(
            [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name, render_mode=None, max_steps = args.max_steps) for i in range(args.num_envs)]
        )
    else:
        envs = gym.vector.SyncVectorEnv(
            [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name, max_steps = args.max_steps) for i in range(args.num_envs)]
        )
    assert envs.single_action_space.shape is not None
    assert isinstance(envs.single_action_space, Discrete), "only discrete action space is supported"

    memory = Memory(envs, args, device)
    agent = Agent(envs, device)
    num_updates = args.total_timesteps // args.batch_size
    optimizer, scheduler = agent.make_optimizer(num_updates, initial_lr=args.learning_rate, end_lr=0.0)
    
    out = wg.Output(layout={"padding": "15px"})
    display(out)
    progress_bar = tqdm(range(num_updates))
    for update in progress_bar:

        agent.rollout(memory, args, envs, trajectory_writer)
        agent.learn(memory, args, optimizer, scheduler)
        
        if args.track:
            memory.log()

        # Print output (different behaviour for probe envs vs normal envs)
        if probe_idx is None:
            output = memory.get_printable_output()
        else:
            output = get_printable_output_for_probe_envs(args, agent, probe_idx, update, num_updates)
        if output:
            with out:
                print(output)
                out.clear_output(wait=True)
            
        memory.reset()

    if trajectory_writer is not None:
        trajectory_writer.write()

    envs.close()
    if args.track:
        wandb.finish()
