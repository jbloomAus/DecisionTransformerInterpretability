# %%
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import List

import gymnasium as gym
import numpy as np
import torch as t
from einops import rearrange
from gymnasium.spaces import Discrete
from torch import nn, optim
from torch.distributions.categorical import Categorical
from torchtyping import TensorType as TT
from tqdm import tqdm
from torchtyping import patch_typeguard
from typeguard import typechecked

import wandb

from .my_probe_envs import Probe1, Probe2, Probe3, Probe4, Probe5
from .utils import (PPOArgs, arg_help, make_env, plot_cartpole_obs_and_dones,
                    set_global_seeds, get_obs_preprocessor)

import warnings
warnings.filterwarnings("ignore", category= DeprecationWarning)


device = t.device("cuda" if t.cuda.is_available() else "cpu")

# %%

MAIN = __name__ == "__main__"

if MAIN:
    for i in range(5):
        probes = [Probe1, Probe2, Probe3, Probe4, Probe5]
        gym.envs.registration.register(id=f"Probe{i+1}-v0", entry_point=probes[i])

# patch_typeguard()

# %%
patch_typeguard()


@dataclass
class Minibatch:
    obs: TT["batch", "obs_shape"]
    actions: TT["batch"]
    logprobs: TT["batch"]
    advantages: TT["batch"]
    values: TT["batch"]
    returns: TT["batch"]

class Memory():

    def __init__(self, envs: gym.vector.SyncVectorEnv, args: PPOArgs, device: t.device):
        self.envs = envs
        self.args = args
        self.next_obs = None
        self.next_done = None
        self.next_value = None
        self.device = device
        self.global_step = 0
        self.obs_preprocessor = get_obs_preprocessor(envs.observation_space)
        self.reset()

    def add(self, *data: t.Tensor):
        '''Adds an experience to storage. Called during the rollout phase.
        '''
        info, *experiences = data
        self.experiences.append(experiences)
        if info and isinstance(info, dict):
            if "final_info" in info.keys():

                for item in info["final_info"]:
                    if isinstance(item, dict):
                        if "episode" in item.keys():
                            self.episode_lengths.append(item["episode"]["l"])
                            self.episode_returns.append(item["episode"]["r"])
                            self.add_vars_to_log(
                                episode_length = item["episode"]["l"],
                                episode_return = item["episode"]["r"],
                            )

                    self.global_step += 1 
                
    def sample_experiences(self):
        '''Helper function to print out experiences, as a sanity check!
        '''
        idx = np.random.randint(0, len(self.experiences))
        print(f"Sample {idx+1}/{len(self.experiences)}:")
        for i, n in enumerate(["obs", "done", "action", "logprob", "value", "reward"]):
            print(f"{n:8}: {self.experiences[idx][i].cpu().numpy().tolist()}")

    def get_minibatch_indexes(self, batch_size: int, minibatch_size: int) -> List[np.ndarray]:
        '''Return a list of length (batch_size // minibatch_size) where each element is an array of indexes into the batch.

        Each index should appear exactly once.
        '''
        assert batch_size % minibatch_size == 0
        indices = np.random.permutation(batch_size)
        indices = rearrange(indices, "(mb_num mb_size) -> mb_num mb_size", mb_size=minibatch_size)
        return list(indices)

    def compute_advantages( 
        self,  
        next_value: TT["env"], 
        next_done: TT["env"], 
        rewards: TT["T", "env"], 
        values: TT["T", "env"], 
        dones: TT["T", "env"], 
        device: t.device, 
        gamma: float, 
        gae_lambda: float 
    ) -> TT["T", "env"]:
        '''Compute advantages using Generalized Advantage Estimation.
        '''
        T = values.shape[0]
        next_values = t.concat([values[1:], next_value.unsqueeze(0)])
        next_dones = t.concat([dones[1:], next_done.unsqueeze(0)])
        deltas = rewards + gamma * next_values * (1.0 - next_dones) - values
        advantages = t.zeros_like(deltas).to(device)
        advantages[-1] = deltas[-1]
        for t_ in reversed(range(1, T)):
            advantages[t_-1] = deltas[t_-1] + gamma * gae_lambda * (1.0 - dones[t_]) * advantages[t_]
        return advantages

    def get_minibatches(self) -> List[Minibatch]:
        '''Computes advantages, and returns minibatches to be used in the 
        learning phase.
        '''
        obs, dones, actions, logprobs, values, rewards = [t.stack(arr) for arr in zip(*self.experiences)]
        advantages = self.compute_advantages(self.next_value, self.next_done, rewards, values, dones, self.device, self.args.gamma, self.args.gae_lambda)
        returns = advantages + values
        indexes = self.get_minibatch_indexes(self.args.batch_size, self.args.minibatch_size)
        return [
            Minibatch(*[
                arr.flatten(0, 1)[ind] 
                for arr in [obs, actions, logprobs, advantages, values, returns]
            ])
            for ind in indexes
        ]

    def get_printable_output(self) -> str:
        '''Sets a new progress bar description, if any episodes have terminated. 
        If not, then the bar's description won't change.
        '''
        if self.episode_lengths:
            global_step = self.global_step
            avg_episode_length = np.mean(self.episode_lengths)
            avg_episode_return = np.mean(self.episode_returns)
            return f"{global_step=:<06}\n{avg_episode_length=:<3.0f}\n{avg_episode_return=:<3.0f}"

    def reset(self) -> None:
        '''Function to be called at the end of each rollout period, to make 
        space for new experiences to be generated.
        '''
        self.experiences = []
        self.vars_to_log = defaultdict(dict)
        self.episode_lengths = []
        self.episode_returns = []
        if self.next_obs is None:
            (obs, info) = self.envs.reset()
            obs = self.obs_preprocessor(obs)
            self.next_obs = t.tensor(obs).to(self.device)
            self.next_done = t.zeros(self.envs.num_envs).to(self.device, dtype=t.float)

    def add_vars_to_log(self, **kwargs):
        '''Add variables to storage, for eventual logging (if args.track=True).
        '''
        self.vars_to_log[self.global_step] |= kwargs

    def log(self) -> None:
        '''Logs variables to wandb.
        '''
        for step, vars_to_log in self.vars_to_log.items():
            wandb.log(vars_to_log, step=step)


# %%

class PPOScheduler:
    def __init__(self, optimizer: optim.Optimizer, initial_lr: float, end_lr: float, num_updates: int):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.end_lr = end_lr
        self.num_updates = num_updates
        self.n_step_calls = 0

    def step(self):
        '''Implement linear learning rate decay so that after num_updates calls to step, the learning rate is end_lr.
        '''
        self.n_step_calls += 1
        frac = self.n_step_calls / self.num_updates
        assert frac <= 1
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.initial_lr + frac * (self.end_lr - self.initial_lr)

# %%

class Agent(nn.Module):
    critic: nn.Sequential
    actor: nn.Sequential

    def __init__(self, envs: gym.vector.SyncVectorEnv, device: t.device):
        super().__init__()
        # obs_shape will be a tuple (e.g. for RGB images this would be an array (h, w, c))
        if isinstance(envs.single_observation_space, gym.spaces.Box):
            self.obs_shape = envs.single_observation_space.shape
        elif isinstance(envs.single_observation_space, gym.spaces.Discrete):
            self.obs_shape = (envs.single_observation_space.n,)
        elif isinstance(envs.single_observation_space, gym.spaces.Dict):
            self.obs_shape = envs.single_observation_space.spaces["image"].shape
        else:
            raise ValueError("Unsupported observation space")
        # num_obs is num elements in observations (e.g. for RGB images this would be h * w * c)
        self.num_obs = np.array(self.obs_shape).prod()
        # assuming a discrete action space
        self.num_actions = envs.single_action_space.n

        self.critic = nn.Sequential(
            nn.Flatten(),
            self.layer_init(nn.Linear(self.num_obs, 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, 1), std=1.0)
        )
        self.actor = nn.Sequential(
            nn.Flatten(),
            self.layer_init(nn.Linear(self.num_obs, 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, self.num_actions), std=0.01)
        )

        self.device = device
        self.to(device)

    def layer_init(self, layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
        t.nn.init.orthogonal_(layer.weight, std)
        t.nn.init.constant_(layer.bias, bias_const)
        return layer

    def make_optimizer(self, num_updates: int, initial_lr: float, end_lr: float) -> tuple[optim.Optimizer, PPOScheduler]:
        '''Return an appropriately configured Adam with its attached scheduler.
        '''
        optimizer = optim.Adam(self.parameters(), lr=initial_lr, eps=1e-5, maximize=True)
        scheduler = PPOScheduler(optimizer, initial_lr, end_lr, num_updates)
        return (optimizer, scheduler)

    def rollout(self, memory: Memory, args: PPOArgs, envs: gym.vector.SyncVectorEnv) -> None:
        '''Performs the rollout phase, as described in '37 Implementational Details'.
        '''
        device = memory.device
        obs = memory.next_obs
        done = memory.next_done

        for step in range(args.num_steps):

            # Generate the next set of new experiences (one for each env)
            with t.inference_mode():
                # Our actor generates logits over actions which we can then sample from
                logits = self.actor(obs)
                # Our critic generates a value function (which we use in the value loss, and to estimate advantages)
                value = self.critic(obs).flatten()
            probs = Categorical(logits=logits)
            action = probs.sample()
            logprob = probs.log_prob(action)
            next_obs, reward, next_done, next_truncated, info = envs.step(action.cpu().numpy())
            next_obs = memory.obs_preprocessor(next_obs)
            reward = t.from_numpy(reward).to(device)

            # Store (s_t, d_t, a_t, logpi(a_t|s_t), v(s_t), r_t+1)
            memory.add(info, obs, done, action, logprob, value, reward)

            obs = t.from_numpy(next_obs).to(device)
            done = t.from_numpy(next_done).to(device, dtype=t.float)

        # Store last (obs, done, value) tuple, since we need it to compute advantages
        memory.next_obs = obs
        memory.next_done = done
        with t.inference_mode():
            memory.next_value = self.critic(obs).flatten()

    def learn(self, memory: Memory, args: PPOArgs, optimizer: optim.Optimizer, scheduler: PPOScheduler) -> None:
        '''Performs the learning phase, as described in '37 Implementational Details'.
        '''
        for _ in range(args.update_epochs):
            minibatches = memory.get_minibatches()
            # Compute loss on each minibatch, and step the optimizer
            for mb in minibatches:
                logits = self.actor(mb.obs)
                probs = Categorical(logits=logits)
                values = self.critic(mb.obs).squeeze()
                clipped_surrogate_objective = calc_clipped_surrogate_objective(probs, mb.actions, mb.advantages, mb.logprobs, args.clip_coef)
                value_loss = calc_value_function_loss(values, mb.returns, args.vf_coef)
                entropy_bonus = calc_entropy_bonus(probs, args.ent_coef)
                total_objective_function = clipped_surrogate_objective - value_loss + entropy_bonus
                optimizer.zero_grad()
                total_objective_function.backward()
                nn.utils.clip_grad_norm_(self.parameters(), args.max_grad_norm)
                optimizer.step()

        # Step the scheduler
        scheduler.step()

        # Get debug variables, for just the most recent minibatch (otherwise there's too much logging!)
        if args.track:
            with t.inference_mode():
                newlogprob = probs.log_prob(mb.actions)
                logratio = newlogprob - mb.logprobs
                ratio = logratio.exp()
                approx_kl = (ratio - 1 - logratio).mean().item()
                clipfracs = [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]
            memory.add_vars_to_log(
                learning_rate = optimizer.param_groups[0]["lr"],
                avg_value = values.mean().item(),
                value_loss = value_loss.item(),
                clipped_surrogate_objective = clipped_surrogate_objective.item(),
                entropy = entropy_bonus.item(),
                approx_kl = approx_kl,
                clipfrac = np.mean(clipfracs)
            )


# %%

if MAIN:
    # Code to check that memory experiences are working correctly
    num_envs = 4
    run_name = "test-run"
    envs = gym.vector.SyncVectorEnv(
        [make_env("CartPole-v1", i, i, False, run_name) for i in range(num_envs)]
    )
    args = PPOArgs()
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    memory = Memory(envs, args, device)
    agent = Agent(envs).to(device)
    agent.rollout(memory, args, envs)

    obs = t.stack([e[0] for e in memory.experiences])
    done = t.stack([e[1] for e in memory.experiences])
    plot_cartpole_obs_and_dones(obs, done)

# %%

def calc_clipped_surrogate_objective(
    probs: Categorical, mb_action: t.Tensor, mb_advantages: t.Tensor, mb_logprobs: t.Tensor, clip_coef: float
) -> t.Tensor:
    '''Return the clipped surrogate objective, suitable for maximisation with gradient ascent.

    probs: a distribution containing the actor's unnormalized logits of shape (minibatch, num_actions)
    clip_coef: amount of clipping, denoted by epsilon in Eq 7.
    '''
    logits_diff = probs.log_prob(mb_action) - mb_logprobs

    r_theta = t.exp(logits_diff)

    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 10e-8)

    non_clipped = r_theta * mb_advantages
    clipped = t.clip(r_theta, 1-clip_coef, 1+clip_coef) * mb_advantages

    return t.minimum(non_clipped, clipped).mean()

# %%

@typechecked
def calc_value_function_loss(values: TT["batch"], mb_returns: TT["batch"], vf_coef: float) -> t.Tensor:
    '''Compute the value function portion of the loss function.

    vf_coef: the coefficient for the value loss, which weights its contribution to the overall loss. Denoted by c_1 in the paper.
    '''
    return 0.5 * vf_coef * (values - mb_returns).pow(2).mean()

# %%

def calc_entropy_bonus(probs: Categorical, ent_coef: float):
    '''Return the entropy bonus term, suitable for gradient ascent.

    ent_coef: the coefficient for the entropy loss, which weights its contribution to the overall loss. Denoted by c_2 in the paper.
    '''
    return ent_coef * probs.entropy().mean()


# %%

from IPython.display import display
import ipywidgets as wg

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


def train_ppo(args: PPOArgs):

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
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
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

        agent.rollout(memory, args, envs)
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

    envs.close()
    if args.track:
        wandb.finish()


# %%

if MAIN:
    args = PPOArgs()
    args.track = False
    # args.env_id = "Probe1-v0" # "CartPole-v1"
    arg_help(args)
    train_ppo(args)

