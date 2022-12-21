
import argparse
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch as t
import torch.nn as nn
import torch.optim as optim
from einops import rearrange
from gym.envs.classic_control.cartpole import CartPoleEnv
from gym.spaces import Discrete
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from .ppo_utils import make_env

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"
RUNNING_FROM_FILE = "ipykernel_launcher" in os.path.basename(sys.argv[0])

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    t.nn.init.orthogonal_(layer.weight, std)
    t.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    critic: nn.Sequential
    actor: nn.Sequential

    def __init__(self, envs: gym.vector.SyncVectorEnv):
        super().__init__()
        self.obs_shape = envs.single_observation_space.shape
        self.num_obs = np.array(self.obs_shape).item()
        self.num_actions = envs.single_action_space.n
        self.critic = nn.Sequential(
            # nn.Flatten(),
            layer_init(nn.Linear(self.num_obs, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0)
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(self.num_obs, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, self.num_actions), std=0.01)
        )

def compute_advantages(
    next_value: t.Tensor,
    next_done: t.Tensor,
    rewards: t.Tensor,
    values: t.Tensor,
    dones: t.Tensor,
    device: t.device,
    gamma: float,
    gae_lambda: float,
) -> t.Tensor:
    """Compute advantages using Generalized Advantage Estimation.
    next_value: shape (1, env) - represents V(s_{t+1}) which is needed for the last advantage term
    next_done: shape (env,)
    rewards: shape (t, env)
    values: shape (t, env)
    dones: shape (t, env)
    Return: shape (t, env)
    """
    "SOLUTION"
    T = values.shape[0]
    next_values = torch.concat([values[1:], next_value])
    next_dones = torch.concat([dones[1:], next_done.unsqueeze(0)])
    deltas = rewards + gamma * next_values * (1.0 - next_dones) - values

    advantages = deltas.clone().to(device)
    for t in reversed(range(1, T)):
        advantages[t-1] = deltas[t-1] + gamma * gae_lambda * (1.0 - dones[t]) * advantages[t]
    return advantages

@dataclass
class Minibatch:
    obs: t.Tensor
    logprobs: t.Tensor
    actions: t.Tensor
    advantages: t.Tensor
    returns: t.Tensor
    values: t.Tensor

def minibatch_indexes(batch_size: int, minibatch_size: int) -> list[np.ndarray]:
    '''Return a list of length (batch_size // minibatch_size) where each element is an array of indexes into the batch.

    Each index should appear exactly once.
    '''
    assert batch_size % minibatch_size == 0
    
    indices = np.random.permutation(batch_size)
    indices = rearrange(indices, "(mb_num mb_size) -> mb_num mb_size", mb_size=minibatch_size)
    return list(indices)

def make_minibatches(
    obs: t.Tensor,
    logprobs: t.Tensor,
    actions: t.Tensor,
    advantages: t.Tensor,
    values: t.Tensor,
    obs_shape: tuple,
    action_shape: tuple,
    batch_size: int,
    minibatch_size: int,
) -> list[Minibatch]:
    '''Flatten the environment and steps dimension into one batch dimension, then shuffle and split into minibatches.'''
    returns = advantages + values

    data = (obs, logprobs, actions, advantages, returns, values)
    shapes = (obs_shape, (), action_shape, (), (), ())
    return [
        Minibatch(*[d.reshape((-1,) + s)[ind] for d, s in zip(data, shapes)])
        for ind in minibatch_indexes(batch_size, minibatch_size)
    ]

def calc_policy_loss(
    probs: Categorical, mb_action: t.Tensor, mb_advantages: t.Tensor, mb_logprobs: t.Tensor, clip_coef: float
) -> t.Tensor:
    '''Return the policy loss, suitable for maximisation with gradient ascent.

    probs: a distribution containing the actor's unnormalized logits of shape (minibatch, num_actions)

    clip_coef: amount of clipping, denoted by epsilon in Eq 7.

    normalize: if true, normalize mb_advantages to have mean 0, variance 1
    '''
    logits_diff = (probs.log_prob(mb_action) - mb_logprobs)

    r_theta = t.exp(logits_diff)

    mb_advantages = (mb_advantages - mb_advantages.mean()) / mb_advantages.std()

    non_clipped = r_theta * mb_advantages
    clipped = t.clip(r_theta, 1-clip_coef, 1+clip_coef) * mb_advantages

    return t.minimum(non_clipped, clipped).mean()

def calc_value_function_loss(critic: nn.Sequential, mb_obs: t.Tensor, mb_returns: t.Tensor, vf_coef: float) -> t.Tensor:
    '''Compute the value function portion of the loss function.

    vf_coef: the coefficient for the value loss, which weights its contribution to the overall loss. Denoted by c_1 in the paper.
    '''
    critic_prediction = critic(mb_obs)

    return 0.5 * vf_coef * (critic_prediction - mb_returns).pow(2).mean()

def calc_entropy_loss(probs: Categorical, ent_coef: float):
    '''Return the entropy loss term.

    ent_coef: the coefficient for the entropy loss, which weights its contribution to the overall loss. Denoted by c_2 in the paper.
    '''
    return ent_coef * probs.entropy().mean()

class PPOScheduler:
    def __init__(self, optimizer, initial_lr: float, end_lr: float, num_updates: int):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.end_lr = end_lr
        self.num_updates = num_updates
        self.n_step_calls = 0

    def step(self):
        '''Implement linear learning rate decay so that after num_updates calls to step, the learning rate is end_lr.'''
        self.n_step_calls += 1
        frac = self.n_step_calls / self.num_updates
        assert frac <= 1
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.initial_lr + frac * (self.end_lr - self.initial_lr)

def make_optimizer(agent: Agent, num_updates: int, initial_lr: float, end_lr: float) -> tuple[optim.Adam, PPOScheduler]:
    '''Return an appropriately configured Adam with its attached scheduler.'''
    optimizer = optim.Adam(agent.parameters(), lr=initial_lr, eps=1e-5, maximize=True)
    scheduler = PPOScheduler(optimizer, initial_lr, end_lr, num_updates)
    return (optimizer, scheduler)

@dataclass
class PPOArgs:
    exp_name: str = os.path.basename(globals().get("__file__", "PPO_implementation").rstrip(".py"))
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = True
    wandb_project_name: str = "PPOCart"
    wandb_entity: str = None
    capture_video: bool = True
    env_id: str = "CartPole-v1"
    total_timesteps: int = 500000
    learning_rate: float = 0.00025
    num_envs: int = 4
    num_steps: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    batch_size: int = 512
    minibatch_size: int = 128

def ppo_parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="CartPole-v1",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=500000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=4,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return PPOArgs(**vars(args))

def train_ppo(args):
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % "\n".join([f"|{key}|{value}|" for (key, value) in vars(args).items()]),
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    action_shape = envs.single_action_space.shape
    assert action_shape is not None
    assert isinstance(envs.single_action_space, Discrete), "only discrete action space is supported"
    agent = Agent(envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    (optimizer, scheduler) = make_optimizer(agent, num_updates, args.learning_rate, 0.0)
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + action_shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    global_step = 0
    old_approx_kl = 0.0
    approx_kl = 0.0
    value_loss = t.tensor(0.0)
    policy_loss = t.tensor(0.0)
    entropy_loss = t.tensor(0.0)
    clipfracs = []
    info = []
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    if RUNNING_FROM_FILE:
        from tqdm import tqdm
        progress_bar = tqdm(range(num_updates))
        range_object = progress_bar
    else:
        range_object = range(num_updates)
    
    for _ in range_object:
        for i in range(0, args.num_steps):

            global_step += args.num_envs

            "(1) YOUR CODE: Rollout phase (see detail #1)"
            obs[i] = next_obs
            dones[i] = next_done
            
            with t.inference_mode():
                next_values = agent.critic(next_obs).flatten()
                logits = agent.actor(next_obs)
            probs = Categorical(logits=logits)
            action = probs.sample()
            logprob = probs.log_prob(action)
            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            
            rewards[i] = t.from_numpy(reward)
            actions[i] = action
            logprobs[i] = logprob
            values[i] = next_values

            next_obs = t.from_numpy(next_obs).to(device)
            next_done = t.from_numpy(done).float().to(device)

            for item in info:
                if "episode" in item.keys():
                    log_string = f"global_step={global_step}, episodic_return={int(item['episode']['r'])}"
                    if RUNNING_FROM_FILE:
                        progress_bar.set_description(log_string)
                    else:
                        print(log_string)
                    writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                    break
        with t.inference_mode():
            next_value = rearrange(agent.critic(next_obs), "env 1 -> 1 env")
        advantages = compute_advantages(
            next_value, next_done, rewards, values, dones, device, args.gamma, args.gae_lambda
        )
        clipfracs.clear()
        for _ in range(args.update_epochs):
            minibatches = make_minibatches(
                obs,
                logprobs,
                actions,
                advantages,
                values,
                envs.single_observation_space.shape,
                action_shape,
                args.batch_size,
                args.minibatch_size,
            )
            for mb in minibatches:

                "(2) YOUR CODE: compute loss on the minibatch and step the optimizer (not the scheduler). Do detail #11 (global gradient clipping) here using nn.utils.clip_grad_norm_."
                logits = agent.actor(mb.obs)
                probs = Categorical(logits=logits)
                policy_loss = calc_policy_loss(probs, mb.actions, mb.advantages, mb.logprobs, args.clip_coef)
                value_function_loss = calc_value_function_loss(agent.critic, mb.obs, mb.returns, args.vf_coef)
                entropy_loss = calc_entropy_loss(probs, args.ent_coef)
                total_loss = policy_loss - value_function_loss + entropy_loss
                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        scheduler.step()
        (y_pred, y_true) = (mb.values.cpu().numpy(), mb.returns.cpu().numpy())
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        with torch.no_grad():
            newlogprob: t.Tensor = probs.log_prob(mb.actions)
            logratio = newlogprob - mb.logprobs
            ratio = logratio.exp()
            old_approx_kl = (-logratio).mean().item()
            approx_kl = (ratio - 1 - logratio).mean().item()
            clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", value_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl, global_step)
        writer.add_scalar("losses/approx_kl", approx_kl, global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        # if global_step % 10 == 0:
        #     print("steps per second (SPS):", int(global_step / (time.time() - start_time)))

    "If running one of the Probe environments, will test if the learned q-values are\n    sensible after training. Useful for debugging."
    obs_for_probes = [[[0.0]], [[-1.0], [+1.0]], [[0.0], [1.0]], [[0.0]], [[0.0], [1.0]]]
    expected_value_for_probes = [[[1.0]], [[-1.0], [+1.0]], [[args.gamma], [1.0]], [[-1.0, 1.0]], [[1.0, -1.0], [-1.0, 1.0]]]
    tolerances = [5e-4, 5e-4, 5e-4, 5e-4, 1e-3]
    match = re.match(r"Probe(\d)-v0", args.env_id)
    if match:
        probe_idx = int(match.group(1)) - 1
        obs = t.tensor(obs_for_probes[probe_idx]).to(device)
        value = agent.critic(obs)
        print("Value: ", value)
        expected_value = t.tensor(expected_value_for_probes[probe_idx]).to(device)
        t.testing.assert_close(value, expected_value, atol=tolerances[probe_idx], rtol=0)

    envs.close()
    writer.close()

if MAIN:
    if RUNNING_FROM_FILE:
        filename = globals().get("__file__", "<filename of this script>")
        print(f"Try running this file from the command line instead:\n\tpython {os.path.basename(filename)} --help")
        args = PPOArgs()
    else:
        args = ppo_parse_args()
    train_ppo(args)


