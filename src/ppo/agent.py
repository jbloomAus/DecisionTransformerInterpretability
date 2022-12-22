import gymnasium as gym
import numpy as np
import torch as t
from torch import nn, optim
from torch.distributions.categorical import Categorical
from torchtyping import patch_typeguard
from typeguard import typechecked
from torchtyping import TensorType as TT

from .memory import Memory
from .utils import PPOArgs


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

    def rollout(self, memory: Memory, args: PPOArgs, envs: gym.vector.SyncVectorEnv, trajectory_writer = None) -> None:
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

            if trajectory_writer is not None:
                trajectory_writer.accumulate_trajectory(
                    next_obs = next_obs, # t + 1
                    reward = reward.detach().numpy(), # t + 1
                    done = next_done, # t + 1
                    action = action.detach().numpy(), # t
                    truncated = next_truncated, # t + 1
                    info = info # t
                ) # trajectory is stored as S,A,R not R,S,A! 

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

patch_typeguard()

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

@typechecked
def calc_value_function_loss(values: TT["batch"], mb_returns: TT["batch"], vf_coef: float) -> t.Tensor:
    '''Compute the value function portion of the loss function.

    vf_coef: the coefficient for the value loss, which weights its contribution to the overall loss. Denoted by c_1 in the paper.
    '''
    return 0.5 * vf_coef * (values - mb_returns).pow(2).mean()

def calc_entropy_bonus(probs: Categorical, ent_coef: float):
    '''Return the entropy bonus term, suitable for gradient ascent.

    ent_coef: the coefficient for the entropy loss, which weights its contribution to the overall loss. Denoted by c_2 in the paper.
    '''
    return ent_coef * probs.entropy().mean()

