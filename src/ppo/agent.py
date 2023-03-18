import abc
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch as t
from torch import nn, optim
from torch.distributions.categorical import Categorical
from torchtyping import TensorType as TT
from torchtyping import patch_typeguard
from typeguard import typechecked

from .memory import Memory

from src.models.trajectory_model import ActorTransformer
from src.config import TransformerModelConfig, EnvironmentConfig, OnlineTrainConfig


class PPOScheduler:
    def __init__(self, optimizer: optim.Optimizer, initial_lr: float, end_lr: float, num_updates: int):
        '''
        A learning rate scheduler for a Proximal Policy Optimization (PPO) algorithm.

        Args:
        - optimizer (optim.Optimizer): the optimizer to use for updating the learning rate.
        - initial_lr (float): the initial learning rate.
        - end_lr (float): the end learning rate.
        - num_updates (int): the number of updates to perform before the learning rate reaches end_lr.
        '''
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.end_lr = end_lr
        self.num_updates = num_updates
        self.n_step_calls = 0

    def step(self):
        '''
        Implement linear learning rate decay so that after num_updates calls to step, the learning rate is end_lr.
        '''
        self.n_step_calls += 1
        frac = self.n_step_calls / self.num_updates
        assert frac <= 1
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.initial_lr + \
                frac * (self.end_lr - self.initial_lr)


class PPOAgent(nn.Module):
    critic: nn.Module
    actor: nn.Module

    @abc.abstractmethod
    def __init__(self, envs: gym.vector.SyncVectorEnv, device):
        super().__init__()
        self.envs = envs
        self.device = device

        self.critic = nn.Sequential()
        self.actor = nn.Sequential()

    @abc.abstractmethod
    def layer_init(self, layer: nn.Linear, std: float, bias_const: float) -> nn.Linear:
        pass

    def make_optimizer(self,
                       num_updates: int,
                       initial_lr: float,
                       end_lr: float) -> Tuple[optim.Optimizer, PPOScheduler]:
        """Returns an Adam optimizer with a learning rate schedule for updating the agent's parameters.

        Args:
            num_updates (int): The total number of updates to be performed.
            initial_lr (float): The initial learning rate.
            end_lr (float): The final learning rate.

        Returns:
            Tuple[optim.Optimizer, PPOScheduler]: A tuple containing the optimizer and its attached scheduler.
        """
        optimizer = optim.Adam(
            self.parameters(), lr=initial_lr, eps=1e-5, maximize=True)
        scheduler = PPOScheduler(optimizer, initial_lr, end_lr, num_updates)
        return (optimizer, scheduler)

    @abc.abstractmethod
    def rollout(self, memory, args, envs, trajectory_writer) -> None:
        pass

    @abc.abstractmethod
    def learn(self, memory, args, optimizer, scheduler) -> None:
        pass


class FCAgent(PPOAgent):
    critic: nn.Sequential
    actor: nn.Sequential

    def __init__(self, envs: gym.vector.SyncVectorEnv, device=t.device, hidden_dim: int = 64):
        '''
        An agent for a Proximal Policy Optimization (PPO) algorithm.

        Args:
        - envs (gym.vector.SyncVectorEnv): the environment(s) to interact with.
        - device (t.device): the device on which to run the agent.
        - hidden_dim (int): the number of neurons in the hidden layer.
        '''
        super().__init__(envs=envs, device=device)

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
        self.hidden_dim = hidden_dim

        self.critic = nn.Sequential(
            nn.Flatten(),
            self.layer_init(nn.Linear(self.num_obs, self.hidden_dim)),
            nn.Tanh(),
            self.layer_init(nn.Linear(self.hidden_dim, self.hidden_dim)),
            nn.Tanh(),
            self.layer_init(nn.Linear(self.hidden_dim, 1), std=1.0)
        )
        self.actor = nn.Sequential(
            nn.Flatten(),
            self.layer_init(nn.Linear(self.num_obs, self.hidden_dim)),
            nn.Tanh(),
            self.layer_init(nn.Linear(self.hidden_dim, self.hidden_dim)),
            nn.Tanh(),
            self.layer_init(nn.Linear(self.hidden_dim,
                                      self.num_actions), std=0.01)
        )

        self.device = device
        self.to(device)

    # TODO work out why this is std not gain for orthogonal init
    def layer_init(self, layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
        """Initializes the weights of a linear layer with orthogonal
        initialization and the biases with a constant value.

        Args:
            layer (nn.Linear): The linear layer to be initialized.
            std (float, optional): The standard deviation of the
                distribution used to initialize the weights. Defaults to np.sqrt(2).
            bias_const (float, optional): The constant value to initialize the biases with. Defaults to 0.0.

        Returns:
            nn.Linear: The initialized linear layer.
        """
        t.nn.init.orthogonal_(layer.weight, std)
        t.nn.init.constant_(layer.bias, bias_const)
        return layer

    def rollout(self, memory: Memory, num_steps: int, envs: gym.vector.SyncVectorEnv, trajectory_writer=None) -> None:
        """Performs the rollout phase of the PPO algorithm, collecting experience by interacting with the environment.

        Args:
            memory (Memory): The replay buffer to store the experiences.
            num_steps (int): The number of steps to collect.
            envs (gym.vector.SyncVectorEnv): The vectorized environment to interact with.
            trajectory_writer (TrajectoryWriter, optional): The writer to log the
                collected trajectories. Defaults to None.
        """

        device = memory.device
        if isinstance(device, str):
            device = t.device(device)
        cuda = device.type == "cuda"
        obs = memory.next_obs
        done = memory.next_done

        for step in range(num_steps):

            # Generate the next set of new experiences (one for each env)
            with t.inference_mode():
                # Our actor generates logits over actions which we can then sample from
                logits = self.actor(obs)
                # Our critic generates a value function (which we use in the value loss, and to estimate advantages)
                value = self.critic(obs).flatten()
            probs = Categorical(logits=logits)
            action = probs.sample()
            logprob = probs.log_prob(action)
            next_obs, reward, next_done, next_truncated, info = envs.step(
                action.cpu().numpy())
            next_obs = memory.obs_preprocessor(next_obs)
            reward = t.from_numpy(reward).to(device)

            # TODO refactor to use ternary statements
            if trajectory_writer is not None:
                # first_obs = obs
                if not cuda:
                    trajectory_writer.accumulate_trajectory(
                        # the observation stored with an action and reward is
                        # the observation which the agent responded to.
                        next_obs=obs.detach().numpy(),
                        # the reward stored with an action and observation is
                        # the reward the agent received for taking that action in that state
                        reward=reward.detach().numpy(),
                        # the action stored with an observation and reward is
                        # the action the agent took to get to that reward
                        action=action.detach().numpy(),
                        # the done stored with an action and observation is
                        # the done the agent received for taking that action in that state
                        done=next_done,
                        truncated=next_truncated,
                        info=info
                    )
                else:
                    trajectory_writer.accumulate_trajectory(
                        # the observation stored with an action and reward
                        # is the observation which the agent responded to.
                        next_obs=obs.detach().cpu().numpy(),
                        # the reward stored with an action and observation
                        # is the reward the agent received for taking that action in that state
                        reward=reward.detach().cpu().numpy(),
                        # the action stored with an observation and reward
                        # is the action the agent took to get to that reward
                        action=action.detach().cpu().numpy(),
                        # the done stored with an action and observation
                        # is the done the agent received for taking that action in that state
                        done=next_done,
                        truncated=next_truncated,
                        info=info
                    )

            # Store (s_t, d_t, a_t, logpi(a_t|s_t), v(s_t), r_t+1)
            memory.add(info, obs, done, action, logprob, value, reward)
            obs = t.from_numpy(next_obs).to(device)
            done = t.from_numpy(next_done).to(device, dtype=t.float)

        # Store last (obs, done, value) tuple, since we need it to compute advantages
        memory.next_obs = obs
        memory.next_done = done
        with t.inference_mode():
            memory.next_value = self.critic(obs).flatten()

    def learn(self,
              memory: Memory,
              args: OnlineTrainConfig,
              optimizer: optim.Optimizer,
              scheduler: PPOScheduler,
              track: bool) -> None:
        """Performs the learning phase of the PPO algorithm, updating the agent's parameters
        using the collected experience.

        Args:
            memory (Memory): The replay buffer containing the collected experiences.
            args (OnlineTrainConfig): The configuration for the training.
            optimizer (optim.Optimizer): The optimizer to update the agent's parameters.
            scheduler (PPOScheduler): The scheduler attached to the optimizer.
            track (bool): Whether to track the training progress.
        """
        for _ in range(args.update_epochs):
            minibatches = memory.get_minibatches()
            # Compute loss on each minibatch, and step the optimizer
            for mb in minibatches:
                logits = self.actor(mb.obs)
                probs = Categorical(logits=logits)
                values = self.critic(mb.obs).squeeze()
                clipped_surrogate_objective = calc_clipped_surrogate_objective(
                    probs, mb.actions, mb.advantages, mb.logprobs, args.clip_coef)
                value_loss = calc_value_function_loss(
                    values, mb.returns, args.vf_coef)
                entropy_bonus = calc_entropy_bonus(probs, args.ent_coef)
                total_objective_function = clipped_surrogate_objective - value_loss + entropy_bonus
                optimizer.zero_grad()
                total_objective_function.backward()
                nn.utils.clip_grad_norm_(self.parameters(), args.max_grad_norm)
                optimizer.step()

        # Step the scheduler
        scheduler.step()

        # Get debug variables, for just the most recent minibatch (otherwise there's too much logging!)
        if track:
            with t.inference_mode():
                newlogprob = probs.log_prob(mb.actions)
                logratio = newlogprob - mb.logprobs
                ratio = logratio.exp()
                approx_kl = (ratio - 1 - logratio).mean().item()
                clipfracs = [
                    ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]
            memory.add_vars_to_log(
                learning_rate=optimizer.param_groups[0]["lr"],
                avg_value=values.mean().item(),
                value_loss=value_loss.item(),
                clipped_surrogate_objective=clipped_surrogate_objective.item(),
                entropy=entropy_bonus.item(),
                approx_kl=approx_kl,
                clipfrac=np.mean(clipfracs)
            )


class TrajPPOAgent(PPOAgent):
    def __init__(self,
                 envs: gym.vector.SyncVectorEnv,
                 environment_config: EnvironmentConfig,
                 transformer_model_config: TransformerModelConfig,
                 device=t.device
                 ):
        '''
        An agent for a Proximal Policy Optimization (PPO) algorithm.

        Args:
        - envs (gym.vector.SyncVectorEnv): the environment(s) to interact with.
        - device (t.device): the device on which to run the agent.
        - environment_config (EnvironmentConfig): the configuration for the environment.
        - transformer_model_config (TransformerModelConfig): the configuration for the transformer model.
        - device (t.device): the device on which to run the agent.
        '''
        super().__init__(envs=envs, device=device)
        self.environment_config = environment_config
        self.transformer_model_config = transformer_model_config

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

        self.hidden_dim = transformer_model_config.d_model
        self.critic = nn.Sequential(
            nn.Flatten(),
            self.layer_init(nn.Linear(self.num_obs, self.hidden_dim)),
            nn.Tanh(),
            self.layer_init(nn.Linear(self.hidden_dim, self.hidden_dim)),
            nn.Tanh(),
            self.layer_init(nn.Linear(self.hidden_dim, 1), std=1.0)
        )

        actor_transformer = ActorTransformer(
            transformer_config=transformer_model_config,
            environment_config=environment_config,
        )

        self.layer_init(actor_transformer.action_predictor, std=0.01)
        self.actor = actor_transformer

        self.device = device
        self.to(device)

    # TODO work out why this is std not gain for orthogonal init
    def layer_init(self, layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
        """Initializes the weights of a linear layer with orthogonal
        initialization and the biases with a constant value.

        Args:
            layer (nn.Linear): The linear layer to be initialized.
            std (float, optional): The standard deviation of the distribution used to
                initialize the weights. Defaults to np.sqrt(2).
            bias_const (float, optional): The constant value to initialize the biases with. Defaults to 0.0.

        Returns:
            nn.Linear: The initialized linear layer.
        """
        t.nn.init.orthogonal_(layer.weight, std)
        t.nn.init.constant_(layer.bias, bias_const)
        return layer

    def rollout(self,
                memory: Memory,
                num_steps: int,
                envs: gym.vector.SyncVectorEnv,
                trajectory_writer=None) -> None:
        """Performs the rollout phase of the PPO algorithm, collecting experience by interacting with the environment.

        Args:
            memory (Memory): The replay buffer to store the experiences.
            num_steps (int): The number of steps to collect.
            envs (gym.vector.SyncVectorEnv): The vectorized environment to interact with.
            trajectory_writer (TrajectoryWriter, optional): The writer to
                log the collected trajectories. Defaults to None.
        """

        device = memory.device
        obs = memory.next_obs
        done = memory.next_done
        context_window_size = self.actor.transformer_config.n_ctx
        n_envs = envs.num_envs
        if isinstance(device, str):
            device = t.device(device)
        cuda = device.type == "cuda"

        for step in range(num_steps):

            if len(memory.experiences) == 0:
                with t.inference_mode():
                    logits = self.actor.forward(
                        states=obs.unsqueeze(1),
                        actions=None,
                        timesteps=t.tensor([0]).repeat(n_envs, 1, 1)
                    )
                    value = self.critic(obs).flatten()
            else:

                # if no experience,s you have one timestep and you buffer out to the context size.
                # if you have some experiences, than you fill out at much of the context window as you can.
                # if you have more experiences than the context window, you have to truncate.

                # we have one more obs than action
                obs_timesteps = (context_window_size - 1) // 2 + \
                    1  # (the current obs)
                actions_timesteps = obs_timesteps - 1

                obss = memory.get_obs_traj(
                    steps=step,
                    pad_to_length=obs_timesteps
                )

                obss = t.cat((obss[:, 1:], obs.unsqueeze(1)), dim=1)

                if actions_timesteps == 0:
                    actions = None
                else:
                    actions = memory.get_act_traj(
                        steps=step - 1,
                        pad_to_length=actions_timesteps
                    ).to(dtype=t.long)

                timesteps = memory.get_timestep_traj(
                    steps=step,
                    pad_to_length=obs_timesteps,
                )

                # Generate the next set of new experiences (one for each env)
                with t.inference_mode():
                    # Our actor generates logits over actions which we can then sample from
                    logits = self.actor.forward(
                        states=obss,
                        actions=actions.unsqueeze(
                            -1) if actions is not None else None,
                        timesteps=t.tensor([0]).repeat(
                            n_envs, obss.shape[1], 1).to(int)
                    )
                    # Our critic generates a value function (which we use in the value loss, and to estimate advantages)
                    value = self.critic(obs).flatten()

            # get the last state action prediction
            probs = Categorical(logits=logits[:, -1])
            action = probs.sample()
            logprob = probs.log_prob(action)
            next_obs, reward, next_done, next_truncated, info = envs.step(
                action.cpu().numpy())
            next_obs = memory.obs_preprocessor(next_obs)
            reward = t.from_numpy(reward).to(device)

            # TODO refactor to use ternary statements
            if trajectory_writer is not None:
                # first_obs = obs
                if not cuda:
                    trajectory_writer.accumulate_trajectory(
                        # the observation stored with an action and reward is
                        # the observation which the agent responded to.
                        next_obs=obs.detach().numpy(),
                        # the reward stored with an action and observation is
                        # the reward the agent received for taking that action in that state
                        reward=reward.detach().numpy(),
                        # the action stored with an observation and reward is
                        # the action the agent took to get to that reward
                        action=action.detach().numpy(),
                        # the done stored with an action and observation is
                        # the done the agent received for taking that action in that state
                        done=next_done,
                        truncated=next_truncated,
                        info=info
                    )
                else:
                    trajectory_writer.accumulate_trajectory(
                        # the observation stored with an action and reward is
                        # the observation which the agent responded to.
                        next_obs=obs.detach().cpu().numpy(),
                        # the reward stored with an action and observation is
                        # the reward the agent received for taking that action in that state
                        reward=reward.detach().cpu().numpy(),
                        # the action stored with an observation and reward
                        # is the action the agent took to get to that reward
                        action=action.detach().cpu().numpy(),
                        # the done stored with an action and observation is
                        # the done the agent received for taking that action in that state
                        done=next_done,
                        truncated=next_truncated,
                        info=info
                    )

            # Store (s_t, d_t, a_t, logpi(a_t|s_t), v(s_t), r_t+1)
            memory.add(info, obs, done, action, logprob, value, reward)
            obs = t.from_numpy(next_obs).to(device)
            done = t.from_numpy(next_done).to(device, dtype=t.float)

        # Store last (obs, done, value) tuple, since we need it to compute advantages
        memory.next_obs = obs
        memory.next_done = done
        with t.inference_mode():
            memory.next_value = self.critic(obs).flatten()

    def learn(self,
              memory: Memory,
              args: OnlineTrainConfig,
              optimizer: optim.Optimizer,
              scheduler: PPOScheduler,
              track: bool) -> None:
        """Performs the learning phase of the PPO algorithm, updating the agent's parameters
        using the collected experience.

        Args:
            memory (Memory): The replay buffer containing the collected experiences.
            args (OnlineTrainConfig): The configuration for the training.
            optimizer (optim.Optimizer): The optimizer to update the agent's parameters.
            scheduler (PPOScheduler): The scheduler attached to the optimizer.
            track (bool): Whether to track the training progress.
        """
        for _ in range(args.update_epochs):
            minibatches = memory.get_trajectory_minibatches(
                (self.actor.transformer_config.n_ctx - 1) // 2 +
                1,  # this is max timesteps
                prob_go_from_end=args.prob_go_from_end,
            )
            # Compute loss on each minibatch, and step the optimizer
            for mb in minibatches:

                logits = self.actor(
                    states=mb.obs,
                    # these should be the previous actions
                    actions=mb.actions[:, :- \
                                       1].unsqueeze(-1).to(int) if mb.actions.shape[1] > 1 else None,
                    timesteps=t.tensor([0]).repeat(
                        mb.obs.shape[0], mb.obs.shape[1], 1).to(int)
                    # t.zeros_like(mb.obs).to(int)  # mb.timesteps[:, :-1] / mb.timesteps.max()
                )

                # squeeze sequence dimension
                probs = Categorical(logits=logits[:, -1])
                # critic is a DNN so let's the last state obs at each time step.
                values = self.critic(mb.obs[:, -1]).squeeze()
                clipped_surrogate_objective = calc_clipped_surrogate_objective(
                    probs=probs,
                    # these should be the current actions
                    mb_action=mb.actions[:, -1].squeeze(-1),
                    mb_advantages=mb.advantages,
                    mb_logprobs=mb.logprobs,
                    clip_coef=args.clip_coef)

                value_loss = calc_value_function_loss(
                    values, mb.returns, args.vf_coef)
                entropy_bonus = calc_entropy_bonus(probs, args.ent_coef)
                total_objective_function = clipped_surrogate_objective - value_loss + entropy_bonus
                optimizer.zero_grad()
                total_objective_function.backward()
                nn.utils.clip_grad_norm_(self.parameters(), args.max_grad_norm)
                optimizer.step()

        # Step the scheduler
        scheduler.step()

        # Get debug variables, for just the most recent minibatch (otherwise there's too much logging!)
        if track:
            with t.inference_mode():
                newlogprob = probs.log_prob(mb.actions.unsqueeze(-1))
                logratio = newlogprob - mb.logprobs
                ratio = logratio.exp()
                approx_kl = (ratio - 1 - logratio).mean().item()
                clipfracs = [
                    ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]
            memory.add_vars_to_log(
                learning_rate=optimizer.param_groups[0]["lr"],
                avg_value=values.mean().item(),
                value_loss=value_loss.item(),
                clipped_surrogate_objective=clipped_surrogate_objective.item(),
                entropy=entropy_bonus.item(),
                approx_kl=approx_kl,
                clipfrac=np.mean(clipfracs)
            )


patch_typeguard()


def calc_clipped_surrogate_objective(
    probs: Categorical, mb_action: t.Tensor, mb_advantages: t.Tensor, mb_logprobs: t.Tensor, clip_coef: float
) -> t.Tensor:
    '''
    Return the clipped surrogate objective, suitable for maximisation with gradient ascent.

    Args:
        probs (Categorical): A distribution containing the actor's
            unnormalized logits of shape (minibatch, num_actions).
        mb_action (Tensor): A tensor of shape (minibatch,) containing the actions taken by the agent in the minibatch.
        mb_advantages (Tensor): A tensor of shape (minibatch,) containing the
            advantages estimated for each state in the minibatch.
        mb_logprobs (Tensor): A tensor of shape (minibatch,) containing the
            log probabilities of the actions taken by the agent in the minibatch.
        clip_coef (float): Amount of clipping, denoted by epsilon in Eq 7.

    Returns:
        Tensor: The clipped surrogate objective computed over the minibatch, with shape ().

    '''
    logits_diff = probs.log_prob(mb_action) - mb_logprobs

    r_theta = t.exp(logits_diff)

    mb_advantages = (mb_advantages - mb_advantages.mean()) / \
        (mb_advantages.std() + 10e-8)

    non_clipped = r_theta * mb_advantages
    clipped = t.clip(r_theta, 1 - clip_coef, 1 + clip_coef) * mb_advantages

    return t.minimum(non_clipped, clipped).mean()


@typechecked
def calc_value_function_loss(values: TT["batch"], mb_returns: TT["batch"], vf_coef: float) -> t.Tensor:  # noqa: F821
    '''
    Compute the value function portion of the loss function.

    Args:
        values (Tensor): A tensor of shape (minibatch,) containing the value function
            estimates for the states in the minibatch.
        mb_returns (Tensor): A tensor of shape (minibatch,) containing the discounted
            returns estimated for each state in the minibatch.
        vf_coef (float): The coefficient for the value loss, which weights its
            contribution to the overall loss. Denoted by c_1 in the paper.

    Returns:
        Tensor: The value function loss computed over the minibatch, with shape ().

    '''
    return 0.5 * vf_coef * (values - mb_returns).pow(2).mean()


def calc_entropy_bonus(probs: Categorical, ent_coef: float):
    '''
    Return the entropy bonus term, suitable for gradient ascent.

    Args:
        probs (Categorical): A distribution containing the actor's unnormalized
            logits of shape (minibatch, num_actions).
        ent_coef (float): The coefficient for the entropy loss, which weights its
            contribution to the overall loss. Denoted by c_2 in the paper.

    Returns:
        Tensor: The entropy bonus computed over the minibatch, with shape ().
    '''
    return ent_coef * probs.entropy().mean()
