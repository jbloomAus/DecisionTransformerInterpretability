import abc
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch as t
from torch import nn, optim
from torch.distributions.categorical import Categorical


from .memory import Memory
from .utils import get_obs_shape
from .loss_functions import calc_clipped_surrogate_objective, calc_value_function_loss, calc_entropy_bonus

from src.models.trajectory_model import ActorTransformer, CriticTransfomer
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

        self.obs_shape = get_obs_shape(envs.single_observation_space)
        self.num_obs = np.array(self.obs_shape).prod()
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
        cuda = device.type == "cuda"
        obs = memory.next_obs
        done = memory.next_done

        for _ in range(num_steps):
            with t.inference_mode():
                logits = self.actor(obs)
                value = self.critic(obs).flatten()
            probs = Categorical(logits=logits)
            action = probs.sample()
            logprob = probs.log_prob(action)
            next_obs, reward, next_done, next_truncated, info = envs.step(
                action.cpu().numpy())
            next_obs = memory.obs_preprocessor(next_obs)
            reward = t.from_numpy(reward).to(device)

            if trajectory_writer is not None:
                obs_np = obs.detach().cpu().numpy() if cuda else obs.detach().numpy()
                reward_np = reward.detach().cpu().numpy() if cuda else reward.detach().numpy()
                action_np = action.detach().cpu().numpy() if cuda else action.detach().numpy()
                trajectory_writer.accumulate_trajectory(
                    next_obs=obs_np,
                    reward=reward_np,
                    action=action_np,
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
        self.obs_shape = get_obs_shape(envs.single_observation_space)
        self.num_obs = np.array(self.obs_shape).prod()
        self.num_actions = envs.single_action_space.n
        self.hidden_dim = transformer_model_config.d_model
        self.critic = CriticTransfomer(
            transformer_config=transformer_model_config,
            environment_config=environment_config,
        )
        self.layer_init(self.critic.value_predictor, std=0.01)
        self.actor = ActorTransformer(
            transformer_config=transformer_model_config,
            environment_config=environment_config,
        )
        self.layer_init(self.actor.action_predictor, std=0.01)
        self.device = device
        self.to(device)

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
                states = obs.unsqueeze(1)
                timesteps = t.tensor([0]).repeat(n_envs, 1, 1)
                with t.inference_mode():
                    logits = self.actor(states, None, timesteps)
                    values = self.critic(states, None, timesteps)
                    value = values[:, -1].squeeze(-1)  # value is scalar
            else:
                # we have one more obs than action
                obs_timesteps = (context_window_size - 1) // 2 + \
                    1  # (the current obs)
                actions_timesteps = obs_timesteps - 1

                obss = memory.get_obs_traj(step, pad_to_length=obs_timesteps)
                obss = t.cat((obss[:, 1:], obs.unsqueeze(1)), dim=1)
                if actions_timesteps == 0:
                    actions = None
                else:
                    actions = memory.get_act_traj(
                        steps=step - 1,
                        pad_to_length=actions_timesteps
                    ).to(dtype=t.long)

                # Generate the next set of new experiences (one for each env)
                with t.inference_mode():
                    # Our actor generates logits over actions which we can then sample from
                    timesteps = t.tensor([0]).repeat(
                        n_envs, obss.shape[1], 1).to(int)
                    logits = self.actor(
                        states=obss, actions=actions, timesteps=timesteps)
                    # Our critic generates a value function (which we use in the value loss, and to estimate advantages)
                    values = self.critic(
                        states=obss, actions=actions, timesteps=timesteps)
                    values = values[:, -1].squeeze(-1)  # value is scalar

            # get the last state action prediction
            probs = Categorical(logits=logits[:, -1])
            action = probs.sample()
            logprob = probs.log_prob(action)
            next_obs, reward, next_done, next_truncated, info = envs.step(
                action.cpu().numpy())
            next_obs = memory.obs_preprocessor(next_obs)
            reward = t.from_numpy(reward).to(device)

            if trajectory_writer is not None:
                obs_np = obs.detach().cpu().numpy() if cuda else obs.detach().numpy()
                reward_np = reward.detach().cpu().numpy() if cuda else reward.detach().numpy()
                action_np = action.detach().cpu().numpy() if cuda else action.detach().numpy()
                trajectory_writer.accumulate_trajectory(
                    next_obs=obs_np,
                    reward=reward_np,
                    action=action_np,
                    done=next_done,
                    truncated=next_truncated,
                    info=info
                )

            # Store (s_t, d_t, a_t, logpi(a_t|s_t), v(s_t), r_t+1)
            memory.add(info, obs, done, action, logprob, value, reward)
            previous_obs = obs
            obs = t.from_numpy(next_obs).to(device)
            done = t.from_numpy(next_done).to(device, dtype=t.float)

        # Store last (obs, done, value) tuple, since we need it to compute advantages
        memory.next_obs = obs
        memory.next_done = done
        with t.inference_mode():

            # get obss and actions whether the trajectories have been instantiated or not
            obss = self.append_obs(obss, previous_obs, obs)
            actions = self.append_action(actions, action.unsqueeze(-1))

            # TODO: ensure that if the critic context window is small enough, we truncate appropriately
            obss = self.truncate_obss(obss)
            actions = self.truncate_actions(actions)
            timesteps = t.tensor([0]).repeat(n_envs, obss.shape[1], 1)
            values = self.critic(obss, actions, timesteps)
            memory.next_value = values[:, -1].squeeze(-1)

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
            n_timesteps = (self.actor.transformer_config.n_ctx - 1) // 2 + 1
            minibatches = memory.get_trajectory_minibatches(
                n_timesteps, args.prob_go_from_end)

            # Compute loss on each minibatch, and step the optimizer
            for mb in minibatches:
                obs = mb.obs
                actions = mb.actions[:, :-1].unsqueeze(-1).to(
                    int) if mb.actions.shape[1] > 1 else None,
                timesteps = t.tensor([0]).repeat(
                    mb.obs.shape[0], mb.obs.shape[1], 1).to(int)

                logits = self.actor(obs, actions, timesteps)
                values = self.critic(obs, actions, timesteps)
                values = values[:, -1].squeeze(-1)

                probs = Categorical(logits=logits[:, -1])

                clipped_surrogate_objective = calc_clipped_surrogate_objective(
                    probs=probs,
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

    def truncate_obss(self, obss):
        '''assume actor and critic have same config'''
        max_obss_length = (self.actor.transformer_config.n_ctx + 1) // 2
        if obss.shape[1] > max_obss_length:
            obss = obss[:, -max_obss_length:, :]
        return obss

    def truncate_actions(self, actions):
        '''assume actor and critic have same config'''
        max_actions_length = (self.actor.transformer_config.n_ctx + 1) // 2 - 1
        if max_actions_length == 0:
            return None
        elif actions.shape[1] > max_actions_length:
            actions = actions[:, -max_actions_length:]
        return actions

    def append_obs(self, obss, previous_obs, obs):
        '''
        Args:
            obss: (n_envs, n_timesteps, obs_dim) A tensor of observations.
            previous_obs: (n_envs, obs_dim) A tensor of the previous observations.
            obs: (n_envs, obs_dim) A tensor of the current observations.

        Returns:
            obss: (n_envs, n_timesteps + 1, obs_dim) A tensor of observations.
        '''
        if obss is None:
            obss = t.cat((previous_obs.unsqueeze(1), obs.unsqueeze(1)), dim=1)
        else:
            obss = t.cat((obss, obs.unsqueeze(1)), dim=1)
        return obss

    def append_action(self, actions, action):
        '''
        Args:
            actions: (n_envs, n_timesteps, action_dim) A tensor of the previous actions.
            action: (n_envs, action_dim) A tensor of the current action just taken

        Returns:
            actions: (n_envs, n_timesteps + 1, action_dim) A tensor of actions.
        '''
        if actions is None:
            actions = action.unsqueeze(1)
        else:
            actions = t.cat((actions, action.unsqueeze(1)), dim=1)
        return actions
