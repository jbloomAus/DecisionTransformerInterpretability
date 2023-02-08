import numpy as np
import torch as t
import torch.nn as nn
import time
import wandb

from typing import Callable
from .model import DecisionTransformer


class Trainer:
    '''I like this abstraction from the original paper so let's emulate it.
    '''

    def __init__(
        self,
        model: DecisionTransformer,
        optimizer: t.optim.Optimizer,
        batch_size: int,
        max_len: int,
        mask_action: int,
        get_batch: Callable,
        action_loss_fn: Callable = nn.CrossEntropyLoss(),
        state_loss_fn: Callable = nn.CrossEntropyLoss(),
        reward_loss_fn: Callable = nn.MSELoss(),
        action_coef=1.0,
        state_coef=1.0,
        reward_coef=1.0,
        scheduler=None,
        track=False,
    ):

        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.max_len = max_len
        self.mask_action = mask_action
        self.get_batch = get_batch
        self.action_loss_fn = action_loss_fn
        self.state_loss_fn = state_loss_fn
        self.reward_loss_fn = reward_loss_fn
        self.action_coef = action_coef,
        self.state_coef = state_coef,
        self.reward_coef = reward_coef,
        self.scheduler = scheduler
        self.track = track

    def train_step(self, step):
        states, actions, _, _, rtgs, timesteps, attention_mask = self.get_batch(
            self.batch_size, max_len=self.max_len)

        # fix this later
        actions[actions == -10] = self.mask_action

        state_target, action_target, rtg_target = t.clone(
            states), t.clone(actions), t.clone(rtgs)

        state_preds, action_preds, reward_preds = self.model.forward(
            states=state_target,
            actions=action_target.to(t.int32).unsqueeze(-1),
            rtgs=rtg_target[:, 1:, :],
            timesteps=timesteps.unsqueeze(-1)
        )

        state_loss = self.state_loss_fn(
            state_target,  # [:,1:],
            state_preds.flatten(2),
        )

        action_loss = self.action_loss_fn(
            action_preds,
            action_target,
        )

        reward_loss = self.reward_loss_fn(
            reward_preds,
            rtg_target[:, 1:],
        )

        loss = self.state_coef*state_loss + self.action_coef * \
            action_loss + self.reward_coef*reward_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.track:
            wandb.log({
                "train/state_loss": state_loss,
                "train/action_loss": action_loss,
                "train/reward_loss": reward_loss,
                "train/loss": loss,
            }, step=step)

        return loss.detach().cpu().item()
