from abc import abstractmethod
from typing import Tuple, Union

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from gymnasium.spaces import Box, Dict
from torchtyping import TensorType as TT
from transformer_lens import HookedTransformer, HookedTransformerConfig

from src.config import EnvironmentConfig, TransformerModelConfig


class TrajectoryTransformer(nn.Module):
    """
    Base Class for trajectory modelling transformers including:
        - Decision Transformer (offline, RTG, (R,s,a))
        - Online Transformer (online, reward, (s,a,r) or (s,a))
    """

    def __init__(
        self,
        transformer_config: TransformerModelConfig,
        environment_config: EnvironmentConfig,
    ):
        super().__init__()

        self.transformer_config = transformer_config
        self.environment_config = environment_config

        self.action_embedding = nn.Sequential(
            nn.Embedding(
                environment_config.action_space.n + 1,
                self.transformer_config.d_model,
            )
        )
        self.time_embedding = self.initialize_time_embedding()
        self.state_embedding = self.initialize_state_embedding()

        # Initialize weights
        nn.init.normal_(
            self.action_embedding[0].weight,
            mean=0.0,
            std=1
            / (
                (environment_config.action_space.n + 1 + 1)
                * self.transformer_config.d_model
            ),
        )

        self.transformer = self.initialize_easy_transformer()

        self.action_predictor = nn.Linear(
            self.transformer_config.d_model, environment_config.action_space.n
        )
        self.initialize_state_predictor()

    def get_time_embedding(self, timesteps):
        assert (
            timesteps.max() <= self.environment_config.max_steps
        ), "timesteps must be less than max_timesteps"

        block_size = timesteps.shape[1]
        timesteps = rearrange(
            timesteps, "batch block time-> (batch block) time"
        )
        time_embeddings = self.time_embedding(timesteps)
        if self.transformer_config.time_embedding_type != "linear":
            time_embeddings = time_embeddings.squeeze(-2)
        time_embeddings = rearrange(
            time_embeddings,
            "(batch block) n_embd -> batch block n_embd",
            block=block_size,
        )
        return time_embeddings

    def get_state_embedding(self, states):
        # embed states and recast back to (batch, block_size, n_embd)
        block_size = states.shape[1]
        if self.transformer_config.state_embedding_type == "CNN":
            states = rearrange(
                states,
                "batch block height width channel -> (batch block) channel height width",
            )
            state_embeddings = self.state_embedding(
                states.type(torch.float32).contiguous()
            )  # (batch * block_size, n_embd)
        elif self.transformer_config.state_embedding_type == "grid":
            states = rearrange(
                states,
                "batch block height width channel -> (batch block) (channel height width)",
            )
            state_embeddings = self.state_embedding(
                states.type(torch.float32).contiguous()
            )  # (batch * block_size, n_embd)
        else:
            states = rearrange(
                states, "batch block state_dim -> (batch block) state_dim"
            )
            state_embeddings = self.state_embedding(
                states.type(torch.float32).contiguous()
            )
        state_embeddings = rearrange(
            state_embeddings,
            "(batch block) n_embd -> batch block n_embd",
            block=block_size,
        )
        return state_embeddings

    def get_action_embedding(self, actions):
        block_size = actions.shape[1]
        if block_size == 0:
            return None  # no actions to embed
        actions = rearrange(
            actions, "batch block action -> (batch block) action"
        )
        # I don't see why we need this but we do? Maybe because of the sequential?
        action_embeddings = self.action_embedding(actions).flatten(1)
        action_embeddings = rearrange(
            action_embeddings,
            "(batch block) n_embd -> batch block n_embd",
            block=block_size,
        )
        return action_embeddings

    def predict_states(self, x):
        return self.state_predictor(x)

    def predict_actions(self, x):
        return self.action_predictor(x)

    @abstractmethod
    def get_token_embeddings(
        self, state_embeddings, time_embeddings, action_embeddings, **kwargs
    ):
        """
        Returns the token embeddings for the transformer input.
        Note that different subclasses will have different token embeddings
        such as the DecisionTransformer which will use RTG (placed before the
        state embedding).

        Args:
            states: (batch, position, state_dim)
            actions: (batch, position)
            timesteps: (batch, position)
        Kwargs:
            rtgs: (batch, position) (only for DecisionTransformer)

        Returns:
            token_embeddings: (batch, position, n_embd)
        """
        pass

    @abstractmethod
    def get_action(self, **kwargs) -> int:
        """
        Returns the action given the state.
        """
        pass

    def initialize_time_embedding(self):
        if not (self.transformer_config.time_embedding_type == "linear"):
            self.time_embedding = nn.Embedding(
                self.environment_config.max_steps + 1,
                self.transformer_config.d_model,
            )
        else:
            self.time_embedding = nn.Linear(1, self.transformer_config.d_model)

        return self.time_embedding

    def initialize_state_embedding(self):
        if self.transformer_config.state_embedding_type == "CNN":
            state_embedding = StateEncoder(self.transformer_config.d_model)
        else:
            if isinstance(self.environment_config.observation_space, Dict):
                n_obs = np.prod(
                    self.environment_config.observation_space["image"].shape
                )
            else:
                n_obs = np.prod(
                    self.environment_config.observation_space.shape
                )

            state_embedding = nn.Linear(
                n_obs, self.transformer_config.d_model, bias=False
            )

            nn.init.normal_(state_embedding.weight, mean=0.0, std=0.02)

        return state_embedding

    def initialize_state_predictor(self):
        if isinstance(self.environment_config.observation_space, Box):
            self.state_predictor = nn.Linear(
                self.transformer_config.d_model,
                np.prod(self.environment_config.observation_space.shape),
            )
        elif isinstance(self.environment_config.observation_space, Dict):
            self.state_predictor = nn.Linear(
                self.transformer_config.d_model,
                np.prod(
                    self.environment_config.observation_space["image"].shape
                ),
            )

    def initialize_easy_transformer(self):
        # Transformer
        cfg = HookedTransformerConfig(
            n_layers=self.transformer_config.n_layers,
            d_model=self.transformer_config.d_model,
            d_head=self.transformer_config.d_head,
            n_heads=self.transformer_config.n_heads,
            d_mlp=self.transformer_config.d_mlp,
            d_vocab=self.transformer_config.d_model,
            # 3x the max timestep so we have room for an action, reward, and state per timestep
            n_ctx=self.transformer_config.n_ctx,
            act_fn="relu",
            normalization_type="LN"
            if self.transformer_config.layer_norm
            else None,
            attention_dir="causal",
            d_vocab_out=self.transformer_config.d_model,
            seed=self.transformer_config.seed,
            device=self.transformer_config.device,
        )

        assert (
            cfg.attention_dir == "causal"
        ), "Attention direction must be causal"
        # assert cfg.normalization_type is None, "Normalization type must be None"

        transformer = HookedTransformer(cfg)

        # Because we passing in tokens, turn off embedding and update the position embedding
        transformer.embed = nn.Identity()
        transformer.pos_embed = PosEmbedTokens(cfg)
        # initialize position embedding
        nn.init.normal_(transformer.pos_embed.W_pos, cfg.initializer_range)
        # don't unembed, we'll do that ourselves.
        transformer.unembed = nn.Identity()

        return transformer


class DecisionTransformer(TrajectoryTransformer):
    def __init__(self, environment_config, transformer_config, **kwargs):
        super().__init__(
            environment_config=environment_config,
            transformer_config=transformer_config,
            **kwargs,
        )
        self.model_type = "decision_transformer"
        self.reward_embedding = nn.Sequential(
            nn.Linear(1, self.transformer_config.d_model, bias=False)
        )
        self.reward_predictor = nn.Linear(self.transformer_config.d_model, 1)

        # n_ctx include full timesteps except for the last where it doesn't know the action
        assert (transformer_config.n_ctx - 2) % 3 == 0

        nn.init.normal_(
            self.reward_embedding[0].weight,
            mean=0.0,
            std=1 / self.transformer_config.d_model,
        )

    def predict_rewards(self, x):
        return self.reward_predictor(x)

    def get_token_embeddings(
        self,
        state_embeddings,
        time_embeddings,
        reward_embeddings,
        action_embeddings=None,
        targets=None,
    ):
        """
        We need to compose the embeddings for:
            - states
            - actions
            - rewards
            - time

        Handling the cases where:
        1. we are training:
            1. we may not have action yet (reward, state)
            2. we have (action, state, reward)...
        2. we are evaluating:
            1. we have a target "a reward" followed by state

        1.1 and 2.1 are the same, but we need to handle the target as the initial reward.

        """
        batches = state_embeddings.shape[0]
        timesteps = time_embeddings.shape[1]

        reward_embeddings = reward_embeddings + time_embeddings
        state_embeddings = state_embeddings + time_embeddings

        if action_embeddings is not None:
            if action_embeddings.shape[1] < timesteps:
                assert (
                    action_embeddings.shape[1] == timesteps - 1
                ), "Action embeddings must be one timestep less than state embeddings"
                action_embeddings = (
                    action_embeddings
                    + time_embeddings[:, : action_embeddings.shape[1]]
                )
                trajectory_length = timesteps * 3 - 1
            else:
                action_embeddings = action_embeddings + time_embeddings
                trajectory_length = timesteps * 3
        else:
            trajectory_length = 2  # one timestep, no action yet

        if targets:
            targets = targets + time_embeddings

        # create the token embeddings
        token_embeddings = torch.zeros(
            (batches, trajectory_length, self.transformer_config.d_model),
            dtype=torch.float32,
            device=state_embeddings.device,
        )  # batches, blocksize, n_embd

        if action_embeddings is not None:
            token_embeddings[:, ::3, :] = reward_embeddings
            token_embeddings[:, 1::3, :] = state_embeddings
            token_embeddings[:, 2::3, :] = action_embeddings
        else:
            token_embeddings[:, 0, :] = reward_embeddings[:, 0, :]
            token_embeddings[:, 1, :] = state_embeddings[:, 0, :]

        if targets is not None:
            target_embedding = self.reward_embedding(targets)
            token_embeddings[:, 0, :] = target_embedding[:, 0, :]

        return token_embeddings

    def to_tokens(self, states, actions, rtgs, timesteps):
        # embed states and recast back to (batch, block_size, n_embd)
        state_embeddings = self.get_state_embedding(
            states
        )  # batch_size, block_size, n_embd
        action_embeddings = (
            self.get_action_embedding(actions) if actions is not None else None
        )  # batch_size, block_size, n_embd or None
        reward_embeddings = self.get_reward_embedding(
            rtgs
        )  # batch_size, block_size, n_embd
        time_embeddings = self.get_time_embedding(
            timesteps
        )  # batch_size, block_size, n_embd

        # use state_embeddings, actions, rewards to go and
        token_embeddings = self.get_token_embeddings(
            state_embeddings=state_embeddings,
            action_embeddings=action_embeddings,
            reward_embeddings=reward_embeddings,
            time_embeddings=time_embeddings,
        )
        return token_embeddings

    def get_action(self, states, actions, rewards, timesteps):
        state_preds, action_preds, reward_preds = self.forward(
            states, actions, rewards, timesteps
        )

        # get the action prediction
        action_preds = action_preds[:, -1, :]  # (batch, n_actions)
        action = torch.argmax(action_preds, dim=-1)  # (batch)
        return action

    def get_reward_embedding(self, rtgs):
        block_size = rtgs.shape[1]
        rtgs = rearrange(rtgs, "batch block rtg -> (batch block) rtg")
        rtg_embeddings = self.reward_embedding(rtgs.type(torch.float32))
        rtg_embeddings = rearrange(
            rtg_embeddings,
            "(batch block) n_embd -> batch block n_embd",
            block=block_size,
        )
        return rtg_embeddings

    def get_logits(self, x, batch_size, seq_length, no_actions: bool):
        if no_actions is False:
            # TODO replace with einsum
            if (x.shape[1] % 3 != 0) and ((x.shape[1] + 1) % 3 == 0):
                x = torch.concat((x, x[:, -2].unsqueeze(1)), dim=1)

            x = x.reshape(
                batch_size, seq_length, 3, self.transformer_config.d_model
            )
            x = x.permute(0, 2, 1, 3)

            # predict next return given state and action
            reward_preds = self.predict_rewards(x[:, 2])
            # predict next state given state and action
            state_preds = self.predict_states(x[:, 2])
            # predict next action given state and RTG
            action_preds = self.predict_actions(x[:, 1])
            return state_preds, action_preds, reward_preds

        else:
            # TODO replace with einsum
            x = x.reshape(
                batch_size, seq_length, 2, self.transformer_config.d_model
            )
            x = x.permute(0, 2, 1, 3)
            # predict next action given state and RTG
            action_preds = self.predict_actions(x[:, 1])
            return None, action_preds, None

    def forward(
        self,
        # has variable shape, starting with batch, position
        states: TT[...],  # noqa: F821
        actions: TT["batch", "position"],  # noqa: F821
        rtgs: TT["batch", "position"],  # noqa: F821
        timesteps: TT["batch", "position"],  # noqa: F821
        pad_action: bool = True,
    ) -> Tuple[
        TT[...], TT["batch", "position"], TT["batch", "position"]  # noqa: F821
    ]:
        batch_size = states.shape[0]
        seq_length = states.shape[1]
        no_actions = actions is None

        if no_actions is False:
            if actions.shape[1] < seq_length - 1:
                raise ValueError(
                    f"Actions required for all timesteps except the last, got {actions.shape[1]} and {seq_length}"
                )

            # if actions.shape[1] == seq_length - 1:
            #     if pad_action:
            #         print(
            #             "Warning: actions are missing for the last timestep, padding with zeros")
            #         # This means that you can't interpret Reward or State predictions for the last timestep!!!
            #         actions = torch.cat([actions, torch.zeros(
            #             batch_size, 1, 1, dtype=torch.long, device=actions.device)], dim=1)

        # embed states and recast back to (batch, block_size, n_embd)
        token_embeddings = self.to_tokens(states, actions, rtgs, timesteps)
        x = self.transformer(token_embeddings)
        state_preds, action_preds, reward_preds = self.get_logits(
            x, batch_size, seq_length, no_actions=no_actions
        )

        return state_preds, action_preds, reward_preds


class CloneTransformer(TrajectoryTransformer):
    """
    Behavioral clone modelling transformer including:
        - CloneTransformer (offline, (s,a))
    """

    def __init__(
        self,
        transformer_config: TransformerModelConfig,
        environment_config: EnvironmentConfig,
    ):
        super().__init__(transformer_config, environment_config)
        self.model_type = "clone_transformer"
        # n_ctx must be odd (previous state, action, next state)
        assert (transformer_config.n_ctx - 1) % 2 == 0
        self.transformer = (
            self.initialize_easy_transformer()
        )  # this might not be needed?

    def get_token_embeddings(
        self, state_embeddings, time_embeddings, action_embeddings=None
    ):
        """
        Returns the token embeddings for the transformer input.
        Note that different subclasses will have different token embeddings
        such as the DecisionTransformer which will use RTG (placed before the
        state embedding).

        Args:
            states: (batch, position, state_dim)
            actions: (batch, position)

        Returns:
            token_embeddings: (batch, position, n_embd)
        """
        batches = state_embeddings.shape[0]
        timesteps = time_embeddings.shape[1]

        state_embeddings = state_embeddings + time_embeddings

        if action_embeddings is not None:
            if action_embeddings.shape[1] == time_embeddings.shape[1] - 1:
                # missing action for last t-step.
                action_embeddings = action_embeddings + time_embeddings[:, :-1]
                # repeat the last action embedding for the last timestep
                action_embeddings = torch.cat(
                    [
                        action_embeddings,
                        action_embeddings[:, -1, :].unsqueeze(1),
                    ],
                    dim=1,
                )
                # now the last action and second last are duplicates but we can fix this later. (TODO)
                trajectory_length = timesteps * 2
            else:
                action_embeddings = action_embeddings + time_embeddings
                trajectory_length = timesteps * 2
        else:
            trajectory_length = 1  # one timestep, no action yet

        # create the token embeddings
        token_embeddings = torch.zeros(
            (batches, trajectory_length, self.transformer_config.d_model),
            dtype=torch.float32,
            device=state_embeddings.device,
        )  # batches, blocksize, n_embd

        if action_embeddings is not None:
            token_embeddings[:, 0::2, :] = state_embeddings
            token_embeddings[:, 1::2, :] = action_embeddings
        else:
            token_embeddings[:, 0, :] = state_embeddings[:, 0, :]

        return token_embeddings

    def to_tokens(self, states, actions, timesteps):
        # embed states and recast back to (batch, block_size, n_embd)
        state_embeddings = self.get_state_embedding(
            states
        )  # batch_size, block_size, n_embd
        action_embeddings = (
            self.get_action_embedding(actions) if actions is not None else None
        )  # batch_size, block_size, n_embd or None
        time_embeddings = self.get_time_embedding(
            timesteps
        )  # batch_size, block_size, n_embd

        # use state_embeddings, actions, rewards to go and
        token_embeddings = self.get_token_embeddings(
            state_embeddings=state_embeddings,
            action_embeddings=action_embeddings,
            time_embeddings=time_embeddings,
        )
        return token_embeddings

    def forward(
        self,
        # has variable shape, starting with batch, position
        states: TT[...],
        actions: TT["batch", "position"],  # noqa: F821
        timesteps: TT["batch", "position"],  # noqa: F821
        pad_action: bool = True,
    ) -> Tuple[
        TT[...], TT["batch", "position"], TT["batch", "position"]  # noqa: F821
    ]:
        batch_size = states.shape[0]
        seq_length = states.shape[1]

        if (
            seq_length + (seq_length - 1) * (actions is not None)
            > self.transformer_config.n_ctx
        ):
            raise ValueError(
                f"Sequence length is too long for transformer, got {seq_length} and {self.transformer_config.n_ctx}"
            )

        no_actions = (actions is None) or (actions.shape[1] == 0)

        if no_actions is False:
            if actions.shape[1] < seq_length - 1:
                raise ValueError(
                    f"Actions required for all timesteps except the last, got {actions.shape[1]} and {seq_length}"
                )

            if actions.shape[1] != seq_length - 1:
                if pad_action:
                    print(
                        "Warning: actions are missing for the last timestep, padding with zeros"
                    )
                    # This means that you can't interpret Reward or State predictions for the last timestep!!!
                    actions = torch.cat(
                        [
                            torch.zeros(
                                batch_size,
                                1,
                                1,
                                dtype=torch.long,
                                device=actions.device,
                            ),
                            actions,
                        ],
                        dim=1,
                    )

        # embed states and recast back to (batch, block_size, n_embd)
        token_embeddings = self.to_tokens(states, actions, timesteps)

        if no_actions is False:
            if actions.shape[1] == states.shape[1] - 1:
                x = self.transformer(token_embeddings[:, :-1])
                # concat last action embedding to the end of the transformer output x[:,-2].unsqueeze(1)
                x = torch.cat(
                    [x, token_embeddings[:, -2, :].unsqueeze(1)], dim=1
                )
                state_preds, action_preds = self.get_logits(
                    x, batch_size, seq_length, no_actions=no_actions
                )
            else:
                x = self.transformer(token_embeddings)
                state_preds, action_preds = self.get_logits(
                    x, batch_size, seq_length, no_actions=no_actions
                )
        else:
            x = self.transformer(token_embeddings)
            state_preds, action_preds = self.get_logits(
                x, batch_size, seq_length, no_actions=no_actions
            )

        return state_preds, action_preds

    def get_action(self, states, actions, timesteps):
        state_preds, action_preds = self.forward(states, actions, timesteps)

        # get the action prediction
        action_preds = action_preds[:, -1, :]  # (batch, n_actions)
        action = torch.argmax(action_preds, dim=-1)  # (batch)
        return action

    def get_logits(self, x, batch_size, seq_length, no_actions: bool):
        # TODO replace with einsum
        if not no_actions:
            x = x.reshape(
                batch_size, seq_length, 2, self.transformer_config.d_model
            ).permute(0, 2, 1, 3)
            # predict next return given state and action
            # reward_preds = self.predict_rewards(x[:, 2])
            # predict next state given state and action
            state_preds = self.predict_states(x[:, 1])
            # predict next action given state
            action_preds = self.predict_actions(x[:, 0])

            return state_preds, action_preds
        else:
            x = x.reshape(
                batch_size, seq_length, 1, self.transformer_config.d_model
            ).permute(0, 2, 1, 3)

            # predict next return given state and action
            # reward_preds = self.predict_rewards(x[:, 2])
            # predict next state given state and action
            # predict next action given state
            action_preds = self.predict_actions(x[:, 0])

            return None, action_preds


class ActorTransformer(CloneTransformer):
    """
    Identical to clone transformer but forward pass can only return action predictions
    """

    def __init__(
        self,
        transformer_config: TransformerModelConfig,
        environment_config: EnvironmentConfig,
    ):
        super().__init__(transformer_config, environment_config)

    def forward(
        self,
        # has variable shape, starting with batch, position
        states: TT[...],
        actions: TT["batch", "position"],  # noqa: F821
        timesteps: TT["batch", "position"],  # noqa: F821
        pad_action: bool = True,
    ) -> TT["batch", "position"]:  # noqa: F821
        _, action_preds = super().forward(
            states, actions, timesteps, pad_action=pad_action
        )

        return action_preds


class CriticTransfomer(CloneTransformer):
    """
    Identical to clone transformer but forward pass can only return state predictions
    """

    def __init__(
        self,
        transformer_config: TransformerModelConfig,
        environment_config: EnvironmentConfig,
    ):
        super().__init__(transformer_config, environment_config)
        self.value_predictor = nn.Linear(
            transformer_config.d_model, 1, bias=True
        )

    def forward(
        self,
        # has variable shape, starting with batch, position
        states: TT[...],
        actions: TT["batch", "position"],  # noqa: F821
        timesteps: TT["batch", "position"],  # noqa: F821
        pad_action: bool = True,
    ) -> TT[...]:  # noqa: F821
        _, value_pred = super().forward(
            states, actions, timesteps, pad_action=pad_action
        )

        return value_pred

    # hacky way to predict values instead of actions with same information
    def predict_actions(self, x):
        return self.value_predictor(x)


class StateEncoder(nn.Module):
    def __init__(self, n_embed):
        super(StateEncoder, self).__init__()
        self.n_embed = n_embed
        # input has shape 56 x 56 x 3
        # output has shape 1 x 1 x 512
        self.conv1 = nn.Conv2d(3, 32, 8, stride=4, padding=0)  # 56 -> 13
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=0)  # 13 -> 5
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=0)  # 5 -> 3
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(576, n_embed)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.fc(x)
        x = F.relu(x)
        return x


class PosEmbedTokens(nn.Module):
    def __init__(self, cfg: Union[Dict, HookedTransformerConfig]):
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = HookedTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        self.W_pos = nn.Parameter(
            torch.empty(self.cfg.n_ctx, self.cfg.d_model)
        )

    def forward(
        self,
        tokens: TT["batch", "position"],  # noqa: F821
        past_kv_pos_offset: int = 0,
    ) -> TT["batch", "position", "d_model"]:  # noqa: F821
        """Tokens have shape [batch, pos]
        Output shape [pos, d_model] - will be broadcast along batch dim"""

        tokens_length = tokens.size(-2)
        pos_embed = self.W_pos[:tokens_length, :]  # [pos, d_model]
        broadcast_pos_embed = einops.repeat(
            pos_embed, "pos d_model -> batch pos d_model", batch=tokens.size(0)
        )  # [batch, pos, d_model]
        return broadcast_pos_embed
