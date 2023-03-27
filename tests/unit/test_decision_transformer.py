import pytest
import torch as t
import numpy as np
from einops import rearrange
import gymnasium as gym
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper, OneHotPartialObsWrapper
from src.environments.wrappers import ViewSizeWrapper, RenderResizeWrapper
from src.models.trajectory_model import DecisionTransformer, CloneTransformer, StateEncoder, ActorTransformer
from src.config import EnvironmentConfig, TransformerModelConfig


def test_state_encoder():

    env = gym.make('MiniGrid-Empty-8x8-v0')
    env = RGBImgPartialObsWrapper(env)  # Get pixel observations
    env = ImgObsWrapper(env)  # Get rid of the 'mission' field
    obs, _ = env.reset()  # This now produces an RGB tensor only

    state_encoder = StateEncoder(64)
    assert state_encoder is not None

    x = t.tensor(obs).unsqueeze(0).to(t.float32)
    x = rearrange(x, 'b h w c-> b c h w')
    x = state_encoder(x)
    assert x.shape == (1, 64)


def test_decision_transformer_img_obs_forward():

    env = gym.make('MiniGrid-Empty-8x8-v0')
    env = RGBImgPartialObsWrapper(env)  # Get pixel observations
    env = ImgObsWrapper(env)  # Get rid of the 'mission' field
    obs, _ = env.reset()  # This now produces an RGB tensor only

    transformer_config = TransformerModelConfig()
    environment_config = EnvironmentConfig(
        env_id='MiniGrid-Empty-8x8-v0',
        img_obs=True,
    )

    decision_transformer = DecisionTransformer(
        transformer_config=transformer_config,
        environment_config=environment_config
    )

    assert decision_transformer is not None

    # our model should have the following:
    assert decision_transformer.state_embedding is not None
    assert decision_transformer.reward_embedding is not None
    assert decision_transformer.action_embedding is not None
    assert decision_transformer.time_embedding is not None

    # a GPT2-like transformer
    assert decision_transformer.transformer is not None
    assert type(decision_transformer.transformer).__name__ == 'HookedTransformer'

    # a linear layer to predict the next action
    assert decision_transformer.action_predictor is not None

    # a linear layer to predict the next reward
    assert decision_transformer.reward_predictor is not None

    # a linear layer to predict the next state
    assert decision_transformer.state_predictor is not None

    states = t.tensor(obs).unsqueeze(0).unsqueeze(0)  # add block, add batch
    # actions = t.tensor([0]).unsqueeze(0).unsqueeze(0)  # add block, add batch
    rewards = t.tensor([0]).unsqueeze(0).unsqueeze(0)  # add block, add batch
    timesteps = t.tensor([0]).unsqueeze(0).unsqueeze(0)  # add block, add batch

    state_preds, action_preds, reward_preds = decision_transformer.forward(
        states=states,
        actions=None,
        rtgs=rewards,
        timesteps=timesteps
    )

    assert state_preds is None  # no action or reward preds if no actions are given
    assert action_preds.shape == (1, 1, 7)
    assert reward_preds is None  # no action or reward preds if no actions are given


def test_decision_transformer_grid_obs_forward():

    env = gym.make('MiniGrid-Empty-8x8-v0')
    obs, _ = env.reset()  # This now produces an RGB tensor only

    transformer_config = TransformerModelConfig()
    environment_config = EnvironmentConfig(
        env_id='MiniGrid-Empty-8x8-v0',
        img_obs=False,
    )

    decision_transformer = DecisionTransformer(
        transformer_config=transformer_config,
        environment_config=environment_config
    )

    assert decision_transformer is not None

    # our model should have the following:
    assert decision_transformer.state_embedding is not None
    assert decision_transformer.reward_embedding is not None
    assert decision_transformer.action_embedding is not None
    assert decision_transformer.time_embedding is not None

    # a GPT2-like transformer
    assert decision_transformer.transformer is not None
    assert type(decision_transformer.transformer).__name__ == 'HookedTransformer'

    # a linear layer to predict the next action
    assert decision_transformer.action_predictor is not None

    # a linear layer to predict the next reward
    assert decision_transformer.reward_predictor is not None

    # a linear layer to predict the next state
    assert decision_transformer.state_predictor is not None

    states = t.tensor(obs['image']).unsqueeze(
        0).unsqueeze(0)  # add block, add batch
    rewards = t.tensor([0]).unsqueeze(0).unsqueeze(0)  # add block, add batch
    timesteps = t.tensor([0]).unsqueeze(0).unsqueeze(0)  # add block, add batch

    state_preds, action_preds, reward_preds = decision_transformer.forward(
        states=states,
        actions=None,
        rtgs=rewards,
        timesteps=timesteps
    )

    assert state_preds is None  # no action or reward preds if no actions are given
    assert action_preds.shape == (1, 1, 7)
    assert reward_preds is None  # no action or reward preds if no actions are given


def test_decision_transformer_grid_one_hot_forward():

    env = gym.make('MiniGrid-Empty-8x8-v0')
    env = OneHotPartialObsWrapper(env)
    obs, _ = env.reset()  # This now produces an RGB tensor only

    transformer_config = TransformerModelConfig()
    environment_config = EnvironmentConfig(
        env_id='MiniGrid-Empty-8x8-v0',
        one_hot_obs=True,
    )

    decision_transformer = DecisionTransformer(
        transformer_config=transformer_config,
        environment_config=environment_config
    )

    assert decision_transformer is not None

    # our model should have the following:
    assert decision_transformer.state_embedding is not None
    assert decision_transformer.reward_embedding is not None
    assert decision_transformer.action_embedding is not None
    assert decision_transformer.time_embedding is not None

    # a GPT2-like transformer
    assert decision_transformer.transformer is not None
    assert type(decision_transformer.transformer).__name__ == 'HookedTransformer'

    # a linear layer to predict the next action
    assert decision_transformer.action_predictor is not None

    # a linear layer to predict the next reward
    assert decision_transformer.reward_predictor is not None

    # a linear layer to predict the next state
    assert decision_transformer.state_predictor is not None

    states = t.tensor(obs['image']).unsqueeze(
        0).unsqueeze(0)  # add block, add batch
    rewards = t.tensor([0]).unsqueeze(0).unsqueeze(0)  # add block, add batch
    timesteps = t.tensor([0]).unsqueeze(0).unsqueeze(0)  # add block, add batch

    state_preds, action_preds, reward_preds = decision_transformer.forward(
        states=states,
        actions=None,
        rtgs=rewards,
        timesteps=timesteps
    )

    assert state_preds is None  # no action or reward preds if no actions are given
    assert action_preds.shape == (1, 1, 7)
    assert reward_preds is None  # no action or reward preds if no actions are given


def test_decision_transformer_view_size_change_forward():

    env = gym.make('MiniGrid-Empty-8x8-v0')
    env = ViewSizeWrapper(env, agent_view_size=3)
    obs, _ = env.reset()  # This now produces an RGB tensor only

    transformer_config = TransformerModelConfig()
    environment_config = EnvironmentConfig(
        env_id='MiniGrid-Empty-8x8-v0',
        view_size=3,
    )

    decision_transformer = DecisionTransformer(
        transformer_config=transformer_config,
        environment_config=environment_config
    )

    assert decision_transformer is not None

    # our model should have the following:
    assert decision_transformer.state_embedding is not None
    assert decision_transformer.reward_embedding is not None
    assert decision_transformer.action_embedding is not None
    assert decision_transformer.time_embedding is not None

    # a GPT2-like transformer
    assert decision_transformer.transformer is not None
    assert type(decision_transformer.transformer).__name__ == 'HookedTransformer'

    # a linear layer to predict the next action
    assert decision_transformer.action_predictor is not None

    # a linear layer to predict the next reward
    assert decision_transformer.reward_predictor is not None

    # a linear layer to predict the next state
    assert decision_transformer.state_predictor is not None

    states = t.tensor(obs['image']).unsqueeze(
        0).unsqueeze(0)  # add block, add batch
    rewards = t.tensor([0]).unsqueeze(0).unsqueeze(0)  # add block, add batch
    timesteps = t.tensor([0]).unsqueeze(0).unsqueeze(0)  # add block, add batch

    state_preds, action_preds, reward_preds = decision_transformer.forward(
        states=states,
        actions=None,
        rtgs=rewards,
        timesteps=timesteps
    )

    assert state_preds is None  # no action or reward preds if no actions are given
    assert action_preds.shape == (1, 1, 7)
    assert reward_preds is None  # no action or reward preds if no actions are given


def test_decision_transformer_grid_obs_no_action_forward():

    env = gym.make('MiniGrid-Empty-8x8-v0')
    obs, _ = env.reset()  # This now produces an RGB tensor only

    transformer_config = TransformerModelConfig()
    environment_config = EnvironmentConfig(
        env_id='MiniGrid-Empty-8x8-v0',
        img_obs=False,
    )

    decision_transformer = DecisionTransformer(
        transformer_config=transformer_config,
        environment_config=environment_config
    )

    assert decision_transformer is not None

    # our model should have the following:
    assert decision_transformer.state_embedding is not None
    assert decision_transformer.reward_embedding is not None
    assert decision_transformer.action_embedding is not None
    assert decision_transformer.time_embedding is not None

    # a GPT2-like transformer
    assert decision_transformer.transformer is not None
    assert type(decision_transformer.transformer).__name__ == 'HookedTransformer'

    # a linear layer to predict the next action
    assert decision_transformer.action_predictor is not None

    # a linear layer to predict the next reward
    assert decision_transformer.reward_predictor is not None

    # a linear layer to predict the next state
    assert decision_transformer.state_predictor is not None

    states = t.tensor(obs['image']).unsqueeze(
        0).unsqueeze(0)  # add block, add batch
    # actions = t.tensor([0]).unsqueeze(0).unsqueeze(0)  # add block, add batch
    rewards = t.tensor([0]).unsqueeze(0).unsqueeze(0)  # add block, add batch
    timesteps = t.tensor([0]).unsqueeze(0).unsqueeze(0)  # add block, add batch

    state_preds, action_preds, reward_preds = decision_transformer.forward(
        states=states,
        actions=None,
        rtgs=rewards,
        timesteps=timesteps
    )

    assert state_preds is None
    assert action_preds.shape == (1, 1, 7)
    assert reward_preds is None


def test_decision_transformer_grid_obs_one_fewer_action_forward():

    env = gym.make('MiniGrid-Empty-8x8-v0')
    obs, _ = env.reset()  # This now produces an RGB tensor only

    transformer_config = TransformerModelConfig(n_ctx=5)
    environment_config = EnvironmentConfig(
        env_id='MiniGrid-Empty-8x8-v0',
        img_obs=False,
    )

    decision_transformer = DecisionTransformer(
        transformer_config=transformer_config,
        environment_config=environment_config
    )

    assert decision_transformer is not None

    # our model should have the following:
    assert decision_transformer.state_embedding is not None
    assert decision_transformer.reward_embedding is not None
    assert decision_transformer.action_embedding is not None
    assert decision_transformer.time_embedding is not None

    # a GPT2-like transformer
    assert decision_transformer.transformer is not None
    assert type(decision_transformer.transformer).__name__ == 'HookedTransformer'

    # a linear layer to predict the next action
    assert decision_transformer.action_predictor is not None

    # a linear layer to predict the next reward
    assert decision_transformer.reward_predictor is not None

    # a linear layer to predict the next state
    assert decision_transformer.state_predictor is not None

    states = t.tensor([obs['image'], obs['image']]).unsqueeze(
        0)  # add block, add batch
    actions = t.tensor([0]).unsqueeze(0).unsqueeze(0)  # add block, add batch
    rewards = t.tensor([[0], [0]]).unsqueeze(0)  # add block, add batch
    timesteps = t.tensor([[0], [1]]).unsqueeze(0)  # add block, add batch

    state_preds, action_preds, reward_preds = decision_transformer.forward(
        states=states,
        actions=actions,
        rtgs=rewards,
        timesteps=timesteps
    )

    assert state_preds.shape == (1, 2, 147)
    assert action_preds.shape == (1, 2, 7)
    assert reward_preds.shape == (1, 2, 1)


def test_clone_transformer_grid_obs_no_action_forward():

    env = gym.make('MiniGrid-Empty-8x8-v0')
    obs, _ = env.reset()  # This now produces an RGB tensor only

    transformer_config = TransformerModelConfig(n_ctx=1)
    environment_config = EnvironmentConfig(
        env_id='MiniGrid-Empty-8x8-v0',
        img_obs=False,
    )

    clone_transformer = CloneTransformer(
        transformer_config=transformer_config,
        environment_config=environment_config
    )

    assert clone_transformer is not None

    # our model should have the following:
    assert clone_transformer.state_embedding is not None
    assert clone_transformer.action_embedding is not None
    assert clone_transformer.time_embedding is not None

    # a GPT2-like transformer
    assert clone_transformer.transformer is not None
    assert type(clone_transformer.transformer).__name__ == 'HookedTransformer'

    # a linear layer to predict the next action
    assert clone_transformer.action_predictor is not None

    # a linear layer to predict the next state
    assert clone_transformer.state_predictor is not None

    states = t.tensor(obs['image']).unsqueeze(
        0).unsqueeze(0)  # add block, add batch
    timesteps = t.tensor([0]).unsqueeze(0).unsqueeze(0)  # add block, add batch

    _, action_preds = clone_transformer.forward(
        states=states,
        actions=None,
        timesteps=timesteps
    )

    assert action_preds.shape == (1, 1, 7)


def test_clone_transformer_grid_obs_one_fewer_action_forward():

    env = gym.make('MiniGrid-Empty-8x8-v0')
    obs, _ = env.reset()  # This now produces an RGB tensor only

    transformer_config = TransformerModelConfig(n_ctx=7)
    environment_config = EnvironmentConfig(
        env_id='MiniGrid-Empty-8x8-v0',
        img_obs=False,
    )

    clone_transformer = CloneTransformer(
        transformer_config=transformer_config,
        environment_config=environment_config
    )

    assert clone_transformer is not None

    # our model should have the following:
    assert clone_transformer.state_embedding is not None
    assert clone_transformer.action_embedding is not None
    assert clone_transformer.time_embedding is not None

    # a GPT2-like transformer
    assert clone_transformer.transformer is not None
    assert type(clone_transformer.transformer).__name__ == 'HookedTransformer'

    # a linear layer to predict the next action
    assert clone_transformer.action_predictor is not None

    # a linear layer to predict the next state
    assert clone_transformer.state_predictor is not None

    states = t.tensor([obs['image'], obs['image']]).unsqueeze(
        0)  # add block, add batch
    actions = t.tensor([0]).unsqueeze(0).unsqueeze(0)  # add block, add batch
    timesteps = t.tensor([[0], [1]]).unsqueeze(0)  # add block, add batch

    state_preds, action_preds = clone_transformer.forward(
        states=states,
        actions=actions,
        timesteps=timesteps
    )

    assert state_preds.shape == (1, 2, 147)
    assert action_preds.shape == (1, 2, 7)


def test_actor_transformer():

    env = gym.make('MiniGrid-Empty-8x8-v0')
    obs, _ = env.reset()  # This now produces an RGB tensor only

    transformer_config = TransformerModelConfig(n_ctx=1)
    environment_config = EnvironmentConfig(
        env_id='MiniGrid-Empty-8x8-v0',
        img_obs=False,
    )

    actor_transformer = ActorTransformer(
        transformer_config=transformer_config,
        environment_config=environment_config
    )

    assert actor_transformer is not None

    # our model should have the following:
    assert actor_transformer.state_embedding is not None
    assert actor_transformer.action_embedding is not None
    assert actor_transformer.time_embedding is not None

    # a GPT2-like transformer
    assert actor_transformer.transformer is not None
    assert type(actor_transformer.transformer).__name__ == 'HookedTransformer'

    # a linear layer to predict the next action
    assert actor_transformer.action_predictor is not None

    # a linear layer to predict the next state
    assert actor_transformer.state_predictor is not None

    states = t.tensor(obs['image']).unsqueeze(
        0).unsqueeze(0)  # add block, add batch
    # t.tensor([0]).unsqueeze(0).unsqueeze(0)  # add block, add batch
    actions = None
    timesteps = t.tensor([0]).unsqueeze(0).unsqueeze(0)  # add block, add batch

    action_preds = actor_transformer.forward(
        states=states,
        actions=actions,
        timesteps=timesteps
    )

    assert action_preds.shape == (1, 1, 7)
