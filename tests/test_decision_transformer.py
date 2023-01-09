import pytest
import torch as t
import numpy as np 
from einops import rearrange
import gymnasium as gym
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
from src.decision_transformer.model import DecisionTransformer, StateEncoder


def test_state_encoder():

    env = gym.make('MiniGrid-Empty-8x8-v0')
    env = RGBImgPartialObsWrapper(env) # Get pixel observations
    env = ImgObsWrapper(env) # Get rid of the 'mission' field
    obs, _ = env.reset() # This now produces an RGB tensor only

    state_encoder = StateEncoder(64)
    assert state_encoder is not None

    x = t.tensor(obs).unsqueeze(0).to(t.float32)
    x = rearrange(x, 'b h w c-> b c h w')
    x = state_encoder(x)
    assert x.shape == (1, 64)

# first, we want to know that we can initialize the model
def test_decision_transformer_init():

    env = gym.make('MiniGrid-Empty-8x8-v0')
    env = RGBImgPartialObsWrapper(env) # Get pixel observations
    env = ImgObsWrapper(env) # Get rid of the 'mission' field
    obs, _ = env.reset() # This now produces an RGB tensor only

    decision_transformer = DecisionTransformer(env)
    assert decision_transformer is not None

    # our model should have the following:

    # state_encoder
    assert decision_transformer.state_encoder is not None

    # reward_embedding
    assert decision_transformer.reward_embedding is not None

    # action_embedding
    assert decision_transformer.action_embedding is not None

    # time_embedding
    assert decision_transformer.time_embedding is not None

    # a GPT2-like transformer
    assert decision_transformer.transformer is not None
    assert type(decision_transformer.transformer).__name__ == 'HookedTransformer'

def test_get_state_embeddings_image():
    
    env = gym.make('MiniGrid-Empty-8x8-v0')
    env = RGBImgPartialObsWrapper(env) # Get pixel observations
    env = ImgObsWrapper(env) # Get rid of the 'mission' field
    obs, _ = env.reset() # This now produces an RGB tensor only

    decision_transformer = DecisionTransformer(env)
    assert decision_transformer is not None

    # get state embeddings
    obs = t.tensor(obs).unsqueeze(0).unsqueeze(0) # add block, add batch
    state_embeddings = decision_transformer.get_state_embeddings(obs)
    assert state_embeddings.shape == (1, 1, 64)

def test_get_state_embeddings_grid():
    
    env = gym.make('MiniGrid-Empty-8x8-v0')
    # env = RGBImgPartialObsWrapper(env) # Get pixel observations
    # env = ImgObsWrapper(env) # Get rid of the 'mission' field
    obs, _ = env.reset() # This now produces an RGB tensor only

    decision_transformer = DecisionTransformer(env, state_embedding_type='grid')
    assert decision_transformer is not None

    # get state embeddings
    obs = t.tensor(obs['image']).unsqueeze(0).unsqueeze(0) # add block, add batch
    state_embeddings = decision_transformer.get_state_embeddings(obs)
    assert state_embeddings.shape == (1, 1, 64)

def test_get_action_embeddings():
        
    env = gym.make('MiniGrid-Empty-8x8-v0')
    env = RGBImgPartialObsWrapper(env) # Get pixel observations
    env = ImgObsWrapper(env) # Get rid of the 'mission' field
    obs, _ = env.reset() # This now produces an RGB tensor only

    decision_transformer = DecisionTransformer(env)
    assert decision_transformer is not None

    # get action embeddings
    action = t.tensor([0]).unsqueeze(0).unsqueeze(0) # add block, add batch
    action_embeddings = decision_transformer.get_action_embeddings(action)
    assert action_embeddings.shape == (1, 1, 64)

def test_get_reward_embeddings():

    env = gym.make('MiniGrid-Empty-8x8-v0')
    env = RGBImgPartialObsWrapper(env) # Get pixel observations
    env = ImgObsWrapper(env) # Get rid of the 'mission' field
    obs, _ = env.reset() # This now produces an RGB tensor only

    decision_transformer = DecisionTransformer(env)
    assert decision_transformer is not None

    # get reward embeddings
    rtgs = t.tensor([0]).unsqueeze(0).unsqueeze(0) # add block, add batch
    reward_embeddings = decision_transformer.get_reward_embeddings(rtgs)
    assert reward_embeddings.shape == (1, 1, 64)

def test_get_time_embeddings():
    
    env = gym.make('MiniGrid-Empty-8x8-v0')
    env = RGBImgPartialObsWrapper(env) # Get pixel observations
    env = ImgObsWrapper(env) # Get rid of the 'mission' field
    obs, _ = env.reset() # This now produces an RGB tensor only

    decision_transformer = DecisionTransformer(env)
    assert decision_transformer is not None

    # get time embeddings
    times = t.tensor([0], dtype=t.long).unsqueeze(0).unsqueeze(0) # add block, add batch
    time_embeddings = decision_transformer.get_time_embeddings(times)
    assert time_embeddings.shape == (1, 1, 64)

def test_get_token_embeddings_single_batch():
        
    env = gym.make('MiniGrid-Empty-8x8-v0')
    env = RGBImgPartialObsWrapper(env) # Get pixel observations
    env = ImgObsWrapper(env) # Get rid of the 'mission' field
    obs, _ = env.reset() # This now produces an RGB tensor only

    decision_transformer = DecisionTransformer(env)
    assert decision_transformer is not None

    # get token embeddings
    obs = t.tensor(obs).unsqueeze(0).unsqueeze(0) # add block, add batch
    action = t.tensor([0]).unsqueeze(0).unsqueeze(0) # add block, add batch
    rtgs = t.tensor([0]).unsqueeze(0).unsqueeze(0) # add block, add batch
    times = t.tensor([0], dtype=t.long).unsqueeze(0).unsqueeze(0) # add block, add batch

    state_embedding = decision_transformer.get_state_embeddings(obs)
    action_embedding = decision_transformer.get_action_embeddings(action)
    reward_embedding = decision_transformer.get_reward_embeddings(rtgs)
    time_embedding = decision_transformer.get_time_embeddings(times)

    token_embeddings = decision_transformer.get_token_embeddings(
        state_embeddings=state_embedding,
        action_embeddings=action_embedding,
        reward_embeddings=reward_embedding,
        time_embeddings=time_embedding
    )

    assert token_embeddings.shape == (1, 3, 64)

    # now assert that each of the embeddings can be decomposed into state/action/reward + time
    # we can do this by checking that the sum of the embeddings is equal to the token embeddings
    t.testing.assert_close(token_embeddings[0][0]- time_embedding, reward_embedding)
    t.testing.assert_close(token_embeddings[0][1]- time_embedding, state_embedding)
    t.testing.assert_close(token_embeddings[0][2]- time_embedding, action_embedding)

def test_get_token_embeddings_multi_batch():
        
    env = gym.make('MiniGrid-Empty-8x8-v0')
    env = RGBImgPartialObsWrapper(env) # Get pixel observations
    env = ImgObsWrapper(env) # Get rid of the 'mission' field
    obs, _ = env.reset() # This now produces an RGB tensor only

    decision_transformer = DecisionTransformer(env)

    # take several actions, store the observations, actions, returns and timesteps
    all_obs = []
    all_actions = []
    all_returns = []
    all_timesteps = []

    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        all_obs.append(obs)
        all_actions.append(action)
        all_returns.append(reward)
        all_timesteps.append(i)

    # convert to tensors.unsqueeze(0)
    all_obs = t.tensor(np.array(all_obs)).to(t.float32).unsqueeze(0)
    all_actions = t.tensor(all_actions).reshape(-1, 1).unsqueeze(0)
    all_returns = t.randn((10, 1))
    all_returns_to_go = all_returns.flip(0).cumsum(0).flip(0).reshape(-1, 1).unsqueeze(0)
    all_timesteps = t.tensor(all_timesteps).reshape(-1, 1).unsqueeze(0)
    
    state_embedding = decision_transformer.get_state_embeddings(all_obs)
    action_embedding = decision_transformer.get_action_embeddings(all_actions)
    reward_embedding = decision_transformer.get_reward_embeddings(all_returns_to_go)
    time_embedding = decision_transformer.get_time_embeddings(all_timesteps)

    token_embeddings = decision_transformer.get_token_embeddings(
        state_embeddings=state_embedding,
        action_embeddings=action_embedding,
        reward_embeddings=reward_embedding,
        time_embeddings=time_embedding
    )

    assert token_embeddings.shape == (1, 30, 64)

    # now assert that each of the embeddings can be decomposed into state/action/reward + time
    # we can do this by checking that the sum of the embeddings is equal to the token embeddings
    t.testing.assert_close(token_embeddings[:,::3,:],  reward_embedding + time_embedding)
    t.testing.assert_close(token_embeddings[:,1::3,:], state_embedding + time_embedding)
    t.testing.assert_close(token_embeddings[:,2::3,:], action_embedding + time_embedding)

def test_forward():

    env = gym.make('MiniGrid-Empty-8x8-v0')
    env = RGBImgPartialObsWrapper(env) # Get pixel observations
    env = ImgObsWrapper(env) # Get rid of the 'mission' field
    obs, _ = env.reset() # This now produces an RGB tensor only


    # take several actions, store the observations, actions, returns and timesteps
    all_obs = []
    all_actions = []
    all_returns = []
    all_timesteps = []


    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        all_obs.append(obs)
        all_actions.append(action)
        all_returns.append(reward)
        all_timesteps.append(i)

    # convert to tensors.unsqueeze(0)
    all_obs = t.tensor(np.array(all_obs)).to(t.float32).unsqueeze(0)
    all_actions = t.tensor(all_actions).reshape(-1, 1).unsqueeze(0)
    all_returns = t.randn((10, 1))
    all_returns_to_go = all_returns.flip(0).cumsum(0).flip(0).reshape(-1, 1).unsqueeze(0)
    all_timesteps = t.tensor(all_timesteps).reshape(-1, 1).unsqueeze(0)
    
    decision_transformer = DecisionTransformer(env)

    if t.cuda.is_available():
        decision_transformer = decision_transformer.cuda()
        all_obs = all_obs.cuda()
        all_actions = all_actions.cuda()
        all_returns_to_go = all_returns_to_go.cuda()
        all_timesteps = all_timesteps.cuda()

    _, action_logits, _= decision_transformer(
        states = all_obs,
        actions = all_actions,
        rtgs = all_returns_to_go,
        timesteps = all_timesteps
    )

    assert action_logits.shape == (1, 10, env.action_space.n)
