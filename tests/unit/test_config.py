import pytest
import gymnasium as gym

from src.config import TransformerModelConfig, EnvironmentConfig


def test_transformer_model_config():
    # test existence of properties
    config = TransformerModelConfig()
    assert hasattr(config, 'd_model')
    assert hasattr(config, 'n_heads')
    assert hasattr(config, 'd_mlp')
    assert hasattr(config, 'n_layers')
    assert hasattr(config, 'n_ctx')
    assert hasattr(config, 'layer_norm')
    assert hasattr(config, 'linear_time_embedding')
    assert hasattr(config, 'state_embedding_type')
    assert hasattr(config, 'time_embedding_type')
    assert hasattr(config, 'seed')
    assert hasattr(config, 'device')
    assert hasattr(config, 'd_head')

    # test post __init__ method
    with pytest.raises(AssertionError):
        # d_model is not divisible by n_heads
        TransformerModelConfig(d_model=100, n_heads=3)


def test_environment_config():
    # test existence of properties
    config = EnvironmentConfig()
    assert hasattr(config, 'env')
    assert hasattr(config, 'env_id')
    assert hasattr(config, 'one_hot')
    assert hasattr(config, 'fully_observed')
    assert hasattr(config, 'max_steps')
    assert hasattr(config, 'seed')
    assert hasattr(config, 'view_size')
    assert hasattr(config, 'capture_video')
    assert hasattr(config, 'video_dir')
    assert hasattr(config, 'render_mode')
    assert hasattr(config, 'num_parralel_envs')
    assert hasattr(config, 'action_space')
    assert hasattr(config, 'observation_space')
    assert hasattr(config, 'device')

    # test post __init__ method
    config = EnvironmentConfig(env_id='MiniGrid-Empty-8x8-v0')
    env = gym.make(config.env_id)
    assert config.action_space == env.action_space
    assert config.observation_space == env.observation_space
