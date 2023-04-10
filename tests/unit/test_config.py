import gymnasium as gym
import pytest

from src.config import (
    EnvironmentConfig,
    LSTMModelConfig,
    OfflineTrainConfig,
    OnlineTrainConfig,
    RunConfig,
    TransformerModelConfig,
)


def test_environment_config():
    # test existence of properties
    config = EnvironmentConfig()
    assert hasattr(config, "env_id")
    assert hasattr(config, "one_hot_obs")
    assert hasattr(config, "img_obs")
    assert hasattr(config, "fully_observed")
    assert hasattr(config, "max_steps")
    assert hasattr(config, "seed")
    assert hasattr(config, "view_size")
    assert hasattr(config, "capture_video")
    assert hasattr(config, "video_dir")
    assert hasattr(config, "render_mode")
    assert hasattr(config, "action_space")
    assert hasattr(config, "observation_space")
    assert hasattr(config, "device")

    # test post __init__ method
    config = EnvironmentConfig(env_id="MiniGrid-Empty-8x8-v0")
    env = gym.make(config.env_id)
    assert config.action_space == env.action_space
    assert config.observation_space == env.observation_space


def test_transformer_model_config():
    # test existence of properties
    config = TransformerModelConfig()
    assert hasattr(config, "d_model")
    assert hasattr(config, "n_heads")
    assert hasattr(config, "d_mlp")
    assert hasattr(config, "n_layers")
    assert hasattr(config, "n_ctx")
    assert hasattr(config, "layer_norm")
    assert hasattr(config, "state_embedding_type")
    assert hasattr(config, "time_embedding_type")
    assert hasattr(config, "seed")
    assert hasattr(config, "device")
    assert hasattr(config, "d_head")

    # test post __init__ method
    with pytest.raises(AssertionError):
        # d_model is not divisible by n_heads
        TransformerModelConfig(d_model=100, n_heads=3)


def test_lstm_model_config():
    environment_config = EnvironmentConfig()
    config = LSTMModelConfig(environment_config)

    assert hasattr(config, "environment_config")
    assert hasattr(config, "image_dim")
    assert hasattr(config, "memory_dim")
    assert hasattr(config, "instr_dim")
    assert hasattr(config, "lang_model")
    assert hasattr(config, "use_memory")
    assert hasattr(config, "recurrence")
    assert hasattr(config, "arch")
    assert hasattr(config, "aux_info")
    assert hasattr(config, "endpool")
    assert hasattr(config, "bow")
    assert hasattr(config, "pixel")
    assert hasattr(config, "res")


def test_online_train_config():
    # test existence of properties
    config = OnlineTrainConfig()
    assert hasattr(config, "batch_size")
    assert hasattr(config, "minibatch_size")
    assert hasattr(config, "hidden_size")
    assert hasattr(config, "total_timesteps")
    assert hasattr(config, "learning_rate")
    assert hasattr(config, "decay_lr")
    assert hasattr(config, "num_envs")
    assert hasattr(config, "num_steps")
    assert hasattr(config, "gamma")
    assert hasattr(config, "gae_lambda")
    assert hasattr(config, "num_minibatches")
    assert hasattr(config, "update_epochs")
    assert hasattr(config, "clip_coef")
    assert hasattr(config, "ent_coef")
    assert hasattr(config, "vf_coef")
    assert hasattr(config, "max_grad_norm")
    assert hasattr(config, "trajectory_path")
    assert hasattr(config, "fully_observed")
    assert hasattr(config, "batch_size")
    assert hasattr(config, "minibatch_size")


def test_offline_train_config():
    # test existence of properties
    config = OfflineTrainConfig(trajectory_path="tests/data/trajectories")
    assert hasattr(config, "batch_size")
    assert hasattr(config, "lr")
    assert hasattr(config, "weight_decay")
    assert hasattr(config, "device")
    assert hasattr(config, "track")
    assert hasattr(config, "train_epochs")
    assert hasattr(config, "test_epochs")
    assert hasattr(config, "test_frequency")
    assert hasattr(config, "eval_frequency")
    assert hasattr(config, "eval_episodes")
    assert hasattr(config, "initial_rtg")
    assert hasattr(config, "eval_max_time_steps")
    assert hasattr(config, "trajectory_path")
    assert hasattr(config, "convert_to_one_hot")
    assert hasattr(config, "eval_num_envs")


def test_offline_train_config_raise_error_no_traj_path():
    with pytest.raises(TypeError):
        OfflineTrainConfig()


def test_run_config():
    # test existence of properties
    config = RunConfig()
    assert hasattr(config, "exp_name")
    assert hasattr(config, "seed")
    assert hasattr(config, "device")
    assert hasattr(config, "track")
    assert hasattr(config, "wandb_project_name")
    assert hasattr(config, "wandb_entity")
    assert not hasattr(config, "trajectory_path")
