import json
import os

import gymnasium as gym
import pytest
import torch
import wandb

from src.config import (
    EnvironmentConfig,
    OfflineTrainConfig,
    OnlineTrainConfig,
    RunConfig,
    TransformerModelConfig,
)

from src.environments.environments import make_env
from src.decision_transformer.offline_dataset import TrajectoryDataset, one_hot_encode_observation
from src.decision_transformer.train import train
from src.decision_transformer.utils import get_max_len_from_model_type, load_decision_transformer, store_model_checkpoint, store_transformer_model
from src.models.trajectory_transformer import DecisionTransformer


@pytest.fixture()
def cleanup_test_results() -> None:
    if not os.path.exists("tmp"):
        os.mkdir("tmp")
    if os.path.exists("tmp/model_data.pt"):
        os.remove("tmp/model_data.pt")


@pytest.fixture()
def run_config() -> RunConfig:
    run_config = RunConfig(
        exp_name="Test-PPO-Basic",
        seed=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        track=False,
        wandb_project_name="PPO-MiniGrid",
        wandb_entity=None,
    )

    return run_config


@pytest.fixture()
def environment_config() -> EnvironmentConfig:
    environment_config = EnvironmentConfig(
        env_id="MiniGrid-Dynamic-Obstacles-8x8-v0",
        view_size=7,
        max_steps=300,
        one_hot_obs=False,
        fully_observed=False,
        render_mode="rgb_array",
        capture_video=True,
        video_dir="videos",
    )
    return environment_config


@pytest.fixture()
def online_config() -> OnlineTrainConfig:
    online_config = OnlineTrainConfig(
        hidden_size=64,
        total_timesteps=2000,
        learning_rate=0.00025,
        decay_lr=True,
        num_envs=30,
        num_steps=64,
        gamma=0.99,
        gae_lambda=0.95,
        num_minibatches=30,
        update_epochs=4,
        clip_coef=0.4,
        ent_coef=0.25,
        vf_coef=0.5,
        max_grad_norm=2,
        trajectory_path="trajectories/MiniGrid-DoorKey-8x8-trajectories.pkl",
    )
    return online_config


@pytest.fixture()
def transformer_config() -> TransformerModelConfig:
    transformer_config = TransformerModelConfig(
        d_model=128,
        n_heads=4,
        d_mlp=256,
        n_layers=2,
        n_ctx=5,
        layer_norm=None,
        state_embedding_type="grid",
        time_embedding_type="embedding",
        seed=1,
        device="cpu",
    )

    return transformer_config


@pytest.fixture()
def offline_config() -> OfflineTrainConfig:
    offline_config = OfflineTrainConfig(
        trajectory_path="trajectories/MiniGrid-DoorKey-8x8-trajectories.pkl",
        batch_size=128,
        lr=0.0001,
        weight_decay=0.0,
        pct_traj=1.0,
        prob_go_from_end=0.0,
        device="cpu",
        track=False,
        train_epochs=100,
        test_epochs=10,
        test_frequency=10,
        eval_frequency=10,
        eval_episodes=10,
        model_type="decision_transformer",
        initial_rtg=[0.0, 1.0],
        eval_max_time_steps=100,
        eval_num_envs=8,
    )
    return offline_config


def test_load_decision_transformer(
    transformer_config,
    offline_config,
    environment_config,
    cleanup_test_results,
):
    model = DecisionTransformer(
        environment_config=environment_config,
        transformer_config=transformer_config,
    )

    path = "tmp/model_data.pt"
    store_transformer_model(
        path=path,
        model=model,
        offline_config=offline_config,
    )

    new_model = load_decision_transformer(path)

    assert_state_dicts_are_equal(new_model.state_dict(), model.state_dict())

    assert new_model.transformer_config == transformer_config
    assert new_model.environment_config == environment_config


def test_decision_transformer_checkpoint_saving_and_loading(
        transformer_config, environment_config, offline_config, run_config
):
    wandb.init(mode="offline")
    model = DecisionTransformer(
        environment_config=environment_config,
        transformer_config=transformer_config
    )
    checkpoint_artifact = wandb.Artifact(
        f"{run_config.exp_name}_checkpoints", type="model"
    )
    checkpoint_num = 1

    checkpoint_num = store_model_checkpoint(
        model=model,
        exp_name=run_config.exp_name,
        offline_config=offline_config,
        checkpoint_num=checkpoint_num,
        checkpoint_artifact=checkpoint_artifact
    )

    assert checkpoint_num == 2

    loaded_model = load_decision_transformer(f"models/{run_config.exp_name}_01.pt")

    assert are_state_dicts_equal(loaded_model.state_dict(), model.state_dict())

    assert loaded_model.transformer_config == transformer_config
    assert loaded_model.environment_config == environment_config


def test_decision_transformer_checkpoint_training(
        transformer_config, environment_config, offline_config, run_config
):
    offline_config.num_checkpoints = 5
    wandb.init(mode="offline")
    model = DecisionTransformer(
        environment_config=environment_config,
        transformer_config=transformer_config
    )
    checkpoint_artifact = wandb.Artifact(
        f"{run_config.exp_name}_checkpoints", type="model"
    )

    env = make_env(environment_config, seed=0, idx=0, run_name="dev")
    env = env()

    preprocess_observations = (
        None
        if not offline_config.convert_to_one_hot
        else one_hot_encode_observation
    )
    max_len = get_max_len_from_model_type(
        offline_config.model_type, transformer_config.n_ctx
    )
    trajectory_data_set = TrajectoryDataset(
        trajectory_path=offline_config.trajectory_path,
        max_len=max_len,
        pct_traj=offline_config.pct_traj,
        prob_go_from_end=offline_config.prob_go_from_end,
        device="cpu",
        preprocess_observations=preprocess_observations,
    )

    train(model, trajectory_data_set, offline_config, env, make_env)

    for i in range(offline_config.num_checkpoints):
        assert os.path.exists(f"models/{run_config.exp_name}_{i+1:0>2}.pt")

    early_model = load_decision_transformer(f"models/{run_config.exp_name}_01.pt")
    late_model = load_decision_transformer(f"models/{run_config.exp_name}_05.pt")

    assert not are_state_dicts_equal(early_model.state_dict(), model.state_dict())
    assert are_state_dicts_equal(late_model.state_dict(), model.state_dict())

    assert early_model.transformer_config == late_model.transformer_config == transformer_config
    assert early_model.environment_config == late_model.environment_config == environment_config

def are_state_dicts_equal(dict1, dict2):
    keys1 = sorted(dict1.keys())
    keys2 = sorted(dict2.keys())

    if keys1 != keys2:
        return False
    results = [dict1[key1].equal(dict2[key2]) for key1, key2 in zip(keys1, keys2)]
    return False not in results
