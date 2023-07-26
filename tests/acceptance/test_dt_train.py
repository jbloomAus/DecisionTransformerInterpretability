import copy
import numpy as np 
import os
import pickle

import pytest
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import WeightedRandomSampler

from src.config import EnvironmentConfig, OnlineTrainConfig, RunConfig, TransformerModelConfig
from src.decision_transformer.offline_dataset import TrajectoryDataset
from src.decision_transformer.train import test
from src.decision_transformer.eval import evaluate_dt_agent
from src.environments.environments import make_env
from src.models.trajectory_transformer import (
    CloneTransformer,
    DecisionTransformer,
)
from src.utils.trajectory_writer import TrajectoryWriter


@pytest.fixture
def trajectory_data_set():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trajectory_path = "tests/fixtures/test_trajectories.pkl"
    trajectory_data_set = TrajectoryDataset(
        trajectory_path, pct_traj=1, device=device
    )
    return trajectory_data_set


@pytest.fixture
def run_config():
    run_config = RunConfig(
        exp_name="test",
        seed=1,
        track=False,
        wandb_project_name="test",
        wandb_entity="test",
    )

    return run_config


@pytest.fixture
def environment_config(trajectory_data_set):
    env_id = trajectory_data_set.metadata["args"]["env_id"]
    environment_config = EnvironmentConfig(
        env_id=env_id,
        one_hot_obs=trajectory_data_set.observation_type == "one_hot",
        view_size=7,
        fully_observed=False,
        capture_video=False,
        render_mode="rgb_array",
        max_steps=1000,
    )

    return environment_config


@pytest.fixture
def online_config():
    online_config = OnlineTrainConfig(
        use_trajectory_model=False,
        hidden_size=64,
        total_timesteps=180000,
        learning_rate=0.00025,
        decay_lr=False,
        num_envs=4,
        num_steps=128,
        gamma=0.99,
        gae_lambda=0.95,
        num_minibatches=4,
        update_epochs=4,
        clip_coef=0.4,
        ent_coef=0.2,
        vf_coef=0.5,
        max_grad_norm=2,
        trajectory_path=None,
        fully_observed=False,
        device=torch.device("cpu"),
    )

    return online_config


@pytest.fixture
def transformer_model_config():
    transformer_model_config = TransformerModelConfig(
        d_model=128,
        n_heads=4,
        d_mlp=256,
        n_layers=2,
        state_embedding_type="grid",  # hard-coded for now to minigrid.
        n_ctx=2,  # one timestep of context
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    return transformer_model_config


@pytest.fixture
def models(environment_config, transformer_model_config):
    dt1 = DecisionTransformer(
        environment_config=environment_config,
        transformer_config=copy.deepcopy(transformer_model_config),
    )

    transformer_model_config.n_ctx = 5
    dt2 = DecisionTransformer(
        environment_config=environment_config,
        transformer_config=copy.deepcopy(transformer_model_config),
    )

    transformer_model_config.n_ctx = 8
    dt3 = DecisionTransformer(
        environment_config=environment_config,
        transformer_config=copy.deepcopy(transformer_model_config),
    )

    transformer_model_config.n_ctx = 1
    bc1 = CloneTransformer(
        environment_config=environment_config,
        transformer_config=copy.deepcopy(transformer_model_config),
    )

    transformer_model_config.n_ctx = 3
    bc2 = CloneTransformer(
        environment_config=environment_config,
        transformer_config=copy.deepcopy(transformer_model_config),
    )

    transformer_model_config.n_ctx = 9
    bc3 = CloneTransformer(
        environment_config=environment_config,
        transformer_config=copy.deepcopy(transformer_model_config),
    )

    models = {
        "dt1 nctx = 2": dt1,
        "dt2 nctx = 5": dt2,
        "dt3 nctx = 8": dt3,
        "bc1 nctx = 1": bc1,
        "bc2 nctx = 3": bc2,
        "bc3 nctx = 9": bc3,
    }
    return models


@pytest.fixture
def dt(request, models):
    return models.get(request.param)


@pytest.fixture
def env(environment_config):
    env = make_env(
        environment_config, seed=0, idx=0, run_name=f"dt_train_videos_0"
    )
    env = env()
    return env


@pytest.fixture
def test_data_loader(trajectory_data_set):
    train_dataset, test_dataset = random_split(
        trajectory_data_set, [0.95, 0.05]
    )

    # Create the test DataLoader
    test_sampler = WeightedRandomSampler(
        weights=trajectory_data_set.sampling_probabilities[
            test_dataset.indices
        ],
        num_samples=len(test_dataset),
        replacement=True,
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=64, sampler=test_sampler
    )

    return test_dataloader


@pytest.mark.parametrize(
    "dt",
    [
        pytest.param("dt1 nctx = 2"),
        pytest.param("dt2 nctx = 5"),
        pytest.param("dt3 nctx = 8"),
        pytest.param("bc1 nctx = 1"),
        pytest.param("bc2 nctx = 3"),
        pytest.param("bc3 nctx = 9"),
    ],
    indirect=True,
)
def test_test_dt_agent(test_data_loader, dt, env):
    mean_loss, accuracy = test(
        model=dt,
        dataloader=test_data_loader,
        env=env,
        epochs=1,
        track=False,
        batch_number=0,
    )

    assert isinstance(mean_loss, float)
    assert isinstance(accuracy, float)

    assert accuracy >= 0
    assert accuracy <= 1
    assert mean_loss >= 0


@pytest.mark.parametrize(
    "dt",
    [
        pytest.param("dt1 nctx = 2"),
        pytest.param("dt2 nctx = 5"),
        pytest.param("dt3 nctx = 8"),
        pytest.param("bc1 nctx = 1"),
        pytest.param("bc2 nctx = 3"),
        pytest.param("bc3 nctx = 9"),
    ],
    indirect=True,
)
def test_evaluate_dt_agent(environment_config, dt):
    environment_config.max_steps = 10  # speed up test
    batch = 0
    eval_env_func = make_env(
        environment_config,
        seed=batch,
        idx=0,
        run_name=f"dt_eval_videos_{batch}",
    )

    statistics = evaluate_dt_agent(
        env_id=environment_config.env_id,
        model=dt,
        env_func=eval_env_func,
        track=False,
        initial_rtg=1,
        trajectories=10,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    assert statistics["prop_completed"] == 0.0
    assert statistics["prop_truncated"] == 1.0
    assert statistics["mean_reward"] == 0.0
    assert statistics["prop_positive_reward"] == 0.0
    # traj length approx 10
    assert statistics["mean_traj_length"] == pytest.approx(10.0, 1.0)


@pytest.mark.parametrize(
    "dt",
    [
        pytest.param("dt2 nctx = 5"),
        pytest.param("dt3 nctx = 8"),
    ],
    indirect=True,
)
def test_evaluate_dt_agent_with_trajectory_writer(environment_config, dt, run_config, online_config):
    trajectory_path = "tmp/test_trajectory_writer_writer.pkl"
    n_ctx = dt.transformer_config.n_ctx
    try:
        environment_config.max_steps = 10  # speed up test
        batch = 0
        eval_env_func = make_env(
            environment_config,
            seed=batch,
            idx=0,
            run_name=f"dt_eval_videos_{batch}",
        )

        trajectory_writer = TrajectoryWriter(
            path=trajectory_path,
            run_config=run_config,
            environment_config=environment_config,
            online_config=online_config,
            model_config=None,
        )

        statistics = evaluate_dt_agent(
            env_id=environment_config.env_id,
            model=dt,
            env_func=eval_env_func,
            track=False,
            initial_rtg=1,
            trajectories=10,
            device="cuda" if torch.cuda.is_available() else "cpu",
            trajectory_writer=trajectory_writer
        )

        assert os.path.getsize(trajectory_path) > 0
        with open(trajectory_path, "rb") as f:
            data = pickle.load(f)
            obs = data["data"]["observations"]
            assert type(obs) == np.ndarray
            assert obs.dtype == np.float64
            assert obs.shape == (16, 1 + n_ctx // 3, 7, 7, 3), f"obs.shape is {obs.shape}"

            rewards = data["data"]["rewards"]
            assert type(rewards) == np.ndarray
            assert rewards.dtype == np.float64
            assert rewards.shape == (16,), f"rewards.shape is {rewards.shape}"

            actions = data["data"]["actions"]
            assert type(actions) == np.ndarray
            assert actions.dtype == np.int64
            assert actions.shape == (16, n_ctx // 3), f"actions.shape is {actions.shape}"
    finally:
        if os.path.exists(trajectory_path):
            os.remove(trajectory_path)
