import pytest
import torch
import copy
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import random_split, DataLoader
from src.config import TransformerModelConfig, EnvironmentConfig
from src.models.trajectory_transformer import (
    DecisionTransformer,
    CloneTransformer,
)
from src.environments.environments import make_env
from src.decision_transformer.train import evaluate_dt_agent, test
from src.decision_transformer.offline_dataset import TrajectoryDataset


@pytest.fixture
def trajectory_data_set():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trajectory_path = "tests/fixtures/test_trajectories.pkl"
    trajectory_data_set = TrajectoryDataset(
        trajectory_path, pct_traj=1, device=device
    )
    return trajectory_data_set


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
