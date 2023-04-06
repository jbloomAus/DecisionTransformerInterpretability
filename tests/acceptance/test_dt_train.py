import pytest
import torch
from src.config import TransformerModelConfig, EnvironmentConfig
from src.models.trajectory_transformer import DecisionTransformer
from src.environments.environments import make_env
from src.decision_transformer.train import evaluate_dt_agent
from src.decision_transformer.offline_dataset import TrajectoryDataset


@pytest.fixture
def trajectory_data_set():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trajectory_path = "tests/fixtures/test_trajectories.pkl"
    trajectory_data_set = TrajectoryDataset(
        trajectory_path,
        pct_traj=1, device=device)
    return trajectory_data_set


@pytest.fixture
def environment_config(trajectory_data_set):

    env_id = trajectory_data_set.metadata['args']['env_id']
    environment_config = EnvironmentConfig(
        env_id=env_id,
        one_hot_obs=trajectory_data_set.observation_type == "one_hot",
        view_size=7,
        fully_observed=False,
        capture_video=False,
        render_mode='rgb_array',
        max_steps=10)

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


def test_evaluate_dt_agent(trajectory_data_set, environment_config, transformer_model_config):

    dt = DecisionTransformer(
        environment_config=environment_config,
        transformer_config=transformer_model_config)

    batch = 0
    eval_env_func = make_env(
        environment_config, seed=batch, idx=0,
        run_name=f"dt_eval_videos_{batch}"
    )

    statistics = evaluate_dt_agent(
        env_id=environment_config.env_id,
        model=dt,
        env_func=eval_env_func,
        track=False,
        initial_rtg=1,
        trajectories=10,
        device="cuda" if torch.cuda.is_available() else "cpu")

    assert statistics["prop_completed"] == 0.0
    assert statistics["prop_truncated"] == 1.0
    assert statistics["mean_reward"] == 0.0
    assert statistics["prop_positive_reward"] == 0.0
    # traj length approx 10
    assert statistics["mean_traj_length"] == pytest.approx(10.0, 1.0)
