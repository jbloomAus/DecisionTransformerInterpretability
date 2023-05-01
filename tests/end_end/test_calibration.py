import pytest
import os
from argparse import Namespace

from src.run_calibration import runner
from src.config import (
    EnvironmentConfig,
    OfflineTrainConfig,
    TransformerModelConfig,
)

from src.models.trajectory_transformer import DecisionTransformer
from src.decision_transformer.runner import store_transformer_model


@pytest.fixture()
def cleanup_test_results() -> None:
    yield
    os.remove("tmp/model_data.pt")


@pytest.fixture()
def args():
    return Namespace(
        env_id="MiniGrid-DoorKey-8x8-v0",
        model_path="to_be_filled_in",
        max_len=1,
        n_trajectories=20,
        initial_rtg_min=0,
        initial_rtg_max=1,
        initial_rtg_step=0.5,
        num_envs=4,
    )


@pytest.fixture()
def environment_config() -> EnvironmentConfig:
    environment_config = EnvironmentConfig(
        env_id="MiniGrid-Dynamic-Obstacles-8x8-v0",
        view_size=3,
        max_steps=300,
        one_hot_obs=True,
        fully_observed=False,
        render_mode="rgb_array",
        capture_video=True,
        video_dir="videos",
    )
    return environment_config


@pytest.fixture()
def transformer_config() -> TransformerModelConfig:
    transformer_config = TransformerModelConfig(
        d_model=128,
        n_heads=4,
        d_mlp=256,
        n_layers=2,
        n_ctx=2,
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


@pytest.fixture()
def saved_model_path(
    offline_config,
    environment_config,
    transformer_config,
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

    return path


def test_calibration_current(args, saved_model_path):
    args.model_path = saved_model_path
    runner(args)
