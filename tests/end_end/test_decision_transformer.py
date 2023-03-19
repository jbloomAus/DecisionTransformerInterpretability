import pytest
import os
# from src.decision_transformer.utils import DTArgs
from src.config import RunConfig, TransformerModelConfig, EnvironmentConfig, OfflineTrainConfig
from src.run_decision_transformer import run_decision_transformer
from src.environments.environments import make_env


@pytest.fixture
def download_training_data() -> None:
    ''' uses gdown to get data'''
    if not os.path.exists("trajectories/MiniGrid-Dynamic-Obstacles-8x8-v0bd60729d-dc0b-4294-9110-8d5f672aa82c.pkl"):
        os.system("gdown 1UBMuhRrM3aYDdHeJBFdTn1RzXDrCL_sr -O trajectories/MiniGrid-Dynamic-Obstacles-8x8-v0bd60729d-dc0b-4294-9110-8d5f672aa82c.pkl")


def test_decision_transformer(download_training_data):

    run_config = RunConfig(
        exp_name="Test-DT",
        wandb_project_name="DecisionTransformerInterpretability",
        seed=1,
        track=True,
    )

    transformer_model_config = TransformerModelConfig(
        d_model=128,
        n_heads=2,
        d_mlp=256,
        n_layers=1,
        n_ctx=3,
        time_embedding_type="embedding",
        seed=1,
        device="cpu"
    )

    offline_config = OfflineTrainConfig(
        trajectory_path="trajectories/MiniGrid-Dynamic-Obstacles-8x8-v0bd60729d-dc0b-4294-9110-8d5f672aa82c.pkl",
        batch_size=128,
        lr=0.0001,
        weight_decay=0.001,
        pct_traj=1,
        prob_go_from_end=0.1,
        device="cpu",
        track=True,
        train_epochs=500,
        test_epochs=10,
        test_frequency=100,
        eval_frequency=100,
        eval_episodes=10,
        initial_rtg=[-1, 0, 1],
        eval_max_time_steps=1000
    )

    run_decision_transformer(
        run_config=run_config,
        transformer_config=transformer_model_config,
        offline_config=offline_config,
        make_env=make_env)

    print("Test passed! Look at wandb and compare to the previous run.")


def test_clone_transformer(download_training_data):

    run_config = RunConfig(
        exp_name="Test-BC",
        wandb_project_name="DecisionTransformerInterpretability",
        seed=1,
        track=True,

    )

    transformer_model_config = TransformerModelConfig(
        d_model=128,
        n_heads=2,
        d_mlp=256,
        n_layers=1,
        n_ctx=3,
        time_embedding_type="embedding",
        seed=1,
        device="cpu"
    )

    offline_config = OfflineTrainConfig(
        trajectory_path="trajectories/MiniGrid-Dynamic-Obstacles-8x8-v0bd60729d-dc0b-4294-9110-8d5f672aa82c.pkl",
        model_type="clone_transformer",
        batch_size=128,
        lr=0.0001,
        weight_decay=0.001,
        pct_traj=0.10,
        prob_go_from_end=0.1,
        device="cpu",
        track=run_config.track,
        train_epochs=500,
        test_epochs=10,
        test_frequency=100,
        eval_frequency=100,
        eval_episodes=10,
        eval_max_time_steps=1000
    )

    run_decision_transformer(
        run_config=run_config,
        transformer_config=transformer_model_config,
        offline_config=offline_config,
        make_env=make_env)

    print("Test passed! Look at wandb and compare to the previous run.")
