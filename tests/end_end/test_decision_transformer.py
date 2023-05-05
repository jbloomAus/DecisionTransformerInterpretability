import os

import pytest

from src.config import (
    EnvironmentConfig,
    OfflineTrainConfig,
    RunConfig,
    TransformerModelConfig,
)
from src.environments.environments import make_env
from src.run_decision_transformer import run_decision_transformer


@pytest.fixture
def download_training_data() -> None:
    """uses gdown to get data"""
    if not os.path.exists(
        "trajectories/MiniGrid-Dynamic-Obstacles-8x8-v0bd60729d-dc0b-4294-9110-8d5f672aa82c.pkl"
    ):
        os.system("pip show gdown || pip install gdown")
        os.system(
            "gdown 1UBMuhRrM3aYDdHeJBFdTn1RzXDrCL_sr -O trajectories/MiniGrid-Dynamic-Obstacles-8x8-v0bd60729d-dc0b-4294-9110-8d5f672aa82c.pkl"
        )


@pytest.fixture
def transformer_model_config():
    transformer_model_config = TransformerModelConfig(
        d_model=128,
        n_heads=2,
        d_mlp=256,
        n_layers=1,
        n_ctx=2,
        time_embedding_type="embedding",
        seed=1,
        device="cpu",
    )
    return transformer_model_config


@pytest.fixture
def offline_config():
    offline_config = OfflineTrainConfig(
        trajectory_path="trajectories/MiniGrid-Dynamic-Obstacles-8x8-v0bd60729d-dc0b-4294-9110-8d5f672aa82c.pkl",
        batch_size=128,
        lr=0.001,
        weight_decay=0.001,
        pct_traj=1,
        prob_go_from_end=0.1,
        device="cpu",
        track=False,
        train_epochs=500,
        test_epochs=10,
        test_frequency=100,
        eval_frequency=100,
        eval_episodes=10,
        initial_rtg=[-1, 0, 1],
        eval_max_time_steps=50,
        model_type="decision_transformer",
    )
    return offline_config


@pytest.mark.parametrize("n_ctx", [2, 5, 8])
def test_decision_transformer(
    download_training_data, n_ctx, transformer_model_config, offline_config
):
    run_config = RunConfig(
        exp_name="Test-DT-n_ctx" + str(n_ctx),
        wandb_project_name="DecisionTransformerInterpretability",
        seed=1,
        track=True,
    )

    transformer_model_config.n_ctx = n_ctx
    offline_config.track = run_config.track

    run_decision_transformer(
        run_config=run_config,
        transformer_config=transformer_model_config,
        offline_config=offline_config,
        make_env=make_env,
    )

    print("Test passed! Look at wandb and compare to the previous run.")


@pytest.mark.parametrize("n_ctx", [2, 5, 8])
def test_decision_transformer_preln(
    download_training_data, n_ctx, transformer_model_config, offline_config
):
    run_config = RunConfig(
        exp_name="Test-DT-LNPre-n_ctx-" + str(n_ctx),
        wandb_project_name="DecisionTransformerInterpretability",
        seed=1,
        track=False,
    )

    transformer_model_config.n_ctx = n_ctx
    transformer_model_config.layer_norm = "LNPre"
    offline_config.track = run_config.track

    run_decision_transformer(
        run_config=run_config,
        transformer_config=transformer_model_config,
        offline_config=offline_config,
        make_env=make_env,
    )

    print("Test passed! Look at wandb and compare to the previous run.")


@pytest.mark.parametrize("optimizer", ["sgd", "adam", "adamw"])
def test_decision_transformer_optimizer(
    download_training_data, optimizer, transformer_model_config, offline_config
):
    run_config = RunConfig(
        exp_name="Test-DT-n_ctx-" + str(2) + "-" + optimizer,
        wandb_project_name="DecisionTransformerInterpretability",
        seed=1,
        track=True,
    )

    transformer_model_config.layer_norm = "LNPre"
    offline_config.optimizer = optimizer
    offline_config.track = run_config.track

    run_decision_transformer(
        run_config=run_config,
        transformer_config=transformer_model_config,
        offline_config=offline_config,
        make_env=make_env,
    )

    print("Test passed! Look at wandb and compare to the previous run.")


@pytest.mark.parametrize(
    "scheduler",
    [
        "Constant",
        "ConstantWithWarmUp",
        "LinearWarmUpDecay",
        "CosineAnnealing",
        "CosineAnnealingWarmUp",
        "CosineAnnealingWarmRestarts",
    ],
)
def test_decision_transformer_scheduler(
    download_training_data, scheduler, transformer_model_config, offline_config
):
    run_config = RunConfig(
        exp_name="Test-DT-n_ctx-" + str(2) + "-" + scheduler,
        wandb_project_name="DecisionTransformerInterpretability",
        seed=1,
        track=True,
    )

    # Interesting to run this with/without LNPre
    transformer_model_config.layer_norm = "LNPre"
    offline_config.scheduler = scheduler
    offline_config.track = run_config.track

    run_decision_transformer(
        run_config=run_config,
        transformer_config=transformer_model_config,
        offline_config=offline_config,
        make_env=make_env,
    )

    print("Test passed! Look at wandb and compare to the previous run.")


@pytest.mark.parametrize(
    "activation_fn",
    ["relu", "gelu", "silu", "gelu_new", "solu_ln", "gelu_fast"],
)
def test_decision_transformer_activation_fn(
    download_training_data,
    activation_fn,
    transformer_model_config,
    offline_config,
):
    run_config = RunConfig(
        exp_name="Test-DT-n_ctx-" + str(2) + "-" + activation_fn,
        wandb_project_name="DecisionTransformerInterpretability",
        seed=1,
        track=True,
    )

    # Interesting to run this with/without LNPre
    transformer_model_config.layer_norm = "LNPre"
    transformer_model_config.activation_fn = activation_fn
    offline_config.track = run_config.track

    run_decision_transformer(
        run_config=run_config,
        transformer_config=transformer_model_config,
        offline_config=offline_config,
        make_env=make_env,
    )

    print("Test passed! Look at wandb and compare to the previous run.")


@pytest.mark.parametrize("gated_mlp", [False, True])
def test_decision_transformer_gated_mlp(
    download_training_data, gated_mlp, transformer_model_config, offline_config
):
    run_config = RunConfig(
        exp_name="Test-DT-n_ctx-" + str(2) + "-GatedMLP-" + str(gated_mlp),
        wandb_project_name="DecisionTransformerInterpretability",
        seed=1,
        track=True,
    )

    # Interesting to run this with/without LNPre
    transformer_model_config.layer_norm = "LNPre"
    transformer_model_config.activation_fn = "gelu"
    transformer_model_config.gated_mlp = gated_mlp
    offline_config.track = run_config.track

    run_decision_transformer(
        run_config=run_config,
        transformer_config=transformer_model_config,
        offline_config=offline_config,
        make_env=make_env,
    )

    print("Test passed! Look at wandb and compare to the previous run.")


@pytest.mark.parametrize("state_embedding_type", ["CNN", "ViT", "Grid"])
def test_decision_transformer_state_embedding(
    download_training_data,
    state_embedding_type,
    transformer_model_config,
    offline_config,
):
    run_config = RunConfig(
        exp_name="Test-DT-n_ctx-"
        + str(2)
        + "-state_type-"
        + state_embedding_type,
        wandb_project_name="DecisionTransformerInterpretability",
        seed=1,
        track=True,
    )

    transformer_model_config.state_embedding_type = state_embedding_type
    offline_config.track = run_config.track

    run_decision_transformer(
        run_config=run_config,
        transformer_config=transformer_model_config,
        offline_config=offline_config,
        make_env=make_env,
    )

    print("Test passed! Look at wandb and compare to the previous run.")


@pytest.mark.parametrize("n_ctx", [1, 3, 9])
def test_clone_transformer(download_training_data, n_ctx):
    run_config = RunConfig(
        exp_name="Test-BC-n_ctx" + str(n_ctx),
        wandb_project_name="DecisionTransformerInterpretability",
        seed=1,
        track=True,
        device="cpu",
    )

    transformer_model_config = TransformerModelConfig(
        d_model=128,
        n_heads=2,
        d_mlp=256,
        n_layers=1,
        n_ctx=n_ctx,  # see current state, previous action and state
        time_embedding_type="embedding",
        seed=1,
        device="cpu",
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
        eval_max_time_steps=1000,
    )

    run_decision_transformer(
        run_config=run_config,
        transformer_config=transformer_model_config,
        offline_config=offline_config,
        make_env=make_env,
    )

    print("Test passed! Look at wandb and compare to the previous run.")


# use this test when wanting to debug a specific script
def test_decision_transformer_bespoke():
    run_config = RunConfig(
        exp_name="MiniGrid-MemoryS7FixedStart-v0",
        wandb_project_name="DecisionTransformerInterpretability",
        seed=1,
        track=True,
    )

    transformer_model_config = TransformerModelConfig(
        d_model=128,
        n_heads=4,
        d_mlp=256,
        n_layers=2,
        n_ctx=38,
        time_embedding_type="embedding",
        seed=1,
        device="cpu",
    )

    offline_config = OfflineTrainConfig(
        trajectory_path="trajectories/MiniGrid-MemoryS7FixedStart-v0-Checkpoint11.gz",
        batch_size=32,
        lr=0.0001,
        weight_decay=0.01,
        pct_traj=1,
        prob_go_from_end=0.5,
        device="cpu",
        track=run_config.track,
        train_epochs=30,
        test_epochs=1,
        test_frequency=3,
        eval_frequency=3,
        eval_episodes=10,
        initial_rtg=[0, 1],
        eval_max_time_steps=50,
        model_type="decision_transformer",
        eval_num_envs=16,
        convert_to_one_hot=True,
    )

    run_decision_transformer(
        run_config=run_config,
        transformer_config=transformer_model_config,
        offline_config=offline_config,
        make_env=make_env,
    )
