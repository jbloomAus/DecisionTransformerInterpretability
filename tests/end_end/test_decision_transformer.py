import pytest
from src.decision_transformer.utils import DTArgs
from src.run_decision_transformer import run_decision_transformer
from src.environments.environments import make_env


def test_decision_transformer():

    args = DTArgs(
        exp_name="Test",
        trajectory_path="trajectories/MiniGrid-Dynamic-Obstacles-8x8-v0bd60729d-dc0b-4294-9110-8d5f672aa82c.pkl",
        d_model=128,
        n_heads=2,
        d_mlp=256,
        n_layers=1,
        learning_rate=0.0001,
        batch_size=128,
        train_epochs=500,
        test_epochs=10,
        n_ctx=3,
        pct_traj=1,
        weight_decay=0.001,
        seed=1,
        linear_time_embedding=False,
        wandb_project_name="DecisionTransformerInterpretability",
        test_frequency=100,
        eval_frequency=100,
        eval_episodes=10,
        initial_rtg=[-1, 0, 1],
        prob_go_from_end=0.1,
        eval_max_time_steps=1000,
        track=True
    )

    run_decision_transformer(args, make_env)

    print("Test passed! Look at wandb and compare to the previous run.")
