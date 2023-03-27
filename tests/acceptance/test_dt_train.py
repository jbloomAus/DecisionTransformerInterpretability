import pytest
import torch
from src.config import TransformerModelConfig, EnvironmentConfig
from src.models.trajectory_model import DecisionTransformer
from src.environments.environments import make_env
from src.decision_transformer.train import evaluate_dt_agent
from src.decision_transformer.offline_dataset import TrajectoryDataset

# need an agent.

# def test_train():
#     pass

# def test_test():
#     pass


def test_evaluate_dt_agent():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trajectory_path = "tests/fixtures/test_trajectories.pkl"
    trajectory_data_set = TrajectoryDataset(
        trajectory_path,
        pct_traj=1, device=device)

    env_id = trajectory_data_set.metadata['args']['env_id']
    env = make_env(env_id, seed=1, idx=0, capture_video=False,
                   run_name="dev", fully_observed=False, max_steps=30)
    env = env()

    # dt = DecisionTransformer(
    #     env=env,
    #     d_model=128,
    #     n_heads=4,
    #     d_mlp=256,
    #     n_layers=2,
    #     state_embedding_type="grid",  # hard-coded for now to minigrid.
    #     max_timestep=1000,
    #     n_ctx=3,  # one timestep of context
    #     device="cuda" if torch.cuda.is_available() else "cpu",
    # )  # Our DT must have a context window large enough

    dt = DecisionTransformer(
        environment_config=EnvironmentConfig(
            env_id=env_id,
            one_hot_obs=trajectory_data_set.observation_type == "one_hot",
            view_size=7,  # trajectory_data_set.metadata['args']['view_size'],
            fully_observed=False,
            capture_video=False,
            render_mode='rgb_array',
            max_steps=1000),
        transformer_config=TransformerModelConfig(
            d_model=128,
            n_heads=4,
            d_mlp=256,
            n_layers=2,
            state_embedding_type="grid",  # hard-coded for now to minigrid.
            # max_timestep=1000,
            n_ctx=2,  # one timestep of context
            device="cuda" if torch.cuda.is_available() else "cpu",
        ))

    dt = dt.to(device)

    if hasattr(dt, "environment_config"):
        max_steps = min(dt.environment_config.max_steps, 10)
    else:
        max_steps = min(dt.max_timestep, 10)

    batch = 0
    eval_env_func = make_env(
        env_id=env.spec.id,
        seed=batch,
        idx=0,
        capture_video=True,
        max_steps=max_steps,
        run_name=f"dt_eval_videos_{batch}",
        fully_observed=False,
        flat_one_hot=(trajectory_data_set.observation_type == "one_hot"),
    )

    statistics = evaluate_dt_agent(
        env_id=env_id,
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
