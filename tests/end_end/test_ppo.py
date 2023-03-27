import os
import pytest

import gymnasium as gym

from src.config import (EnvironmentConfig, OnlineTrainConfig, RunConfig,
                        TransformerModelConfig)
from src.ppo.my_probe_envs import Probe1, Probe2, Probe3, Probe4, Probe5, Probe6
from src.ppo.runner import ppo_runner

# load probe envs so we can use them to debug.
for i in range(6):
    probes = [Probe1, Probe2, Probe3, Probe4, Probe5, Probe6]
    gym.envs.registration.register(id=f"Probe{i+1}-v0", entry_point=probes[i])


def test_ppo_runner():
    run_config = RunConfig(
        exp_name="Test-PPO-Basic",
        seed=1,
        cuda=True,
        track=True,
        wandb_project_name="PPO-MiniGrid",
        wandb_entity=None,
    )

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

    online_config = OnlineTrainConfig(
        hidden_size=64,
        total_timesteps=200000,
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
        trajectory_path="trajectories/test/test_ppo.gz",
    )

    agent = ppo_runner(
        run_config=run_config,
        environment_config=environment_config,
        online_config=online_config,
        transformer_model_config=None
    )

    assert os.path.exists(
        "trajectories/test/test_ppo.gz"), "Trajectory file not saved"


@pytest.mark.skip(reason="Traj PPO not working")
def test_ppo_runner_traj_model():
    run_config = RunConfig(
        exp_name="Test-PPO-Traj",
        seed=1,
        cuda=True,
        track=True,
        wandb_project_name="PPO-MiniGrid",
        wandb_entity=None,
    )

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

    online_config = OnlineTrainConfig(
        use_trajectory_model=True,
        hidden_size=64,
        total_timesteps=200000,
        learning_rate=0.00025,
        decay_lr=True,
        num_envs=30,
        num_steps=64,
        gamma=0.99,
        gae_lambda=0.95,
        num_minibatches=30,
        update_epochs=4,
        clip_coef=0.4,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=2,
        trajectory_path=None,
        prob_go_from_end=0.1,
    )

    transformer_model_config = TransformerModelConfig(
        d_model=128,
        n_heads=2,
        d_mlp=256,
        n_layers=1,
        n_ctx=1,
        time_embedding_type="embedding",
        state_embedding_type="grid",
        seed=1,
        device="cpu"
    )

    ppo_runner(
        run_config=run_config,
        environment_config=environment_config,
        online_config=online_config,
        transformer_model_config=transformer_model_config
    )


@pytest.mark.skip(reason="Traj PPO not working")
def test_ppo_runner_traj_model_memory():
    run_config = RunConfig(
        exp_name="Test-PPO-Traj-Memory",
        seed=1,
        cuda=True,
        track=True,
        wandb_project_name="PPO-MiniGrid",
        wandb_entity=None,
    )

    environment_config = EnvironmentConfig(
        # env_id="MiniGrid-RedBlueDoors-6x6-v0",
        env_id="MiniGrid-MemoryS7-v0",
        # env_id="MiniGrid-Dynamic-Obstacles-8x8-v0",
        # env_id="Probe6-v0",
        view_size=7,
        max_steps=50,
        one_hot_obs=False,
        fully_observed=False,
        render_mode="rgb_array",
        capture_video=True,
        video_dir="videos",
    )

    online_config = OnlineTrainConfig(
        use_trajectory_model=True,
        hidden_size=64,
        total_timesteps=200000,
        learning_rate=0.00025,
        decay_lr=True,
        num_envs=14,
        num_steps=256,
        gamma=0.99,
        gae_lambda=0.95,
        num_minibatches=14,
        update_epochs=4,
        clip_coef=0.6,
        ent_coef=0.10,
        vf_coef=0.5,
        max_grad_norm=2,
        trajectory_path=None,
        prob_go_from_end=0.2,
    )

    transformer_model_config = TransformerModelConfig(
        d_model=128,
        n_heads=4,
        d_mlp=256,
        n_layers=2,
        n_ctx=15,
        time_embedding_type="embedding",
        state_embedding_type="grid",
        seed=1,
        device="cpu"
    )

    ppo_runner(
        run_config=run_config,
        environment_config=environment_config,
        online_config=online_config,
        transformer_model_config=transformer_model_config
    )
