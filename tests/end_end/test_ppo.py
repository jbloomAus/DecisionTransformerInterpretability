
from src.ppo.runner import ppo_runner
from src.ppo.utils import PPOArgs
from src.config import EnvironmentConfig, RunConfig, OnlineTrainConfig


def test_ppo_runner():
    args = PPOArgs(
        exp_name="Test",
        seed=1,
        cuda=True,
        track=True,
        wandb_project_name="PPO-MiniGrid",
        # wandb_entity=None,
        capture_video=True,
        env_id="MiniGrid-Dynamic-Obstacles-8x8-v0",
        view_size=3,
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
        max_steps=300,
        one_hot_obs=True,
        trajectory_path=None,
        fully_observed=False,
    )

    run_config = RunConfig(
        exp_name=args.exp_name,
        seed=args.seed,
        cuda=args.cuda,
        track=args.track,
        wandb_project_name=args.wandb_project_name,
        wandb_entity=args.wandb_entity,
    )

    environment_config = EnvironmentConfig(
        env_id=args.env_id,
        view_size=args.view_size,
        max_steps=args.max_steps,
        one_hot_obs=args.one_hot_obs,
        fully_observed=args.fully_observed,
        render_mode="rgb_array",
        capture_video=args.capture_video,
        video_dir="videos",
    )

    model_config = None

    online_config = OnlineTrainConfig(
        hidden_size=64,
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        decay_lr=args.decay_lr,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        num_minibatches=args.num_minibatches,
        update_epochs=args.update_epochs,
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        trajectory_path=args.trajectory_path,
    )

    ppo_runner(
        run_config=run_config,
        environment_config=environment_config,
        online_config=online_config,
        transformer_model_config=model_config
    )
