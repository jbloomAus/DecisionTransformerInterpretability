
from src.run_ppo import ppo_runner
from src.ppo.utils import PPOArgs


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

    ppo_runner(args)
