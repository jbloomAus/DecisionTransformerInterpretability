# @dataclass
# class PPOArgs:
#     exp_name: str = 'MiniGrid-Dynamic-Obstacles-8x8-v0'
#     seed: int = 1
#     cuda: bool = True
#     track: bool = True
#     wandb_project_name: str = "PPO-MiniGrid"
#     wandb_entity: str = None
#     capture_video: bool = True
#     env_id: str = 'MiniGrid-Dynamic-Obstacles-8x8-v0'
#     total_timesteps: int = 1000000
#     learning_rate: float = 0.00025
#     num_envs: int = 30
#     num_steps: int = 248
#     gamma: float = 0.99
#     gae_lambda: float = 0.95
#     num_minibatches: int = 30
#     update_epochs: int = 4
#     clip_coef: float = 0.4
#     ent_coef: float = 0.01
#     vf_coef: float = 0.5
#     max_grad_norm: float = 2
#     max_steps: int = 2000
#     trajectory_path: str = None
#     fully_observed: bool = False

python src/run_ppo.py --exp_name "MiniGrid-Dynamic-Obstacles-8x8-v0" \
    --seed 1 \
    --cuda \
    --track \
    --wandb_project_name "PPO-MiniGrid" \
    --capture_video \
    --env_id "MiniGrid-Dynamic-Obstacles-8x8-v0" \
    --total_timesteps 200000 \
    --learning_rate 0.00025 \
    --num_envs 30 \
    --num_steps 64 \
    --gamma 0.99 \
    --gae_lambda 0.95 \
    --num_minibatches 30 \
    --update_epochs 4 \
    --clip_coef 0.4 \
    --ent_coef 0.25 \
    --vf_coef 0.5 \
    --max_grad_norm 2 \
    --max_steps 300 \
    --one_hot_obs
    # --trajectory_path None 
    # --fully_observed False
