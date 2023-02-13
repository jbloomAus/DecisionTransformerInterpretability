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
#     num_minibatches: int = 30
#     update_epochs: int = 4
#     clip_coef: float = 0.4
#     ent_coef: float = 0.01
#     vf_coef: float = 0.5
#     max_steps: int = 2000
#     trajectory_path: str = None
#     fully_observed: bool = False

python src/run_ppo.py --exp_name "MiniGrid-DoorKey-16x16-v0" \
    --seed 1 \
    --cuda \
    --track \
    --wandb_project_name "PPO-MiniGrid" \
    --env_id "MiniGrid-DoorKey-8x8-v0" \
    --view_size 5 \
    --total_timesteps 350000 \
    --learning_rate 0.00025 \
    --num_envs 4 \
    --num_steps 128 \
    --num_minibatches 4 \
    --update_epochs 4 \
    --clip_coef 0.2 \
    --ent_coef 0.01 \
    --vf_coef 0.5 \
    --max_steps 1000 \
    --one_hot_obs
