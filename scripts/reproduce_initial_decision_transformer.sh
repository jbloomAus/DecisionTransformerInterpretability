# creating the training data (from a ppo run)
# if you don't have enough CPU's you won't get a successful run. There are other hyper par configs that work for fewer CPU's but weren't what was used in the post.

# generate the trajectory data
/home/src/run_ppo.py --exp_name MiniGrid-Dynamic-Obstacles-8x8-v0 --seed 1 --cuda --track --wandb_project_name PPO-MiniGrid --capture_video --env_id MiniGrid-Dynamic-Obstacles-8x8-v0 --total_timesteps 200000 --learning_rate 0.00025 --num_envs 30 --num_steps 64 --gamma 0.99 --gae_lambda 0.95 --num_minibatches 30 --update_epochs 4 --clip_coef 0.4 --ent_coef 0.25 --vf_coef 0.5 --max_grad_norm 2 --max_steps 300

## Creating the Decision Transformer

# get the original training data:  gdown 1UBMuhRrM3aYDdHeJBFdTn1RzXDrCL_sr

# Decent length training run that gets performance close to the paper
python src/run_decision_transformer.py --exp_name Test --trajectory_path trajectories/MiniGrid-Dynamic-Obstacles-8x8-v0bd60729d-dc0b-4294-9110-8d5f672aa82c.pkl --d_model 128 --n_heads 2 --d_mlp 256 --n_layers 1 --learning_rate 0.0001 --batch_size 128 --train_epochs 5000 --test_epochs 10 --n_ctx 3 --pct_traj 1 --weight_decay 0.001 --seed 1 --wandb_project_name DecisionTransformerInterpretability --test_frequency 1000 --eval_frequency 1000 --eval_episodes 10 --initial_rtg 1 --prob_go_from_end 0.1 --eval_max_time_steps 1000 --track True

# Shorter Version for testing. Should still see successful agent.
python -m src.run_decision_transformer --exp_name Test --trajectory_path trajectories/MiniGrid-Dynamic-Obstacles-8x8-v0bd60729d-dc0b-4294-9110-8d5f672aa82c.pkl --d_model 128 --n_heads 2 --d_mlp 256 --n_layers 1 --learning_rate 0.0001 --batch_size 128 --train_epochs 500 --test_epochs 10 --n_ctx 3 --pct_traj 1 --weight_decay 0.001 --seed 1 --wandb_project_name DecisionTransformerInterpretability --test_frequency 100 --eval_frequency 100 --eval_episodes 10 --initial_rtg 1 --prob_go_from_end 0.1 --eval_max_time_steps 1000 --track True
