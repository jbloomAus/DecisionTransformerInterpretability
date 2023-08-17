# download training data:  gdown 1UBMuhRrM3aYDdHeJBFdTn1RzXDrCL_sr

# python -m src.run_decision_transformer \
#     --exp_name MiniGrid-Dynamic-Obstacles-8x8-v0-Refactor \
#     --trajectory_path trajectories/MiniGrid-Dynamic-Obstacles-8x8-v0bd60729d-dc0b-4294-9110-8d5f672aa82c.pkl \
#     --d_model 128 \
#     --n_heads 2 \
#     --d_mlp 256 \
#     --n_layers 1 \
#     --learning_rate 0.0001 \
#     --batch_size 128 \
#     --train_epochs 5000 \
#     --test_epochs 10 \
#     --n_ctx 3 \
#     --pct_traj 1 \
#     --weight_decay 0.001 \
#     --seed 1 \
#     --wandb_project_name DecisionTransformerInterpretability-Dev \
#     --test_frequency 1000 \
#     --eval_frequency 1000 \
#     --eval_episodes 10 \
#     --initial_rtg -1 \
#     --initial_rtg 0 \
#     --initial_rtg 1 \
#     --prob_go_from_end 0.1 \
#     --eval_max_time_steps 1000 \
#     --track


python -m src.run_decision_transformer \
    --exp_name MiniGrid-MemoryS7FixedStart-v0 \
    --trajectory_path trajectories/MiniGrid-MemoryS7FixedStart-v0-Checkpoint6.gz \
    --d_model 128 \
    --n_heads 4 \
    --d_mlp 256 \
    --n_layers 3 \
    --learning_rate 0.0001 \
    --batch_size 32 \
    --train_epochs 30 \
    --test_epochs 1 \
    --n_ctx 26 \
    --pct_traj 1 \
    --weight_decay 0.01 \
    --seed 1 \
    --wandb_project_name DecisionTransformerInterpretability \
    --test_frequency 1 \
    --eval_frequency 1 \
    --eval_episodes 10 \
    --initial_rtg 0 \
    --initial_rtg 1 \
    --prob_go_from_end 0.3 \
    --eval_max_time_steps 50 \
    --eval_num_envs 16 \
    --state_embedding grid \
    --layer_norm LNPre \
    --scheduler CosineAnnealingWarmup \
    --warm_up_steps 1000 \
    --track

# src.run_decision_transformer \
#     --exp_name MiniGrid-MemoryS7FixedStart-v0 \
#     --trajectory_path trajectories/MiniGrid-MemoryS7FixedStart-v0-Checkpoint6.gz \
#     --d_model 128 --n_heads 4 --d_mlp 256 --n_layers 3 \
#     --learning_rate 0.0001 --batch_size 32 --train_epochs 200 \
#     --test_epochs 3 --n_ctx 26 --pct_traj 1 --weight_decay 0.01 \
#     --seed 1 --wandb_project_name DecisionTransformerInterpretability \
#     --test_frequency 10 --eval_frequency 10 --eval_episodes 5 --initial_rtg 0 \
#     --initial_rtg 1 --prob_go_from_end 0.3 --eval_max_time_steps 50 --eval_num_envs 16 --convert_to_one_hot --track



    # --convert_to_one_hot \
    # --trajectory_path trajectories/MiniGrid-MemoryS7FixedStart-v0-Checkpoint6.gz \
# python -m src.run_decision_transformer \
#     --exp_name Memory-VariedSamplingStrategies \
#     --trajectory_path trajectories/MiniGrid-MemoryS7FixedStart-v0-Checkpoint11-VariedSamplingStrategies.gz \
#     --d_model 128 \
#     --n_heads 8 \
#     --d_mlp 512 \
#     --n_layers 2 \
#     --learning_rate 0.0001 \
#     --batch_size 128 \
#     --train_epochs 90 \
#     --test_epochs 1 \
#     --n_ctx 26 \
#     --pct_traj 1 \
#     --weight_decay 0.001 \
#     --seed 1 \
#     --wandb_project_name DecisionTransformerInterpretability \
#     --test_frequency 3 \
#     --eval_frequency 3 \
#     --eval_episodes 10 \
#     --initial_rtg 0 \
#     --initial_rtg 1 \
#     --prob_go_from_end 0.1 \
#     --eval_max_time_steps 50 \
#     --eval_num_envs 10 \
#     --convert_to_one_hot \
#     --track
