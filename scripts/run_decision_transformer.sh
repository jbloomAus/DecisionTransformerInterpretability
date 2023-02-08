python src/run_decision_transformer.py \
    --exp_name "MiniGrid-DoorKey-8x8" \
    --trajectory_path "trajectories/MiniGrid-DoorKey-8x8-v00fe27f46-3e0c-4e13-a2a6-d58b36c7a236.gz" \
    --d_model 128 \
    --n_heads 4 \
    --d_mlp 256 \
    --n_layers 2 \
    --learning_rate 0.0001 \
    --batch_size 128 \
    --batches 15001 \
    --n_ctx 3 \
    --pct_traj 1 \
    --weight_decay 0.1 \
    --seed 1 \
    --wandb_project_name "DecisionTransformerInterpretability" \
    --test_frequency 100 \./sdcr
    --test_batches 10 \
    --eval_frequency 100 \
    --eval_episodes 10 \
    --initial_rtg 1 \
    --prob_go_from_end 0.01 \
    --eval_max_time_steps 500 \
    --cuda True \
    --track True
    # --linear_time_embedding \

# python src/run_decision_transformer.py \
#     --exp_name "MiniGrid-DoorKey-8x8" \
#     --trajectory_path "trajectories/MiniGrid-DoorKey-8x8-v0fefa0263-af92-4438-83b2-37295b29ea50.xz" \
#     --d_model 64 \
#     --n_heads 2 \
#     --d_mlp 256 \
#     --n_layers 1 \
#     --learning_rate 0.0001 \
#     --batch_size 16 \
#     --batches 1000 \
#     --n_ctx 3 \
#     --pct_traj 1 \
#     --weight_decay 0.01 \
#     --seed 1 \
#     --wandb_project_name "DecisionTransformerInterpretability" \
#     --test_frequency 100 \
#     --test_batches 10 \
#     --eval_frequency 100 \
#     --eval_episodes 10 \
#     --initial_rtg 0.9 \
#     --prob_go_from_end 0.0 \
#     --eval_max_time_steps 40 \
#     --cuda False
#     # --track True
#     # --linear_time_embedding \
