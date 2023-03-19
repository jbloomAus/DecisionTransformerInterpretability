# download training data:  gdown 1UBMuhRrM3aYDdHeJBFdTn1RzXDrCL_sr

python -m src.run_decision_transformer \
    --exp_name MiniGrid-Dynamic-Obstacles-8x8-v0-Refactor \
    --trajectory_path trajectories/MiniGrid-Dynamic-Obstacles-8x8-v0bd60729d-dc0b-4294-9110-8d5f672aa82c.pkl \
    --d_model 128 \
    --n_heads 2 \
    --d_mlp 256 \
    --n_layers 1 \
    --learning_rate 0.0001 \
    --batch_size 128 \
    --train_epochs 5000 \
    --test_epochs 10 \
    --n_ctx 3 \
    --pct_traj 1 \
    --weight_decay 0.001 \
    --seed 1 \
    --wandb_project_name DecisionTransformerInterpretability-Dev \
    --test_frequency 1000 \
    --eval_frequency 1000 \
    --eval_episodes 10 \
    --initial_rtg -1 \
    --initial_rtg 0 \
    --initial_rtg 1 \
    --prob_go_from_end 0.1 \
    --eval_max_time_steps 1000 \
    --track True
