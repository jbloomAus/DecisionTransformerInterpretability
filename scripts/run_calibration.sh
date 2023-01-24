python src/run_calibration.py \
    --env_id "MiniGrid-Dynamic-Obstacles-8x8-v0" \
    --model_path "models/demo_model_overnight_training.pt" \
    --n_trajectories 500 \
    --initial_rtg_min -1 \
    --initial_rtg_max 1 \
    --initial_rtg_step 0.05 