python -m src.run_calibration \
    --env_id "MiniGrid-Dynamic-Obstacles-8x8-v0" \
    --model_path "models/MiniGrid-MemoryS7FixedStart-v0/WorkingModel.pt" \
    --n_trajectories 20 \
    --initial_rtg_min 0 \
    --initial_rtg_max 1 \
    --initial_rtg_step 0.005
