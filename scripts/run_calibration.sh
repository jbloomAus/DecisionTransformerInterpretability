python -m src.run_calibration \
    --env_id "MiniGrid-Dynamic-Obstacles-8x8-v0" \
    --model_path "artifacts/MiniGrid-Dynamic-Obstacles-8x8-v0__MiniGrid-Dynamic-Obstacles-8x8-v0__1__1675306594:v0/MiniGrid-Dynamic-Obstacles-8x8-v0__MiniGrid-Dynamic-Obstacles-8x8-v0__1__1675306594.pt" \
    --n_trajectories 20 \
    --initial_rtg_min -1 \
    --initial_rtg_max 1 \
    --initial_rtg_step 0.05
