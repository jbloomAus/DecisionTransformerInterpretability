python src/run_calibration.py \
    --env_id "MiniGrid-Dynamic-Obstacles-8x8-v0" \
    --model_path "artifacts/MiniGrid-Dynamic-Obstacles-8x8-v0__MiniGrid-Dynamic-Obstacles-8x8-v0__1__1673546242:v0/MiniGrid-Dynamic-Obstacles-8x8-v0__MiniGrid-Dynamic-Obstacles-8x8-v0__1__1673546242.pt" \
    --n_trajectories 1000 \
    --initial_rtg_min -1 \
    --initial_rtg_max 1 \
    --initial_rtg_step 0.1 