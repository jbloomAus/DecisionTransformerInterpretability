import argparse
import logging
import math
import os
import warnings

import numpy as np
import torch as t

from src.config import EnvironmentConfig
from src.decision_transformer.calibration import (
    calibration_statistics,
    plot_calibration_statistics,
)
from src.decision_transformer.utils import load_decision_transformer
from src.environments.environments import make_env

logging.basicConfig(level=logging.INFO)


def runner(args):
    logger = logging.getLogger(__name__)

    logger.info(f"Loading model from {args.model_path}")
    logger.info(f"Using environment {args.env_id}")

    dt = load_decision_transformer(args.model_path)
    env_func = make_env(
        config=dt.environment_config, seed=1, idx=0, run_name="dev"
    )
    transformer_config = dt.transformer_config

    d_model = transformer_config.d_model
    n_heads = transformer_config.n_heads
    d_mlp = transformer_config.d_mlp
    n_ctx = transformer_config.n_ctx
    n_layers = transformer_config.n_layers
    max_timestep = dt.environment_config.max_steps

    warnings.filterwarnings("ignore", category=UserWarning)
    statistics = calibration_statistics(
        dt,
        args.env_id,
        env_func,
        initial_rtg_range=np.linspace(
            args.initial_rtg_min,
            args.initial_rtg_max,
            int(
                (args.initial_rtg_max - args.initial_rtg_min)
                / args.initial_rtg_step
            ),
        ),
        trajectories=args.n_trajectories,
        num_envs=args.num_envs,
    )

    fig = plot_calibration_statistics(statistics, show_spread=True, CI=0.95)

    # make font title smaller
    fig.update_layout(
        title=f"{args.env_id} - d_model: {d_model} - n_heads: {n_heads} - d_mlp: {d_mlp} - n_ctx: {n_ctx} "
        f"- n_layers: {n_layers} - max_timestep: {max_timestep}",
        title_font_size=10,
    )

    if not os.path.exists("figures"):
        os.mkdir("figures")

    # format the output path according to the input path.
    args.output_path = f"figures/{args.model_path.split('/')[-1].split('.')[0]}_calibration.png"
    fig.write_image(args.output_path)


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(
        prog="Get Calibration of Decision Transformer",
        description="Assess the RTG calibration of a decision transformer",
    )

    parser.add_argument(
        "--env_id",
        type=str,
        default="MiniGrid-Dynamic-Obstacles-8x8-v0",
        help="Environment ID",
    )
    parser.add_argument(
        "--model_path", type=str, default="models/dt.pt", help="Path to model"
    )
    parser.add_argument(
        "--n_trajectories",
        type=int,
        default=100,
        help="Number of trajectories to evaluate",
    )
    parser.add_argument(
        "--initial_rtg_min", type=float, default=-1, help="Minimum initial RTG"
    )
    parser.add_argument(
        "--initial_rtg_max", type=float, default=1, help="Maximum initial RTG"
    )
    parser.add_argument(
        "--initial_rtg_step",
        type=float,
        default=0.1,
        help="Step size for initial RTG",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=8,
        help="How many environments to run in parallel",
    )
    args = parser.parse_args()

    logger.info(f"Loading model from {args.model_path}")
    logger.info(f"Using environment {args.env_id}")

    runner(args)
