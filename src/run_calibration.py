from src.environments.environments import make_env
from src.decision_transformer.utils import load_decision_transformer, model_stored_in_legacy_format, load_model_data
from src.decision_transformer.calibration import calibration_statistics, plot_calibration_statistics
import argparse
import warnings
import numpy as np
import os
import torch as t
import math

# import a  base python logger
import logging

logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(
        prog="Get Calibration of Decision Transformer",
        description="Assess the RTG calibration of a decision transformer")

    parser.add_argument(
        "--env_id", type=str, default="MiniGrid-Dynamic-Obstacles-8x8-v0", help="Environment ID")
    parser.add_argument("--model_path", type=str,
                        default="models/dt.pt", help="Path to model")
    parser.add_argument("--n_trajectories", type=int,
                        default=100, help="Number of trajectories to evaluate")
    parser.add_argument("--initial_rtg_min", type=float,
                        default=-1, help="Minimum initial RTG")
    parser.add_argument("--initial_rtg_max", type=float,
                        default=1, help="Maximum initial RTG")
    parser.add_argument("--initial_rtg_step", type=float,
                        default=0.1, help="Step size for initial RTG")
    args = parser.parse_args()

    logger.info(f"Loading model from {args.model_path}")
    logger.info(f"Using environment {args.env_id}")

    if model_stored_in_legacy_format(args.model_path):
        state_dict = t.load(args.model_path)
        one_hot_encoded = not (
                state_dict["state_encoder.weight"].shape[-1] % 20)  # hack for now

        if one_hot_encoded:
            view_size = int(
                math.sqrt(state_dict["state_encoder.weight"].shape[-1] // 20))
        else:
            view_size = int(
                math.sqrt(state_dict["state_encoder.weight"].shape[-1] // 3))

        max_time_steps = state_dict["time_embedding.weight"].shape[0]
        env_func = make_env(
            args.env_id, seed=1, idx=0, capture_video=False,
            run_name="dev", fully_observed=False, flat_one_hot=one_hot_encoded,
            max_steps=max_time_steps, agent_view_size=args.view_size
        )

        dt = load_decision_transformer(args.model_path, env_func())

        d_model = dt.d_model
        n_heads = dt.n_heads
        d_mlp = dt.d_mlp
        n_ctx = dt.n_ctx
        n_layers = dt.n_layers
        max_timestep = dt.max_timestep
    else:
        state_dict, trajectory_data_set, transformer_config, _ = load_model_data(args.model_path)

        env_func = make_env(
            args.env_id, seed=1, idx=0, capture_video=False,
            run_name="dev", fully_observed=False, flat_one_hot=(trajectory_data_set.observation_type == "one_hot"),
            max_steps=trajectory_data_set.metadata['args']['max_steps'],
            agent_view_size=trajectory_data_set.metadata['args']['view_size']
        )

        dt = load_decision_transformer(args.model_path, env_func())

        d_model = transformer_config.d_model
        n_heads = transformer_config.n_heads
        d_mlp = transformer_config.d_mlp
        n_ctx = transformer_config.n_ctx
        n_layers = transformer_config.n_layers
        max_timestep = trajectory_data_set.metadata['args']['max_steps']

    warnings.filterwarnings("ignore", category=UserWarning)
    statistics = calibration_statistics(
        dt,
        args.env_id,
        env_func,
        initial_rtg_range=np.linspace(args.initial_rtg_min, args.initial_rtg_max, int(
            (args.initial_rtg_max - args.initial_rtg_min) / args.initial_rtg_step)),
        trajectories=args.n_trajectories
    )

    fig = plot_calibration_statistics(statistics, show_spread=True, CI=0.95)
    # add all the hyperparameters to the title (env id, d_model, n_heads, d_mlp, n_ctx, n_layers, max_timestep, layernorm)
    # make font title smaller
    fig.update_layout(
        title=f"{args.env_id} - d_model: {d_model} - n_heads: {n_heads} - d_mlp: {d_mlp} - n_ctx: {n_ctx} "
              f"- n_layers: {n_layers} - max_timestep: {max_timestep}",
        title_font_size=10
    )

    if not os.path.exists("figures"):
        os.mkdir("figures")

    # format the output path according to the input path.
    args.output_path = f"figures/{args.model_path.split('/')[-1].split('.')[0]}_calibration.png"
    fig.write_image(args.output_path)
