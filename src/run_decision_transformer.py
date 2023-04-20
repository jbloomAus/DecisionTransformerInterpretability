"""
This file is the entry point for running the decision transformer.
"""
import torch as t

from .config import OfflineTrainConfig, RunConfig, TransformerModelConfig
from .decision_transformer.runner import run_decision_transformer
from .decision_transformer.utils import parse_args
from .environments.environments import make_env

if __name__ == "__main__":
    args = parse_args()

    run_config = RunConfig(
        exp_name=args.exp_name,
        seed=args.seed,
        device="cuda" if args.cuda and t.cuda.is_available() else "cpu",
        track=args.track,
        wandb_project_name=args.wandb_project_name,
        wandb_entity=args.wandb_entity,

    )

    TIME_EMBEDDING_TYPE = (
        "linear" if args.linear_time_embedding else "embedding"
    )

    transformer_model_config = TransformerModelConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_mlp=args.d_mlp,
        n_layers=args.n_layers,
        layer_norm=args.layer_norm,
        time_embedding_type=TIME_EMBEDDING_TYPE,
        n_ctx=args.n_ctx,
        device=run_config.device
    )

    offline_config = OfflineTrainConfig(
        model_type=args.model_type,
        trajectory_path=args.trajectory_path,
        pct_traj=args.pct_traj,
        train_epochs=args.train_epochs,
        test_epochs=args.test_epochs,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        test_frequency=args.test_frequency,
        eval_frequency=args.eval_frequency,
        eval_episodes=args.eval_episodes,
        eval_num_envs=args.eval_num_envs,
        initial_rtg=args.initial_rtg,
        prob_go_from_end=args.prob_go_from_end,
        eval_max_time_steps=args.eval_max_time_steps,
        track=args.track,
        convert_to_one_hot=args.convert_to_one_hot,
        device=run_config.device
    )

    run_decision_transformer(
        run_config=run_config,
        transformer_config=transformer_model_config,
        offline_config=offline_config,
        make_env=make_env,
    )
