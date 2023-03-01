from decision_transformer.runner import run_decision_transformer
from decision_transformer.utils import DTArgs, parse_args
from environments.environments import make_env

if __name__ == "__main__":

    args = parse_args()
    args = DTArgs(
        exp_name=args.exp_name,
        seed=args.seed,
        cuda=args.cuda,
        track=args.track,
        wandb_project_name=args.wandb_project_name,
        wandb_entity=args.wandb_entity,
        trajectory_path=args.trajectory_path,
        pct_traj=args.pct_traj,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_mlp=args.d_mlp,
        n_layers=args.n_layers,
        layer_norm=args.layer_norm,
        linear_time_embedding=args.linear_time_embedding,
        n_ctx=args.n_ctx,
        train_epochs=args.train_epochs,
        test_epochs=args.test_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        test_frequency=args.test_frequency,
        eval_frequency=args.eval_frequency,
        eval_episodes=args.eval_episodes,
        initial_rtg=args.initial_rtg,
        prob_go_from_end=args.prob_go_from_end,
        eval_max_time_steps=args.eval_max_time_steps
    )

    run_decision_transformer(args, make_env)
