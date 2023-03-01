import torch as t
import warnings
import wandb
import time
import os

from typing import Callable


from .utils import DTArgs
from .model import DecisionTransformer
from .offline_dataset import TrajectoryDataset, TrajectoryVisualizer
from .train import train


def run_decision_transformer(args: DTArgs, make_env: Callable):
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    if args.cuda:
        device = t.device("cuda" if t.cuda.is_available() else "cpu")
    else:
        device = t.device("cpu")

    if args.trajectory_path is None:
        raise ValueError("Must specify a trajectory path.")

    trajectory_data_set = TrajectoryDataset(
        trajectory_path=args.trajectory_path,
        max_len=args.n_ctx // 3,
        pct_traj=args.pct_traj,
        prob_go_from_end=args.prob_go_from_end,
        device=device
    )

    # make an environment
    env_id = trajectory_data_set.metadata['args']['env_id']
    # pretty print the metadata
    print(trajectory_data_set.metadata)

    if not "view_size" in trajectory_data_set.metadata['args']:
        trajectory_data_set.metadata['args']['view_size'] = 7

    env = make_env(
        env_id,
        seed=0,
        idx=0,
        capture_video=False,
        run_name="dev",
        fully_observed=False,
        # detect if we are using flat one-hot observations.
        flat_one_hot=(trajectory_data_set.observation_type == "one_hot"),
        agent_view_size=trajectory_data_set.metadata['args']['view_size'],
    )
    env = env()

    if args.track:
        run_name = f"{env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            name=run_name,
            config=args)
        trajectory_visualizer = TrajectoryVisualizer(trajectory_data_set)
        fig = trajectory_visualizer.plot_reward_over_time()
        wandb.log({"dataset/reward_over_time": wandb.Plotly(fig)})
        fig = trajectory_visualizer.plot_base_action_frequencies()
        wandb.log({"dataset/base_action_frequencies": wandb.Plotly(fig)})
        wandb.log(
            {"dataset/num_trajectories": trajectory_data_set.num_trajectories})

    if args.linear_time_embedding:
        time_embedding_type = "linear"
    else:
        time_embedding_type = "embedding"

    # make a decision transformer
    dt = DecisionTransformer(
        env=env,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_mlp=args.d_mlp,
        n_layers=args.n_layers,
        layer_norm=args.layer_norm,
        time_embedding_type=time_embedding_type,
        state_embedding_type="grid",  # hard-coded for now to minigrid.
        # so we can embed all the timesteps in the training data.
        max_timestep=trajectory_data_set.metadata.get("args").get("max_steps"),
        n_ctx=args.n_ctx,
        device=device
    )

    if args.track:
        wandb.watch(dt, log="parameters")

    dt = train(
        dt=dt,
        trajectory_data_set=trajectory_data_set,
        env=env,
        make_env=make_env,
        device=device,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        track=args.track,
        train_epochs=args.train_epochs,
        test_epochs=args.test_epochs,
        test_frequency=args.test_frequency,
        eval_frequency=args.eval_frequency,
        eval_episodes=args.eval_episodes,
        initial_rtg=args.initial_rtg,
        eval_max_time_steps=args.eval_max_time_steps
    )

    if args.track:
        # save the model with pickle, then upload it as an artifact, then delete it.
        # name it after the run name.
        if not os.path.exists("models"):
            os.mkdir("models")

        model_path = f"models/{run_name}.pt"
        t.save(dt.state_dict(), model_path)
        artifact = wandb.Artifact(run_name, type="model")
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)
        os.remove(model_path)

        wandb.finish()
