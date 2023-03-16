import json
import os
import time
import warnings
from typing import Callable

import torch as t

import wandb
from src.config import (EnvironmentConfig, OfflineTrainConfig, RunConfig,
                        TransformerModelConfig, ConfigJsonEncoder)
from src.models.trajectory_model import CloneTransformer, DecisionTransformer

# from .model import DecisionTransformer
from .offline_dataset import TrajectoryDataset, TrajectoryVisualizer
from .train import train


def run_decision_transformer(
        run_config: RunConfig,
        transformer_config: TransformerModelConfig,
        offline_config: OfflineTrainConfig,
        make_env: Callable):
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    if run_config.cuda:
        device = t.device("cuda" if t.cuda.is_available() else "cpu")
    else:
        device = t.device("cpu")

    if offline_config.trajectory_path is None:
        raise ValueError("Must specify a trajectory path.")

    trajectory_data_set = TrajectoryDataset(
        trajectory_path=offline_config.trajectory_path,
        max_len=transformer_config.n_ctx // 3,
        pct_traj=offline_config.pct_traj,
        prob_go_from_end=offline_config.prob_go_from_end,
        device=device,
    )

    # make an environment
    env_id = trajectory_data_set.metadata['args']['env_id']
    # pretty print the metadata
    print(trajectory_data_set.metadata)

    if "view_size" not in trajectory_data_set.metadata['args']:
        trajectory_data_set.metadata['args']['view_size'] = 7

    environment_config = EnvironmentConfig(
        env_id=trajectory_data_set.metadata['args']['env_id'],
        one_hot_obs=trajectory_data_set.observation_type == "one_hot",
        view_size=trajectory_data_set.metadata['args']['view_size'],
        fully_observed=False,
        capture_video=False,
        render_mode='rgb_array')

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

    wandb_args = run_config.__dict__ | transformer_config.__dict__ | offline_config.__dict__

    if run_config.track:
        run_name = f"{env_id}__{run_config.exp_name}__{run_config.seed}__{int(time.time())}"
        wandb.init(
            project=run_config.wandb_project_name,
            entity=run_config.wandb_entity,
            name=run_name,
            config=wandb_args)
        trajectory_visualizer = TrajectoryVisualizer(trajectory_data_set)
        fig = trajectory_visualizer.plot_reward_over_time()
        wandb.log({"dataset/reward_over_time": wandb.Plotly(fig)})
        fig = trajectory_visualizer.plot_base_action_frequencies()
        wandb.log({"dataset/base_action_frequencies": wandb.Plotly(fig)})
        wandb.log(
            {"dataset/num_trajectories": trajectory_data_set.num_trajectories})

    if offline_config.model_type == "decision_transformer":
        model = DecisionTransformer(
            environment_config=environment_config,
            transformer_config=transformer_config
        )
    else:
        model = CloneTransformer(
            environment_config=environment_config,
            transformer_config=transformer_config
        )

    if run_config.track:
        wandb.watch(model, log="parameters")

    model = train(
        model=model,
        trajectory_data_set=trajectory_data_set,
        env=env,
        make_env=make_env,
        device=device,
        lr=offline_config.lr,
        weight_decay=offline_config.weight_decay,
        batch_size=offline_config.batch_size,
        track=offline_config.track,
        train_epochs=offline_config.train_epochs,
        test_epochs=offline_config.test_epochs,
        test_frequency=offline_config.test_frequency,
        eval_frequency=offline_config.eval_frequency,
        eval_episodes=offline_config.eval_episodes,
        initial_rtg=offline_config.initial_rtg,
        eval_max_time_steps=offline_config.eval_max_time_steps
    )

    if run_config.track:
        # save the model with pickle, then upload it
        # as an artifact, then delete it.
        # name it after the run name.
        if not os.path.exists("models"):
            os.mkdir("models")

        model_path = f"models/{run_name}.pt"
        t.save({
            "model_state_dict": model.state_dict(),
            "transformer_config": json.dumps(transformer_config, cls=ConfigJsonEncoder),
            "offline_config": json.dumps(offline_config, cls=ConfigJsonEncoder),
        }, model_path)
        artifact = wandb.Artifact(run_name, type="model")
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)
        os.remove(model_path)

        wandb.finish()
