
import warnings
import gymnasium as gym
import torch as t
import wandb
import time

from argparse import Namespace
from typing import Optional

from src.config import RunConfig, TransformerModelConfig, EnvironmentConfig, OnlineTrainConfig
from src.ppo.utils import set_global_seeds
from src.ppo.train import train_ppo
from src.utils import TrajectoryWriter
from src.environments.environments import make_env
from src.environments.registration import register_envs

warnings.filterwarnings("ignore", category=DeprecationWarning)
device = t.device("cuda" if t.cuda.is_available() else "cpu")


def ppo_runner(
    run_config: RunConfig,
    environment_config: EnvironmentConfig,
    online_config: OnlineTrainConfig,
    transformer_model_config: Optional[TransformerModelConfig],
):
    '''
    Trains a Proximal Policy Optimization (PPO) algorithm on a specified environment using the given hyperparameters.

    Args:
    - args (PPOArgs): an object that contains hyperparameters and other arguments for PPO training.

    Returns: None.
    '''

    if transformer_model_config is None:
        print("Defaulting to Fully Connected AC model. Not a transformer model.")

    args = run_config.__dict__ | environment_config.__dict__ | online_config.__dict__
    if transformer_model_config is not None:
        args = args | transformer_model_config.__dict__

    args = Namespace(**args)
    if online_config.trajectory_path:
        trajectory_writer = TrajectoryWriter(
            online_config.trajectory_path,
            args=args
        )
    else:
        trajectory_writer = None

    # Verify environment is registered
    register_envs()
    all_envs = [env_spec for env_spec in gym.envs.registry]
    assert environment_config.env_id in all_envs, f"Environment {args.env_id} not registered."

    # wandb initialisation,
    run_name = f"{environment_config.env_id}__{run_config.exp_name}__{run_config.seed}__{int(time.time())}"
    if args.track:
        run = wandb.init(
            project=run_config.wandb_project_name,
            entity=run_config.wandb_entity,
            config=vars(args),  # vars is equivalent to args.__dict__
            name=run_name,
            save_code=True,
        )

    # add run_name to args
    args.run_name = run_name

    # make envs
    set_global_seeds(run_config.seed)

    envs = gym.vector.SyncVectorEnv(
        [make_env(
            env_id=environment_config.env_id,
            seed=environment_config.seed + i,
            idx=i,
            capture_video=environment_config.capture_video,
            run_name=run_name,
            max_steps=environment_config.max_steps,
            fully_observed=environment_config.fully_observed,
            flat_one_hot=environment_config.one_hot_obs,
            agent_view_size=environment_config.view_size,
            render_mode="rgb_array",
        ) for i in range(online_config.num_envs)]
    )

    train_ppo(
        args,
        envs,
        trajectory_writer=trajectory_writer,
        probe_idx=None
    )
    if args.track:
        run.finish()
