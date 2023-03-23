
import warnings
import gymnasium as gym
import torch as t
import wandb
import time

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

    if online_config.trajectory_path:
        trajectory_writer = TrajectoryWriter(
            online_config.trajectory_path,
            run_config=run_config,
            environment_config=environment_config,
            online_config=online_config,
            transformer_model_config=transformer_model_config,
        )
    else:
        trajectory_writer = None

    # Verify environment is registered
    register_envs()
    all_envs = [env_spec for env_spec in gym.envs.registry]
    assert environment_config.env_id in all_envs, f"Environment {environment_config.env_id} not registered."

    # wandb initialisation,
    run_name = f"{environment_config.env_id}__{run_config.exp_name}__{run_config.seed}__{int(time.time())}"
    if run_config.track:
        run = wandb.init(
            project=run_config.wandb_project_name,
            entity=run_config.wandb_entity,
            config=combine_args(
                run_config, environment_config, online_config, transformer_model_config
            ),  # vars is equivalent to args.__dict__
            name=run_name,
            save_code=True,
        )

    # add run_name to args
    run_config.run_name = run_name

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

    agent = train_ppo(
        run_config=run_config,
        online_config=online_config,
        environment_config=environment_config,
        transformer_model_config=transformer_model_config,
        envs=envs,
        trajectory_writer=trajectory_writer
    )
    if run_config.track:
        run.finish()


def combine_args(run_config, environment_config, online_config, transformer_model_config=None):
    args = {}
    args.update(run_config.__dict__)
    args.update(environment_config.__dict__)
    args.update(online_config.__dict__)
    if transformer_model_config is not None:
        args.update(transformer_model_config.__dict__)
    return args
