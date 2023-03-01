
import warnings
import gymnasium as gym
import torch as t
import wandb
import time

from .ppo.utils import PPOArgs, set_global_seeds, parse_args
from .ppo.train import train_ppo
from .utils import TrajectoryWriter
from .environments.environments import make_env
from .environments.registration import register_envs

warnings.filterwarnings("ignore", category=DeprecationWarning)
device = t.device("cuda" if t.cuda.is_available() else "cpu")


def ppo_runner(args: PPOArgs):
    '''
    Trains a Proximal Policy Optimization (PPO) algorithm on a specified environment using the given hyperparameters.

    Args:
    - args (PPOArgs): an object that contains hyperparameters and other arguments for PPO training.

    Returns: None.
    '''

    if args.trajectory_path:
        trajectory_writer = TrajectoryWriter(args.trajectory_path, args)
    else:
        trajectory_writer = None

    # Verify environment is registered
    register_envs()
    all_envs = [env_spec for env_spec in gym.envs.registry]
    assert args.env_id in all_envs, f"Environment {args.env_id} not registered."

    # wandb initialisation,
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),  # vars is equivalent to args.__dict__
            name=run_name,
            # monitor_gym=True,
            save_code=True,
        )

    # add run_name to args
    args.run_name = run_name

    # make envs
    set_global_seeds(args.seed)

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name,
                  max_steps=args.max_steps,
                  fully_observed=args.fully_observed,
                  flat_one_hot=args.one_hot_obs,
                  agent_view_size=args.view_size,
                  render_mode="rgb_array",
                  ) for i in range(args.num_envs)]
    )

    train_ppo(
        args,
        envs,
        trajectory_writer=trajectory_writer,
        probe_idx=None
    )
    if args.track:
        run.finish()


if __name__ == "__main__":

    args = parse_args()
    args = PPOArgs(
        exp_name=args.exp_name,
        seed=args.seed,
        cuda=args.cuda,
        track=args.track,
        wandb_project_name=args.wandb_project_name,
        wandb_entity=args.wandb_entity,
        capture_video=args.capture_video,
        env_id=args.env_id,
        view_size=args.view_size,
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        decay_lr=args.decay_lr,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        num_minibatches=args.num_minibatches,
        update_epochs=args.update_epochs,
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        max_steps=args.max_steps,
        one_hot_obs=args.one_hot_obs,
        trajectory_path=args.trajectory_path,
        fully_observed=args.fully_observed,
    )

    ppo_runner(args)
