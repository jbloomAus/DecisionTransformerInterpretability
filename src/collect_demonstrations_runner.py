import sys
import argparse
from src.ppo.agent import load_saved_checkpoint
from src.utils import TrajectoryWriter
from src.config import RunConfig, OnlineTrainConfig
from src.ppo.memory import Memory


def runner(checkpoint_path, num_envs, rollout_length, trajectory_path=None):
    agent = load_saved_checkpoint(checkpoint_path, num_envs)
    memory = Memory(
        agent.envs, OnlineTrainConfig(num_envs=num_envs), device=agent.device
    )
    if trajectory_path:
        trajectory_writer = TrajectoryWriter(
            path=trajectory_path,
            run_config=RunConfig(track=False),
            environment_config=agent.environment_config,
            online_config=OnlineTrainConfig(num_envs=num_envs),
            model_config=agent.model_config,
        )
    else:
        trajectory_writer = None
    agent.rollout(memory, rollout_length, agent.envs, trajectory_writer)
    if trajectory_writer:
        trajectory_writer.tag_terminated_trajectories()
        trajectory_writer.write(upload_to_wandb=False)
    return memory, trajectory_writer


def main():
    parser = argparse.ArgumentParser(
        description="Run a PPO agent with a saved checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the saved checkpoint.",
    )
    parser.add_argument(
        "--num_envs", type=int, default=16, help="Number of environments."
    )
    parser.add_argument(
        "--rollout_length",
        type=int,
        default=60000,
        help="Length of the rollout.",
    )
    parser.add_argument(
        "--trajectory_path",
        type=str,
        default=None,
        help="Path to save trajectory data.",
    )

    args = parser.parse_args()

    runner(
        args.checkpoint,
        args.num_envs,
        args.rollout_length,
        args.trajectory_path,
    )


if __name__ == "__main__":
    main()
