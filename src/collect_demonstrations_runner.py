import sys
import time
import argparse
from src.ppo.agent import load_saved_checkpoint
from src.utils.trajectory_writer import TrajectoryWriter
from src.config import RunConfig, OnlineTrainConfig
from src.ppo.memory import Memory
from src.environments.registration import register_envs


def runner(
    checkpoint_path: int,
    num_envs: int,
    trajectory_path: str,
    sampling_configs: list,
):
    register_envs()
    agent = load_saved_checkpoint(checkpoint_path, num_envs)
    memory = Memory(
        agent.envs, OnlineTrainConfig(num_envs=num_envs), device=agent.device
    )

    trajectory_writer = TrajectoryWriter(
        path=trajectory_path,
        run_config=RunConfig(track=False),
        environment_config=agent.environment_config,
        online_config=OnlineTrainConfig(num_envs=num_envs),
        model_config=agent.model_config,
    )

    i = 0
    for config in sampling_configs:
        print(f"Sampling with config: {config}")
        agent.rollout(
            memory=memory,
            num_steps=config["rollout_length"],
            envs=agent.envs,
            trajectory_writer=trajectory_writer,
            **config,
        )
        i += 1
        print(f"finished config: {i} out of {len(sampling_configs)}")

    if trajectory_writer:
        trajectory_writer.tag_terminated_trajectories()
        trajectory_writer.write(upload_to_wandb=False)
    return memory, trajectory_writer


def main():
    parser = argparse.ArgumentParser(
        description="Collect demonstrations from a trained agent."
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
        "--trajectory_path",
        type=str,
        default=None,
        help="Path to save trajectory data.",
    )
    parser.add_argument(
        "--basic", type=int, help="Number of steps for basic sampling"
    )
    parser.add_argument(
        "--temp",
        nargs=2,
        action="append",
        metavar=("steps", "temperature"),
        type=float,
        help="Number of steps and temperature for temperature sampling",
    )
    parser.add_argument(
        "--topk",
        nargs=2,
        action="append",
        metavar=("steps", "k"),
        type=int,
        help="Number of steps and temperature for temperature sampling",
    )
    parser.add_argument(
        "--bottomk",
        nargs=2,
        action="append",
        metavar=("steps", "k"),
        type=int,
        help="Number of steps and temperature for temperature sampling",
    )

    args = parser.parse_args()

    sampling_configs = []

    if args.basic:
        sampling_configs.append(
            {"rollout_length": args.basic, "sampling_method": "basic"}
        )

    if args.temp:
        for temp_arg in args.temp:
            steps, temperature = int(temp_arg[0]), temp_arg[1]
            sampling_configs.append(
                {
                    "rollout_length": steps,
                    "sampling_method": "temperature",
                    "temperature": temperature,
                }
            )
    if args.topk:
        for topk_arg in args.topk:
            steps, k = int(topk_arg[0]), topk_arg[1]
            sampling_configs.append(
                {"rollout_length": steps, "sampling_method": "topk", "k": k}
            )
    if args.bottomk:
        for bottomk_arg in args.bottomk:
            steps, k = int(bottomk_arg[0]), bottomk_arg[1]
            sampling_configs.append(
                {"rollout_length": steps, "sampling_method": "bottomk", "k": k}
            )

    runner(
        args.checkpoint, args.num_envs, args.trajectory_path, sampling_configs
    )


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"Total time: {time.time() - start}")
