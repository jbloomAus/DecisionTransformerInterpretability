import pytest
import os
from src.collect_demonstrations_runner import runner
from src.environments.registration import register_envs
from src.ppo.agent import process_memory_vars_to_log


@pytest.fixture(autouse=True)
def register_envs_fixture():
    register_envs()


@pytest.fixture
def checkpoint_path():
    return "models/ppo/memory_lstm_demos/Test-PPO-LSTM_06.pt"


def test_runner(checkpoint_path):
    num_envs = 16
    rollout_length = 200  # Use a smaller rollout length for testing

    # Use a temporary directory for storing the trajectory file
    trajectory_path = "tmp/test_trajectory.gz"

    memory, trajectory_writer = runner(
        checkpoint_path, num_envs, rollout_length, trajectory_path
    )

    assert memory is not None, "Memory object not created"
    assert trajectory_writer is not None, "TrajectoryWriter object not created"
    assert trajectory_writer.path == str(
        trajectory_path
    ), "TrajectoryWriter path does not match"

    # Check that the trajectory file was created
    assert os.path.exists(trajectory_path), "Trajectory file not created"

    # check that the memory object contains the right number of results
    df = process_memory_vars_to_log(memory.vars_to_log)

    assert df.episode_length.sum() == pytest.approx(
        num_envs, rollout_length, abs=1000
    )

    # distribution range
    assert df.episode_return.max() == pytest.approx(1.0, abs=0.2)
    assert df.episode_return.min() == 0.0

    # distribution properties
    assert df.episode_return.std() == pytest.approx(0.55, abs=0.2)
    assert df.episode_return.mean() == pytest.approx(0.43, abs=0.2)
