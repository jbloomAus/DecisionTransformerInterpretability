"""
This testing module is itself quite interesting for the following reasons:
1. We are analysing the performance of difference sampling strategies on trajectories collected from a trained PPO agent.
2. We can achive highly variable performance using the model directly (it's not an optimally trained model) and sampling
    actions according to the logits ("basic sampling"). Alternatively, high temperature or bottom k lead to very low 
    performance since the agent is forced to take suboptimal actions. Low temperature and topk lead to good performance
    as well.
3. This is really valuable information because it tells us that we can use sampling variations on our PPO agent that 
    take it "off distribution" but because we can still score them, generate much more diverse training data for our
    offline agents. While not especially complicated, I see this as a very useful result.
"""
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


@pytest.fixture(
    params=[
        # Basic sampling config
        (
            [{"rollout_length": 200, "sampling_method": "basic"}],
            {"mean": 0.43, "std": 0.55, "max": 1.0, "min": 0.0},
        ),
        # Temperature sampling configs - Should be variably performant
        (
            [
                {
                    "rollout_length": 200,
                    "sampling_method": "temperature",
                    "temperature": 1e-6,
                }
            ],
            {"mean": 0.40, "std": 0.45, "max": 0.95, "min": 0.0},
        ),
        (
            [
                {
                    "rollout_length": 200,
                    "sampling_method": "temperature",
                    "temperature": 3,
                }
            ],
            {"mean": 0.40, "std": 0.45, "max": 0.95, "min": 0.0},
        ),
        (
            [
                {
                    "rollout_length": 200,
                    "sampling_method": "temperature",
                    "temperature": 10,
                }
            ],
            {"mean": 0.01, "std": 0.20, "max": 0.60, "min": 0.0},
        ),
        (
            [
                {
                    "rollout_length": 200,
                    "sampling_method": "temperature",
                    "temperature": 100,
                }
            ],
            {"mean": 0.01, "std": 0.20, "max": 0.51, "min": 0.0},
        ),
        # Bottom k sampling - Should be poorly performant
        (
            [{"rollout_length": 200, "sampling_method": "bottomk", "k": 3}],
            {"mean": 0.01, "std": 0.20, "max": 0.0, "min": 0.0},
        ),
        (
            [{"rollout_length": 200, "sampling_method": "bottomk", "k": 6}],
            {"mean": 0.01, "std": 0.20, "max": 0.31, "min": 0.0},
        ),
        # Top k sampling - Should be pretty performant/on par with basic sampling
        (
            [{"rollout_length": 200, "sampling_method": "topk", "k": 3}],
            {"mean": 0.40, "std": 0.40, "max": 0.8, "min": 0.0},
        ),
        (
            [{"rollout_length": 200, "sampling_method": "topk", "k": 6}],
            {"mean": 0.40, "std": 0.40, "max": 0.8, "min": 0.0},
        ),
    ]
)
def sampling_config_and_expected(request):
    return request.param


def test_runner(checkpoint_path, sampling_config_and_expected):
    num_envs = 16
    sampling_config, expected = sampling_config_and_expected

    # Use a temporary directory for storing the trajectory file
    trajectory_path = "tmp/test_trajectory.gz"

    memory, trajectory_writer = runner(
        checkpoint_path, num_envs, trajectory_path, sampling_config
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
        num_envs, sampling_config[0]["rollout_length"], abs=1000
    )

    # distribution range
    assert df.episode_return.max() == pytest.approx(expected["max"], abs=0.2)
    assert df.episode_return.min() == expected["min"]

    # distribution properties
    assert df.episode_return.std() == pytest.approx(expected["std"], abs=0.2)
    assert df.episode_return.mean() == pytest.approx(expected["mean"], abs=0.2)
