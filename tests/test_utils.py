import pytest 

from dataclasses import dataclass
import numpy as np
import pickle 
import torch
from src.utils import TrajectoryWriter, TrajectoryReader


def test_trajectory_writer_numpy():

    @dataclass
    class DummyArgs:
        pass

    args = DummyArgs()

    trajectory_writer = TrajectoryWriter("tmp/test_trajectory_writer_writer.pkl", args)

    # test accumulate trajectory when all the objects are initialized as np arrays

    trajectory_writer.accumulate_trajectory(
        next_obs=np.array([1, 2, 3]),
        reward=np.array([1, 2, 3]),
        done=np.array([1, 0, 0]),
        truncated=np.array([1, 0, 0]),
        action=np.array([1, 2, 3]),
        info=np.array([{"a": 1, "b": 2, "c": 3}], dtype=object),
    )

    trajectory_writer.write()


    with open("tmp/test_trajectory_writer_writer.pkl", "rb") as f:
        data = pickle.load(f)

        obs = data["data"]["observations"]
        assert type(obs) == np.ndarray
        assert obs.dtype == np.float64

        assert obs[0][0] == 1
        assert obs[0][1] == 2
        assert obs[0][2] == 3

        rewards = data["data"]["rewards"]
        assert type(rewards) == np.ndarray
        assert rewards.dtype == np.float64

        assert rewards[0][0] == 1
        assert rewards[0][1] == 2
        assert rewards[0][2] == 3

        dones = data["data"]["dones"]
        assert type(dones) == np.ndarray
        assert dones.dtype == bool

        assert dones[0][0] 
        assert dones[0][1] == False
        assert dones[0][2] == False

        actions = data["data"]["actions"]
        assert type(actions) == np.ndarray
        assert actions.dtype == np.int64

        assert actions[0][0] == 1
        assert actions[0][1] == 2
        assert actions[0][2] == 3

        infos = data["data"]["infos"]
        assert type(infos) == np.ndarray
        assert infos.dtype == np.object

        assert infos[0][0]["a"] == 1
        assert infos[0][0]["b"] == 2
        assert infos[0][0]["c"] == 3
    
def test_trajectory_writer_torch():

    @dataclass
    class DummyArgs:
        pass

    args = DummyArgs()

    trajectory_writer = TrajectoryWriter("tmp/test_trajectory_writer_writer.pkl", args)

    # test accumulate trajectory when all the objects are initialized as pytorch tensors

    # assert raises type error 
    with pytest.raises(TypeError):
        trajectory_writer.accumulate_trajectory(
            next_obs=torch.tensor([1, 2, 3], dtype=torch.float64),
            reward=torch.tensor([1, 2, 3], dtype=torch.float64),
            done=torch.tensor([1, 0, 0], dtype=torch.bool),
            action=torch.tensor([1, 2, 3], dtype=torch.int64),
            info=[{"a": 1, "b": 2, "c": 3}],
        )

