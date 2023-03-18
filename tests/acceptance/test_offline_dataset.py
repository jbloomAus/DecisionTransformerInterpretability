import pytest

import torch
from src.decision_transformer.offline_dataset import TrajectoryDataset, TrajectoryReader
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import WeightedRandomSampler

PATH = "tests/fixtures/test_trajectories.pkl"
PATH_COMPRESSED = "tests/fixtures/test_trajectories.xz"


def get_len_i_for_i_in_list(l):
    return [len(i) for i in l]


def test_trajectory_reader():

    trajectory_reader = TrajectoryReader(PATH)
    data = trajectory_reader.read()
    assert data is not None


def test_trajectory_reader_xz():

    trajectory_reader = TrajectoryReader(PATH_COMPRESSED)
    data = trajectory_reader.read()
    assert data is not None


def test_trajectory_dataset_init():

    trajectory_data_set = TrajectoryDataset(PATH, pct_traj=1.0, device="cpu")

    assert trajectory_data_set.num_trajectories == 54
    assert trajectory_data_set.num_timesteps == 49920
    assert trajectory_data_set.actions is not None
    assert trajectory_data_set.rewards is not None
    assert trajectory_data_set.dones is not None
    assert trajectory_data_set.returns is not None
    assert trajectory_data_set.states is not None
    assert trajectory_data_set.timesteps is not None
    assert trajectory_data_set.traj_lens.min() > 0

    assert len(trajectory_data_set.actions) == len(trajectory_data_set.rewards)
    assert len(trajectory_data_set.actions) == len(trajectory_data_set.dones)
    assert len(trajectory_data_set.actions) == len(trajectory_data_set.returns)
    assert len(trajectory_data_set.actions) == len(trajectory_data_set.states)

    # lengths match
    assert get_len_i_for_i_in_list(
        trajectory_data_set.actions) == get_len_i_for_i_in_list(trajectory_data_set.states)

    # max traj length is 1000
    assert max(get_len_i_for_i_in_list(trajectory_data_set.actions)
               ) == trajectory_data_set.max_ep_len
    assert trajectory_data_set.max_ep_len == trajectory_data_set.metadata["args"]["max_steps"]


def test_trajectory_dataset_init_xz():

    trajectory_data_set = TrajectoryDataset(
        PATH_COMPRESSED, pct_traj=1.0, device="cpu")

    assert trajectory_data_set.num_trajectories == 238
    assert trajectory_data_set.num_timesteps == 1920
    assert trajectory_data_set.actions is not None
    assert trajectory_data_set.rewards is not None
    assert trajectory_data_set.dones is not None
    assert trajectory_data_set.returns is not None
    assert trajectory_data_set.states is not None
    assert trajectory_data_set.timesteps is not None

    assert len(trajectory_data_set.actions) == len(trajectory_data_set.rewards)
    assert len(trajectory_data_set.actions) == len(trajectory_data_set.dones)
    assert len(trajectory_data_set.actions) == len(trajectory_data_set.returns)
    assert len(trajectory_data_set.actions) == len(trajectory_data_set.states)

    # lengths match
    assert get_len_i_for_i_in_list(
        trajectory_data_set.actions) == get_len_i_for_i_in_list(trajectory_data_set.states)


def test_trajectory_dataset_get_indices_of_top_p_trajectories_1():

    trajectory_data_set = TrajectoryDataset(PATH, pct_traj=1.0, device="cpu")
    indices = trajectory_data_set.get_indices_of_top_p_trajectories(
        pct_traj=1.0)

    # 1. the length of the indices is correct
    assert len(indices) == 54

    # 2. The rewards go in ascending order.
    for i in range(len(indices)-1):
        assert trajectory_data_set.returns[indices[i]
                                           ] <= trajectory_data_set.returns[indices[i+1]]


def test_trajectory_dataset_get_indices_of_top_p_trajectories_01():

    trajectory_data_set = TrajectoryDataset(PATH, pct_traj=1.0, device="cpu")
    indices = trajectory_data_set.get_indices_of_top_p_trajectories(
        pct_traj=0.1)

    # 1. the length of the indices is correct
    assert len(indices) == 7

    # 2. The rewards go in ascending order.
    for i in range(len(indices)-1):
        assert trajectory_data_set.returns[indices[i]
                                           ] <= trajectory_data_set.returns[indices[i+1]]


def test_trajectory_dataset__getitem__():

    trajectory_data_set = TrajectoryDataset(
        PATH, max_len=100, pct_traj=1.0, device="cpu")
    s, a, r, d, rtg, timesteps, mask = trajectory_data_set[0]

    assert isinstance(s, torch.Tensor)
    assert isinstance(a, torch.Tensor)
    assert isinstance(r, torch.Tensor)
    assert isinstance(d, torch.Tensor)
    assert isinstance(rtg, torch.Tensor)
    assert isinstance(timesteps, torch.Tensor)
    assert isinstance(mask, torch.Tensor)

    assert s.shape == (100, 7, 7, 3)
    assert a.shape == (100,)
    assert r.shape == (100, 1)  # flatten this later?
    assert d.shape == (100,)
    assert rtg.shape == (101, 1)  # how did we get the extra timestep?
    assert timesteps.shape == (100,)
    assert mask.shape == (100,)


def test_trajectory_dataset_sampling_probabilities():

    trajectory_data_set = TrajectoryDataset(PATH, pct_traj=1.0, device="cpu")
    assert len(
        trajectory_data_set.sampling_probabilities) == trajectory_data_set.num_trajectories
    prob = trajectory_data_set.traj_lens[trajectory_data_set.indices[0]
                                         ]/trajectory_data_set.num_timesteps
    assert trajectory_data_set.sampling_probabilities[0] == pytest.approx(
        0.02, rel=1e-1)
    assert trajectory_data_set.sampling_probabilities[0] == prob
    assert trajectory_data_set.sampling_probabilities[-1] == pytest.approx(
        0.0055, rel=1e-1)
    assert trajectory_data_set.sampling_probabilities.sum() == pytest.approx(1.0, rel=1e-1)


def test_trajectory_dataset_discount_cumusum_10():

    trajectory_data_set = TrajectoryDataset(PATH, pct_traj=1.0, device="cpu")
    vector = torch.tensor([1, 2, 3], dtype=torch.float32)
    discount = 1.0
    expected = torch.tensor([1, 2, 3], dtype=torch.float32)
    expected[0] = expected.sum()
    expected[1] = expected[1:].sum()
    expected[2] = expected[2:].sum()

    trajectory_data_set = TrajectoryDataset(PATH, pct_traj=1.0, device="cpu")
    actual = trajectory_data_set.discount_cumsum(vector, discount)
    assert actual.shape == expected.shape
    torch.testing.assert_allclose(torch.tensor(actual), expected)


def test_trajectory_dataset_as_dataloader():

    dataset = TrajectoryDataset(PATH, max_len=100, pct_traj=1.0, device="cpu")
    sampler = WeightedRandomSampler(
        weights=dataset.sampling_probabilities,
        num_samples=dataset.num_trajectories,
        replacement=True,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=8, sampler=sampler)

    for i, (s, a, r, d, rtg, timesteps, mask) in enumerate(dataloader):
        assert s.shape == (8, 100, 7, 7, 3), f"i={i}, s.shape={s.shape}"
        assert a.shape == (8, 100)
        assert r.shape == (8, 100, 1)
        assert d.shape == (8, 100)
        assert rtg.shape == (8, 101, 1)
        assert timesteps.shape == (8, 100)
        assert mask.shape == (8, 100)

        assert s.dtype == torch.float32
        assert a.dtype == torch.long
        assert r.dtype == torch.float32
        assert d.dtype == torch.bool
        assert rtg.dtype == torch.float32
        assert timesteps.dtype == torch.int64
        assert mask.dtype == torch.bool

        assert s.device == torch.device("cpu")
        assert a.device == torch.device("cpu")
        assert r.device == torch.device("cpu")
        assert d.device == torch.device("cpu")
        assert rtg.device == torch.device("cpu")
        assert timesteps.device == torch.device("cpu")
        assert mask.device == torch.device("cpu")
        if i > 4:
            break


def test_train_test_split():

    dataset = TrajectoryDataset(PATH, max_len=100, pct_traj=1.0, device="cpu")

    # Split the dataset into training and testing sets
    train_dataset, test_dataset = random_split(dataset, [0.80, 0.20])

    assert len(train_dataset) == pytest.approx(0.80 * len(dataset), abs=1)
    assert len(test_dataset) == pytest.approx(0.20 * len(dataset), abs=1)

    s, a, r, d, rtg, timesteps, mask = train_dataset[0]
    assert s.shape == (100, 7, 7, 3)
    assert a.shape == (100,)
    assert r.shape == (100, 1)
    assert d.shape == (100,)
    assert rtg.shape == (101, 1)
    assert timesteps.shape == (100,)
    assert mask.shape == (100,)

    s, a, r, d, rtg, timesteps, mask = test_dataset[0]
    assert s.shape == (100, 7, 7, 3)
    assert a.shape == (100,)
    assert r.shape == (100, 1)
    assert d.shape == (100,)
    assert rtg.shape == (101, 1)
    assert timesteps.shape == (100,)
    assert mask.shape == (100,)

    # Create the train DataLoader
    train_sampler = torch.utils.data.sampler.WeightedRandomSampler(
        weights=dataset.sampling_probabilities[train_dataset.indices],
        num_samples=len(train_dataset),
        replacement=True,
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=8, sampler=train_sampler)

    # Create the test DataLoader
    test_sampler = torch.utils.data.sampler.WeightedRandomSampler(
        weights=dataset.sampling_probabilities[test_dataset.indices],
        num_samples=len(test_dataset),
        replacement=True,
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=8, sampler=test_sampler)


def test_train_test_split_other_data():

    dataset = TrajectoryDataset(
        PATH_COMPRESSED, max_len=100, pct_traj=1.0, device="cpu")

    # Split the dataset into training and testing sets
    train_dataset, test_dataset = random_split(dataset, [0.80, 0.20])

    assert len(train_dataset) == pytest.approx(0.80 * len(dataset), abs=1)
    assert len(test_dataset) == pytest.approx(0.20 * len(dataset), abs=1)

    s, a, r, d, rtg, timesteps, mask = train_dataset[0]
    assert s.shape == (100, 7, 7, 20)
    assert a.shape == (100,)
    assert r.shape == (100, 1)
    assert d.shape == (100,)
    assert rtg.shape == (101, 1)
    assert timesteps.shape == (100,)
    assert mask.shape == (100,)

    s, a, r, d, rtg, timesteps, mask = test_dataset[0]
    assert s.shape == (100, 7, 7, 20)
    assert a.shape == (100,)
    assert r.shape == (100, 1)
    assert d.shape == (100,)
    assert rtg.shape == (101, 1)
    assert timesteps.shape == (100,)
    assert mask.shape == (100,)

    # Create the train DataLoader
    train_sampler = torch.utils.data.sampler.WeightedRandomSampler(
        weights=dataset.sampling_probabilities[train_dataset.indices],
        num_samples=len(train_dataset),
        replacement=True,
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=8, sampler=train_sampler)

    # Create the test DataLoader
    test_sampler = torch.utils.data.sampler.WeightedRandomSampler(
        weights=dataset.sampling_probabilities[test_dataset.indices],
        num_samples=len(test_dataset),
        replacement=True,
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=8, sampler=test_sampler)
