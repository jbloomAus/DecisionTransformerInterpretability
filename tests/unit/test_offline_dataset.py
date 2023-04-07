import pytest

import torch
from src.decision_transformer.offline_dataset import TrajectoryDataset, TrajectoryReader
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import WeightedRandomSampler


def get_len_i_for_i_in_list(l):
    return [len(i) for i in l]


@pytest.fixture
def trajectory_reader_pkl():
    PATH = "tests/fixtures/test_trajectories.pkl"
    trajectory_reader = TrajectoryReader(PATH)
    return trajectory_reader


@pytest.fixture
def trajectory_reader_xz():
    PATH = "tests/fixtures/test_trajectories.xz"
    trajectory_reader = TrajectoryReader(PATH)
    return trajectory_reader


@pytest.fixture
def trajectory_dataset():
    PATH = "tests/fixtures/test_trajectories.pkl"
    trajectory_dataset = TrajectoryDataset(
        PATH, max_len=100, pct_traj=1.0, device="cpu")
    return trajectory_dataset


@pytest.fixture
def trajectory_dataset_xz():
    PATH = "tests/fixtures/test_trajectories.xz"
    trajectory_dataset = TrajectoryDataset(
        PATH, max_len=100, pct_traj=1.0, device="cpu")
    return trajectory_dataset


def test_trajectory_reader_read(trajectory_reader_pkl):
    data = trajectory_reader_pkl.read()
    assert data is not None


def test_trajectory_reader_xz(trajectory_reader_xz):
    data = trajectory_reader_xz.read()
    assert data is not None


def test_trajectory_dataset_init(trajectory_dataset):

    assert trajectory_dataset.num_trajectories == 54
    assert trajectory_dataset.num_timesteps == 49920
    assert trajectory_dataset.actions is not None
    assert trajectory_dataset.rewards is not None
    assert trajectory_dataset.dones is not None
    assert trajectory_dataset.returns is not None
    assert trajectory_dataset.states is not None
    assert trajectory_dataset.timesteps is not None
    assert trajectory_dataset.traj_lens.min() > 0

    assert len(trajectory_dataset.actions) == len(trajectory_dataset.rewards)
    assert len(trajectory_dataset.actions) == len(trajectory_dataset.dones)
    assert len(trajectory_dataset.actions) == len(trajectory_dataset.returns)
    assert len(trajectory_dataset.actions) == len(trajectory_dataset.states)

    # lengths match
    assert get_len_i_for_i_in_list(
        trajectory_dataset.actions) == get_len_i_for_i_in_list(trajectory_dataset.states)

    # max traj length is 1000
    assert max(get_len_i_for_i_in_list(trajectory_dataset.actions)
               ) == trajectory_dataset.max_ep_len
    assert trajectory_dataset.max_ep_len == trajectory_dataset.metadata["args"]["max_steps"]


def test_trajectory_dataset_init_xz(trajectory_dataset_xz):

    trajectory_dataset = trajectory_dataset_xz

    assert trajectory_dataset.num_trajectories == 238
    assert trajectory_dataset.num_timesteps == 1920
    assert trajectory_dataset.actions is not None
    assert trajectory_dataset.rewards is not None
    assert trajectory_dataset.dones is not None
    assert trajectory_dataset.returns is not None
    assert trajectory_dataset.states is not None
    assert trajectory_dataset.timesteps is not None

    assert len(trajectory_dataset.actions) == len(trajectory_dataset.rewards)
    assert len(trajectory_dataset.actions) == len(trajectory_dataset.dones)
    assert len(trajectory_dataset.actions) == len(trajectory_dataset.returns)
    assert len(trajectory_dataset.actions) == len(trajectory_dataset.states)

    # lengths match
    assert get_len_i_for_i_in_list(
        trajectory_dataset.actions) == get_len_i_for_i_in_list(trajectory_dataset.states)


def test_trajectory_dataset_get_indices_of_top_p_trajectories_1(trajectory_dataset):

    indices = trajectory_dataset.get_indices_of_top_p_trajectories(1.0)

    # 1. the length of the indices is correct
    assert len(indices) == 54

    # 2. The rewards go in ascending order.
    for i in range(len(indices)-1):
        assert trajectory_dataset.returns[indices[i]
                                          ] <= trajectory_dataset.returns[indices[i+1]]


def test_trajectory_dataset_get_indices_of_top_p_trajectories_01(trajectory_dataset):

    indices = trajectory_dataset.get_indices_of_top_p_trajectories(0.1)

    # 1. the length of the indices is correct
    assert len(indices) == 7

    # 2. The rewards go in ascending order.
    for i in range(len(indices)-1):
        assert trajectory_dataset.returns[indices[i]
                                          ] <= trajectory_dataset.returns[indices[i+1]]


def test_trajectory_dataset__getitem__(trajectory_dataset):

    s, a, r, d, rtg, timesteps, mask = trajectory_dataset[0]

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


def test_trajectory_dataset_sampling_probabilities(trajectory_dataset):

    assert len(
        trajectory_dataset.sampling_probabilities) == trajectory_dataset.num_trajectories
    prob = trajectory_dataset.traj_lens[trajectory_dataset.indices[0]
                                        ]/trajectory_dataset.num_timesteps
    assert trajectory_dataset.sampling_probabilities[0] == pytest.approx(
        0.02, rel=1e-1)
    assert trajectory_dataset.sampling_probabilities[0] == prob
    assert trajectory_dataset.sampling_probabilities[-1] == pytest.approx(
        0.0055, rel=1e-1)
    assert trajectory_dataset.sampling_probabilities.sum() == pytest.approx(1.0, rel=1e-1)


def test_trajectory_dataset_discount_cumusum_10(trajectory_dataset):

    vector = torch.tensor([1, 2, 3], dtype=torch.float32)
    discount = 1.0
    expected = torch.tensor([1, 2, 3], dtype=torch.float32)
    expected[0] = expected.sum()
    expected[1] = expected[1:].sum()
    expected[2] = expected[2:].sum()

    actual = trajectory_dataset.discount_cumsum(vector, discount)
    assert actual.shape == expected.shape
    torch.testing.assert_allclose(torch.tensor(actual), expected)


def test_trajectory_dataset_as_dataloader(trajectory_dataset):

    sampler = WeightedRandomSampler(
        weights=trajectory_dataset.sampling_probabilities,
        num_samples=trajectory_dataset.num_trajectories,
        replacement=True,
    )
    dataloader = torch.utils.data.DataLoader(
        trajectory_dataset, batch_size=8, sampler=sampler)

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


def test_train_test_split(trajectory_dataset):

    # Split the dataset into training and testing sets
    train_dataset, test_dataset = random_split(
        trajectory_dataset, [0.80, 0.20])

    assert len(train_dataset) == pytest.approx(
        0.80 * len(trajectory_dataset), abs=1)
    assert len(test_dataset) == pytest.approx(
        0.20 * len(trajectory_dataset), abs=1)

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
        weights=trajectory_dataset.sampling_probabilities[train_dataset.indices],
        num_samples=len(train_dataset),
        replacement=True,
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=8, sampler=train_sampler)

    # Create the test DataLoader
    test_sampler = torch.utils.data.sampler.WeightedRandomSampler(
        weights=trajectory_dataset.sampling_probabilities[test_dataset.indices],
        num_samples=len(test_dataset),
        replacement=True,
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=8, sampler=test_sampler)


def test_train_test_split_other_data(trajectory_dataset_xz):

    # Split the dataset into training and testing sets
    train_dataset, test_dataset = random_split(
        trajectory_dataset_xz, [0.80, 0.20])

    assert len(train_dataset) == pytest.approx(
        0.80 * len(trajectory_dataset_xz), abs=1)
    assert len(test_dataset) == pytest.approx(
        0.20 * len(trajectory_dataset_xz), abs=1)

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
        weights=trajectory_dataset_xz.sampling_probabilities[train_dataset.indices],
        num_samples=len(train_dataset),
        replacement=True,
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=8, sampler=train_sampler)

    # Create the test DataLoader
    test_sampler = torch.utils.data.sampler.WeightedRandomSampler(
        weights=trajectory_dataset_xz.sampling_probabilities[test_dataset.indices],
        num_samples=len(test_dataset),
        replacement=True,
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=8, sampler=test_sampler)
