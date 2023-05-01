import torch
import pytest
import numpy as np

from src.decision_transformer.utils import (
    get_max_len_from_model_type,
    initialize_padding_inputs,
    get_optimizer,
)


def test_get_max_len_from_model_type_dt():
    assert get_max_len_from_model_type("decision_transformer", 2) == 1
    assert get_max_len_from_model_type("decision_transformer", 3) == 2
    assert get_max_len_from_model_type("decision_transformer", 4) == 2
    assert get_max_len_from_model_type("decision_transformer", 5) == 2
    assert get_max_len_from_model_type("decision_transformer", 6) == 3
    assert get_max_len_from_model_type("decision_transformer", 7) == 3
    assert get_max_len_from_model_type("decision_transformer", 8) == 3
    assert get_max_len_from_model_type("decision_transformer", 9) == 4
    assert get_max_len_from_model_type("decision_transformer", 10) == 4
    assert get_max_len_from_model_type("decision_transformer", 11) == 4
    assert get_max_len_from_model_type("decision_transformer", 12) == 5
    assert get_max_len_from_model_type("decision_transformer", 13) == 5


def test_get_max_len_from_model_type_bc():
    assert get_max_len_from_model_type("clone_transformer", 1) == 1
    assert get_max_len_from_model_type("clone_transformer", 2) == 2
    assert get_max_len_from_model_type("clone_transformer", 3) == 2
    assert get_max_len_from_model_type("clone_transformer", 4) == 3
    assert get_max_len_from_model_type("clone_transformer", 5) == 3
    assert get_max_len_from_model_type("clone_transformer", 6) == 4
    assert get_max_len_from_model_type("clone_transformer", 7) == 4
    assert get_max_len_from_model_type("clone_transformer", 8) == 5
    assert get_max_len_from_model_type("clone_transformer", 9) == 5
    assert get_max_len_from_model_type("clone_transformer", 10) == 6
    assert get_max_len_from_model_type("clone_transformer", 11) == 6
    assert get_max_len_from_model_type("clone_transformer", 12) == 7
    assert get_max_len_from_model_type("clone_transformer", 13) == 7


@pytest.mark.parametrize(
    "max_len, initial_obs, action_pad_token",
    [
        (3, {"image": np.zeros((3, 2, 2))}, 0),
        (5, {"image": np.ones((3, 2, 2))}, -1),
        (7, {"image": np.random.randn(2, 3, 4, 5)}, 2),
    ],
)
def test_initialize_padding_inputs(max_len, initial_obs, action_pad_token):
    initial_rtg = np.random.uniform(low=-1.0, high=1.0)
    batch_size = (
        initial_obs["image"].shape[0] if initial_obs["image"].ndim == 4 else 1
    )

    obs, actions, reward, rtg, timesteps, mask = initialize_padding_inputs(
        max_len=max_len,
        initial_obs=initial_obs,
        initial_rtg=initial_rtg,
        action_pad_token=action_pad_token,
        batch_size=batch_size,
    )

    # Test shapes
    dim_obs = initial_obs["image"].shape[-3:]
    assert obs.shape == (batch_size, max_len, *dim_obs)
    assert actions.shape == (batch_size, max_len - 1, 1)
    assert reward.shape == (batch_size, max_len, 1)
    assert rtg.shape == (batch_size, max_len, 1)
    assert timesteps.shape == (batch_size, max_len, 1)
    assert mask.shape == (batch_size, max_len)

    # Test types
    assert obs.dtype == torch.float64
    assert actions.dtype == torch.int64
    assert reward.dtype == torch.float32
    assert rtg.dtype == torch.float32
    assert timesteps.dtype == torch.int64
    assert mask.dtype == torch.bool

    # Test values
    assert np.all(obs.numpy()[:, -1, ...] == initial_obs["image"])
    assert np.all(actions.numpy() == action_pad_token)
    assert np.all(reward.numpy() == 0.0)
    assert np.all(rtg.numpy() == initial_rtg)
    assert np.all(timesteps.numpy() == 0)
    assert np.all(
        mask.numpy()
        == np.array(
            [
                [0] * (max_len - 1) + [1],
            ]
        )
    )


def test_initialize_padding_inputs_batch():
    max_len = 4
    initial_obs = {"image": np.zeros((2, 3, 2, 2))}
    initial_rtg = np.random.uniform(low=-1.0, high=1.0)
    action_pad_token = 0
    batch_size = 2

    obs, actions, reward, rtg, timesteps, mask = initialize_padding_inputs(
        max_len=max_len,
        initial_obs=initial_obs,
        initial_rtg=initial_rtg,
        action_pad_token=action_pad_token,
        batch_size=batch_size,
    )

    # Test shapes
    dim_obs = initial_obs["image"].shape[-3:]
    assert obs.shape == (batch_size, max_len, *dim_obs)
    assert actions.shape == (batch_size, max_len - 1, 1)
    assert reward.shape == (batch_size, max_len, 1)
    assert rtg.shape == (batch_size, max_len, 1)
    assert timesteps.shape == (batch_size, max_len, 1)
    assert mask.shape == (batch_size, max_len)

    # Test values
    assert np.all(obs.numpy()[:, -1, ...] == initial_obs["image"])
    assert np.all(actions.numpy() == action_pad_token)
    assert np.all(reward.numpy() == 0.0)
    assert np.all(rtg.numpy() == initial_rtg)
    assert np.all(timesteps.numpy() == 0)
    assert np.all(
        mask.numpy()
        == np.array(
            [
                [0] * (max_len - 1) + [1],
                [0] * (max_len - 1) + [1],
            ]
        )
    )


def test_get_optimizer():
    dummy_model = torch.nn.Linear(1, 1)
    assert isinstance(
        get_optimizer("Adam")(dummy_model.parameters(), 0.01), torch.optim.Adam
    )
    assert isinstance(
        get_optimizer("SGD")(dummy_model.parameters(), 0.01), torch.optim.SGD
    )
    assert isinstance(
        get_optimizer("AdamW")(dummy_model.parameters(), 0.01),
        torch.optim.AdamW,
    )
