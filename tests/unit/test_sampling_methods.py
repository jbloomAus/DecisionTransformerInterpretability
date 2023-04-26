import numpy as np
import pytest
import torch
from torch.distributions.categorical import Categorical

from src.utils.sampling_methods import (
    sample_from_categorical,
    basic_sample,
    bottomk_sample,
    greedy_sample,
    temp_sample,
    topk_sample,
)


@pytest.fixture
def random_probs():
    return torch.rand(5, 10)  # 5 is the batch size


@pytest.fixture
def categorical_random(random_probs):
    return Categorical(probs=random_probs)


def test_basic(categorical_random, random_probs):
    indices = basic_sample(categorical_random)
    assert indices.shape == random_probs.shape[:-1]
    assert (indices >= 0).all() and (indices < len(random_probs[0])).all()


def test_greedy(categorical_random, random_probs):
    indices = greedy_sample(categorical_random)
    assert indices.shape == random_probs.shape[:-1]
    assert torch.equal(indices, random_probs.argmax(dim=-1))


@pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0])
def test_temperature(categorical_random, temperature):
    indices = temp_sample(categorical_random, temperature)
    assert indices.shape == categorical_random.probs.shape[:-1]
    assert (indices >= 0).all() and (
        indices < len(categorical_random.probs[0])
    ).all()


def test_temperature_sampling_distributions(categorical_random, random_probs):
    # just use the first batch dim to keep things simple here
    random_probs = random_probs[0]
    categorical_random = Categorical(probs=random_probs)

    # Now lets test the properties of temperature sampling
    num_samples = 1000

    # Test for temperature close to zero (near-greedy behavior)
    low_temperature = 1e-6
    samples_low_temp = np.array(
        [
            temp_sample(categorical_random, low_temperature)
            for _ in range(num_samples)
        ]
    )
    unique_samples_low_temp = np.unique(samples_low_temp)
    assert (
        len(unique_samples_low_temp) == 1
        and unique_samples_low_temp[0] == random_probs.argmax().item()
    )

    # Test for high temperature (diverse sampling)
    high_temperature = 5.0
    samples_high_temp = np.array(
        [
            temp_sample(categorical_random, high_temperature)
            for _ in range(num_samples)
        ]
    )
    unique_samples_high_temp = np.unique(samples_high_temp)
    assert (
        len(unique_samples_high_temp) > 1
    )  # More than one unique sample should be observed

    # Test for very high temperature (closer to uniform sampling)
    very_high_temperature = 50.0
    samples_very_high_temp = np.array(
        [
            temp_sample(categorical_random, very_high_temperature)
            for _ in range(num_samples)
        ]
    )
    counts = np.bincount(samples_very_high_temp)
    normalized_counts = counts / num_samples
    uniform_prob = 1 / len(random_probs)
    tolerance = 0.05  # Tolerance for deviation from uniform probability
    assert np.allclose(normalized_counts, uniform_prob, atol=tolerance)


@pytest.mark.parametrize("k", [1, 3, 5])
def test_topk_sample(categorical_random, k):
    indices = topk_sample(categorical_random, k)
    assert indices.shape == categorical_random.probs.shape[:-1]
    _, top_k_indices = categorical_random.probs.topk(k, dim=-1)

    for i in range(len(indices)):
        assert (
            indices[i] in top_k_indices[i]
        ), f"{indices[i]} not in {top_k_indices[i]}"


@pytest.mark.parametrize("k", [1, 3, 5])
def test_bottomk_sample(categorical_random, k):
    indices = bottomk_sample(categorical_random, k)
    assert indices.shape == categorical_random.probs.shape[:-1]
    _, bottom_k_indices = categorical_random.probs.topk(
        k, dim=-1, largest=False
    )

    for i in range(len(indices)):
        assert indices[i] in bottom_k_indices[i]


def test_sample_from_categorical_with_batch_dim(random_probs):
    random_probs_batch = random_probs
    categorical_random = Categorical(probs=random_probs_batch)

    # Test basic sampling
    indices = sample_from_categorical(categorical_random, method="basic")
    assert torch.all((indices >= 0) & (indices < len(random_probs_batch[0])))

    # Test greedy sampling
    indices = sample_from_categorical(categorical_random, method="greedy")
    assert torch.all(indices == random_probs_batch.argmax(dim=-1))

    # Test temperature sampling
    temperature = 0.5
    indices = sample_from_categorical(
        categorical_random, method="temp", temperature=temperature
    )
    assert torch.all((indices >= 0) & (indices < len(random_probs_batch[0])))

    # Test top-k sampling
    k = 3
    indices = sample_from_categorical(categorical_random, method="topk", k=k)
    _, top_k_indices = random_probs_batch.topk(k, dim=-1)
    assert torch.all(torch.any(indices.unsqueeze(-1) == top_k_indices, dim=-1))

    # Test bottom-k sampling
    k = 3
    indices = sample_from_categorical(
        categorical_random, method="bottomk", k=k
    )
    _, bottom_k_indices = random_probs_batch.topk(k, dim=-1, largest=False)
    assert torch.all(
        torch.any(indices.unsqueeze(-1) == bottom_k_indices, dim=-1)
    )

    # Test invalid method
    with pytest.raises(ValueError):
        sample_from_categorical(categorical_random, method="invalid_method")
