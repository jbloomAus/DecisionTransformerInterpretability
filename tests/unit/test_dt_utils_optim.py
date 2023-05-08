import torch
import pytest
from src.decision_transformer.utils import get_optim_groups


# Create a simple model for testing
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.layer_norm = torch.nn.LayerNorm(2)
        self.embedding = torch.nn.Embedding(10, 2)


# Create a dummy offline_config object for testing
class DummyOfflineConfig:
    def __init__(self, weight_decay):
        self.weight_decay = weight_decay


@pytest.fixture
def model():
    return SimpleModel()


@pytest.fixture
def offline_config():
    return DummyOfflineConfig(weight_decay=0.01)


def test_get_optim_groups(model, offline_config):
    optim_groups = get_optim_groups(model, offline_config)

    # Check if the length of optim_groups is 2
    assert len(optim_groups) == 2

    # Check if the weight_decay values are set correctly
    assert optim_groups[0]["weight_decay"] == offline_config.weight_decay
    assert optim_groups[1]["weight_decay"] == 0.0

    # Check if the parameters are correctly separated into decay and no_decay sets
    decay_params = {
        pn
        for pn, _ in model.linear.named_parameters()
        if pn.endswith("weight")
    }
    no_decay_params = {
        pn
        for m in [model.layer_norm, model.embedding]
        for pn, _ in m.named_parameters()
    }
    no_decay_params |= {
        pn for pn, _ in model.linear.named_parameters() if pn.endswith("bias")
    }

    # use shapes as a proxy for tensor origin:

    assert len(decay_params) == 1
    assert len(no_decay_params) == 2

    assert optim_groups[0]["params"][0].shape == model.linear.weight.shape

    assert optim_groups[1]["params"][0].shape == model.embedding.weight.shape
    assert optim_groups[1]["params"][2].shape == model.linear.bias.shape
    assert optim_groups[1]["params"][3].shape == model.layer_norm.bias.shape
