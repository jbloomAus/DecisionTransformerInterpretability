import pytest
from src.decision_transformer.utils import get_optim_groups
from src.models.trajectory_transformer import (
    DecisionTransformer,
    CloneTransformer,
)
from src.config import (
    EnvironmentConfig,
    OfflineTrainConfig,
    TransformerModelConfig,
)


@pytest.fixture
def environment_config():
    env_config = EnvironmentConfig()
    return env_config


@pytest.fixture
def transformer_config():
    transformer_config = TransformerModelConfig(n_ctx=26)
    return transformer_config


@pytest.fixture
def offline_config():
    offline_config = OfflineTrainConfig(trajectory_path="./tmp")
    return offline_config


@pytest.fixture
def decision_transformer(environment_config, transformer_config):
    dt = DecisionTransformer(environment_config, transformer_config)
    return dt


@pytest.fixture
def decision_transformer_ln(environment_config, transformer_config):
    transformer_config.layer_norm = "LN"
    dt = DecisionTransformer(environment_config, transformer_config)
    return dt


@pytest.fixture
def clone_transformer(environment_config, transformer_config):
    transformer_config.n_ctx = 13
    ct = CloneTransformer(transformer_config, environment_config)
    return ct


def test_get_optim_groups_dt(decision_transformer, offline_config):
    optim_groups = get_optim_groups(decision_transformer, offline_config)

    # Check if the length of optim_groups is 2
    assert len(optim_groups) == 2

    # Check if the weight_decay values are set correctly
    assert optim_groups[0]["weight_decay"] == offline_config.weight_decay
    assert optim_groups[1]["weight_decay"] == 0.0


def test_get_optim_groups_dt_ln(decision_transformer_ln, offline_config):
    optim_groups = get_optim_groups(decision_transformer_ln, offline_config)

    # Check if the length of optim_groups is 2
    assert len(optim_groups) == 2

    # Check if the weight_decay values are set correctly
    assert optim_groups[0]["weight_decay"] == offline_config.weight_decay
    assert optim_groups[1]["weight_decay"] == 0.0


def test_get_optim_groups_bc(clone_transformer, offline_config):
    optim_groups = get_optim_groups(clone_transformer, offline_config)

    # Check if the length of optim_groups is 2
    assert len(optim_groups) == 2

    # Check if the weight_decay values are set correctly
    assert optim_groups[0]["weight_decay"] == offline_config.weight_decay
    assert optim_groups[1]["weight_decay"] == 0.0
