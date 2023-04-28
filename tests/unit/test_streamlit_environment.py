import gymnasium as gym
import os
import pytest

from src.config import (
    EnvironmentConfig,
    OfflineTrainConfig,
    RunConfig,
    TransformerModelConfig,
)
from src.models.trajectory_transformer import DecisionTransformer

from src.decision_transformer.utils import store_transformer_model
from src.streamlit_app.environment import get_env_and_dt
from src.environments.registration import register_envs


@pytest.fixture()
def decision_transformer_path():
    register_envs()

    environment_config = EnvironmentConfig("MiniGrid-MemoryS7FixedStart-v0")
    offline_config = OfflineTrainConfig("test.pkl")
    run_config = RunConfig()
    transformer_config = TransformerModelConfig()

    model = DecisionTransformer(
        environment_config=environment_config,
        transformer_config=transformer_config,
    )
    if not os.path.exists("tmp"):
        os.mkdir("tmp")

    path = "tmp/dt.pt"
    store_transformer_model(
        path=path,
        model=model,
        offline_config=offline_config,
    )

    return path


def test_get_env_and_dt(decision_transformer_path):
    env, dt = get_env_and_dt(decision_transformer_path)

    assert isinstance(dt, DecisionTransformer)
    assert isinstance(env, gym.Env)
