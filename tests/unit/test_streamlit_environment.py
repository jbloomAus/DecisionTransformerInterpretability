import gymnasium as gym
import pytest

from src.decision_transformer.model import DecisionTransformer
from src.streamlit_app.environment import get_env_and_dt


def test_get_env_and_dt():
    env, dt = get_env_and_dt(
        "models/MiniGrid-Dynamic-Obstacles-8x8-v0/demo_model_overnight_training.pt"
    )

    assert isinstance(dt, DecisionTransformer)
    assert isinstance(env, gym.Env)
    assert dt.env == env
