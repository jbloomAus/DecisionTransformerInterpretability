import pytest

import warnings
import torch as t
import numpy as np
from src.environments import make_env
from src.decision_transformer.utils import load_decision_transformer
from src.decision_transformer.calibration import calibration_statistics, plot_calibration_statistics


def test_calibration_end_to_end():

    env_id = "MiniGrid-Dynamic-Obstacles-8x8-v0"
    model_path = "models/MiniGrid-Dynamic-Obstacles-8x8-v0/demo_model_overnight_training.pt"
    state_dict = t.load(model_path)
    one_hot_encoded = state_dict["state_encoder.weight"].shape[-1] == 980
    max_time_steps = state_dict["time_embedding.weight"].shape[0]
    env_func = make_env(
        env_id, seed=1, idx=0,
        capture_video=False, run_name="dev",
        fully_observed=False, flat_one_hot=one_hot_encoded, max_steps=max_time_steps)
    env = env_func()

    state_dict = t.load(model_path)
    one_hot_encoded = state_dict["state_encoder.weight"].shape[-1] == 980

    dt = load_decision_transformer(model_path, env)

    warnings.filterwarnings("ignore", category=UserWarning)
    statistics = calibration_statistics(
        dt,
        env_id,
        env_func=env_func,
        initial_rtg_range=np.linspace(-1, 1, 10),
        trajectories=3,
    )

    assert statistics is not None
    assert len(statistics) == 10
    fig = plot_calibration_statistics(statistics)

    assert fig is not None


def test_calibration_end_to_end_one_hot_model():

    env_id = "MiniGrid-Dynamic-Obstacles-8x8-v0"
    model_path = "models/MiniGrid-Dynamic-Obstacles-8x8-v0/demo_model_one_hot_overnight.pt"
    state_dict = t.load(model_path)
    one_hot_encoded = state_dict["state_encoder.weight"].shape[-1] == 980
    max_time_steps = state_dict["time_embedding.weight"].shape[0]
    env_func = make_env(
        env_id, seed=1, idx=0,
        capture_video=False, run_name="dev",
        fully_observed=False, flat_one_hot=one_hot_encoded, max_steps=max_time_steps)
    env = env_func()

    state_dict = t.load(model_path)
    one_hot_encoded = state_dict["state_encoder.weight"].shape[-1] == 980

    dt = load_decision_transformer(model_path, env)

    warnings.filterwarnings("ignore", category=UserWarning)
    statistics = calibration_statistics(
        dt,
        env_id,
        env_func=env_func,
        initial_rtg_range=np.linspace(-1, 1, 10),
        trajectories=3,
    )

    assert statistics is not None
    assert len(statistics) == 10
    fig = plot_calibration_statistics(statistics)

    assert fig is not None
