import warnings

import numpy as np

from src.decision_transformer.calibration import (
    calibration_statistics,
    plot_calibration_statistics,
)
from src.decision_transformer.utils import load_decision_transformer
from src.environments.environments import make_env


def test_calibration_end_to_end():
    env_id = "MiniGrid-Dynamic-Obstacles-8x8-v0"
    model_path = "models/MiniGrid-Dynamic-Obstacles-8x8-v0/ReproduceOriginalPostShort.pt"

    dt = load_decision_transformer(model_path)

    env_func = make_env(dt.environment_config, seed=1, idx=0, run_name="dev")

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
