import json
import os
import pytest
import torch

from src.config import EnvironmentConfig, ConfigJsonEncoder, TransformerModelConfig, OfflineTrainConfig
from src.decision_transformer.offline_dataset import TrajectoryDataset
from src.models.trajectory_model import DecisionTransformer
from src.utils import load_model_data


@pytest.fixture()
def cleanup_test_results() -> None:
    yield
    os.remove('models/model_data.pt')


def test_load_model_data(cleanup_test_results):
    transformer_config = TransformerModelConfig(
        d_model=128,
        n_heads=4,
        d_mlp=256,
        n_layers=2,
        n_ctx=3,
        layer_norm=False,
        state_embedding_type='grid',
        time_embedding_type='embedding',
        seed=1,
        device='cpu'
    )

    offline_config = OfflineTrainConfig(
        trajectory_path='trajectories/MiniGrid-DoorKey-8x8-trajectories.pkl',
        batch_size=128,
        lr=0.0001,
        weight_decay=0.0,
        pct_traj=1.0,
        prob_go_from_end=0.0,
        device='cpu',
        track=False,
        train_epochs=100,
        test_epochs=10,
        test_frequency=10,
        eval_frequency=10,
        eval_episodes=10,
        model_type='decision_transformer',
        initial_rtg=[0.0, 1.0],
        eval_max_time_steps=100
    )

    trajectory_data_set = TrajectoryDataset(
        trajectory_path=offline_config.trajectory_path,
        max_len=transformer_config.n_ctx // 3,
        pct_traj=offline_config.pct_traj,
        prob_go_from_end=offline_config.prob_go_from_end,
        device=transformer_config.device,
    )

    environment_config = EnvironmentConfig(
        env_id=trajectory_data_set.metadata['args']['env_id'],
        one_hot_obs=trajectory_data_set.observation_type == "one_hot",
        view_size=trajectory_data_set.metadata['args']['view_size'],
        fully_observed=False,
        capture_video=False,
        render_mode='rgb_array')

    model = DecisionTransformer(
        environment_config=environment_config,
        transformer_config=transformer_config
    )

    torch.save({
        "model_state_dict": model.state_dict(),
        "transformer_config": json.dumps(transformer_config, cls=ConfigJsonEncoder),
        "offline_config": json.dumps(offline_config, cls=ConfigJsonEncoder),
    }, "models/model_data.pt")

    state_dict, _, loaded_transformer_config, loaded_offline_config = \
        load_model_data("models/model_data.pt")

    assert_state_dicts_are_equal(state_dict, model.state_dict())
    assert loaded_transformer_config == transformer_config
    assert loaded_offline_config == offline_config


def assert_state_dicts_are_equal(dict1, dict2):
    keys1 = sorted(dict1.keys())
    keys2 = sorted(dict2.keys())

    assert keys1 == keys2

    for key1, key2 in zip(keys1, keys2):
        assert dict1[key1].equal(dict2[key2])
