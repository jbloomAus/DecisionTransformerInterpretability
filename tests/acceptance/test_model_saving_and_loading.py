import json
import os

import gymnasium as gym
import pytest
import torch

from src.config import (
    EnvironmentConfig,
    OfflineTrainConfig,
    OnlineTrainConfig,
    RunConfig,
    TransformerModelConfig,
)

from src.decision_transformer.runner import store_transformer_model
from src.decision_transformer.offline_dataset import TrajectoryDataset
from src.decision_transformer.utils import load_decision_transformer
from src.models.trajectory_transformer import DecisionTransformer


@pytest.fixture()
def cleanup_test_results() -> None:
    if not os.path.exists("tmp"):
        os.mkdir("tmp")
    if os.path.exists("tmp/model_data.pt"):
        os.remove("tmp/model_data.pt")


@pytest.fixture()
def run_config() -> RunConfig:
    run_config = RunConfig(
        exp_name="Test-PPO-Basic",
        seed=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        track=False,
        wandb_project_name="PPO-MiniGrid",
        wandb_entity=None,
    )

    return run_config


@pytest.fixture()
def environment_config() -> EnvironmentConfig:
    environment_config = EnvironmentConfig(
        env_id="MiniGrid-Dynamic-Obstacles-8x8-v0",
        view_size=7,
        max_steps=300,
        one_hot_obs=False,
        fully_observed=False,
        render_mode="rgb_array",
        capture_video=True,
        video_dir="videos",
    )
    return environment_config


@pytest.fixture()
def online_config() -> OnlineTrainConfig:
    online_config = OnlineTrainConfig(
        hidden_size=64,
        total_timesteps=2000,
        learning_rate=0.00025,
        decay_lr=True,
        num_envs=30,
        num_steps=64,
        gamma=0.99,
        gae_lambda=0.95,
        num_minibatches=30,
        update_epochs=4,
        clip_coef=0.4,
        ent_coef=0.25,
        vf_coef=0.5,
        max_grad_norm=2,
        trajectory_path="trajectories/MiniGrid-DoorKey-8x8-trajectories.pkl",
    )
    return online_config


@pytest.fixture()
def transformer_config() -> TransformerModelConfig:
    transformer_config = TransformerModelConfig(
        d_model=128,
        n_heads=4,
        d_mlp=256,
        n_layers=2,
        n_ctx=5,
        layer_norm=None,
        state_embedding_type="grid",
        time_embedding_type="embedding",
        seed=1,
        device="cpu",
    )

    return transformer_config


@pytest.fixture()
def offline_config() -> OfflineTrainConfig:
    offline_config = OfflineTrainConfig(
        trajectory_path="trajectories/MiniGrid-DoorKey-8x8-trajectories.pkl",
        batch_size=128,
        lr=0.0001,
        weight_decay=0.0,
        pct_traj=1.0,
        prob_go_from_end=0.0,
        device="cpu",
        track=False,
        train_epochs=100,
        test_epochs=10,
        test_frequency=10,
        eval_frequency=10,
        eval_episodes=10,
        model_type="decision_transformer",
        initial_rtg=[0.0, 1.0],
        eval_max_time_steps=100,
        eval_num_envs=8,
    )
    return offline_config


def test_load_decision_transformer(
    transformer_config,
    offline_config,
    environment_config,
    cleanup_test_results,
):
    model = DecisionTransformer(
        environment_config=environment_config,
        transformer_config=transformer_config,
    )

    path = "tmp/model_data.pt"
    store_transformer_model(
        path=path,
        model=model,
        offline_config=offline_config,
    )

    new_model = load_decision_transformer(path)

    assert_state_dicts_are_equal(new_model.state_dict(), model.state_dict())

    assert new_model.transformer_config == transformer_config
    assert new_model.environment_config == environment_config


def test_load_decision_transformer_with_processing(
    transformer_config,
    offline_config,
    environment_config,
    cleanup_test_results,
):
    transformer_config.layer_norm = "LN"
    model = DecisionTransformer(
        environment_config=environment_config,
        transformer_config=transformer_config,
    )

    # in order to ensure that this test works, perturb the weights of the model (specifically, the ln_final weights)
    # and then check that the weights are corrected when the model is loaded.
    # use torch init on ln_final w and b
    torch.nn.init.normal_(model.transformer.ln_final.w, mean=0.0, std=1.0)
    torch.nn.init.normal_(model.transformer.ln_final.b, mean=0.0, std=1.0)
    # also do blocks.0.ln1.w for blocks 0 and 1 and b
    torch.nn.init.normal_(model.transformer.blocks[0].ln1.w, mean=0.0, std=1.0)
    torch.nn.init.normal_(model.transformer.blocks[0].ln1.b, mean=0.0, std=1.0)
    torch.nn.init.normal_(model.transformer.blocks[1].ln1.w, mean=0.0, std=1.0)
    torch.nn.init.normal_(model.transformer.blocks[1].ln1.b, mean=0.0, std=1.0)

    path = "tmp/model_data.pt"
    store_transformer_model(
        path=path,
        model=model,
        offline_config=offline_config,
    )

    new_model = load_decision_transformer(path, tlens_weight_processing=True)

    transformer_config.layer_norm = "LNPre"  # we expect this to change.
    assert new_model.transformer_config == transformer_config
    assert new_model.environment_config == environment_config

    # the state dicts should diverge now make it hard to compare them.
    # what might work is to test that the forward pass produces the same results.
    # yet again I wish I had a convenience method for generating inputs.
    env = gym.make("MiniGrid-Empty-8x8-v0")
    obs, _ = env.reset()
    states = torch.tensor([obs["image"], obs["image"]]).unsqueeze(
        0
    )  # add block, add batch
    actions = (
        torch.tensor([0]).unsqueeze(0).unsqueeze(0)
    )  # add block, add batch
    rewards = torch.tensor([[0], [0]]).unsqueeze(0)  # add block, add batch
    timesteps = torch.tensor([[0], [1]]).unsqueeze(0)  # add block, add batch

    _, action_preds_original, _ = model.forward(
        states=states, actions=actions, rtgs=rewards, timesteps=timesteps
    )

    _, action_preds_new, _ = new_model.forward(
        states=states, actions=actions, rtgs=rewards, timesteps=timesteps
    )

    assert torch.allclose(action_preds_original, action_preds_new)
    assert not model.state_dict().keys() == new_model.state_dict().keys()
    # assert_state_dicts_are_equal(new_model.state_dict(), model.state_dict())

    # these shouldn't be the same.
    old_reward_predictor_weight = model.reward_predictor.weight
    new_reward_predictor_weight = new_model.reward_predictor.weight
    assert not torch.allclose(
        old_reward_predictor_weight, new_reward_predictor_weight
    )

    ln_final_b = model.transformer.ln_final.b
    ln_final_w = model.transformer.ln_final.w

    # state_dict[f"unembed.W_U"] = (
    #     state_dict[f"unembed.W_U"] * state_dict[f"ln_final.w"][:, None]
    # )
    expected_reward_predictor_weight = (
        old_reward_predictor_weight * ln_final_w[:, None].T
    )

    #     state_dict[f"unembed.b_U"] = state_dict[f"unembed.b_U"] + (
    #         state_dict[f"unembed.W_U"] * state_dict[f"ln_final.b"][:, None]
    #     ).sum(dim=-2)
    #     del state_dict[f"ln_final.b"]
    expected_reward_predictor_bias = model.reward_predictor.bias + (
        (old_reward_predictor_weight * ln_final_b[:, None].T).sum(dim=-1)
    )

    assert not torch.allclose(ln_final_b, torch.zeros_like(ln_final_b))
    assert not torch.allclose(ln_final_w, torch.ones_like(ln_final_w))
    assert torch.allclose(
        expected_reward_predictor_weight, new_reward_predictor_weight
    )
    assert torch.allclose(
        expected_reward_predictor_bias, new_model.reward_predictor.bias
    )

    # # Center the weights that read in from the LayerNormPre
    # state_dict[f"unembed.W_U"] -= einops.reduce(
    #     state_dict[f"unembed.W_U"],
    #     "d_model d_vocab -> 1 d_vocab",
    #     "mean")

    # del state_dict[f"ln_final.w"]


def assert_state_dicts_are_equal(dict1, dict2):
    keys1 = sorted(dict1.keys())
    keys2 = sorted(dict2.keys())

    assert keys1 == keys2

    for key1, key2 in zip(keys1, keys2):
        assert dict1[key1].equal(dict2[key2])
