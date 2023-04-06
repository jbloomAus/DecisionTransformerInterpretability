import os
from dataclasses import dataclass

import gymnasium as gym
import pandas as pd
import pytest
import torch
from gymnasium.spaces import Discrete

import wandb
from src.config import (EnvironmentConfig, LSTMModelConfig,
                        TransformerModelConfig)
from src.environments.environments import make_env
from src.environments.registration import register_envs
from src.ppo.agent import (FCAgent, LSTMPPOAgent, TransformerPPOAgent,
                           load_all_agents_from_checkpoints,
                           load_saved_checkpoint, sample_from_agents)
from src.ppo.memory import Memory, Minibatch, TrajectoryMinibatch
from src.ppo.train import train_ppo
from src.ppo.utils import store_model_checkpoint

register_envs()


def compare_state_dicts(state_dict1, state_dict2):
    for key in state_dict1:
        if key not in state_dict2:
            print(f"Key {key} not found in state_dict2.")
            return False
        if not torch.allclose(state_dict1[key], state_dict2[key]):
            print(
                f"Value of key {key} is different in state_dict1 and state_dict2.")
            return False
    for key in state_dict2:
        if key not in state_dict1:
            print(f"Key {key} not found in state_dict1.")
            return False
    return True


@pytest.fixture
def run_config():
    @dataclass
    class DummyRunConfig:
        exp_name: str = 'test'
        seed: int = 1
        track: bool = False
        wandb_project_name: str = 'test'
        wandb_entity: str = 'test'
        device = torch.device('cpu')

    return DummyRunConfig()


@pytest.fixture
def environment_config():
    @dataclass
    class DummyEnvironmentConfig:
        env_id: str = 'MiniGrid-Dynamic-Obstacles-8x8-v0'
        one_hot_obs: bool = False
        img_obs: bool = False
        fully_observed: bool = False
        max_steps: int = 30
        seed: int = 1
        view_size: int = 7
        capture_video: bool = False
        video_dir: str = 'videos'
        render_mode: str = 'rgb_array'
        action_space: None = None
        observation_space: None = None
        device: torch.device = torch.device("cpu")

    return DummyEnvironmentConfig()


@pytest.fixture
def online_config():
    @dataclass
    class DummyOnlineConfig:
        use_trajectory_model: bool = False
        hidden_size: int = 64
        total_timesteps: int = 1000
        learning_rate: float = 0.00025
        decay_lr: bool = False,
        num_envs: int = 10
        num_steps: int = 128
        gamma: float = 0.99
        gae_lambda: float = 0.95
        num_minibatches: int = 10
        update_epochs: int = 4
        clip_coef: float = 0.2
        ent_coef: float = 0.01
        vf_coef: float = 0.5
        max_grad_norm: float = 2
        trajectory_path: str = None
        fully_observed: bool = False
        batch_size: int = 64
        minibatch_size: int = 4
        prob_go_from_end: float = 0.0
        device: torch.device = torch.device("cpu")

    return DummyOnlineConfig()


@pytest.fixture
def transformer_model_config():
    @dataclass
    class DummyTransformerModelConfig:
        d_model: int = 128
        n_heads: int = 2
        d_mlp: int = 256
        n_layers: int = 1
        n_ctx: int = 1
        time_embedding_type: str = "embedding"
        state_embedding_type: str = "grid"
        seed: int = 1
        device: str = "cpu"
        d_head: int = 64  # d_model // n_heads
        layer_norm = False

    return DummyTransformerModelConfig()


@pytest.fixture
def large_transformer_model_config():
    @dataclass
    class DummyTransformerModelConfig:
        d_model: int = 128
        n_heads: int = 2
        d_mlp: int = 256
        n_layers: int = 1
        n_ctx: int = 1
        time_embedding_type: str = "embedding"
        state_embedding_type: str = "grid"
        seed: int = 1
        device: str = "cpu"
        d_head: int = 64  # d_model // n_heads
        layer_norm = False

    return DummyTransformerModelConfig()


@pytest.fixture
def lstm_config(environment_config):
    @dataclass
    class DummyLSTMConfig:
        environment_config = environment_config
        image_dim: int = 128
        memory_dim: int = 128
        instr_dim: int = 128
        use_instr: bool = False
        lang_model: str = 'gru'
        use_memory: bool = False
        recurrence: int = 4
        arch: str = "bow_endpool_res"
        aux_info: bool = False
        endpool: bool = True
        bow: bool = True
        pixel: bool = False
        res: bool = True
        device = torch.device("cpu")

    config = DummyLSTMConfig()
    config.environment_config = environment_config
    config.action_space = environment_config.action_space
    config.observation_space = environment_config.observation_space

    return config


@pytest.fixture()
def fc_agent():
    environment_config = EnvironmentConfig()
    envs = gym.vector.SyncVectorEnv(
        [make_env(environment_config, 1, 1, "test") for i in range(2)]
    )
    return FCAgent(envs, environment_config, device="cpu")


@pytest.fixture()
def transformer_agent():
    envs = gym.vector.SyncVectorEnv(
        [make_env(EnvironmentConfig(), 1, 1, "test") for i in range(2)]
    )
    transformer_agent = TransformerPPOAgent(envs, environment_config=EnvironmentConfig(),
                                            transformer_model_config=TransformerModelConfig(
                                                n_ctx=3),
                                            device=torch.device("cpu"))
    return transformer_agent


@pytest.fixture()
def lstm_agent():
    envs = gym.vector.SyncVectorEnv(
        [make_env(EnvironmentConfig(), 1, 1, "test") for i in range(2)]
    )
    lstm_agent = LSTMPPOAgent(envs, environment_config=EnvironmentConfig(),
                              lstm_config=LSTMModelConfig(EnvironmentConfig()),
                              device=torch.device("cpu"))
    return lstm_agent


@pytest.fixture()
def lstm_agents():
    path = "models/ppo/memory_lstm_demos"
    agents = load_all_agents_from_checkpoints(path, 4)
    return agents


def test_empty_env(run_config, environment_config, online_config):

    env_name = "MiniGrid-Empty-5x5-v0"

    envs = gym.vector.SyncVectorEnv(
        [make_env(environment_config, i, i, "test") for i in range(4)]
    )

    environment_config.env_id = env_name
    ppo = train_ppo(
        run_config=run_config,
        online_config=online_config,
        environment_config=environment_config,
        model_config=None,
        envs=envs
    )


def test_empty_env_flat_one_hot(run_config, environment_config, online_config):

    env_name = "MiniGrid-Empty-5x5-v0"
    envs = gym.vector.SyncVectorEnv(
        [make_env(environment_config, i, i, "test") for i in range(4)]
    )

    environment_config.env_id = env_name
    environment_config.one_hot_obs = True
    ppo = train_ppo(
        run_config=run_config,
        online_config=online_config,
        environment_config=environment_config,
        model_config=None,
        envs=envs
    )


def test_ppo_agent_gym():

    env_config = EnvironmentConfig(
        env_id='CartPole-v1', capture_video=False, max_steps=None)
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_config, 1, 1, "test") for i in range(2)]
    )
    assert envs.single_action_space.shape is not None
    assert isinstance(envs.single_action_space,
                      Discrete), "only discrete action space is supported"

    # memory = Memory(envs, args, "cpu")
    agent = FCAgent(envs, env_config, device="cpu")

    assert agent.num_obs == 4
    assert agent.num_actions == 2


def test_ppo_agent_minigrid():

    env_config = EnvironmentConfig(
        env_id='MiniGrid-Empty-8x8-v0', capture_video=False, max_steps=None)
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_config, 1, 1, "test") for i in range(2)]
    )
    assert envs.single_action_space.shape is not None
    assert isinstance(envs.single_action_space,
                      Discrete), "only discrete action space is supported"

    # memory = Memory(envs, args, "cpu")
    agent = FCAgent(envs, environment_config, device="cpu")

    # depends on whether you wrapped in Fully observed or not
    assert agent.num_obs == 7 * 7 * 3
    assert agent.num_actions == 7


def test_ppo_agent_rollout_minibatches_minigrid(online_config):

    env_config = EnvironmentConfig(
        env_id='MiniGrid-Empty-8x8-v0', capture_video=False, max_steps=None)
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_config, 1, 1, "test") for i in range(2)]
    )
    assert envs.single_action_space.shape is not None
    assert isinstance(envs.single_action_space,
                      Discrete), "only discrete action space is supported"

    memory = Memory(envs, online_config)

    agent = FCAgent(envs, env_config, "cpu")
    agent.rollout(memory, online_config.num_steps, envs, None)

    minibatches = memory.get_minibatches()
    assert len(
        minibatches) == online_config.batch_size // online_config.minibatch_size

    observation_shape = envs.single_observation_space['image'].shape
    minibatch = minibatches[0]
    assert isinstance(minibatch, Minibatch)
    assert minibatch.obs.shape == (
        online_config.minibatch_size, *observation_shape), "obs shape is wrong"
    assert minibatch.actions.shape == (
        online_config.minibatch_size, ), "actions shape is wrong"
    assert minibatch.values.shape == (
        online_config.minibatch_size, ), "values shape is wrong"
    assert minibatch.logprobs.shape == (
        online_config.minibatch_size, ), "log_probs shape is wrong"
    assert minibatch.returns.shape == (
        online_config.minibatch_size, ), "returns shape is wrong"
    assert minibatch.advantages.shape == (
        online_config.minibatch_size, ), "advantages shape is wrong"


@pytest.mark.parametrize("n_ctx", [1, 3, 9])
def test_ppo_traj_agent_rollout_minibatches(
        online_config, environment_config, transformer_model_config, n_ctx):

    transformer_model_config.n_ctx = n_ctx
    envs = gym.vector.SyncVectorEnv(
        [make_env(environment_config, 1, 1, "test") for i in range(2)]
    )
    memory = Memory(envs, online_config)
    environment_config.action_space = envs.single_action_space
    environment_config.observation_space = envs.single_observation_space
    agent = TransformerPPOAgent(
        envs, environment_config, transformer_model_config)
    agent.rollout(memory, online_config.num_steps, envs, None)

    timesteps = (transformer_model_config.n_ctx - 1) // 2 + 1
    minibatches = memory.get_trajectory_minibatches(timesteps)
    assert len(minibatches) > 0

    minibatch = minibatches[0]

    assert isinstance(minibatch, TrajectoryMinibatch)
    assert minibatch.obs.shape[1] == timesteps
    assert minibatch.obs.shape[0] == minibatch.actions.shape[0] == minibatch.logprobs.shape[0] == \
        minibatch.advantages.shape[0] == minibatch.values.shape[0] == minibatch.returns.shape[0]
    assert minibatch.obs.ndim == 2 + \
        len(envs.single_observation_space['image'].shape)
    assert minibatch.actions.ndim == 2
    assert minibatch.logprobs.ndim == minibatch.advantages.ndim == minibatch.values.ndim == minibatch.returns.ndim == 1
    assert minibatch.timesteps.shape == minibatch.obs.shape[:2]
    assert minibatch.timesteps.max() <= environment_config.max_steps


@pytest.mark.parametrize("n_ctx", [1, 3, 9])
def test_ppo_traj_agent_rollout_and_learn_minibatches(
        online_config, environment_config, transformer_model_config, n_ctx):

    transformer_model_config.n_ctx = n_ctx
    envs = gym.vector.SyncVectorEnv(
        [make_env(environment_config, 1, 1, "test") for i in range(2)]
    )

    memory = Memory(envs, online_config)
    environment_config.action_space = envs.single_action_space
    environment_config.observation_space = envs.single_observation_space
    agent = TransformerPPOAgent(
        envs, environment_config, transformer_model_config)

    num_updates = online_config.total_timesteps // online_config.batch_size
    optimizer, scheduler = agent.make_optimizer(
        num_updates=num_updates,
        initial_lr=online_config.learning_rate,
        end_lr=online_config.learning_rate if not online_config.decay_lr else 0.0)

    agent.rollout(memory, online_config.num_steps, envs, None)
    agent.learn(memory, online_config, optimizer, scheduler, track=False)


# skip this test for now
@pytest.mark.skip(reason="not implemented yet -> need to add environment config to bring it in line with other two agent_classes")
def test_fc_agent_model_checkpoint_saving_and_loading(fc_agent, run_config, online_config):

    wandb.init(mode="offline")
    run_config.track = True
    checkpoint_artifact = wandb.Artifact(
        f"{run_config.exp_name}_checkpoints", type="model")
    checkpoint_num = 1

    # save checkpoint
    checkpoint_num = store_model_checkpoint(
        fc_agent, online_config, run_config, checkpoint_num, checkpoint_artifact)

    assert checkpoint_num == 2


def test_traj_ppo_model_checkpoint_saving_and_loading(transformer_agent, run_config, online_config):

    wandb.init(mode="offline")
    run_config.track = True
    run_config.exp_name = "TRANSFORMERTEST"
    checkpoint_artifact = wandb.Artifact(
        f"{run_config.exp_name}_checkpoints", type="model")
    checkpoint_num = 1

    # save checkpoint
    checkpoint_num = store_model_checkpoint(
        transformer_agent, online_config, run_config, checkpoint_num, checkpoint_artifact)

    assert checkpoint_num == 2

    agent = load_saved_checkpoint(
        "./models/TRANSFORMERTEST_01.pt", online_config.num_envs)

    assert transformer_agent.environment_config == agent.environment_config
    assert compare_state_dicts(
        transformer_agent.state_dict(), agent.state_dict())


def test_lstm_ppo_model_checkpoint_saving_and_loading(lstm_agent, run_config, online_config):

    wandb.init(mode="offline")
    run_config.track = True
    run_config.exp_name = "LSTMTEST"
    checkpoint_artifact = wandb.Artifact(
        f"{run_config.exp_name}_checkpoints", type="model")
    checkpoint_num = 1

    # save checkpoint
    checkpoint_num = store_model_checkpoint(
        lstm_agent, online_config, run_config, checkpoint_num, checkpoint_artifact)

    assert checkpoint_num == 2

    agent = load_saved_checkpoint(
        "./models/LSTMTEST_01.pt", online_config.num_envs)

    assert lstm_agent.environment_config == agent.environment_config
    assert compare_state_dicts(
        lstm_agent.model.state_dict(), agent.model.state_dict())


def test_lstm_ppo_model_load_saved_checkpoints():

    path = "models/ppo/memory_lstm_demos"
    agents = load_all_agents_from_checkpoints(path, 4)
    agent = agents[0]

    assert len(agents) == 2
    assert isinstance(agent, LSTMPPOAgent)
    assert len(agent.envs.envs) == 4


def test_sample_from_agents(lstm_agents):

    rollout_length = 200
    max_steps = lstm_agents[0].environment_config.max_steps
    num_envs = 4

    all_episode_lengths, all_episode_returns = sample_from_agents(
        lstm_agents,
        rollout_length=rollout_length,
        trajectory_path="tmp/test_sample_from_agents",
        num_envs=num_envs,
    )

    # list all of the files in that folder
    files = os.listdir("tmp/test_sample_from_agents")
    assert len(files) == 2
    # assert they end in gz
    assert files[0].endswith(".gz")
    assert files[1].endswith(".gz")

    # assert all_episode_lengths is a list of pandas dataframes
    assert isinstance(all_episode_lengths, list)
    assert isinstance(all_episode_lengths[0], pd.Series)

    # do the same for all_episode_returns
    assert isinstance(all_episode_returns, list)
    assert isinstance(all_episode_returns[0], pd.Series)

    for agent_index in range(len(lstm_agents)):
        assert len(all_episode_lengths[agent_index]) == len(
            all_episode_returns[agent_index])
        # we expect to finish enough episodes to have almost the rollout_length * num_envs steps
        assert pytest.approx(all_episode_lengths[agent_index].sum(
        ), abs=100) == rollout_length*num_envs - max_steps*num_envs//2
