from dataclasses import dataclass

import gymnasium as gym
import pytest
from gymnasium.spaces import Discrete

from src.environments.environments import make_env
from src.ppo.agent import FCAgent, TrajPPOAgent
from src.ppo.memory import Memory, Minibatch, TrajectoryMinibatch
from src.ppo.train import train_ppo


@pytest.fixture
def run_config():
    @dataclass
    class DummyRunConfig:
        exp_name: str = 'test'
        seed: int = 1
        track: bool = False
        wandb_project_name: str = 'test'
        wandb_entity: str = 'test'

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
        device: str = 'cpu'

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


def test_empty_env(run_config, environment_config, online_config):

    env_name = "MiniGrid-Empty-5x5-v0"

    envs = gym.vector.SyncVectorEnv(
        [make_env(env_name, i, i, False, "test",
                  render_mode=None, max_steps=None, fully_observed=False, flat_one_hot=False) for i in range(4)]
    )

    environment_config.env_id = env_name
    ppo = train_ppo(
        run_config=run_config,
        online_config=online_config,
        environment_config=environment_config,
        transformer_model_config=None,
        envs=envs
    )


def test_empty_env_flat_one_hot(run_config, environment_config, online_config):

    env_name = "MiniGrid-Empty-5x5-v0"

    envs = gym.vector.SyncVectorEnv(
        [make_env(env_name, i, i, False, "test",
                  render_mode=None, max_steps=None, fully_observed=False, flat_one_hot=True) for i in range(4)]
    )

    environment_config.env_id = env_name
    environment_config.one_hot_obs = True
    ppo = train_ppo(
        run_config=run_config,
        online_config=online_config,
        environment_config=environment_config,
        transformer_model_config=None,
        envs=envs
    )


def test_ppo_agent_gym():

    envs = gym.vector.SyncVectorEnv(
        [make_env('CartPole-v1', 1, 1, False, "test", max_steps=None)
         for i in range(2)]
    )
    assert envs.single_action_space.shape is not None
    assert isinstance(envs.single_action_space,
                      Discrete), "only discrete action space is supported"

    # memory = Memory(envs, args, "cpu")
    agent = FCAgent(envs, "cpu")

    assert agent.num_obs == 4
    assert agent.num_actions == 2


def test_ppo_agent_minigrid():

    envs = gym.vector.SyncVectorEnv(
        [make_env('MiniGrid-Empty-8x8-v0', 1, 1, False, "test")
         for i in range(2)]
    )
    assert envs.single_action_space.shape is not None
    assert isinstance(envs.single_action_space,
                      Discrete), "only discrete action space is supported"

    # memory = Memory(envs, args, "cpu")
    agent = FCAgent(envs, "cpu")

    # depends on whether you wrapped in Fully observed or not
    assert agent.num_obs == 7 * 7 * 3
    assert agent.num_actions == 7


def test_ppo_agent_rollout_minibatches_minigrid(online_config):

    envs = gym.vector.SyncVectorEnv(
        [make_env('MiniGrid-Empty-8x8-v0', 1, 1, False, "test")
         for i in range(2)]
    )
    assert envs.single_action_space.shape is not None
    assert isinstance(envs.single_action_space,
                      Discrete), "only discrete action space is supported"

    memory = Memory(envs, online_config)

    agent = FCAgent(envs)
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
        [make_env('MiniGrid-Dynamic-Obstacles-8x8-v0', 1, 1, False, "test",
                  max_steps=environment_config.max_steps)
         for i in range(2)]
    )
    memory = Memory(envs, online_config)
    environment_config.action_space = envs.single_action_space
    environment_config.observation_space = envs.single_observation_space
    agent = TrajPPOAgent(envs, environment_config, transformer_model_config)
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
        [make_env('MiniGrid-Dynamic-Obstacles-8x8-v0', 1, 1, False, "test",
                  max_steps=environment_config.max_steps)
         for i in range(2)]
    )

    memory = Memory(envs, online_config)
    environment_config.action_space = envs.single_action_space
    environment_config.observation_space = envs.single_observation_space
    agent = TrajPPOAgent(envs, environment_config, transformer_model_config)

    num_updates = online_config.total_timesteps // online_config.batch_size
    optimizer, scheduler = agent.make_optimizer(
        num_updates=num_updates,
        initial_lr=online_config.learning_rate,
        end_lr=online_config.learning_rate if not online_config.decay_lr else 0.0)

    agent.rollout(memory, online_config.num_steps, envs, None)
    agent.learn(memory, online_config, optimizer, scheduler, track=False)
