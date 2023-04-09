import pytest
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass

from src.ppo.agent import (
    PPOScheduler,
    PPOAgent,
    FCAgent,
    TransformerPPOAgent,
    LSTMPPOAgent,
    get_agent,
)

# only use get_agent class test
from src.config import LSTMModelConfig, TransformerModelConfig
from src.models.trajectory_transformer import TrajectoryTransformer
from src.models.trajectory_lstm import TrajectoryLSTM
from src.ppo.memory import Memory


@pytest.fixture
def online_config():
    @dataclass
    class DummyOnlineConfig:
        use_trajectory_model: bool = False
        hidden_size: int = 64
        total_timesteps: int = 180000
        learning_rate: float = 0.0001
        decay_lr: bool = (False,)
        num_envs: int = 16
        num_steps: int = 128
        gamma: float = 0.99
        gae_lambda: float = 0.95
        num_minibatches: int = 4
        update_epochs: int = 4
        clip_coef: float = 0.4
        ent_coef: float = 0.2
        vf_coef: float = 0.5
        max_grad_norm: float = 2
        trajectory_path: str = None
        fully_observed: bool = False
        batch_size: int = 2048
        minibatch_size = 512
        prob_go_from_end = 0.1
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
def big_transformer_model_config():
    @dataclass
    class DummyTransformerModelConfig:
        d_model: int = 128
        n_heads: int = 2
        d_mlp: int = 256
        n_layers: int = 1
        # look at previous state (s,) and 4 full timesteps (s,a) before that.
        n_ctx: int = 9
        time_embedding_type: str = "embedding"
        state_embedding_type: str = "grid"
        seed: int = 1
        device: str = "cpu"
        d_head: int = 64  # d_model // n_heads
        layer_norm = False

    return DummyTransformerModelConfig()


@pytest.fixture
def environment_config():
    env_id = "MiniGrid-Dynamic-Obstacles-8x8-v0"
    env = gym.make(env_id)

    @dataclass
    class DummyEnvironmentConfig:
        env_id: str = "MiniGrid-Dynamic-Obstacles-8x8-v0"
        one_hot_obs: bool = False
        img_obs: bool = False
        fully_observed: bool = False
        max_steps: int = 1000
        seed: int = 1
        view_size: int = 7  # 7 ensure view size wrapper isn't added
        capture_video: bool = False
        video_dir: str = "videos"
        render_mode: str = "rgb_array"
        action_space: None = None
        observation_space: None = None
        device: str = "cpu"
        action_space = env.action_space
        observation_space = env.observation_space

    return DummyEnvironmentConfig()


@pytest.fixture
def lstm_config(environment_config):
    @dataclass
    class DummyLSTMConfig:
        environment_config = environment_config
        image_dim: int = 128
        memory_dim: int = 128
        instr_dim: int = 128
        use_instr: bool = False
        lang_model: str = "gru"
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
    config.obs_space = environment_config.observation_space

    return config


@pytest.fixture
def optimizer():
    return optim.Adam(params=[torch.tensor([1.0], requires_grad=True)])


@pytest.fixture
def fc_agent(environment_config):
    envs = gym.vector.SyncVectorEnv(
        [lambda: gym.make("CartPole-v0") for _ in range(4)]
    )
    device = torch.device("cpu")
    hidden_dim = 32
    environment_config.env_id = "CartPole-v0"
    return FCAgent(
        envs, environment_config, device=device, hidden_dim=hidden_dim
    )


@pytest.fixture
def transformer_agent(
    online_config, transformer_model_config, environment_config
):
    envs = gym.vector.SyncVectorEnv(
        [
            lambda: gym.make(environment_config.env_id)
            for _ in range(online_config.num_envs)
        ]
    )
    device = torch.device("cpu")
    environment_config.action_space = envs.single_action_space
    environment_config.observation_space = envs.single_observation_space
    return TransformerPPOAgent(
        envs, environment_config, transformer_model_config, device
    )


@pytest.fixture
def big_transformer_agent(
    online_config, big_transformer_model_config, environment_config
):
    envs = gym.vector.SyncVectorEnv(
        [
            lambda: gym.make(environment_config.env_id)
            for _ in range(online_config.num_envs)
        ]
    )
    device = torch.device("cpu")
    environment_config.action_space = envs.single_action_space
    environment_config.observation_space = envs.single_observation_space
    return TransformerPPOAgent(
        envs, environment_config, big_transformer_model_config, device
    )


@pytest.fixture
def lstm_agent(online_config, lstm_config, environment_config):
    envs = gym.vector.SyncVectorEnv(
        [
            lambda: gym.make(environment_config.env_id)
            for _ in range(online_config.num_envs)
        ]
    )
    device = torch.device("cpu")
    return LSTMPPOAgent(envs, environment_config, lstm_config, device)


def test_ppo_scheduler_step(optimizer):
    scheduler = PPOScheduler(
        optimizer=optimizer, initial_lr=1e-3, end_lr=1e-5, num_updates=1000
    )
    for i in range(1000):
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-5)


def test_ppo_agent_init():
    envs = gym.vector.SyncVectorEnv(
        [lambda: gym.make("CartPole-v0") for _ in range(4)]
    )
    device = torch.device("cpu")
    agent = PPOAgent(envs, device)

    assert isinstance(agent, nn.Module)
    assert isinstance(agent.critic, nn.Module)
    assert isinstance(agent.actor, nn.Module)


def test_fc_agent_init(environment_config):
    envs = gym.vector.SyncVectorEnv(
        [lambda: gym.make("CartPole-v0") for _ in range(4)]
    )
    device = torch.device("cpu")
    hidden_dim = 32
    agent = FCAgent(
        envs, environment_config, device=device, hidden_dim=hidden_dim
    )

    assert isinstance(agent, PPOAgent)
    assert isinstance(agent.critic, nn.Sequential)
    assert isinstance(agent.actor, nn.Sequential)
    assert agent.obs_shape == (4,)
    assert agent.num_obs == 4
    assert agent.num_actions == 2
    assert agent.hidden_dim == hidden_dim


def test_fc_agent_layer_init(fc_agent):
    layer = nn.Linear(4, 2)
    std = 0.5
    bias_const = 0.1
    new_layer = fc_agent.layer_init(layer, std, bias_const)

    assert isinstance(new_layer, nn.Linear)
    # assert std == pytest.approx(new_layer.weight.std().detach().item())
    assert bias_const == pytest.approx(new_layer.bias.mean().detach().item())


def test_fc_agent_make_optimizer(fc_agent):
    num_updates = 10
    initial_lr = 0.001
    end_lr = 0.0001
    optimizer, scheduler = fc_agent.make_optimizer(
        num_updates, initial_lr, end_lr
    )

    assert isinstance(optimizer, optim.Adam)
    assert isinstance(scheduler, PPOScheduler)
    assert scheduler.num_updates == num_updates
    assert scheduler.initial_lr == initial_lr
    assert scheduler.end_lr == end_lr


def test_fc_agent_rollout(fc_agent, online_config):
    num_steps = 10
    envs = gym.vector.SyncVectorEnv(
        [
            lambda: gym.make("CartPole-v0")
            for _ in range(online_config.num_envs)
        ]
    )
    memory = Memory(envs=envs, args=online_config, device=fc_agent.device)

    fc_agent.rollout(memory, num_steps, envs)

    assert memory.next_obs.shape[0] == envs.num_envs
    assert memory.next_obs.shape[1] == envs.single_observation_space.shape[0]
    assert len(memory.next_done) == envs.num_envs
    assert len(memory.next_value) == envs.num_envs
    assert len(memory.experiences) == num_steps
    assert len(memory.experiences[0]) == 6


def test_fc_agent_learn(fc_agent, online_config):
    envs = gym.vector.SyncVectorEnv(
        [
            lambda: gym.make("CartPole-v0")
            for _ in range(online_config.num_envs)
        ]
    )

    memory = Memory(envs=envs, args=online_config, device=fc_agent.device)

    num_updates = online_config.total_timesteps // online_config.batch_size
    optimizer, scheduler = fc_agent.make_optimizer(
        num_updates,
        online_config.learning_rate,
        online_config.learning_rate * 1e-4,
    )

    fc_agent.rollout(memory, online_config.num_steps, envs)
    fc_agent.learn(memory, online_config, optimizer, scheduler, track=False)

    assert isinstance(optimizer, optim.Adam)
    assert isinstance(scheduler, PPOScheduler)
    assert len(memory.next_done) == envs.num_envs
    assert len(memory.next_value) == envs.num_envs
    assert len(memory.experiences) == online_config.num_steps
    assert len(memory.experiences[0]) == 6


def test_transformer_agent_init(transformer_agent):
    agent = transformer_agent

    assert isinstance(agent, PPOAgent)
    assert isinstance(agent.critic, TrajectoryTransformer)
    assert isinstance(agent.actor, TrajectoryTransformer)
    assert agent.obs_shape == (7, 7, 3)
    assert agent.num_obs == 147
    assert agent.num_actions == 3
    # assert agent.hidden_dim == hidden_dim


def test_transformer_agent_rollout(transformer_agent, online_config):
    num_steps = 10
    agent = transformer_agent
    memory = Memory(
        envs=transformer_agent.envs, args=online_config, device="cpu"
    )

    agent.rollout(memory, num_steps, transformer_agent.envs)

    assert memory.next_obs.shape[0] == online_config.num_envs
    assert (
        memory.next_obs.shape[1]
        == transformer_agent.envs.single_observation_space["image"].shape[0]
    )
    assert len(memory.next_done) == transformer_agent.envs.num_envs
    assert len(memory.next_value) == transformer_agent.envs.num_envs
    assert len(memory.experiences) == num_steps
    assert len(memory.experiences[0]) == 6


def test_transformer_agent_learn(transformer_agent, online_config):
    num_steps = 10
    agent = transformer_agent

    num_updates = online_config.total_timesteps // online_config.batch_size
    optimizer, scheduler = agent.make_optimizer(
        num_updates,
        online_config.learning_rate,
        online_config.learning_rate * 1e-4,
    )

    memory = Memory(
        envs=transformer_agent.envs, args=online_config, device="cpu"
    )

    agent.rollout(memory, num_steps, transformer_agent.envs)
    agent.learn(memory, online_config, optimizer, scheduler, track=False)

    assert memory.next_obs.shape[0] == transformer_agent.envs.num_envs
    assert (
        memory.next_obs.shape[1]
        == transformer_agent.envs.single_observation_space["image"].shape[0]
    )
    assert len(memory.next_done) == transformer_agent.envs.num_envs
    assert len(memory.next_value) == transformer_agent.envs.num_envs
    assert len(memory.experiences) == num_steps
    assert len(memory.experiences[0]) == 6


def test_transformer_agent_larger_context_init(big_transformer_agent):
    agent = big_transformer_agent

    assert isinstance(agent, PPOAgent)
    assert isinstance(agent.critic, TrajectoryTransformer)
    assert isinstance(agent.actor, TrajectoryTransformer)
    assert agent.obs_shape == (7, 7, 3)
    assert agent.num_obs == 147
    assert agent.num_actions == 3
    # assert agent.hidden_dim == hidden_dim


def test_transformer_agent_larger_context_rollout(big_transformer_agent):
    num_steps = 10
    agent = big_transformer_agent
    memory = Memory(
        envs=big_transformer_agent.envs, args=online_config, device="cpu"
    )

    agent.rollout(memory, num_steps, big_transformer_agent.envs)

    assert memory.next_obs.shape[0] == big_transformer_agent.envs.num_envs
    assert (
        memory.next_obs.shape[1]
        == big_transformer_agent.envs.single_observation_space["image"].shape[
            0
        ]
    )
    assert len(memory.next_done) == big_transformer_agent.envs.num_envs
    assert len(memory.next_value) == big_transformer_agent.envs.num_envs
    assert len(memory.experiences) == num_steps
    assert len(memory.experiences[0]) == 6


def test_transformer_agent_larger_context_learn(
    big_transformer_agent, online_config
):
    num_steps = 10
    agent = big_transformer_agent
    num_updates = online_config.total_timesteps // online_config.batch_size
    optimizer, scheduler = agent.make_optimizer(
        num_updates,
        online_config.learning_rate,
        online_config.learning_rate * 1e-4,
    )

    memory = Memory(
        envs=big_transformer_agent.envs, args=online_config, device="cpu"
    )

    agent.rollout(memory, num_steps, big_transformer_agent.envs)
    agent.learn(memory, online_config, optimizer, scheduler, track=False)

    assert memory.next_obs.shape[0] == big_transformer_agent.envs.num_envs
    assert (
        memory.next_obs.shape[1]
        == big_transformer_agent.envs.single_observation_space["image"].shape[
            0
        ]
    )
    assert len(memory.next_done) == big_transformer_agent.envs.num_envs
    assert len(memory.next_value) == big_transformer_agent.envs.num_envs
    assert len(memory.experiences) == num_steps
    assert len(memory.experiences[0]) == 6


def test_lstm_agent_init(lstm_agent):
    agent = lstm_agent

    assert isinstance(agent, PPOAgent)
    assert isinstance(agent.model, TrajectoryLSTM)
    assert agent.obs_shape == (7, 7, 3)
    assert agent.num_obs == 147
    assert agent.num_actions == 3
    # assert agent.hidden_dim == hidden_dim


def test_lstm_agent_rollout(lstm_agent, online_config):
    num_steps = 10
    agent = lstm_agent
    memory = Memory(
        envs=lstm_agent.envs, args=online_config, device=torch.device("cpu")
    )

    agent.rollout(memory, num_steps, lstm_agent.envs)

    assert memory.next_obs.shape[0] == lstm_agent.envs.num_envs
    assert (
        memory.next_obs.shape[1]
        == lstm_agent.envs.single_observation_space["image"].shape[0]
    )
    assert len(memory.next_done) == lstm_agent.envs.num_envs
    assert len(memory.next_value) == lstm_agent.envs.num_envs
    assert len(memory.experiences) == num_steps
    assert len(memory.experiences[0]) == 8


def test_lstm_agent_rollout_learn(lstm_agent, online_config):
    online_config.batch_size = 128 * 16
    num_steps = 128
    agent = lstm_agent
    num_updates = online_config.total_timesteps // online_config.batch_size
    optimizer, scheduler = agent.make_optimizer(
        num_updates,
        online_config.learning_rate,
        online_config.learning_rate * 1e-4,
    )

    memory = Memory(
        envs=lstm_agent.envs, args=online_config, device=torch.device("cpu")
    )

    agent.rollout(memory, num_steps, lstm_agent.envs)
    agent.learn(memory, online_config, optimizer, scheduler, track=False)

    assert memory.next_obs.shape[0] == lstm_agent.envs.num_envs
    assert (
        memory.next_obs.shape[1]
        == lstm_agent.envs.single_observation_space["image"].shape[0]
    )
    assert len(memory.next_done) == lstm_agent.envs.num_envs
    assert len(memory.next_value) == lstm_agent.envs.num_envs
    assert len(memory.experiences) == num_steps
    assert len(memory.experiences[0]) == 8

    assert scheduler.num_updates == num_updates


def test_get_agent_fc_agent(fc_agent, environment_config, online_config):
    agent = get_agent(
        model_config=None,
        envs=fc_agent.envs,
        environment_config=environment_config,
        online_config=online_config,
    )

    assert isinstance(agent, PPOAgent)
    assert isinstance(agent, FCAgent)


def test_get_agent_transformer_agent(
    transformer_agent,
    transformer_model_config,
    environment_config,
    online_config,
):
    agent = get_agent(
        model_config=TransformerModelConfig(n_ctx=3),
        envs=transformer_agent.envs,
        environment_config=environment_config,
        online_config=online_config,
    )

    assert isinstance(agent, PPOAgent)
    assert isinstance(agent, TransformerPPOAgent)


def test_get_agent_lstm_agent(lstm_agent, environment_config):
    agent = get_agent(
        model_config=LSTMModelConfig(environment_config),
        envs=lstm_agent.envs,
        environment_config=environment_config,
        online_config=online_config,
    )

    assert isinstance(agent, PPOAgent)
    assert isinstance(agent, LSTMPPOAgent)
