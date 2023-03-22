import pytest
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass

from src.ppo.agent import PPOScheduler, PPOAgent, FCAgent, TrajPPOAgent
from src.models.trajectory_model import TrajectoryTransformer
from src.ppo.memory import Memory


@pytest.fixture
def online_config():
    @dataclass
    class DummyOnlineConfig:
        use_trajectory_model: bool = False
        hidden_size: int = 64
        total_timesteps: int = 180000
        learning_rate: float = 0.00025
        decay_lr: bool = False,
        num_envs: int = 4
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
        batch_size: int = 16
        minibatch_size = 4
        prob_go_from_end = 0.1

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
    @dataclass
    class DummyEnvironmentConfig:
        env_id: str = 'MiniGrid-Dynamic-Obstacles-8x8-v0'
        one_hot_obs: bool = False
        img_obs: bool = False
        fully_observed: bool = False
        max_steps: int = 1000
        seed: int = 1
        view_size: int = 7  # 7 ensure view size wrapper isn't added
        capture_video: bool = False
        video_dir: str = 'videos'
        render_mode: str = 'rgb_array'
        action_space: None = None
        observation_space: None = None
        device: str = 'cpu'

    return DummyEnvironmentConfig()


@pytest.fixture
def optimizer():
    return optim.Adam(params=[torch.tensor([1.0], requires_grad=True)])


@pytest.fixture
def fc_agent():
    envs = gym.vector.SyncVectorEnv(
        [lambda: gym.make('CartPole-v0') for _ in range(4)])
    device = torch.device("cpu")
    hidden_dim = 32
    return FCAgent(envs, device, hidden_dim)


def test_ppo_scheduler_step(optimizer):
    scheduler = PPOScheduler(
        optimizer=optimizer, initial_lr=1e-3, end_lr=1e-5, num_updates=1000)
    for i in range(1000):
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-5)


def test_ppo_agent_init():
    envs = gym.vector.SyncVectorEnv(
        [lambda: gym.make('CartPole-v0') for _ in range(4)])
    device = torch.device("cpu")
    agent = PPOAgent(envs, device)

    assert isinstance(agent, nn.Module)
    assert isinstance(agent.critic, nn.Module)
    assert isinstance(agent.actor, nn.Module)


def test_fc_agent_init():
    envs = gym.vector.SyncVectorEnv(
        [lambda: gym.make('CartPole-v0') for _ in range(4)])
    device = torch.device("cpu")
    hidden_dim = 32
    agent = FCAgent(envs, device, hidden_dim)

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
        num_updates, initial_lr, end_lr)

    assert isinstance(optimizer, optim.Adam)
    assert isinstance(scheduler, PPOScheduler)
    assert scheduler.num_updates == num_updates
    assert scheduler.initial_lr == initial_lr
    assert scheduler.end_lr == end_lr


def test_fc_agent_rollout(fc_agent, online_config):

    num_steps = 10
    envs = gym.vector.SyncVectorEnv(
        [lambda: gym.make('CartPole-v0') for _ in range(online_config.num_envs)])
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
        [lambda: gym.make('CartPole-v0') for _ in range(4)])

    memory = Memory(envs=envs, args=online_config, device=fc_agent.device)

    num_updates = online_config.total_timesteps // online_config.batch_size
    optimizer, scheduler = fc_agent.make_optimizer(
        num_updates,
        online_config.learning_rate,
        online_config.learning_rate * 1e-4
    )

    fc_agent.rollout(memory, online_config.num_steps, envs)
    fc_agent.learn(memory, online_config, optimizer, scheduler, track=False)

    assert isinstance(optimizer, optim.Adam)
    assert isinstance(scheduler, PPOScheduler)
    assert len(memory.next_done) == envs.num_envs
    assert len(memory.next_value) == envs.num_envs
    assert len(memory.experiences) == online_config.num_steps
    assert len(memory.experiences[0]) == 6


def test_traj_agent_init(transformer_model_config, environment_config):

    envs = gym.vector.SyncVectorEnv(
        [lambda: gym.make(environment_config.env_id) for _ in range(4)])
    environment_config.action_space = envs.single_action_space
    environment_config.observation_space = envs.single_observation_space

    agent = TrajPPOAgent(
        envs=envs,
        environment_config=environment_config,
        transformer_model_config=transformer_model_config,
        device="cpu"
    )

    assert isinstance(agent, PPOAgent)
    assert isinstance(agent.critic, TrajectoryTransformer)
    assert isinstance(agent.actor, TrajectoryTransformer)
    assert agent.obs_shape == (7, 7, 3)
    assert agent.num_obs == 147
    assert agent.num_actions == 3
    # assert agent.hidden_dim == hidden_dim


def test_traj_agent_rollout(transformer_model_config, environment_config):

    num_steps = 10
    envs = gym.vector.SyncVectorEnv(
        [lambda: gym.make(environment_config.env_id) for _ in range(4)])
    environment_config.action_space = envs.single_action_space
    environment_config.observation_space = envs.single_observation_space

    agent = TrajPPOAgent(
        envs=envs,
        environment_config=environment_config,
        transformer_model_config=transformer_model_config,
        device="cpu"
    )
    memory = Memory(envs=envs, args=online_config, device="cpu")

    agent.rollout(memory, num_steps, envs)

    assert memory.next_obs.shape[0] == envs.num_envs
    assert memory.next_obs.shape[1] == envs.single_observation_space['image'].shape[0]
    assert len(memory.next_done) == envs.num_envs
    assert len(memory.next_value) == envs.num_envs
    assert len(memory.experiences) == num_steps
    assert len(memory.experiences[0]) == 6


def test_traj_agent_learn(transformer_model_config, environment_config, online_config):

    num_steps = 10
    envs = gym.vector.SyncVectorEnv(
        [lambda: gym.make(environment_config.env_id) for _ in range(4)])
    environment_config.action_space = envs.single_action_space
    environment_config.observation_space = envs.single_observation_space

    agent = TrajPPOAgent(
        envs=envs,
        environment_config=environment_config,
        transformer_model_config=transformer_model_config,
        device="cpu"
    )

    num_updates = online_config.total_timesteps // online_config.batch_size
    optimizer, scheduler = agent.make_optimizer(
        num_updates,
        online_config.learning_rate,
        online_config.learning_rate * 1e-4
    )

    memory = Memory(envs=envs, args=online_config, device="cpu")

    agent.rollout(memory, num_steps, envs)
    agent.learn(memory, online_config, optimizer, scheduler, track=False)

    assert memory.next_obs.shape[0] == envs.num_envs
    assert memory.next_obs.shape[1] == envs.single_observation_space['image'].shape[0]
    assert len(memory.next_done) == envs.num_envs
    assert len(memory.next_value) == envs.num_envs
    assert len(memory.experiences) == num_steps
    assert len(memory.experiences[0]) == 6


def test_traj_agent_larger_context_init(big_transformer_model_config, environment_config):

    envs = gym.vector.SyncVectorEnv(
        [lambda: gym.make(environment_config.env_id) for _ in range(4)])
    environment_config.action_space = envs.single_action_space
    environment_config.observation_space = envs.single_observation_space

    agent = TrajPPOAgent(
        envs=envs,
        environment_config=environment_config,
        transformer_model_config=big_transformer_model_config,
        device="cpu"
    )

    assert isinstance(agent, PPOAgent)
    assert isinstance(agent.critic, TrajectoryTransformer)
    assert isinstance(agent.actor, TrajectoryTransformer)
    assert agent.obs_shape == (7, 7, 3)
    assert agent.num_obs == 147
    assert agent.num_actions == 3
    # assert agent.hidden_dim == hidden_dim


def test_traj_agent_larger_context_rollout(big_transformer_model_config, environment_config):

    num_steps = 10
    envs = gym.vector.SyncVectorEnv(
        [lambda: gym.make(environment_config.env_id) for _ in range(4)])
    environment_config.action_space = envs.single_action_space
    environment_config.observation_space = envs.single_observation_space

    agent = TrajPPOAgent(
        envs=envs,
        environment_config=environment_config,
        transformer_model_config=big_transformer_model_config,
        device="cpu"
    )
    memory = Memory(envs=envs, args=online_config, device="cpu")

    agent.rollout(memory, num_steps, envs)

    assert memory.next_obs.shape[0] == envs.num_envs
    assert memory.next_obs.shape[1] == envs.single_observation_space['image'].shape[0]
    assert len(memory.next_done) == envs.num_envs
    assert len(memory.next_value) == envs.num_envs
    assert len(memory.experiences) == num_steps
    assert len(memory.experiences[0]) == 6


def test_traj_agent_larger_context_learn(big_transformer_model_config, environment_config, online_config):

    num_steps = 10
    envs = gym.vector.SyncVectorEnv(
        [lambda: gym.make(environment_config.env_id) for _ in range(4)])
    environment_config.action_space = envs.single_action_space
    environment_config.observation_space = envs.single_observation_space

    agent = TrajPPOAgent(
        envs=envs,
        environment_config=environment_config,
        transformer_model_config=big_transformer_model_config,
        device="cpu"
    )

    num_updates = online_config.total_timesteps // online_config.batch_size
    optimizer, scheduler = agent.make_optimizer(
        num_updates,
        online_config.learning_rate,
        online_config.learning_rate * 1e-4
    )

    memory = Memory(envs=envs, args=online_config, device="cpu")

    agent.rollout(memory, num_steps, envs)
    agent.learn(memory, online_config, optimizer, scheduler, track=False)

    assert memory.next_obs.shape[0] == envs.num_envs
    assert memory.next_obs.shape[1] == envs.single_observation_space['image'].shape[0]
    assert len(memory.next_done) == envs.num_envs
    assert len(memory.next_value) == envs.num_envs
    assert len(memory.experiences) == num_steps
    assert len(memory.experiences[0]) == 6
