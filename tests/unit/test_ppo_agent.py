import pytest
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from src.ppo.utils import PPOArgs
from src.ppo.agent import PPOScheduler, PPOAgent, FCAgent
from src.ppo.memory import Memory


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
    hidden_dim = 32
    agent = PPOAgent(envs, device, hidden_dim)

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


def test_fc_agent_rollout(fc_agent):

    num_steps = 10
    args = PPOArgs(num_steps=num_steps)
    envs = gym.vector.SyncVectorEnv(
        [lambda: gym.make('CartPole-v0') for _ in range(args.num_envs)])
    memory = Memory(envs=envs, args=args, device=fc_agent.device)

    fc_agent.rollout(memory, num_steps, envs)

    assert memory.next_obs.shape[0] == envs.num_envs
    assert memory.next_obs.shape[1] == envs.single_observation_space.shape[0]
    assert len(memory.next_done) == envs.num_envs
    assert len(memory.next_value) == envs.num_envs
    assert len(memory.experiences) == num_steps
    assert len(memory.experiences[0]) == 6


def test_learn(fc_agent):

    envs = gym.vector.SyncVectorEnv(
        [lambda: gym.make('CartPole-v0') for _ in range(4)])
    args = PPOArgs(
        num_steps=10,
        update_epochs=1,
        clip_coef=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        max_grad_norm=0.5,
        track=False,
        cuda=False,
        num_envs=4)
    memory = Memory(envs=envs, args=args, device=fc_agent.device)

    num_updates = args.total_timesteps // args.batch_size
    optimizer, scheduler = fc_agent.make_optimizer(
        num_updates,
        args.learning_rate,
        args.learning_rate * 1e-4
    )

    fc_agent.rollout(memory, args.num_steps, envs)
    fc_agent.learn(memory, args, optimizer, scheduler)

    assert isinstance(optimizer, optim.Adam)
    assert isinstance(scheduler, PPOScheduler)
    assert len(memory.next_done) == envs.num_envs
    assert len(memory.next_value) == envs.num_envs
    assert len(memory.experiences) == args.num_steps
    assert len(memory.experiences[0]) == 6
