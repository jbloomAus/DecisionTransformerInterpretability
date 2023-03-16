import re
from dataclasses import dataclass

import gymnasium as gym
import pytest
import torch as t
from gymnasium.spaces import Discrete

from src.environments.environments import make_env
from src.ppo.agent import FCAgent as Agent
from src.ppo.memory import Memory, Minibatch, TrajectoryMinibatch
from src.ppo.my_probe_envs import Probe1, Probe2, Probe3, Probe4, Probe5
from src.ppo.train import train_ppo

for i in range(5):
    probes = [Probe1, Probe2, Probe3, Probe4, Probe5]
    gym.envs.registration.register(id=f"Probe{i+1}-v0", entry_point=probes[i])


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
        max_steps: int = 1000
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
        num_envs: int = 4
        num_steps: int = 128
        gamma: float = 0.99
        gae_lambda: float = 0.95
        num_minibatches: int = 8
        update_epochs: int = 4
        clip_coef: float = 0.4
        ent_coef: float = 0.00
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


@pytest.mark.parametrize("env_name", ["Probe1-v0", "Probe2-v0", "Probe3-v0", "Probe4-v0", "Probe5-v0"])
def test_probe_envs(env_name, run_config, environment_config, online_config):

    for i in range(5):
        probes = [Probe1, Probe2, Probe3, Probe4, Probe5]
        gym.envs.registration.register(
            id=f"Probe{i+1}-v0", entry_point=probes[i])

    envs = gym.vector.SyncVectorEnv(
        [make_env(env_name, i, i, False, "test",
                  render_mode=None, max_steps=None, fully_observed=False) for i in range(4)]
    )

    # currently, ppo has tests which run inside main if it
    # detects "Probe" in the env name. We will fix this
    # eventually.
    environment_config.env_id = env_name
    environment_config.action_space = envs.single_action_space
    online_config.total_timesteps = 2000
    agent = train_ppo(
        run_config=run_config,
        online_config=online_config,
        environment_config=environment_config,
        transformer_model_config=None,
        envs=envs
    )

    obs_for_probes = [
        [[0.0]],
        [[-1.0], [+1.0]],
        [[0.0], [1.0]],
        [[0.0], [0.0]],
        [[0.0], [1.0]]]

    expected_value_for_probes = [
        [[1.0]],
        [[-1.0], [+1.0]],
        [[online_config.gamma], [1.0]],
        [[+1.0], [+1.0]],  # can achieve high reward independently of obs
        [[+1.0], [+1.0]]
    ]

    tolerances_for_value = [5e-4, 5e-4, 5e-4, 5e-4, 1e-3]

    match = re.match(r"Probe(\d)-v0", env_name)
    probe_idx = int(match.group(1)) - 1
    obs = t.tensor(obs_for_probes[probe_idx])
    value = agent.critic(obs)
    print("Value: ", value)
    expected_value = t.tensor(expected_value_for_probes[probe_idx])
    t.testing.assert_close(value, expected_value,
                           atol=tolerances_for_value[probe_idx], rtol=1)

    if probe_idx == 3:  # probe env 4, action should be +1.0
        action = agent.actor(obs)
        prob = t.nn.functional.softmax(action, dim=-1)
        t.testing.assert_close(
            prob,
            t.tensor([[0.0, 1.0], [0.0, 1.0]]),
            atol=1e-2,
            rtol=1
        )

    if probe_idx == 4:  # probe env 4, action should be +1.0
        action = agent.actor(obs)
        prob = t.nn.functional.softmax(action, dim=-1)
        t.testing.assert_close(
            prob,
            t.tensor([[1.0, 0.0], [0.0, 1.0]]),
            atol=1e-2,
            rtol=1
        )


@pytest.mark.parametrize("env_name", ["Probe1-v0", "Probe2-v0", "Probe3-v0", "Probe4-v0", "Probe5-v0"])
def test_probe_envs_traj_model_1_context(
        env_name,
        run_config,
        environment_config,
        online_config,
        transformer_model_config):

    for i in range(5):
        probes = [Probe1, Probe2, Probe3, Probe4, Probe5]
        gym.envs.registration.register(
            id=f"Probe{i+1}-v0", entry_point=probes[i])

    envs = gym.vector.SyncVectorEnv(
        [make_env(env_name, i, i, False, "test",
                  render_mode=None, max_steps=None, fully_observed=False) for i in range(4)]
    )

    # currently, ppo has tests which run inside main if it
    # detects "Probe" in the env name. We will fix this
    # eventually.
    environment_config.env_id = env_name
    environment_config.action_space = envs.single_action_space
    environment_config.observation_space = envs.single_observation_space
    online_config.use_trajectory_model = True
    transformer_model_config.state_embedding_type = "other"
    online_config.total_timesteps = 2000
    if env_name == "Probe3-v0":
        online_config.total_timesteps = 10000
    if env_name == "Probe5-v0":
        online_config.total_timesteps = 10000
    agent = train_ppo(
        run_config=run_config,
        online_config=online_config,
        environment_config=environment_config,
        transformer_model_config=transformer_model_config,
        envs=envs
    )

    obs_for_probes = [
        [[0.0]],
        [[-1.0], [+1.0]],
        [[0.0], [1.0]],
        [[0.0], [0.0]],
        [[0.0], [1.0]]]

    match = re.match(r"Probe(\d)-v0", env_name)
    probe_idx = int(match.group(1)) - 1
    obs = t.tensor(obs_for_probes[probe_idx])

    # test action quality first since if actions actor isn't working
    # out expected value will be wrong
    if probe_idx == 3:
        action = agent.actor(
            obs.unsqueeze(1),
            actions=None,
            timesteps=t.tensor([[0], [0]]).unsqueeze(-1)
        )
        prob = t.nn.functional.softmax(action, dim=-1).squeeze(1)
        t.testing.assert_close(
            prob,
            t.tensor([[0.0, 1.0], [0.0, 1.0]]),
            atol=1e-2,
            rtol=1
        )

    if probe_idx == 4:
        action = agent.actor(
            obs.unsqueeze(1),
            actions=None,
            timesteps=t.tensor([[0], [0]]).unsqueeze(-1)
        )
        prob = t.nn.functional.softmax(action, dim=-1).squeeze(1)
        t.testing.assert_close(
            prob,
            t.tensor([[1.0, 0.0], [0.0, 1.0]]),
            atol=1e-2,
            rtol=1
        )

    expected_value_for_probes = [
        [[1.0]],
        [[-1.0], [+1.0]],
        [[online_config.gamma], [1.0]],
        [[+1.0], [+1.0]],  # can achieve high reward independently of obs
        [[+1.0], [+1.0]]
    ]

    tolerances_for_value = [5e-4, 5e-4, 5e-4, 5e-4, 1e-3]

    value = agent.critic(obs)
    print("Value: ", value)
    expected_value = t.tensor(expected_value_for_probes[probe_idx])
    t.testing.assert_close(value, expected_value,
                           atol=tolerances_for_value[probe_idx], rtol=0.1)


@pytest.mark.parametrize("env_name", ["Probe1-v0", "Probe2-v0", "Probe3-v0", "Probe4-v0", "Probe5-v0"])
def test_probe_envs_traj_model_2_context(
        env_name,
        run_config,
        environment_config,
        online_config,
        transformer_model_config):

    for i in range(5):
        probes = [Probe1, Probe2, Probe3, Probe4, Probe5]
        gym.envs.registration.register(
            id=f"Probe{i+1}-v0", entry_point=probes[i])

    envs = gym.vector.SyncVectorEnv(
        [make_env(env_name, i, i, False, "test",
                  render_mode=None, max_steps=None, fully_observed=False) for i in range(4)]
    )

    # currently, ppo has tests which run inside main if it
    # detects "Probe" in the env name. We will fix this
    # eventually.
    environment_config.env_id = env_name
    environment_config.action_space = envs.single_action_space
    environment_config.observation_space = envs.single_observation_space
    online_config.use_trajectory_model = True
    transformer_model_config.state_embedding_type = "other"
    online_config.total_timesteps = 2000
    if env_name == "Probe3-v0":
        online_config.total_timesteps = 10000
    agent = train_ppo(
        run_config=run_config,
        online_config=online_config,
        environment_config=environment_config,
        transformer_model_config=transformer_model_config,
        envs=envs
    )

    obs_for_probes = [
        [[0.0]],
        [[-1.0], [+1.0]],
        [[0.0], [1.0]],
        [[0.0], [0.0]],
        [[0.0], [1.0]]]

    match = re.match(r"Probe(\d)-v0", env_name)
    probe_idx = int(match.group(1)) - 1
    obs = t.tensor(obs_for_probes[probe_idx])

    # test action quality first since if actions actor isn't working
    # out expected value will be wrong
    if probe_idx == 3:  # probe env 4, action should be +1.0
        action = agent.actor(
            obs.unsqueeze(1),
            actions=None,
            timesteps=t.tensor([[0], [0]]).unsqueeze(-1)
        )
        prob = t.nn.functional.softmax(action, dim=-1).squeeze(1)
        t.testing.assert_close(
            prob,
            t.tensor([[0.0, 1.0], [0.0, 1.0]]),
            atol=1e-2,
            rtol=1
        )

    if probe_idx == 4:  # probe env 4, action should be +1.0
        action = agent.actor(
            obs.unsqueeze(1),
            actions=None,
            timesteps=t.tensor([[0], [0]]).unsqueeze(-1)
        )
        prob = t.nn.functional.softmax(action, dim=-1).squeeze(1)
        t.testing.assert_close(
            prob,
            t.tensor([[1.0, 0.0], [0.0, 1.0]]),
            atol=1e-2,
            rtol=1
        )

    expected_value_for_probes = [
        [[1.0]],
        [[-1.0], [+1.0]],
        [[online_config.gamma], [1.0]],
        [[+1.0], [+1.0]],  # can achieve high reward independently of obs
        [[+1.0], [+1.0]]
    ]

    tolerances_for_value = [5e-4, 5e-4, 5e-4, 5e-4, 1e-3]

    value = agent.critic(obs)
    print("Value: ", value)
    expected_value = t.tensor(expected_value_for_probes[probe_idx])
    t.testing.assert_close(value, expected_value,
                           atol=tolerances_for_value[probe_idx], rtol=0.1)


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
    agent = Agent(envs, "cpu")

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
    agent = Agent(envs, "cpu")

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

    memory = Memory(
        envs, online_config, "cpu")

    agent = Agent(envs, "cpu")
    agent.rollout(memory, online_config.num_steps, envs, None)

    minibatches = memory.get_minibatches()
    assert len(minibatches) == online_config.num_minibatches

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


def test_ppo_agent_rollout_trajectory_minibatches_minigrid_no_padding(online_config):
    envs = gym.vector.SyncVectorEnv(
        [make_env('MiniGrid-Dynamic-Obstacles-8x8-v0', 1, 1, False, "test")
         for i in range(2)]
    )
    memory = Memory(envs, online_config, "cpu")
    agent = Agent(envs, "cpu")
    agent.rollout(memory, online_config.num_steps, envs, None)

    timesteps = 1
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


def test_ppo_agent_rollout_trajectory_minibatches_minigrid_extra_padding(online_config):
    envs = gym.vector.SyncVectorEnv(
        [make_env('MiniGrid-Dynamic-Obstacles-8x8-v0', 1, 1, False, "test")
         for i in range(2)]
    )
    memory = Memory(envs, online_config, "cpu")
    agent = Agent(envs, "cpu")
    agent.rollout(memory, online_config.num_steps, envs, None)

    timesteps = 10
    minibatches = memory.get_trajectory_minibatches(10)
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
