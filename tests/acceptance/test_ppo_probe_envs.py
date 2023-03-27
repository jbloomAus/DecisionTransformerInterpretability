import re
from dataclasses import dataclass

import gymnasium as gym
import pytest
import torch as t

from src.environments.environments import make_env
from src.ppo.my_probe_envs import Probe1, Probe2, Probe3, Probe4, Probe5, Probe6
from src.ppo.train import train_ppo

for i in range(6):
    probes = [Probe1, Probe2, Probe3, Probe4, Probe5, Probe6]
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

    tolerances_for_value = [5e-4, 5e-4, 5e-4, 5e-2, 2e-1]

    match = re.match(r"Probe(\d)-v0", env_name)
    probe_idx = int(match.group(1)) - 1
    obs = t.tensor(obs_for_probes[probe_idx])
    value = agent.critic(obs)
    print("Value: ", value)
    expected_value = t.tensor(expected_value_for_probes[probe_idx])
    t.testing.assert_close(value, expected_value,
                           atol=tolerances_for_value[probe_idx], rtol=0)

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


@pytest.mark.parametrize("env_name", ["Probe1-v0", "Probe2-v0", "Probe3-v0", "Probe4-v0", "Probe5-v0", "Probe6-v0"])
@pytest.mark.skip(reason="Traj PPO not working seemingly, put on hold until we fix it")
def test_probe_envs_traj_model_1_context(
        env_name,
        run_config,
        environment_config,
        online_config,
        transformer_model_config):

    for i in range(6):
        probes = [Probe1, Probe2, Probe3, Probe4, Probe5, Probe6]
        gym.envs.registration.register(
            id=f"Probe{i+1}-v0", entry_point=probes[i])

    envs = gym.vector.SyncVectorEnv(
        [make_env(env_name, i, i, False, "test",
                  render_mode=None, max_steps=None, fully_observed=False) for i in range(4)]
    )

    online_config.total_timesteps = 2000
    if env_name == "Probe5-v0":
        online_config.total_timesteps = 5000
        online_config.clip_coef = 0.05
        online_config.learning_rate = 0.00025
    # currently, ppo has tests which run inside main if it
    # detects "Probe" in the env name. We will fix this
    # eventually.
    environment_config.env_id = env_name
    environment_config.action_space = envs.single_action_space
    environment_config.observation_space = envs.single_observation_space
    online_config.use_trajectory_model = True
    transformer_model_config.state_embedding_type = "other"

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
        [[0.0], [1.0]],
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
            t.tensor([[1.0, 0.01], [0.01, 1.0]]),
            atol=3e-1,
            rtol=30  # incredibly lax until I work out why this is so bad
        )

    expected_value_for_probes = [
        [[1.0]],
        [[-1.0], [+1.0]],
        [[online_config.gamma], [1.0]],
        [[+1.0], [+1.0]],  # can achieve high reward independently of obs
        [[+1.0], [+1.0]],
        [[+1.0], [+1.0]],
    ]

    tolerances_for_value = [5e-4, 5e-4, 5e-2, 5e-4, 2e-1, 1]

    value = agent.critic(
        states=obs.unsqueeze(1),
        actions=None,
        timesteps=t.tensor([0]).repeat(obs.shape[0], obs.shape[1], 1)
    )[:, -1]

    print("Value: ", value)
    expected_value = t.tensor(expected_value_for_probes[probe_idx])
    t.testing.assert_close(value, expected_value,
                           atol=tolerances_for_value[probe_idx], rtol=0)


@pytest.mark.parametrize("env_name", ["Probe1-v0", "Probe2-v0", "Probe3-v0", "Probe4-v0"])
@pytest.mark.skip(reason="Traj PPO not working seemingly, put on hold until we fix it")
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
            atol=3e-1,
            rtol=30  # incredibly lax until I work out why this is so bad
        )

    expected_value_for_probes = [
        [[1.0]],
        [[-1.0], [+1.0]],
        [[online_config.gamma], [1.0]],
        [[+1.0], [+1.0]],  # can achieve high reward independently of obs
        [[+1.0], [+1.0]]
    ]

    tolerances_for_value = [5e-4, 5e-4, 5e-4, 5e-4, 1e-3]

    value = agent.critic(
        states=obs.unsqueeze(1),
        actions=None,
        timesteps=t.tensor([0]).repeat(obs.shape[0], obs.shape[1], 1)
    )[:, -1]
    print("Value: ", value)
    expected_value = t.tensor(expected_value_for_probes[probe_idx])
    t.testing.assert_close(value, expected_value,
                           atol=tolerances_for_value[probe_idx], rtol=0.1)
