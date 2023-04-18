import pytest
import torch
from minigrid.envs import (
    DynamicObstaclesEnv,
    EmptyEnv,
    FourRoomsEnv,
    MemoryEnv,
)
from minigrid.envs.babyai import (
    BossLevel,
    GoToDoor,
    GoToRedBlueBall,
    UnlockLocal,
)
from minigrid.wrappers import RGBImgPartialObsWrapper

from src.config import LSTMModelConfig

# stuff I need to commit to MiniGrid
from src.environments.wrappers import DictObservationSpaceWrapper
from src.models.trajectory_lstm import TrajectoryLSTM
from src.utils.dictlist import DictList

# create a fixture which can be used to parameterize different minigrid_envs
# @pytest.fixture


def minigrid_envs():
    return [
        MemoryEnv(
            size=7, random_length=False, max_steps=200, render_mode="rgb_array"
        ),
        DynamicObstaclesEnv(
            size=7, n_obstacles=3, max_steps=200, render_mode="rgb_array"
        ),
        EmptyEnv(size=7, max_steps=200, render_mode="rgb_array"),
        FourRoomsEnv(max_steps=200, render_mode="rgb_array"),
    ]


def babyai_envs():
    return [
        GoToDoor(),
        GoToRedBlueBall(),
        UnlockLocal(),
        BossLevel(),
    ]


@pytest.mark.parametrize("env", minigrid_envs())
def test__init___standard_env_without_memory(env):
    obs, info = env.reset()

    config = LSTMModelConfig(env, use_memory=False)
    acmodel = TrajectoryLSTM(config)


@pytest.mark.parametrize("env", minigrid_envs())
def test__init_forward_without_memory(env):
    obs, info = env.reset()

    config = LSTMModelConfig(env, use_memory=False)
    acmodel = TrajectoryLSTM(config)

    obs, info = env.reset()
    obs = DictList(obs)
    obs.image = torch.from_numpy(obs.image).to(dtype=torch.float).unsqueeze(0)
    result = acmodel.forward(obs, torch.zeros(1, 128 * 2))
    result


@pytest.mark.parametrize("env", minigrid_envs())
def test__init___standard_env_with_memory(env):
    obs, info = env.reset()

    config = LSTMModelConfig(env, use_memory=True)
    acmodel = TrajectoryLSTM(config)


@pytest.mark.parametrize("env", minigrid_envs())
def test__init_forward_with_memory(env):
    obs, info = env.reset()

    config = LSTMModelConfig(env, use_memory=True)
    acmodel = TrajectoryLSTM(config)

    obs, info = env.reset()
    obs = DictList(obs)
    obs.image = torch.from_numpy(obs.image).to(dtype=torch.float).unsqueeze(0)
    result = acmodel.forward(obs, torch.zeros(1, 128 * 2))
    result


@pytest.mark.parametrize("env", minigrid_envs())
def test__init___standard_env_with_pixel(env):
    env = RGBImgPartialObsWrapper(env)
    obs, info = env.reset()

    config = LSTMModelConfig(env, arch="pixels_endpool_res")
    acmodel = TrajectoryLSTM(config)


@pytest.mark.parametrize("env", minigrid_envs())
def test__init_forward_with_pixel(env):
    env = RGBImgPartialObsWrapper(env)
    obs, info = env.reset()

    config = LSTMModelConfig(env, use_memory=False, arch="pixels_endpool_res")
    acmodel = TrajectoryLSTM(config)

    obs, info = env.reset()
    obs = DictList(obs)
    obs.image = torch.from_numpy(obs.image).to(dtype=torch.float).unsqueeze(0)
    result = acmodel.forward(obs, torch.zeros(1, 128 * 2))
    result


@pytest.mark.parametrize("env", minigrid_envs())
def test__init___without_endpool_res(env):
    obs, info = env.reset()

    config = LSTMModelConfig(env, use_memory=False, arch="bow")
    acmodel = TrajectoryLSTM(config)


@pytest.mark.parametrize("env", minigrid_envs())
def test__init_forward_wwithout_endpool_res(env):
    obs, info = env.reset()

    config = LSTMModelConfig(env, use_memory=False, arch="bow")
    acmodel = TrajectoryLSTM(config)

    obs, info = env.reset()
    obs = DictList(obs)
    obs.image = torch.from_numpy(obs.image).to(dtype=torch.float).unsqueeze(0)
    result = acmodel.forward(obs, torch.zeros(1, 128 * 2))
    result


@pytest.mark.parametrize("env", babyai_envs())
def test__init___standard_env_with_instruction(env):
    env = DictObservationSpaceWrapper(env)
    obs, info = env.reset()

    config = LSTMModelConfig(env, use_memory=True, use_instr=True)
    acmodel = TrajectoryLSTM(config)


@pytest.mark.parametrize("env", babyai_envs())
def test__init_forward_with_instruction(env):
    env = DictObservationSpaceWrapper(env)
    obs, info = env.reset()

    config = LSTMModelConfig(env, use_memory=False, use_instr=True)
    acmodel = TrajectoryLSTM(config)

    obs, info = env.reset()
    obs = DictList(obs)
    obs.image = torch.from_numpy(obs.image).to(dtype=torch.float).unsqueeze(0)
    obs.mission = torch.tensor(obs.mission).unsqueeze(0)
    result = acmodel.forward(obs, torch.zeros(1, 128 * 2))
    result


@pytest.mark.parametrize("env", babyai_envs())
def test__init___standard_env_with_instruction_bigru(env):
    env = DictObservationSpaceWrapper(env)
    obs, info = env.reset()

    config = LSTMModelConfig(
        env, use_memory=True, use_instr=True, lang_model="bigru"
    )
    acmodel = TrajectoryLSTM(config)


@pytest.mark.parametrize("env", babyai_envs())
def test__init_forward_with_instruction_bigru(env):
    env = DictObservationSpaceWrapper(env)
    obs, info = env.reset()

    config = LSTMModelConfig(
        env, use_memory=False, use_instr=True, lang_model="bigru"
    )
    acmodel = TrajectoryLSTM(config)

    obs, info = env.reset()
    obs = DictList(obs)
    obs.image = torch.from_numpy(obs.image).to(dtype=torch.float).unsqueeze(0)
    obs.mission = torch.tensor(obs.mission).unsqueeze(0)
    result = acmodel.forward(obs, torch.zeros(1, 128 * 2))
    result


@pytest.mark.parametrize("env", babyai_envs())
def test__init___standard_env_with_instruction_attgru(env):
    env = DictObservationSpaceWrapper(env)
    obs, info = env.reset()

    config = LSTMModelConfig(
        env, use_memory=True, use_instr=True, lang_model="attgru"
    )
    acmodel = TrajectoryLSTM(config)


@pytest.mark.parametrize("env", babyai_envs())
def test__init_forward_with_instruction_attgru(env):
    env = DictObservationSpaceWrapper(env)
    obs, info = env.reset()

    config = LSTMModelConfig(
        env, use_memory=False, use_instr=True, lang_model="attgru"
    )
    acmodel = TrajectoryLSTM(config)

    obs, info = env.reset()
    obs = DictList(obs)
    obs.image = torch.from_numpy(obs.image).to(dtype=torch.float).unsqueeze(0)
    obs.mission = torch.tensor(obs.mission).unsqueeze(0)
    result = acmodel.forward(obs, torch.zeros(1, 128 * 2))
    result
