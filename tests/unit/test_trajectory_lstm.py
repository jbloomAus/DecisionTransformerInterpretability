import pytest
import torch

from minigrid.envs import MemoryEnv, DynamicObstaclesEnv, EmptyEnv, FourRoomsEnv
from minigrid.envs.babyai import GoToDoor, GoToRedBlueBall, UnlockLocal, BossLevel
from minigrid.wrappers import RGBImgPartialObsWrapper, DictObservationSpaceWrapper

from src.models.trajectory_lstm import TrajectoryLSTM


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


# create a fixture which can be used to parameterize different minigrid_envs
# @pytest.fixture
def minigrid_envs():
    return [
        MemoryEnv(size=7, random_length=False,
                  max_steps=200, render_mode='rgb_array'),
        DynamicObstaclesEnv(size=7, n_obstacles=3,
                            max_steps=200, render_mode='rgb_array'),
        EmptyEnv(size=7, max_steps=200, render_mode='rgb_array'),
        FourRoomsEnv(max_steps=200, render_mode='rgb_array'),
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

    acmodel = TrajectoryLSTM(
        obs_space=env.observation_space,
        action_space=env.action_space,
        use_memory=False
    )


@pytest.mark.parametrize("env", minigrid_envs())
def test__init_forward_without_memory(env):

    obs, info = env.reset()

    acmodel = TrajectoryLSTM(
        obs_space=env.observation_space,
        action_space=env.action_space,
        use_memory=False
    )

    obs, info = env.reset()
    obs = AttrDict(obs)
    obs['image'] = torch.tensor(obs['image']).unsqueeze(0)
    result = acmodel.forward(obs, torch.zeros(1, 128*2))
    result


@pytest.mark.parametrize("env", minigrid_envs())
def test__init___standard_env_with_memory(env):

    obs, info = env.reset()

    acmodel = TrajectoryLSTM(
        obs_space=env.observation_space,
        action_space=env.action_space,
        use_memory=True
    )


@pytest.mark.parametrize("env", minigrid_envs())
def test__init_forward_with_memory(env):

    obs, info = env.reset()

    acmodel = TrajectoryLSTM(
        obs_space=env.observation_space,
        action_space=env.action_space,
        use_memory=True
    )

    obs, info = env.reset()
    obs = AttrDict(obs)
    obs['image'] = torch.tensor(obs['image']).unsqueeze(0)
    result = acmodel.forward(obs, torch.zeros(1, 128*2))
    result


@pytest.mark.parametrize("env", minigrid_envs())
def test__init___standard_env_with_pixel(env):

    env = RGBImgPartialObsWrapper(env)
    obs, info = env.reset()

    acmodel = TrajectoryLSTM(
        obs_space=env.observation_space,
        action_space=env.action_space,
        use_memory=False,
        arch="pixels_endpool_res"
    )


@pytest.mark.parametrize("env", minigrid_envs())
def test__init_forward_with_pixel(env):

    env = RGBImgPartialObsWrapper(env)
    obs, info = env.reset()

    acmodel = TrajectoryLSTM(
        obs_space=env.observation_space,
        action_space=env.action_space,
        use_memory=False,
        arch="pixels_endpool_res"
    )

    obs, info = env.reset()
    obs = AttrDict(obs)
    obs['image'] = torch.FloatTensor(obs['image']).unsqueeze(0)
    result = acmodel.forward(obs, torch.zeros(1, 128*2))
    result


@pytest.mark.parametrize("env", minigrid_envs())
def test__init___without_endpool_res(env):

    obs, info = env.reset()

    acmodel = TrajectoryLSTM(
        obs_space=env.observation_space,
        action_space=env.action_space,
        use_memory=False,
        arch="bow"
    )


@pytest.mark.parametrize("env", minigrid_envs())
def test__init_forward_wwithout_endpool_res(env):

    obs, info = env.reset()

    acmodel = TrajectoryLSTM(
        obs_space=env.observation_space,
        action_space=env.action_space,
        use_memory=False,
        arch="bow"
    )

    obs, info = env.reset()
    obs = AttrDict(obs)
    obs['image'] = torch.FloatTensor(obs['image']).unsqueeze(0)
    result = acmodel.forward(obs, torch.zeros(1, 128*2))
    result


@pytest.mark.parametrize("env", babyai_envs())
def test__init___standard_env_with_instruction(env):

    env = DictObservationSpaceWrapper(env)
    obs, info = env.reset()

    acmodel = TrajectoryLSTM(
        obs_space=env.observation_space,
        action_space=env.action_space,
        use_memory=True,
        use_instr=True
    )


@pytest.mark.parametrize("env", babyai_envs())
def test__init_forward_with_instruction(env):

    env = DictObservationSpaceWrapper(env)
    obs, info = env.reset()

    acmodel = TrajectoryLSTM(
        obs_space=env.observation_space,
        action_space=env.action_space,
        use_memory=False,
        use_instr=True
    )

    obs, info = env.reset()
    obs = AttrDict(obs)
    obs['image'] = torch.tensor(obs['image']).unsqueeze(0)
    obs['mission'] = torch.tensor(obs['mission']).unsqueeze(0)
    result = acmodel.forward(obs, torch.zeros(1, 128*2))
    result
