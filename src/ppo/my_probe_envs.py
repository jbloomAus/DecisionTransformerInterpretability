from typing import Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete

MAIN = __name__ == "__main__"

Arr = np.ndarray
ObsType = np.ndarray
ActType = int


class Probe1(gym.Env):
    """One action, observation of [0.0], one timestep long, +1 reward.
    We expect the agent to rapidly learn that the value of the constant [0.0] observation is +1.0.
    Note we're using a continuous observation space for consistency with CartPole.
    """

    action_space: Discrete
    observation_space: Box

    def __init__(self):
        super().__init__()
        self.observation_space = Box(np.array([0]), np.array([0]))
        self.action_space = Discrete(1)
        self.reset()

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        return np.array([0]), 1.0, True, False, {}

    def reset(
        self, seed: Optional[int] = None, return_info=True, options=None
    ) -> Union[ObsType, Tuple[ObsType, dict]]:
        if return_info:
            return np.array([0.0]), {}
        return np.array([0.0])


class Probe2(gym.Env):
    """One action, observation of [-1.0] or [+1.0], one timestep long, reward equals observation.
    We expect the agent to rapidly learn the value of each observation is equal to the observation.
    """

    action_space: Discrete
    observation_space: Box

    def __init__(self):
        super().__init__()
        self.observation_space = Box(np.array([-1.0]), np.array([+1.0]))
        self.action_space = Discrete(1)
        self.reset()
        self.reward = None

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        assert self.reward is not None
        return np.array([0]), self.reward, True, False, {}

    def reset(
        self, seed: Optional[int] = None, return_info=True, options=None
    ) -> Union[ObsType, Tuple[ObsType, dict]]:
        super().reset(seed=seed)
        self.reward = 1.0 if self.np_random.random() < 0.5 else -1.0
        if return_info:
            return np.array([self.reward]), {}
        return np.array([self.reward])


class Probe3(gym.Env):
    """One action, [0.0] then [1.0] observation, two timesteps, +1 reward at the end.
    We expect the agent to rapidly learn the discounted value of the initial observation.
    """

    action_space: Discrete
    observation_space: Box

    def __init__(self):
        super().__init__()
        self.observation_space = Box(np.array([-0.0]), np.array([+1.0]))
        self.action_space = Discrete(1)
        self.reset()

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        self.n += 1
        if self.n == 1:
            return np.array([1.0]), 0.0, False, False, {}
        elif self.n == 2:
            return np.array([0]), 1.0, True, False, {}
        raise ValueError(self.n)

    def reset(
        self, seed: Optional[int] = None, return_info=True, options=None
    ) -> Union[ObsType, Tuple[ObsType, dict]]:
        super().reset(seed=seed)
        self.n = 0
        if return_info:
            return np.array([0.0]), {}
        return np.array([0.0])


class Probe4(gym.Env):
    """Two actions, [0.0] observation, one timestep, reward is -1.0 or +1.0 dependent on the action.
    We expect the agent to learn to choose the +1.0 action.
    """

    action_space: Discrete
    observation_space: Box

    def __init__(self):
        super().__init__()
        self.observation_space = Box(np.array([-0.0]), np.array([+0.0]))
        self.action_space = Discrete(2)
        self.reset()

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        reward = 1.0 if action == 1 else -1.0
        return np.array([0.0]), reward, True, False, {}

    def reset(
        self, seed: Optional[int] = None, return_info=True, options=None
    ) -> Union[ObsType, Tuple[ObsType, dict]]:
        super().reset(seed=seed)
        if return_info:
            return np.array([0.0]), {}
        return np.array([0.0])


class Probe5(gym.Env):
    """Two actions, random 0/1 observation, one timestep, reward is 1 if action equals observation otherwise -1.
    We expect the agent to learn to match its action to the observation.
    """

    action_space: Discrete
    observation_space: Box

    def __init__(self):
        super().__init__()
        self.observation_space = Box(np.array([-1.0]), np.array([+1.0]))
        self.action_space = Discrete(2)
        self.reset()

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        reward = 1.0 if action == self.obs else -1.0
        return np.array([-1.0]), reward, True, False, {}

    def reset(
        self, seed: Optional[int] = None, return_info=True, options=None
    ) -> Union[ObsType, Tuple[ObsType, dict]]:
        super().reset(seed=seed)
        self.obs = 0 if self.np_random.random() < 0.5 else 1
        if return_info:
            return np.array([self.obs], dtype=float), {}
        return np.array([self.obs], dtype=float)


class Probe6(gym.Env):
    """Two actions, single float observation that increments by 1 every time step, reward is 1 if action is 1 otherwise 0.
    We expect the agent to learn to choose action 1 when the observation is odd and action 0 when it is even.
    """

    action_space: Discrete
    observation_space: Box

    def __init__(self):
        super().__init__()
        self.observation_space = Box(
            low=0, high=np.inf, shape=(1,), dtype=np.float32
        )
        self.action_space = Discrete(2)
        self.reset()

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        self.time_step += 1
        obs = float(self.time_step)
        self.observation_space.low = np.array([obs + 1], dtype=np.float32)
        self.observation_space.high = np.array([obs + 1], dtype=np.float32)
        if self.time_step == 10:
            reward = 1.0
            return np.array([obs]), reward, True, False, {}
        else:
            reward = 0.0
            return np.array([obs]), reward, False, False, {}

    def reset(
        self, seed: Optional[int] = None, return_info=True, options=None
    ) -> Union[ObsType, Tuple[ObsType, dict]]:
        self.observation_space.low = np.array([0], dtype=np.float32)
        self.observation_space.high = np.array([0], dtype=np.float32)
        self.time_step = 0
        if return_info:
            return np.array([0], dtype=np.float32), {}
        return np.array([0], dtype=np.float32)


class Probe7(gym.Env):
    """
    4 timesteps. Observation at time 0 is samples from 0 or 1 uniformly.
    Reward is 0 at all timesteps except the 5th, when it is 1 if the action is equal to the observation given
    at the first timestep, otherwise 0.
    """

    action_space: Discrete
    observation_space: Box

    def __init__(self):
        super().__init__()
        self.observation_space = Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )
        self.action_space = Discrete(5)
        self.reset()

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        self.time_step += 1

        if self.time_step == 4:
            reward = 1.0 if action == self.initial_obs else 0.0
            return (
                np.array([self.initial_obs], dtype=np.float32),
                reward,
                True,
                False,
                {},
            )
        elif self.time_step == 0:
            return (
                np.array([self.initial_obs], dtype=np.float32),
                0.0,
                False,
                False,
                {},
            )
        else:
            return np.array([0.0], np.float32), 0.0, False, False, {}

    def reset(
        self, seed: Optional[int] = None, return_info=True, options=None
    ) -> Union[ObsType, Tuple[ObsType, dict]]:
        self.time_step = 0
        if seed is not None:
            np.random.seed(seed)
        self.initial_obs = np.array(
            [0 if self.np_random.random() < 0.5 else 1], dtype=np.float32
        )
        if return_info:
            return self.initial_obs, {}
        return self.initial_obs
