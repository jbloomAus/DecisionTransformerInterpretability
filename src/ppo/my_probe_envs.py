# %%
import gym
import numpy as np
from typing import Union
from typing import Optional
from gym.spaces import Discrete, Box

MAIN = __name__ == "__main__"

Arr = np.ndarray
ObsType = np.ndarray
ActType = int

## %

class Probe1(gym.Env):
    """One action, observation of [0.0], one timestep long, +1 reward.
    We expect the agent to rapidly learn that the value of the constant [0.0] observation is +1.0. Note we're using a continuous observation space for consistency with CartPole.
    """

    action_space: Discrete
    observation_space: Box

    def __init__(self):
        super().__init__()
        self.observation_space = Box(np.array([0]), np.array([0]))
        self.action_space = Discrete(1)
        self.seed()
        self.reset()

    def step(self, action: ActType) -> tuple[ObsType, float, bool, dict]:
        return np.array([0]), 1.0, True, {}

    def reset(
        self, seed: Optional[int] = None, return_info=False, options=None
    ) -> Union[ObsType, tuple[ObsType, dict]]:
        if return_info:
            return np.array([0.0]), {}
        return np.array([0.0])


if MAIN:
    env = gym.make("Probe1-v0")
    assert env.observation_space.shape == (1,)
    assert env.action_space.shape == ()

"""
### Additional Probe Environments
Feel free to skip ahead for now, and implement these as needed to debug your model. 
"""
# %%
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
        self.seed()
        self.reset()
        self.reward = None

    def step(self, action: ActType) -> tuple[ObsType, float, bool, dict]:
        assert self.reward is not None
        return np.array([0]), self.reward, True, {}

    def reset(
        self, seed: Optional[int] = None, return_info=False, options=None
    ) -> Union[ObsType, tuple[ObsType, dict]]:
        super().reset(seed=seed)
        self.reward = 1.0 if self.np_random.random() < 0.5 else -1.0
        if return_info:
            return np.array([self.reward]), {}
        return np.array([self.reward])



# %%
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
        self.seed()
        self.reset()

    def step(self, action: ActType) -> tuple[ObsType, float, bool, dict]:
        self.n += 1
        if self.n == 1:
            return np.array([1.0]), 0.0, False, {}
        elif self.n == 2:
            return np.array([0]), 1.0, True, {}
        raise ValueError(self.n)

    def reset(
        self, seed: Optional[int] = None, return_info=False, options=None
    ) -> Union[ObsType, tuple[ObsType, dict]]:
        super().reset(seed=seed)
        self.n = 0
        if return_info:
            return np.array([0.0]), {}
        return np.array([0.0])



# %%
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
        self.seed()
        self.reset()

    def step(self, action: ActType) -> tuple[ObsType, float, bool, dict]:
        reward = 1.0 if action == 1 else -1.0
        return np.array([0.0]), reward, True, {}

    def reset(
        self, seed: Optional[int] = None, return_info=False, options=None
    ) -> Union[ObsType, tuple[ObsType, dict]]:
        super().reset(seed=seed)
        if return_info:
            return np.array([0.0]), {}
        return np.array([0.0])




# %%
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
        self.seed()
        self.reset()

    def step(self, action: ActType) -> tuple[ObsType, float, bool, dict]:
        reward = 1.0 if action == self.obs else -1.0
        return np.array([-1.0]), reward, True, {}

    def reset(
        self, seed: Optional[int] = None, return_info=False, options=None
    ) -> Union[ObsType, tuple[ObsType, dict]]:
        super().reset(seed=seed)
        self.obs = 0 if self.np_random.random() < 0.5 else 1
        if return_info:
            return np.array([self.obs], dtype=float), {}
        return np.array([self.obs], dtype=float)


