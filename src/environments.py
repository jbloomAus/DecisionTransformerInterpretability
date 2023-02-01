
import operator
from functools import reduce

import gymnasium as gym
import minigrid
import numpy as np
from gymnasium import spaces
from minigrid.wrappers import ObservationWrapper
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX, STATE_TO_IDX

def make_env(
    env_id: str, 
    seed: int, 
    idx: int, 
    capture_video: bool, 
    run_name: str, 
    render_mode="rgb_array", 
    max_steps=100, 
    fully_observed=False, 
    flat_one_hot=False,
    video_frequency=50
    ):
    """Return a function that returns an environment after setting up boilerplate."""

    # only one of fully observed or flat one hot can be true.
    assert not (fully_observed and flat_one_hot), "Can't have both fully_observed and flat_one_hot."

    def thunk():

        kwargs = {}
        if render_mode:
            kwargs["render_mode"] = render_mode
        if max_steps:
            kwargs["max_steps"] = max_steps

        env = gym.make(env_id, **kwargs)

        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(
                    env,
                    f"videos/{run_name}",
                    episode_trigger=lambda x: x % video_frequency == 0,  # Video every 50 runs for env #1
                    disable_logger=True
                )

        # hard code for now!
        if env_id.startswith("MiniGrid"):
            if fully_observed:
                env = minigrid.wrappers.FullyObsWrapper(env)
            elif flat_one_hot:
                env = FlatOneHotObsWrapper(env)

        obs = env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


# I decided to rewrite the flat obs wrapper since it bugs me 
# that they only one hot encoded the mission string and not the image.
class FlatOneHotObsWrapper(ObservationWrapper):
    """
    Encode observed "image" state (not image, minigrid schema) using a one-hot scheme and flatten.
    This wrapper is not applicable to BabyAI environments, given that these have their own language component.
    """

    def __init__(self, env, maxStrLen=96):
        super().__init__(env)

        self.maxStrLen = maxStrLen
        self.numCharCodes = 28

        imgSpace = env.observation_space.spaces["image"]
        imgSize = reduce(operator.mul, imgSpace.shape, 1)

        max_one_hot_dim_idx = max(
            [max(OBJECT_TO_IDX.values()), 
            max(STATE_TO_IDX.values()), 
            max(COLOR_TO_IDX.values())]
            )


        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(imgSize*max_one_hot_dim_idx,),
            dtype="uint8",
        )

    def observation(self, obs):
        image = obs["image"]
        

        obs = image.flatten()
        # one hot encode the image and flatten again us vectorization
        obs = np.eye(len(OBJECT_TO_IDX))[image.flatten()]
        obs = obs.flatten()

        return obs