import gymnasium as gym
import numpy as np
from typing import List
from collections import Counter
import uuid
from gymnasium.core import Wrapper
from minigrid.core.mission import MissionSpace
from minigrid.wrappers import OneHotPartialObsWrapper, FullyObsWrapper, ObservationWrapper
from gymnasium import spaces
from dataclasses import dataclass


@dataclass
class EnvironmentArgs:
    env_ids: List[str]
    env_prob: List[float] = None
    seed: int = 1
    capture_video: bool = False
    run_name: str = "test"
    render_mode: str = "rgb_array"
    max_steps: int = 100
    fully_observed: bool = False
    flat_one_hot: bool = False
    agent_view_size: int = 7
    video_frequency: int = 50


class EnvGenerator:
    """Generates a set of gym vectorized environments given a set of input parameters."""

    def __init__(self, env_args: EnvironmentArgs):
        self.env_args = env_args

        if isinstance(self.env_args.env_ids, str):
            self.env_args.env_ids = [self.env_args.env_ids]

        if self.env_args.env_prob is None:
            self.env_args.env_prob = [
                1 / len(self.env_args.env_ids)] * len(self.env_args.env_ids)

    def generate_envs(self, num_envs: int):
        """Generate a set of vectorized environments using the input parameters."""

        # env_funcs = [self._make_env(i) for i in range(num_envs)]
        # # print out each observation space
        # reference = env_funcs[0]()
        # for env_func in env_funcs:
        #     env = env_func()
        #     print(env.observation_space)
        #     print(env)
        #     if env.observation_space != reference.observation_space:
        #         print("Observation spaces must be the same for all environments.\n, {} != {}".format(env.observation_space, reference.observation_space))
        #         print(env.observation_space["mission"])
        #         print(reference.observation_space["mission"])
        #         print(env.observation_space["mission"] == reference.observation_space["mission"])
        #         # quit()

        envs = gym.vector.SyncVectorEnv(
            [self._make_env(i) for i in range(num_envs)])
        return envs

    def _make_env(self, idx: int):
        """Return a function that returns an environment after setting up boilerplate."""

        def thunk():

            kwargs = {}
            if self.env_args.render_mode:
                kwargs["render_mode"] = self.env_args.render_mode
            if self.env_args.max_steps:
                kwargs["max_steps"] = self.env_args.max_steps

            env_id = np.random.choice(
                self.env_args.env_ids,
                p=self.env_args.env_prob,
            )

            env = gym.make(env_id, **kwargs)

            env = gym.wrappers.RecordEpisodeStatistics(env)
            if self.env_args.capture_video:
                if idx == 0:
                    env = gym.wrappers.RecordVideo(
                        env,
                        f"videos/{self.env_args.run_name}",
                        # Video every 50 runs for env #1
                        episode_trigger=lambda x: (
                            x % self.env_args.video_frequency) == 0,
                        disable_logger=True,
                        name_prefix=f"env_{uuid.uuid4()}",
                    )

            # only one of fully observed or flat one hot can be true.
            assert not (
                self.env_args.fully_observed and self.env_args.flat_one_hot), "Can't have both fully_observed and flat_one_hot."

            # hard code for now!
            if env_id.startswith("MiniGrid"):
                if self.env_args.fully_observed:
                    env = FullyObsWrapper(env)
                if self.env_args.agent_view_size != 7:
                    env = ViewSizeWrapper(
                        env, agent_view_size=self.env_args.agent_view_size)
                if self.env_args.flat_one_hot:
                    env = OneHotPartialObsWrapper(env)

            # if more than one env, use the MissionSpaceEqualRedefinitionWrapper
            # if len(set(self.env_args.env_ids)) > 1:
            env = MissionSpaceEqualRedefinitionWrapper(env)

            obs = env.reset()
            # env.action_space.seed(self.env_args.seed)
            # env.observation_space.seed(self.env_args.seed)
            env.run_name = self.env_args.run_name
            return env

        return thunk


class ViewSizeWrapper(ObservationWrapper):
    """
    Wrapper to customize the agent field of view size.
    This cannot be used with fully observable wrappers.

    Example:
        >>> import miniworld
        >>> import gymnasium as gym
        >>> import matplotlib.pyplot as plt
        >>> from minigrid.wrappers import ViewSizeWrapper
        >>> env = gym.make("MiniGrid-LavaCrossingS11N5-v0")
        >>> obs, _ = env.reset()
        >>> obs['image'].shape
        (7, 7, 3)
        >>> env_obs = ViewSizeWrapper(env, agent_view_size=5)
        >>> obs, _ = env_obs.reset()
        >>> obs['image'].shape
        (5, 5, 3)
    """

    def __init__(self, env, agent_view_size=7):
        super().__init__(env)

        assert agent_view_size % 2 == 1
        assert agent_view_size >= 3

        self.agent_view_size = agent_view_size

        # Compute observation space with specified view size
        new_image_space = gym.spaces.Box(
            low=0, high=255, shape=(agent_view_size, agent_view_size, 3), dtype="uint8"
        )

        # Override the environment's observation spaceexit
        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "image": new_image_space}
        )

    def observation(self, obs):
        env = self.unwrapped

        grid, vis_mask = env.gen_obs_grid(self.agent_view_size)

        # Encode the partially observable view into a numpy array
        image = grid.encode(vis_mask)

        return {**obs, "image": image}


class MissionSpaceEqualRedefinitionWrapper(Wrapper):
    """
    Wrapper to redefine the mission space to not return equal even if the mission_funcs are not equal.
    I'd like to keep the dict observation space but make a sync vector with heterogenous mission
    statements
    """

    def __init__(self, env):
        super().__init__(env)

        self.unwrapped.observation_space['mission'] = MissionSpaceIgnoreFuncEqual(
            self.unwrapped.observation_space["mission"].mission_func,
            self.unwrapped.observation_space["mission"].ordered_placeholders
        )


class MissionSpaceIgnoreFuncEqual(MissionSpace):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __eq__(self, other) -> bool:
        """Check whether ``other`` is equivalent to this instance."""
        if isinstance(other, MissionSpace):

            # Check that place holder lists are the same
            if self.ordered_placeholders is not None:
                # Check length
                if (
                    len(self.ordered_placeholders) == len(
                        other.ordered_placeholders)
                ) and (
                    all(
                        set(i) == set(j)
                        for i, j in zip(
                            self.ordered_placeholders, other.ordered_placeholders
                        )
                    )
                ):
                    # Check mission string is the same with dummy space placeholders
                    test_placeholders = [""] * len(self.ordered_placeholders)
                    mission = self.mission_func(*test_placeholders)
                    other_mission = other.mission_func(*test_placeholders)
                    return mission == other_mission
            else:

                # Check that other is also None
                if other.ordered_placeholders is None:

                    # # Check mission string is the same
                    # mission = self.mission_func()
                    # other_mission = other.mission_func()
                    # return mission == other_mission
                    return True

        # If none of the statements above return then False
        return False
