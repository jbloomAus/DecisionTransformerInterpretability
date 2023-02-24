import gymnasium as gym
import numpy as np
from gymnasium import spaces
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import (FullyObsWrapper, ObservationWrapper,
                               OneHotPartialObsWrapper)


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
    agent_view_size=7,
    video_frequency=50
):
    """Return a function that returns an environment after setting up boilerplate."""

    # only one of fully observed or flat one hot can be true.
    assert not (
        fully_observed and flat_one_hot), "Can't have both fully_observed and flat_one_hot."

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
                env = RenderResizeWrapper(env, 256, 256)
                env = gym.wrappers.RecordVideo(
                    env,
                    f"videos/{run_name}",
                    # Video every 50 runs for env #1
                    episode_trigger=lambda x: x % video_frequency == 0,
                    disable_logger=True
                )
        

        # hard code for now!
        if env_id.startswith("MiniGrid"):
            if fully_observed:
                env = FullyObsWrapper(env)
            if agent_view_size != 7:
                env = ViewSizeWrapper(env, agent_view_size=agent_view_size)
            if flat_one_hot:
                env = OneHotPartialObsWrapper(env)

        obs = env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        env.run_name = run_name
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

class RenderResizeWrapper(gym.Wrapper):
    def __init__(self, env: MiniGridEnv, render_width = 256, render_height = 256):
        super().__init__(env)
        self.render_width = render_width
        self.render_height = render_height

    def render(self):

        if self.env.render_mode == "rgb_array":
            img = self.env.render()
            img = np.array(img)

            # Resize image
            img = self._resize_image(img, self.render_width, self.render_height)

            return img

        else:
            return self.env.render()

    def _resize_image(self, image, width, height):
        from PIL import Image

        # Convert to PIL Image
        image = Image.fromarray(image)

        # Resize image
        image = image.resize((width, height), resample=Image.BILINEAR)

        # Convert back to numpy array
        image = np.array(image)

        return image

class MultiEnvSampler(gym.Env):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }
    
    def __init__(self, envs, p=None, render_mode='rgb_array'):
        if len(envs) < 2:
            raise ValueError("MultiEnvSampler requires at least two environments")
        self.envs = envs
        # don't call it num_envs because this interacts badly with RecordEpisodeStatistics wrapper. Solve later.
        self.n_envs = len(envs)
        self.env_names = [env.unwrapped.__class__.__name__ for env in envs]
        self.p = p
        if self.p is None:
            self.p = np.ones(self.n_envs) / self.n_envs
        elif len(self.p) != self.n_envs:
            raise ValueError("The length of p must be equal to the number of environments")

        obs_space = self.envs[0].observation_space
        action_space = self.envs[0].action_space

        self.env_id = 0

        self._homogenize_mission_spaces()
        for env in self.envs[1:]:
            if not env.observation_space == obs_space:
                raise ValueError(
                    f"All environments must have the same observation space\n{env.observation_space} != {obs_space}"
                    )

            if not env.action_space == action_space:
                raise ValueError(
                    f"All environments must have the same action space\n{env.action_space} != {action_space}"
                    )

        self.render_mode = render_mode
        self.observation_space = obs_space
        self.action_space = action_space

    def reset(self, seed= None, all_envs=False, options=None):
        np.random.seed(seed)
        self.env_id = np.random.choice(self.n_envs, p=self.p)
        if all_envs:
            return [env.reset() for env in self.envs]
        return self.envs[self.env_id].reset()

    def step(self, action):
        obs, reward, done, info, truncated = self.envs[self.env_id].step(action)
        return obs, reward, done, info, truncated

    def render(self):
        return self.envs[self.env_id].render()

    def close(self):
        for env in self.envs:
            env.close()

    def get_current_env_name(self):
        return self.env_names[self.env_id]
    
    def _sample_env_id(self):
        env_id = np.random.choice(self.n_envs, p=self.p)
        return env_id

    def _homogenize_mission_spaces(self):
        '''resets all mission spaces to be equal to the first env'''
        # set mission space to the first env with a mission space's mission space
        if self.envs[0].observation_space["mission"] is not None:
            mission_space = self.envs[0].observation_space["mission"]
        else:
            mission_space = None
        for env in self.envs[1:]:
            if env.observation_space["mission"] is not None:
                pass
            env.observation_space["mission"] = mission_space

from gymnasium import register
from minigrid.envs import DynamicObstaclesEnv, CrossingEnv, MultiRoomEnv, DoorKeyEnv, EmptyEnv, FourRoomsEnv


def get_dynamic_obstacles_multi_env(render_mode='rgb_array', max_steps=1000):

    envs = []
    env = DynamicObstaclesEnv(
        size=6,
        n_obstacles=0,
        agent_start_pos = None,
        render_mode=render_mode,
        max_steps=max_steps
    )
    envs.append(env)
    for size in range(6,10):
        for num_obstacles in range(5,7):
            env = DynamicObstaclesEnv(
                size=size,
                n_obstacles=num_obstacles,
                agent_start_pos = None,
                render_mode=render_mode,
                max_steps=max_steps
            )
            envs.append(env)
    
    return MultiEnvSampler(envs)


def get_dynamic_obstacles_multi_env(render_mode='rgb_array', max_steps=1000):

    envs = []
    env = DynamicObstaclesEnv(
        size=6,
        n_obstacles=0,
        agent_start_pos = None,
        render_mode=render_mode,
        max_steps=max_steps
    )
    envs.append(env)
    for size in range(6,10):
        for num_obstacles in range(5,7):
            env = DynamicObstaclesEnv(
                size=size,
                n_obstacles=num_obstacles,
                agent_start_pos = None,
                render_mode=render_mode,
                max_steps=max_steps
            )
            envs.append(env)
    
    return MultiEnvSampler(envs)

from minigrid.core.world_object import Lava, Wall

def get_crossing_multi_env(render_mode='rgb_array', max_steps=1000):

    envs = []
    for size in range(5,14,2):
        for num_crossings in range(0,7):
            env = CrossingEnv(
                size=size,
                num_crossings=num_crossings,
                obstacle_type=Lava,
                render_mode=render_mode,
                max_steps=max_steps
            )
            envs.append(env)
    
    for size in range(5,14,2):
        for num_crossings in range(0,7):
            env = CrossingEnv(
                size=size,
                num_crossings=num_crossings,
                obstacle_type=Wall,
                render_mode=render_mode,
                max_steps=max_steps
            )
            envs.append(env)
    
    return MultiEnvSampler(envs)


#  minNumRooms,
#         maxNumRooms,
#         maxRoomSize=10,
#         max_steps: int | None = None,

def get_multi_room_env(render_mode='rgb_array', max_steps=1000):

    envs = []
    for min_rooms in range(1,5):
        for max_rooms in range(min_rooms,5):
            for max_room_size in range(5,10):
                env = MultiRoomEnv(
                    minNumRooms=min_rooms,
                    maxNumRooms=max_rooms,
                    maxRoomSize=max_room_size,
                    render_mode=render_mode,
                    max_steps=max_steps
                )
                envs.append(env)
    
    return MultiEnvSampler(envs)

print("Registering DynamicObstaclesMultiEnv-v0")
print("Registering CrossingMultiEnv-v0")
def register_envs():
    
    register(
        id='DynamicObstaclesMultiEnv-v0',
        entry_point='environments:get_dynamic_obstacles_multi_env',
    )

    register(
        id='CrossingMultiEnv-v0',
        entry_point='environments:get_crossing_multi_env',
    )

    register(
        id='MultiRoomMultiEnv-v0',
        entry_point='environments:get_multi_room_env',
    )


