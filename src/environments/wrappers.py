import gymnasium as gym
import numpy as np
from gymnasium import spaces
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import ObservationWrapper


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
        current_dim = self.observation_space["image"].shape[2:]
        # Compute observation space with specified view size
        new_image_space = gym.spaces.Box(
            low=0, high=255, shape=(agent_view_size, agent_view_size, *current_dim), dtype="uint8"
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
    def __init__(self, env: MiniGridEnv, render_width=256, render_height=256):
        super().__init__(env)
        self.render_width = render_width
        self.render_height = render_height

    def render(self):

        if self.env.render_mode == "rgb_array":
            img = self.env.render()
            img = np.array(img)

            # Resize image
            img = self._resize_image(
                img, self.render_width, self.render_height)

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
