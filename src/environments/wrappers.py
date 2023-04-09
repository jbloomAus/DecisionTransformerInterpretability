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
            low=0,
            high=255,
            shape=(agent_view_size, agent_view_size, *current_dim),
            dtype="uint8",
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
                img, self.render_width, self.render_height
            )

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


class DictObservationSpaceWrapper(ObservationWrapper):
    """
    Transforms the observation space (that has a textual component) to a fully numerical observation space,
    where the textual instructions are replaced by arrays representing the indices of each word in a fixed vocabulary.

    This wrapper is not applicable to BabyAI environments, given that these have their own language component.

    Example:
        >>> import miniworld
        >>> import gymnasium as gym
        >>> import matplotlib.pyplot as plt
        >>> from minigrid.wrappers import DictObservationSpaceWrapper
        >>> env = gym.make("MiniGrid-LavaCrossingS11N5-v0")
        >>> obs, _ = env.reset()
        >>> obs['mission']
        'avoid the lava and get to the green goal square'
        >>> env_obs = DictObservationSpaceWrapper(env)
        >>> obs, _ = env_obs.reset()
        >>> obs['mission'][:10]
        [19, 31, 17, 36, 20, 38, 31, 2, 15, 35]
    """

    def __init__(self, env, max_words_in_mission=50, word_dict=None):
        """
        max_words_in_mission is the length of the array to represent a mission, value 0 for missing words
        word_dict is a dictionary of words to use (keys=words, values=indices from 1 to < max_words_in_mission),
                  if None, use the Minigrid language
        """
        super().__init__(env)

        if word_dict is None:
            word_dict = self.get_minigrid_words()

        self.max_words_in_mission = max_words_in_mission
        self.word_dict = word_dict

        self.observation_space = spaces.Dict(
            {
                "image": env.observation_space["image"],
                "direction": env.observation_space["direction"],
                "mission": spaces.MultiDiscrete(
                    [len(self.word_dict.keys())] * max_words_in_mission
                ),
            }
        )

    @staticmethod
    def get_minigrid_words():
        colors = ["red", "green", "blue", "yellow", "purple", "grey"]
        objects = [
            "unseen",
            "empty",
            "wall",
            "floor",
            "box",
            "key",
            "ball",
            "door",
            "goal",
            "agent",
            "lava",
        ]

        verbs = [
            "pick",
            "avoid",
            "get",
            "find",
            "put",
            "use",
            "open",
            "go",
            "fetch",
            "reach",
            "unlock",
            "traverse",
        ]

        extra_words = [
            "up",
            "the",
            "a",
            "at",
            ",",
            "square",
            "and",
            "then",
            "to",
            "of",
            "rooms",
            "near",
            "opening",
            "must",
            "you",
            "matching",
            "end",
            "hallway",
            "object",
            "from",
            "room",
            "next",
            "after",
            "behind",
            "on",
            "your",
            "in",
            "front",
            "right",
            "left",
        ]

        all_words = colors + objects + verbs + extra_words
        assert len(all_words) == len(set(all_words))
        return {word: i for i, word in enumerate(all_words)}

    def string_to_indices(self, string, offset=1):
        """
        Convert a string to a list of indices.
        """
        indices = []
        # adding space before and after commas
        string = string.replace(",", " , ")
        for word in string.split():
            if word in self.word_dict.keys():
                indices.append(self.word_dict[word] + offset)
            else:
                raise ValueError(f"Unknown word: {word}")
        return indices

    def observation(self, obs):
        obs["mission"] = self.string_to_indices(obs["mission"])
        assert len(obs["mission"]) < self.max_words_in_mission
        obs["mission"] += [0] * (
            self.max_words_in_mission - len(obs["mission"])
        )

        return obs
