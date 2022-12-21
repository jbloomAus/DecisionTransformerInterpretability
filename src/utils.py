import gymnasium as gym
import torch 
import numpy as np

def get_obs_preprocessor(obs_space):

    # handle cases where obs space is instance of gym.spaces.Box, gym.spaces.Dict, gym.spaces

    if isinstance(obs_space, gym.spaces.Box):
        return lambda x: x

    elif isinstance(obs_space, gym.spaces.Dict):
        obs_space = obs_space.spaces
        if 'image' in obs_space:
            return lambda x: preprocess_images(x['image'])

def preprocess_images(images, device=None):
    # Bug of Pytorch: very slow if not first converted to numpy array
    images = np.array(images)
    return images