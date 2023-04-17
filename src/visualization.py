import numpy as np
import torch
from minigrid.core.constants import IDX_TO_OBJECT


def find_agent(observation):
    height = observation.shape[0]
    width = observation.shape[1]
    for i in range(width):
        for j in range(height):
            object = IDX_TO_OBJECT[int(observation[i, j][0])]
            if object == "agent":
                return i, j

    return -1, -1


def render_minigrid_observation(env, observation):
    if isinstance(observation, np.ndarray):
        observation = (
            observation.copy()
        )  # so we don't edit the original object
    elif isinstance(observation, torch.Tensor):
        observation = observation.numpy().copy()

    agent_pos = find_agent(observation)
    agent_dir = observation[agent_pos[0], agent_pos[1]][2]

    # print(agent_pos, agent_dir)
    # observation[agent_pos[0], agent_pos[1]] = [0, 0, 0]
    # import streamlit as st
    # st.write(env.spec.id)
    # st.write(env.observation_space)
    grid, _ = env.grid.decode(observation.astype(np.uint8))

    i = agent_pos[0]
    j = agent_pos[1]

    return grid.render(32, (i, j), agent_dir=agent_dir)


def render_minigrid_observations(env, observations):
    return np.array(
        [
            render_minigrid_observation(env, observation)
            for observation in observations
        ]
    )
