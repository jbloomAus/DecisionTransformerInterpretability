import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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


# Plotting funcs for the embeddings
def tensor_cosine_similarity_heatmap(tensor):
    # Normalize each row in the tensor
    row_norms = torch.norm(tensor, dim=1, keepdim=True)
    normalized_tensor = tensor / row_norms

    # Calculate the pairwise cosine similarity
    cosine_similarity_matrix = torch.matmul(
        normalized_tensor, normalized_tensor.T
    )

    # Convert the resulting tensor to a Pandas DataFrame
    df = pd.DataFrame(cosine_similarity_matrix.numpy())

    # Visualize the pairwise cosine similarity as a heatmap using Plotly Express
    fig = px.imshow(
        df,
        color_continuous_scale="viridis",
        title="Pairwise Cosine Similarity Heatmap",
    )
    fig.show()


def tensor_2d_embedding_similarity(tensor, x, y, mode="heatmap"):
    # Get the specified embedding
    embedding = tensor[y, x]

    # Normalize the input tensor
    tensor_norms = torch.norm(tensor, dim=-1, keepdim=True)
    normalized_tensor = tensor / tensor_norms

    # Normalize the specified embedding
    embedding_norm = torch.norm(embedding, keepdim=True)
    normalized_embedding = embedding / embedding_norm

    # Calculate the cosine similarity between each position and the specified embedding
    cosine_similarity_matrix = torch.matmul(
        normalized_tensor.view(-1, normalized_tensor.shape[-1]),
        normalized_embedding.view(-1, 1),
    )
    cosine_similarity_matrix = cosine_similarity_matrix.view(
        normalized_tensor.shape[:-1]
    )

    # Convert the resulting tensor to a Pandas DataFrame
    df = pd.DataFrame(cosine_similarity_matrix.numpy())

    if mode == "heatmap":
        # Visualize the pairwise cosine similarity as a heatmap using Plotly Express
        fig = px.imshow(
            df,
            color_continuous_scale="viridis",
            title=f"Cosine Similarity Heatmap for Embedding at ({x}, {y})",
        )

    elif mode == "contour":
        # Create a meshgrid for the contour plot
        x_grid, y_grid = torch.meshgrid(
            torch.arange(tensor.shape[0]), torch.arange(tensor.shape[1])
        )

        # Visualize the cosine similarity as a 3D contour plot using Plotly
        fig = go.Figure(
            data=[
                go.Surface(
                    z=df.values,
                    x=x_grid.numpy(),
                    y=y_grid.numpy(),
                    colorscale="Viridis",
                )
            ]
        )

        fig.update_layout(
            scene=dict(
                zaxis_title="Cosine Similarity",
                xaxis_title="X",
                yaxis_title="Y",
            ),
            title=f"3D Cosine Similarity Contour Plot for Embedding at ({x}, {y})",
        )

    fig.show()
