import gymnasium as gym
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
from minigrid.core.constants import IDX_TO_OBJECT
from src.environments.utils import reverse_one_hot
import itertools


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


def get_cosine_sim_df(tensor, column_labels=None, row_labels=None):
    # Normalize each row in the tensor
    row_norms = torch.norm(tensor, dim=1, keepdim=True)
    normalized_tensor = tensor / row_norms

    # Calculate the pairwise cosine similarity
    cosine_similarity_matrix = torch.matmul(
        normalized_tensor, normalized_tensor.T
    )

    # Convert the resulting tensor to a Pandas DataFrame
    df = pd.DataFrame(cosine_similarity_matrix.numpy())
    if column_labels is not None:
        df.columns = column_labels
    if row_labels is not None:
        df.index = row_labels

    return df


# Plotting funcs for the embeddings
def tensor_cosine_similarity_heatmap(
    tensor,
    labels=None,
    index_labels=None,
):
    df = get_cosine_sim_df(tensor)

    # index labels are a list of lists used to add more detail to the df
    if index_labels:
        indices = list(itertools.product(*index_labels))
        multi_index = pd.MultiIndex.from_tuples(
            indices,
            names=labels,  # use labels differently if we have index labels
        )
        if len(labels) == 3:
            df.index = multi_index.to_series().apply(
                lambda x: "{0}, ({1},{2})".format(*x)
            )
            df.columns = multi_index.to_series().apply(
                lambda x: "{0}, ({1},{2})".format(*x)
            )
        else:
            df.index = multi_index.to_series().apply(
                lambda x: "({0},{1})".format(*x)
            )
            df.columns = multi_index.to_series().apply(
                lambda x: "({0},{1})".format(*x)
            )

    # Visualize the pairwise cosine similarity as a heatmap using Plotly Express
    fig = px.imshow(
        df,
        color_continuous_scale="RdBu",
        title="Pairwise Cosine Similarity Heatmap",
        color_continuous_midpoint=0.0,
        labels={"color": "Cosine Similarity"},
    )
    if labels and not index_labels:
        fig.update_xaxes(
            tickmode="array",
            tickvals=list(range(len(labels))),
            ticktext=labels,
            showgrid=False,
        )
        fig.update_yaxes(
            tickmode="array",
            tickvals=list(range(len(labels))),
            ticktext=labels,
            showgrid=False,
        )
    if index_labels:
        fig.update_xaxes(
            visible=False,
        )
        fig.update_yaxes(
            visible=False,
        )

    return fig


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


def get_param_stats(model):
    param_stats = []

    for name, param in model.named_parameters():
        mean = param.data.mean().item()
        std = param.data.std().item()
        if param.data.dim() > 1:
            norm = torch.norm(param.data, dim=1).mean().item()
        else:
            norm = torch.norm(param.data).item()
        param_stats.append(
            {"name": name, "mean": mean, "std": std, "norm": norm}
        )

    df = pd.DataFrame(param_stats)
    return df


def plot_param_stats(df):
    """
    use get_param stats then this to look at properties of weights.
    """
    # Calculate log of standard deviation
    df["log_std"] = -1 * np.log(df["std"])

    # add color column to df, red for ends with weight, blue for ends with bias, purple if embedding
    df["color"] = "green"
    df.loc[df["name"].str.endswith("weight"), "color"] = "red"
    df.loc[df["name"].str.contains("W_"), "color"] = "red"
    df.loc[df["name"].str.endswith("bias"), "color"] = "blue"
    df.loc[df["name"].str.contains("b_"), "color"] = "blue"
    df.loc[df["name"].str.contains("embedding"), "color"] = "purple"

    # make a name label which is name.split('.')[-1]
    df["name_label"] = df["name"].apply(lambda x: x.split(".")[-1])

    # Create the mean bar chart
    fig_mean = go.Figure()
    fig_mean.add_trace(
        go.Bar(
            x=df["name"],
            y=df["mean"],
            text=df["mean"],
            textposition="outside",
            hovertext=df["name_label"],
            marker_color=df["color"],
        )
    )
    fig_mean.update_traces(
        texttemplate="%{text:.4f}",
        hovertemplate="Parameter: %{hovertext}<br>Mean: %{text:.4f}",
    )
    fig_mean.update_yaxes(title_text="Mean")
    fig_mean.update_xaxes(title_text="Parameter Name")
    fig_mean.update_layout(title_text="Mean of Model Parameters")

    # Create the norm chart
    fig_norm = go.Figure()
    fig_norm.add_trace(
        go.Bar(
            x=df["name"],
            y=df["norm"],
            text=df["norm"],
            textposition="outside",
            hovertext=df["name_label"],
            marker_color=df["color"],
        )
    )
    fig_norm.update_traces(
        texttemplate="%{text:.4f}",
        hovertemplate="Parameter: %{hovertext}<br>Norm: %{text:.4f}",
    )
    fig_norm.update_yaxes(title_text="Norm")
    fig_norm.update_xaxes(title_text="Parameter Name")
    fig_norm.update_layout(title_text="Norm of Model Parameters")

    # Create the log of standard deviation bar chart
    fig_log_std = go.Figure()
    fig_log_std.add_trace(
        go.Bar(
            x=df["name"],
            y=df["log_std"],
            text=df["log_std"],
            textposition="outside",
            hovertext=df["name_label"],
            marker_color=df["color"],
        )
    )
    fig_log_std.update_traces(
        texttemplate="%{text:.4f}",
        hovertemplate="Parameter: %{hovertext}<br>Log Std: %{text:.4f}",
    )
    fig_log_std.update_yaxes(title_text="Log of Standard Deviation")
    fig_log_std.update_xaxes(title_text="Parameter Name")
    fig_log_std.update_layout(
        title_text="Log of Standard Deviation of Model Parameters"
    )
    # add a horizontal line at y = 1.69
    fig_log_std.add_shape(
        type="line",
        x0=0,
        y0=-np.log(0.02),
        x1=len(df["name"]),
        y1=-np.log(0.02),
        line=dict(color="red", width=2, dash="dash"),
    )

    return fig_mean, fig_log_std, fig_norm


# used by streamlit app
def get_rendered_obs(
    env: gym.Env,
    obs: torch.Tensor,
):
    obs = obs.clone()
    if obs.shape[-1] == 20:
        obs = reverse_one_hot(observation=obs)
    image = render_minigrid_observation(env, obs)
    return image


# used by streamlit app, works on many obs
def get_rendered_obss(
    env: gym.Env,
    obs: torch.Tensor,
):
    obs = [get_rendered_obs(env, frame) for frame in obs]
    obs = np.stack(obs)
    return obs
