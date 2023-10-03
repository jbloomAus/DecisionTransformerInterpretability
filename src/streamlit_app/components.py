import itertools
import torch
import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
import uuid

from .environment import get_action_preds_from_app_state
from .utils import read_index_html
from .visualizations import (
    plot_action_preds,
    render_env,
    plot_heatmap,
)
from src.visualization import get_rendered_obs


def render_game_screen(dt, env):
    columns = st.columns(2)
    with columns[0]:
        st.write(f"Current RTG: {round(st.session_state.rtg[0][-1].item(),2)}")
        action_preds, x, cache, tokens = get_action_preds_from_app_state(dt)
        plot_action_preds(action_preds)
    with columns[1]:
        current_time = st.session_state.timesteps
        st.write(f"Current Time: {int(current_time[0][-1].item())}")
        fig = render_env(env)
        st.pyplot(fig)

    return x, cache, tokens


def hyperpar_side_bar():
    with st.sidebar:
        st.subheader("Hyperparameters")
        initial_rtg = st.slider(
            "Initial RTG",
            min_value=0.0,
            max_value=1.0,
            value=0.9,
            step=0.01,
        )
        if "rtg" in st.session_state:
            # get cumulative reward
            cumulative_reward = st.session_state.reward.cumsum(dim=1)
            rtg = initial_rtg * torch.ones(
                (1, cumulative_reward.shape[1], 1), dtype=torch.float
            )  # no reward yet
            st.session_state.rtg = rtg - cumulative_reward

    return initial_rtg


def render_trajectory_details():
    with st.expander("Trajectory Details"):
        # write out actions, rtgs, rewards, and timesteps
        st.write(f"max timeteps: {st.session_state.max_len}")
        st.write(f"trajectory length: {len(st.session_state.obs[0])}")
        if st.session_state.a is not None:
            st.write(f"actions: {st.session_state.a[0].squeeze(-1).tolist()}")
        st.write(f"rtgs: {st.session_state.rtg[0].squeeze(-1).tolist()}")
        st.write(f"rewards: {st.session_state.reward[0].squeeze(-1).tolist()}")
        st.write(
            f"timesteps: {st.session_state.timesteps[0].squeeze(-1).tolist()}"
        )


def reset_button():
    if st.button("reset", key=uuid.uuid4()):
        reset_env_dt()
        st.experimental_rerun()


def record_keypresses():
    components.html(
        read_index_html(),
        height=0,
        width=0,
    )


def reset_env_dt():
    if "env" in st.session_state:
        del st.session_state.env
    if "dt" in st.session_state:
        del st.session_state.dt


def model_info():
    """
    A module which shows information about the current model.

    """
    dt = st.session_state.dt

    with st.expander("Model Info"):
        config_tab, model_tab = st.tabs(["Config", "Model"])

        with config_tab:
            st.write(f"Model Type: {dt.model_type}")
            # convert into pandas df
            transformer_config = dt.transformer_config.__dict__
            transformer_config = {
                k: [v] for k, v in transformer_config.items()
            }
            transformer_config = pd.DataFrame(transformer_config)
            st.write(transformer_config)
            # for k, v in dt.transformer_config.__dict__.items():
            #     st.write(f"{k}: {v}")

        if "model_summary" in st.session_state:
            with model_tab:
                st.write("Only summarizes the transformer.")
                st.write(
                    "```" + str(st.session_state.model_summary) + "```",
                    unsafe_allow_html=True,
                )
        else:
            st.warning(
                "No model summary available. Please run a trajectory take an action."
            )


def show_history():
    with st.expander("Show history"):
        rendered_obss = st.session_state.rendered_obs
        trajectory_length = rendered_obss.shape[0]

        state_tab, obs_tab = st.tabs(["World State", "Agent POV"])

        # st.write(
        #     {i : st.session_state.labels[1::3][i] for i in range(trajectory_length)}
        #     )
        mapping = {i: i for i in list(range(trajectory_length))}
        for i in range(st.session_state.max_len, 0, -1):
            mapping[trajectory_length - i] = st.session_state.labels[1::3][
                i - 1
            ]

        st.write("Use this mapping to match these frames to tokens")
        st.write(mapping)

        with state_tab:
            st.plotly_chart(
                px.imshow(rendered_obss[:, :, :, :], animation_frame=0),
                use_container_width=True,
            )

        with obs_tab:
            env = st.session_state.env
            pov_obs = [
                st.session_state.obs[0][-i - 1]
                for i in range(trajectory_length)
            ][::-1]
            env = st.session_state.env
            pov_obs = [get_rendered_obs(env, obs) for obs in pov_obs]
            pov_obs = np.stack(pov_obs)
            st.plotly_chart(
                px.imshow(pov_obs[:, :, :, :], animation_frame=0),
                use_container_width=True,
            )


# When doing decompositions, want these variables
def decomp_configuration_ui(key=""):
    st.write("Please note that the full decomposition is slow to compute")
    cola, colb = st.columns(2)
    with cola:
        decomp_level = st.selectbox(
            "Decomposition Level",
            ["Full", "MLP"],
            key=key + "decomp_level",
        )
    with colb:
        cluster = st.checkbox("Cluster", value=False, key=key + "cluster")
        normalize = st.checkbox(
            "Normalize", value=False, key=key + "normalize"
        )

    return decomp_level, cluster, normalize


def get_decomp_scan(rtg, cache, logit_dir, decomp_level, normalize=False):
    if decomp_level == "Reduced":
        results, labels = cache.decompose_resid(
            apply_ln=False, return_labels=True
        )
    elif decomp_level == "Full":
        results, labels = cache.get_full_resid_decomposition(
            apply_ln=False,
            return_labels=True,
            expand_neurons=False,  # if you don't set this, you'll crash your browser.
        )
    elif decomp_level == "MLP":
        results, labels = cache.get_full_resid_decomposition(
            apply_ln=False, return_labels=True, expand_neurons=True
        )

    attribution = results[:, :, -1, :] @ logit_dir

    df = pd.DataFrame(attribution.T.detach().cpu().numpy(), columns=labels)
    df.index = rtg[:, -1].squeeze(1).detach().cpu().numpy()

    if normalize:
        # center around zero
        df = df - df.min(axis=0)
        # divide each column by its norm:
        df = df / (df.max(axis=0) - df.min(axis=0))

    return df


def plot_decomp_scan_line(
    df,
    x="RTG",
    labels={"index": "RTG", "value": "Logit Difference"},
    title="Residual Stream Contributions in Directional Analysis",
):
    fig = px.line(
        df,
        labels=labels,
        title=title,
    )

    if df.index.min() < -1:
        fig.add_vline(x=-1, line_dash="dot", line_width=1, line_color="white")

    if df.index.min() < 0:
        fig.add_vline(x=0, line_dash="dot", line_width=1, line_color="white")

    if df.index.max() > 1:
        fig.add_vline(x=1, line_dash="dot", line_width=1, line_color="white")

    # # add a little more margin to the top
    fig.update_layout(margin=dict(t=50))

    return fig


def plot_decomp_scan_corr(df, cluster=False, x="RTG"):
    # drop the column "embed"
    df = df.drop(columns=["embed", "bias", "pos_embed"], errors="ignore")
    df_corr = df.corr()
    # find any NaNs and remove those rows
    fig2 = plot_heatmap(df_corr, cluster=cluster)

    return fig2


def plot_attention_patterns_by_rtg(dt, rtgs=11):
    x = [
        [[] for _ in range(dt.transformer_config.n_heads)]
        for _ in range(dt.transformer_config.n_layers)
    ]
    y = [
        [[] for _ in range(dt.transformer_config.n_heads)]
        for _ in range(dt.transformer_config.n_layers)
    ]
    lines = [
        [[] for _ in range(dt.transformer_config.n_heads)]
        for _ in range(dt.transformer_config.n_layers)
    ]
    rtg_labels = [
        [[] for _ in range(dt.transformer_config.n_heads)]
        for _ in range(dt.transformer_config.n_layers)
    ]

    initial_rtg = st.session_state.rtg
    rtgs = max(2, rtgs)  # RTG should be a minimum of 2, for [0, 1].
    rtg_caches = []

    # [S1, A1, R1, S2, A2, R2, ... Sn, An, Rn]. Rn won't be used.
    step_vals = list(
        np.array(
            [
                [f"S{i+1}", f"A{i+1}", f"R{i+1}"]
                for i in range(1 + dt.transformer_config.n_ctx // 3)
            ]
        ).flatten()
    )

    # Run the model for each RTG value and collect its cache.
    for r in [i / (rtgs - 1) for i in range(rtgs)]:
        _, _, cache, _ = get_action_preds_from_app_state(dt)
        cumulative_reward = st.session_state.reward.cumsum(dim=1)
        rtg = r * torch.ones(
            (1, cumulative_reward.shape[1], 1), dtype=torch.float
        )
        st.session_state.rtg = rtg - cumulative_reward
        cache = torch.stack(
            [
                cache[f"blocks.{str(layer)}.attn.hook_pattern"]
                for layer in range(dt.transformer_config.n_layers)
            ]
        )
        rtg_caches.append(cache.squeeze(1))  # (layers, heads, n_ctx, n_ctx)

    rtg_cache = torch.stack(rtg_caches)  # (rtgs, layers, heads, n_ctx, n_ctx)

    st.session_state.rtg = initial_rtg  # Set RTG back to original value.

    layers = range(dt.transformer_config.n_layers)
    heads = range(dt.transformer_config.n_heads)
    rtg_nums = range(rtg_cache.shape[0])
    rows = range(rtg_cache.shape[-1])

    for layer, head, rtg, row in itertools.product(
        layers, heads, rtg_nums, rows
    ):
        data = (
            rtg_cache[rtg, layer, head, row, :].detach().cpu().numpy()
        )  # Values of a given row.
        x[layer][head].extend(
            [step_vals[i] for i in range(len(data))]
        )  # 0-(num_rows-1)
        y[layer][head].extend(data)
        lines[layer][head].extend(
            [step_vals[row] for _ in range(len(data))]
        )  # Frame is based on row number.
        rtg_labels[layer][head].extend(
            round(rtg / (rtgs - 1), 3) for _ in range(len(data))
        )  # Get RTG vals to 3 digits only.

    return x, y, lines, rtg_labels


# Searching data frames
def search_dataframe(df: pd.DataFrame, query: str) -> pd.DataFrame:
    df_str = df.astype(str).apply(lambda x: " ".join(x), axis=1)
    mask = df_str.str.contains(query, case=False)
    return df[mask]


def create_search_component(df: pd.DataFrame, title: str, key=""):
    a, b, c = st.columns(3)

    with a:
        # Define your search bar
        query = st.text_input(title, key=key + "search")
    with b:
        sort_by = st.selectbox(
            "Sort by",
            options=df.columns,
            index=len(df.columns) - 1,
            key=key + "sort_by",
        )
    with c:
        ascending = st.checkbox("Ascending", key=key + "ascending")
        show_top_10_only = st.checkbox("Show top 10 only", key=key + "top_10")

    if query:
        # Call your function and print output
        search_result = search_dataframe(df, query)
        if not search_result.empty:
            if show_top_10_only:
                st.write(
                    search_result.sort_values(
                        sort_by, ascending=ascending
                    ).head(10)
                )
            else:
                st.write(
                    search_result.sort_values(sort_by, ascending=ascending)
                )
        else:
            st.write("No results found.")
