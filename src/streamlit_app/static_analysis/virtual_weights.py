import streamlit as st
import torch
from fancy_einsum import einsum
import numpy as np
from src.streamlit_app.constants import (
    STATE_EMBEDDING_LABELS,
    IDX_TO_ACTION,
    twenty_idx_format_func,
    three_channel_schema,
)

from src.streamlit_app.utils import tensor_to_long_data_frame
from src.streamlit_app.components import create_search_component
from src.streamlit_app.features import load_features

# from src.streamlit_app
import plotly.express as px

from .gridmaps import ov_gridmap_component


def show_ov_state_action_component(_dt):
    df = get_full_ov_state_action(_dt)

    unstructured_tab, comparison_tab = st.tabs(
        ["Unstructured", "Comparison View"]
    )

    with unstructured_tab:
        # reset index
        df = df.reset_index(drop=True)

        # make a strip plot
        fig = px.scatter(
            df.sort_values(by=["Layer", "Head", "Channel"]),
            y="Score",
            color="Head",
            hover_data=["Head", "Channel", "X", "Y", "Action"],
            labels={"value": "Congruence"},
        )

        # update x axis to hide the tick labels, and remove the label
        fig.update_xaxes(showticklabels=False, title=None)
        st.plotly_chart(fig, use_container_width=True)

    with comparison_tab:
        # for one of the heads selected, and a pair of the actins selected,
        # we want a scatter plot of score vs score
        # use a multiselect for each, but them in three  columns

        b, c, d, a = st.columns(4)

        with b:
            action_1 = st.selectbox(
                "Select Action 1",
                options=df.Action.unique(),
                index=0,
            )
        with c:
            action_2 = st.selectbox(
                "Select Action 2",
                options=df.Action.unique(),
                index=1,
            )
        with d:
            # channel selection
            selected_channels_2 = st.multiselect(
                "Select Channels",
                options=df.Channel.unique(),
                key="channels ov comparison",
                default=["key", "ball"],
            )

        with a:
            use_small_multiples = st.checkbox("Use Small Multiples")

        # filter the dataframe
        filtered_df = df[
            (df["Action"].isin([action_1, action_2]))
            & (df["Channel"].isin(selected_channels_2))
        ]

        # reshape the df so we have the scores of one action in one column and the scores of the other in another
        filtered_df = filtered_df.pivot_table(
            index=["Head", "Embedding"],
            columns="Action",
            values="Score",
        ).reset_index()
        # rename the columns

        filtered_df.columns = [
            "Head",
            "Embedding",
            action_1,
            action_2,
        ]

        if use_small_multiples:
            # make a scatter plot of the two scores
            fig = px.scatter(
                filtered_df,
                x=action_1,
                y=action_2,
                color="Head",
                hover_data=["Head", "Embedding"],
                facet_col="Head",
                facet_col_wrap=4,
                labels={
                    "value": "Congruence",
                },
            )
            # make plot taller
            fig.update_layout(height=800)
        else:
            fig = px.scatter(
                filtered_df,
                x=action_1,
                y=action_2,
                color="Head",
                hover_data=["Head", "Embedding"],
                labels={
                    "value": "Congruence",
                },
            )

        st.plotly_chart(fig, use_container_width=True)

    gridmap_tab, search_tab = st.tabs(["Gridmap", "Search"])

    with search_tab:
        create_search_component(
            df[["Head", "Embedding", "Score"]],
            title="Search Full OV Circuit",
            key="Search Full OV Circuit",
        )

    with gridmap_tab:
        ov_gridmap_component(df, key="ov")


def show_ov_action_action_component(_dt):
    df = get_full_ov_action_action(_dt)

    unstructured_tab, comparison_tab = st.tabs(
        ["Unstructured", "Comparison View"]
    )

    with unstructured_tab:
        # reset index
        df = df.reset_index(drop=True)

        # make a strip plot
        fig = px.scatter(
            df.sort_values(by=["Layer", "Head", "Action_Out"]),
            y="Score",
            color="Head",
            hover_data=["Head", "Action_In", "Action_Out"],
            labels={"value": "Congruence"},
        )

        # update x axis to hide the tick labels, and remove the label
        fig.update_xaxes(showticklabels=False, title=None)
        st.plotly_chart(fig, use_container_width=True)

    with comparison_tab:
        # for one of the heads selected, and a pair of the actins selected,
        # we want a scatter plot of score vs score
        # use a multiselect for each, but them in three  columns

        b, c, d = st.columns(3)

        with b:
            action_1 = st.selectbox(
                "Select Out Action 1",
                options=df.Action_Out.unique(),
                index=0,
                key="action 1 ov comparison",
            )
        with c:
            action_2 = st.selectbox(
                "Select Out Action 2",
                options=df.Action_Out.unique(),
                index=1,
                key="action 2 ov comparison",
            )
        with d:
            use_small_multiples = st.checkbox(
                "Use Small Multiples",
                key="small multiples ov comparison action",
            )

        # filter the dataframe
        filtered_df = df[(df["Action_Out"].isin([action_1, action_2]))]

        # reshape the df so we have the scores of one action in one column and the scores of the other in another
        filtered_df = filtered_df.pivot_table(
            index=["Head", "Action_In"],
            columns="Action_Out",
            values="Score",
        ).reset_index()
        # rename the columns

        filtered_df.columns = [
            "Head",
            "Action_In",
            action_1,
            action_2,
        ]

        if use_small_multiples:
            # make a scatter plot of the two scores
            fig = px.scatter(
                filtered_df,
                x=action_1,
                y=action_2,
                color="Head",
                hover_data=["Head", "Action_In"],
                facet_col="Head",
                facet_col_wrap=4,
                labels={
                    "value": "Congruence",
                },
            )
            # make plot taller
            fig.update_layout(height=800)
        else:
            fig = px.scatter(
                filtered_df,
                x=action_1,
                y=action_2,
                color="Head",
                hover_data=["Head", "Action_In"],
                labels={
                    "value": "Congruence",
                },
            )

        st.plotly_chart(fig, use_container_width=True)


def show_ov_rtg_action_component(_dt):
    df = get_full_ov_rtg_action(_dt)

    fig = px.scatter(
        df,
        y="Score",
        color="Head",
        hover_data=["Head", "Action"],
        labels={
            "value": "Congruence",
        },
    )

    st.plotly_chart(fig, use_container_width=True)


def show_ov_pos_action_component(_dt):
    df = get_full_ov_pos_action(_dt)

    fig = px.scatter(
        df,
        y="Score",
        color="Head",
        hover_data=["Head", "Position", "Action"],
        labels={
            "value": "Congruence",
        },
    )

    st.plotly_chart(fig, use_container_width=True)


def show_ov_feature_action_component(_dt):
    df = get_full_ov_feature_action(_dt)

    unstructured_tab, comparison_tab = st.tabs(
        ["Unstructured", "Comparison View"]
    )

    st.write(df)

    with unstructured_tab:
        fig = px.scatter(
            df,
            y="Score",
            color="Head",
            hover_data=["Head", "Feature", "Action"],
            labels={
                "value": "Congruence",
            },
        )

        st.plotly_chart(fig, use_container_width=True)

    with comparison_tab:
        b, c, d, a = st.columns(4)

        with b:
            action_1 = st.selectbox(
                "Select Action 1",
                options=df.Action.unique(),
                index=0,
                key="action 1 ov comparison, feature",
            )
        with c:
            action_2 = st.selectbox(
                "Select Action 2",
                options=df.Action.unique(),
                index=1,
                key="action 2 ov comparison, feature",
            )
        with d:
            use_small_multiples = st.checkbox(
                "Use Small Multiples",
                key="small multiples ov comparison, feature",
            )
        with a:
            color_by_feature = st.checkbox(
                "Color by Feature", key="color by feature"
            )

        # # filter the dataframe
        # filtered_df = df[
        #     (df["Action"].isin([action_1, action_2]))
        #     & (df["Channel"].isin(selected_channels_2))
        # ]
        filtered_df = df[(df["Action"].isin([action_1, action_2]))]

        # reshape the df so we have the scores of one action in one column and the scores of the other in another
        filtered_df = filtered_df.pivot_table(
            index=["Head", "Feature"],
            columns="Action",
            values="Score",
        ).reset_index()
        # rename the columns

        filtered_df.columns = [
            "Head",
            "Feature",
            action_1,
            action_2,
        ]

        if use_small_multiples:
            # make a scatter plot of the two scores
            fig = px.scatter(
                filtered_df,
                x=action_1,
                y=action_2,
                color="Head" if not color_by_feature else "Feature",
                hover_data=["Head", "Feature"],
                facet_col="Head",
                facet_col_wrap=4,
                labels={
                    "value": "Congruence",
                },
            )
            # make plot taller
            fig.update_layout(height=800)
        else:
            fig = px.scatter(
                filtered_df,
                x=action_1,
                y=action_2,
                color="Head" if not color_by_feature else "Feature",
                hover_data=["Head", "Feature"],
                labels={
                    "value": "Congruence",
                },
            )

        st.plotly_chart(fig, use_container_width=True)


def get_ov_circuit(_dt):
    # stack the heads
    W_V = torch.stack([block.attn.W_V for block in _dt.transformer.blocks])
    W_0 = torch.stack([block.attn.W_O for block in _dt.transformer.blocks])

    # centre
    # W_0 = W_0 - W_0.mean(2, keepdim=True)
    # W_V = W_V - W_V.mean(2, keepdim=True)

    # inner OV circuits.
    W_OV = torch.einsum("lhmd,lhdn->lhmn", W_V, W_0)

    return W_OV


def get_qk_circuit(_dt):
    # stack the heads
    W_Q = torch.stack([block.attn.W_Q for block in _dt.transformer.blocks])
    W_K = torch.stack([block.attn.W_K for block in _dt.transformer.blocks])

    # # centre
    # W_K = W_K - W_K.mean(2, keepdim=True)
    # W_Q = W_Q - W_Q.mean(2, keepdim=True)

    # st.write(W_Q.shape)
    # st.write(W_Q.mean(2).shape)

    # inner QK circuits.
    W_QK = einsum(
        "layer head d_model1 d_head, layer head d_model2 d_head -> layer head d_model1 d_model2",
        W_Q,
        W_K,
    )

    return W_QK


def get_full_ov_state_action(_dt):
    W_OV = get_ov_circuit(_dt)
    W_U = _dt.action_predictor.weight
    W_E = _dt.state_embedding.weight

    height, width, channels = _dt.environment_config.observation_space[
        "image"
    ].shape
    n_layers = _dt.transformer_config.n_layers
    n_actions = _dt.environment_config.action_space.n
    n_heads = _dt.transformer_config.n_heads

    format_func = (
        lambda x: three_channel_schema(x)
        if channels == 3
        else twenty_idx_format_func(x)
    )

    # st.plotly_chart(px.imshow(W_OV.detach().numpy(), facet_col=0), use_container_width=True)
    OV_circuit_full = einsum(
        "d_res_in emb, layer head d_res_in d_res_out, action d_res_out -> layer head emb action",
        W_E,
        W_OV,
        W_U,
    )

    OV_circuit_full = OV_circuit_full.reshape(
        n_layers, n_heads, channels, height, width, n_actions
    )

    df = tensor_to_long_data_frame(
        OV_circuit_full,
        dimension_names=["Layer", "Head", "Channel", "X", "Y", "Action"],
    )
    df = df.sort_values(by=["Layer", "Head", "Action"])
    df["Channel"] = df["Channel"].map(format_func)
    df["Layer"] = df["Layer"].map(lambda x: f"L{x}")
    df["Head"] = df["Layer"] + df["Head"].map(lambda x: f"H{x}")
    df["Action"] = df["Action"].map(IDX_TO_ACTION)
    df["Embedding"] = (
        df["Channel"]
        + ", ("
        + df["X"].astype(str)
        + ", "
        + df["Y"].astype(str)
        + ")"
    )

    return df


def get_full_ov_action_action(_dt):
    W_OV = get_ov_circuit(_dt)
    W_U = _dt.action_predictor.weight
    W_E = _dt.action_embedding[0].weight[:-1, :]

    W_OV_full = einsum(
        "action_out d_model_a, layer head d_model_a d_model_b, action_in d_model_b -> layer head action_out action_in",
        W_U,
        W_OV,
        W_E,
    )

    W_OV_full_df = tensor_to_long_data_frame(
        W_OV_full,
        dimension_names=[
            "Layer",
            "Head",
            "Action_Out",
            "Action_In",
        ],
    )

    W_OV_full_df["Layer"] = W_OV_full_df["Layer"].map(lambda x: f"L{x}")
    W_OV_full_df["Head"] = W_OV_full_df["Layer"] + W_OV_full_df["Head"].map(
        lambda x: f"H{x}"
    )
    W_OV_full_df["Action_Out"] = W_OV_full_df["Action_Out"].map(
        lambda x: IDX_TO_ACTION[x]
    )
    W_OV_full_df["Action_In"] = W_OV_full_df["Action_In"].map(
        lambda x: IDX_TO_ACTION[x]
    )

    return W_OV_full_df


def get_full_ov_rtg_action(_dt):
    W_OV = get_ov_circuit(_dt)
    W_U = _dt.action_predictor.weight
    W_E = _dt.reward_embedding[0].weight

    W_OV_full = einsum(
        "d_model_a position, layer head d_model_a d_model_b, action d_model_b -> layer head action",
        W_E,
        W_OV,
        W_U,
    )

    W_OV_full_df = tensor_to_long_data_frame(
        W_OV_full,
        dimension_names=[
            "Layer",
            "Head",
            "Action",
        ],
    )

    W_OV_full_df["Layer"] = W_OV_full_df["Layer"].map(lambda x: f"L{x}")
    W_OV_full_df["Head"] = W_OV_full_df["Layer"] + W_OV_full_df["Head"].map(
        lambda x: f"H{x}"
    )
    W_OV_full_df["Action"] = W_OV_full_df["Action"].map(
        lambda x: IDX_TO_ACTION[x]
    )

    return W_OV_full_df


def get_full_ov_pos_action(_dt):
    W_OV = get_ov_circuit(_dt)
    W_U = _dt.action_predictor.weight
    W_E = _dt.transformer.W_pos

    W_OV_full = einsum(
        "position d_model_a, layer head d_model_a d_model_b, action d_model_b -> layer head position action",
        W_E,
        W_OV,
        W_U,
    )

    W_OV_full_df = tensor_to_long_data_frame(
        W_OV_full,
        dimension_names=[
            "Layer",
            "Head",
            "Position",
            "Action",
        ],
    )

    W_OV_full_df["Layer"] = W_OV_full_df["Layer"].map(lambda x: f"L{x}")
    W_OV_full_df["Head"] = W_OV_full_df["Layer"] + W_OV_full_df["Head"].map(
        lambda x: f"H{x}"
    )
    W_OV_full_df["Action"] = W_OV_full_df["Action"].map(
        lambda x: IDX_TO_ACTION[x]
    )

    position_names = list(
        np.array(
            [
                [f"R{i+1}", f"S{i+1}", f"A{i+1}"]
                for i in range(1 + _dt.transformer_config.n_ctx // 3)
            ]
        ).flatten()
    )[:-1]

    W_OV_full_df["Position"] = W_OV_full_df["Position"].map(
        lambda x: position_names[x]
    )

    return W_OV_full_df


def get_full_qk_state_state(_dt, q_filter=None, k_filter=None):
    W_QK = get_qk_circuit(_dt)
    W_E_state = _dt.state_embedding.weight

    W_E_state_q = W_E_state[:, q_filter] if q_filter is not None else W_E_state
    W_E_state_k = W_E_state[:, k_filter] if k_filter is not None else W_E_state

    W_QK_full = einsum(
        "d_model_Q n_emb_Q, layer head d_model_Q d_model_K, d_model_K n_emb_K -> layer head n_emb_Q n_emb_K",
        W_E_state_q,
        W_QK,
        W_E_state_k,
    )

    W_QK_full_df = tensor_to_long_data_frame(
        W_QK_full,
        dimension_names=[
            "Layer",
            "Head",
            "Embedding-Q",
            "Embedding-K",
        ],
    )

    filtered_embedding_labels_k = (
        [STATE_EMBEDDING_LABELS[i] for i in k_filter]
        if k_filter is not None
        else STATE_EMBEDDING_LABELS
    )
    filtered_embedding_labels_q = (
        [STATE_EMBEDDING_LABELS[i] for i in q_filter]
        if q_filter is not None
        else STATE_EMBEDDING_LABELS
    )

    W_QK_full_df["Layer"] = W_QK_full_df["Layer"].map(lambda x: f"L{x}")
    W_QK_full_df["Head"] = W_QK_full_df["Layer"] + W_QK_full_df["Head"].map(
        lambda x: f"H{x}"
    )
    W_QK_full_df["Embedding-Q"] = W_QK_full_df["Embedding-Q"].map(
        lambda x: filtered_embedding_labels_q[x]
    )
    W_QK_full_df["Embedding-K"] = W_QK_full_df["Embedding-K"].map(
        lambda x: filtered_embedding_labels_k[x]
    )
    W_QK_full_df["Channel_Q"] = W_QK_full_df["Embedding-Q"].map(
        lambda x: x.split(",")[0]
    )
    W_QK_full_df["X_Q"] = W_QK_full_df["Embedding-Q"].map(
        lambda x: x.split(",")[1].split("(")[1]
    )
    W_QK_full_df["Y_Q"] = W_QK_full_df["Embedding-Q"].map(
        lambda x: x.split(",")[2].split(")")[0]
    )

    W_QK_full_df["Channel_K"] = W_QK_full_df["Embedding-K"].map(
        lambda x: x.split(",")[0]
    )
    W_QK_full_df["X_K"] = W_QK_full_df["Embedding-K"].map(
        lambda x: x.split(",")[1].split("(")[1]
    )
    W_QK_full_df["Y_K"] = W_QK_full_df["Embedding-K"].map(
        lambda x: x.split(",")[2].split(")")[0]
    )

    return W_QK_full_df


def get_full_ov_feature_action(_dt):
    features, feature_metadata = load_features()

    W_OV = get_ov_circuit(_dt)
    W_U = _dt.action_predictor.weight
    W_E = features

    OV_circuit_full = einsum(
        "emb d_res_in, layer head d_res_in d_res_out, action d_res_out -> layer head emb action",
        W_E,
        W_OV,
        W_U,
    )

    df = tensor_to_long_data_frame(
        OV_circuit_full,
        dimension_names=["Layer", "Head", "Feature", "Action"],
    )
    df = df.sort_values(by=["Layer", "Head", "Action"])
    df["Layer"] = df["Layer"].map(lambda x: f"L{x}")
    df["Head"] = df["Layer"] + df["Head"].map(lambda x: f"H{x}")
    df["Action"] = df["Action"].map(IDX_TO_ACTION)
    df["Feature"] = df["Feature"].map(
        lambda x: feature_metadata.feature_names[x]
    )
    return df
