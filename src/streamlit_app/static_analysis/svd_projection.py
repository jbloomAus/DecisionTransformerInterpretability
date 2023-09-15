import pandas as pd
import plotly.express as px
import streamlit as st
import torch
from fancy_einsum import einsum

from src.streamlit_app.components import create_search_component

from src.streamlit_app.constants import (
    ACTION_NAMES,
    IDX_TO_ACTION,
    IDX_TO_STATE,
    POSITION_NAMES,
    SPARSE_CHANNEL_NAMES,
    STATE_EMBEDDING_LABELS,
    get_all_neuron_labels,
    three_channel_schema,
    twenty_idx_format_func,
)

from src.streamlit_app.utils import tensor_to_long_data_frame

from .gridmaps import svd_projection_gridmap_component
from .ui import layer_head_k_selector_ui


def embedding_projection_onto_svd_component(
    _dt, _reading_svd_projection, key="embeddings"
):
    U = _reading_svd_projection
    W_E_state = _dt.state_embedding.weight.detach().T
    W_E_action = _dt.action_embedding[0].weight.detach().T
    W_E_reward = _dt.reward_embedding[0].weight.detach().T

    state_tab, rtg_tab = st.tabs(["State", "RTG"])  # , "Action"]

    with state_tab:
        left_svd_vectors = st.slider(
            "Number of Singular Directions",
            min_value=3,
            max_value=_dt.transformer_config.d_head,
            key=f"state out, head svd, {key}",
        )

        U_filtered = U[:, :, :left_svd_vectors, :]

        activations = einsum(
            "n_emb d_model, layer head d_head_in d_model -> layer head n_emb d_head_in",
            W_E_state,
            U_filtered,
        )

        activations = tensor_to_long_data_frame(
            activations,
            [
                "Layer",
                "Head",
                "Embedding",
                "Direction",
            ],
        )
        activations["Head"] = activations["Layer"].map(
            lambda x: f"L{x}"
        ) + activations["Head"].map(lambda x: f"H{x}")
        activations["Embedding"] = activations["Embedding"].map(
            lambda x: STATE_EMBEDDING_LABELS[x]
        )
        activations["Direction"] = activations["Head"] + activations[
            "Direction"
        ].map(lambda x: f"D{x}")
        fig = px.scatter(
            activations.sort_values(by="Direction"),
            x=activations.index,
            color="Head",
            y="Score",
            hover_data=["Embedding", "Direction"],
            labels={"Score": "Congruence"},
            render_mode="webgl",
        )

        st.plotly_chart(fig, use_container_width=True)

        search_tab, visualization_tab = st.tabs(["search", "gridmap"])

        with search_tab:
            # add search box
            create_search_component(
                activations[["Direction", "Embedding", "Score"]],
                "Search Bar (eg: L0H1)",
                key=f"search bar  state in, svd {key}",
            )

        with visualization_tab:
            svd_projection_gridmap_component(activations, key="state in" + key)

    with rtg_tab:
        W_E = _dt.reward_embedding[0].weight.T
        # W_pos_e = dt.transformer.W_pos
        # W_E = W_E.T + W_pos_e

        left_svd_vectors = st.slider(
            "Number of Singular Directions",
            min_value=3,
            max_value=_dt.transformer_config.d_head,
            key=f"embed rtg, left svd {key}",
        )

        U_filtered = U[:, :, :left_svd_vectors, :]
        activations = einsum(
            "n_emb d_model, layer head d_head_in d_model -> layer head n_emb d_head_in",
            W_E,
            U_filtered,
        )

        activations = tensor_to_long_data_frame(
            activations,
            [
                "Layer",
                "Head",
                "Embedding",
                "Direction",
            ],
        )
        activations["Head"] = activations["Layer"].map(
            lambda x: f"L{x}"
        ) + activations["Head"].map(lambda x: f"H{x}")
        activations["Embedding"] = activations["Embedding"].map(
            lambda x: STATE_EMBEDDING_LABELS[x]
        )

        activations["Direction"] = activations["Head"] + activations[
            "Direction"
        ].map(lambda x: f"D{x}")

        fig = px.scatter(
            activations,
            x=activations.index,
            color="Head",
            y="Score",
            hover_data=["Direction"],
            labels={"Score": "Congruence"},
            render_mode="webgl",
        )

        st.plotly_chart(fig, use_container_width=True)


def svd_out_to_svd_in_component(
    _dt,
    writing_svd_projection,
    _reading_svd_projection,
    key="composition type",
):
    U = _reading_svd_projection
    V = writing_svd_projection

    a, b = st.columns(2)
    with a:
        left_svd_vectors = st.slider(
            "Number of Singular Directions (out)",
            min_value=3,
            max_value=_dt.transformer_config.d_head,
            key="head head left" + key,
        )
    with b:
        right_svd_vectors = st.slider(
            "Number of Singular Directions (in)",
            min_value=3,
            max_value=_dt.transformer_config.d_head,
            key="head head right" + key,
        )

    U_filtered = U[:, :, :left_svd_vectors, :]
    V_filtered = V[:, :, :right_svd_vectors, :]

    activations = einsum(
        "l1 h1 d_head_out d_res, l2 h2 d_head_in d_res  -> l2 l1 h1 h2 d_head_out d_head_in",
        V_filtered,
        U_filtered,
    )

    activations = tensor_to_long_data_frame(
        activations,
        [
            "head_layer_in",
            "head_layer_out",
            "head_in",
            "head_out",
            "Direction In",
            "Direction Out",
        ],
    )

    # remove rows where head in layer is <= head out layer
    activations = activations[
        activations["head_layer_in"] > activations["head_layer_out"]
    ].reset_index(drop=True)

    activations["Head Out"] = activations["head_layer_out"].map(
        lambda x: f"L{x}"
    ) + activations["head_out"].map(lambda x: f"H{x}")

    activations["Head In"] = activations["head_layer_in"].map(
        lambda x: f"L{x}"
    ) + activations["head_in"].map(lambda x: f"H{x}")

    activations["Direction Out"] = activations["Head Out"] + activations[
        "Direction Out"
    ].map(lambda x: f"D{x}")

    activations["Direction In"] = activations["Head In"] + activations[
        "Direction In"
    ].map(lambda x: f"D{x}")

    activations = activations[
        ["Head Out", "Direction Out", "Head In", "Direction In", "Score"]
    ]

    group_by = st.selectbox(
        "Group By", options=["Head Out", "Head In"], key="head in out" + key
    )
    if group_by == "Head Out":
        # reorder rows
        activations = activations.sort_values(
            by=["Head Out", "Head In", "Direction Out", "Direction In"]
        ).reset_index(drop=True)

    fig = px.scatter(
        activations,
        x=activations.index,
        color=group_by,
        y="Score",
        hover_data=["Head Out", "Direction Out", "Head In", "Direction In"],
        labels={"Score": "Congruence"},
        render_mode="webgl",
    )

    st.plotly_chart(fig, use_container_width=True)

    # add search box
    create_search_component(
        activations,
        "Head Out Singular Value Projections onto Head In",
        key="svd, head head" + key,
    )


def svd_out_to_mlp_in_component(_dt, V_OV):
    right_svd_vectors = st.slider(
        "Number of Singular Directions",
        min_value=3,
        max_value=_dt.transformer_config.d_head,
        key="svd out to mlp in",
    )

    MLP_in = torch.stack([block.mlp.W_in for block in _dt.transformer.blocks])
    V_filtered = V_OV[:, :, :right_svd_vectors, :]
    activations = einsum(
        "l1 h1 d_head_out d_head_ext, l2 d_head_ext d_mlp_in -> l2 l1 h1 d_head_out d_mlp_in",
        V_filtered,
        MLP_in,
    )

    activations = tensor_to_long_data_frame(
        activations,
        [
            "mlp_layer",
            "head_layer",
            "Head",
            "Direction",
            "Neuron",
        ],
    )

    # remove rows where mlp_layer < head_layer
    activations = activations[
        activations["mlp_layer"] >= activations["head_layer"]
    ]

    activations["Head"] = activations["head_layer"].map(
        lambda x: f"L{x}"
    ) + activations["Head"].map(lambda x: f"H{x}")

    activations["Neuron"] = activations["mlp_layer"].map(
        lambda x: f"L{x}"
    ) + activations["Neuron"].map(lambda x: f"N{x}")
    activations["Direction"] = activations["Head"] + activations[
        "Direction"
    ].map(lambda x: f"D{x}")

    # drop head head_layer, mlp_layer, head_out_dim
    activations.drop(["head_layer", "mlp_layer"], axis=1, inplace=True)

    activations = activations.sort_values(
        ["Head", "Direction", "Neuron", "Score"]
    )

    activations.reset_index(drop=True, inplace=True)
    fig = px.scatter(
        activations,
        x=activations.index,
        color="Head",
        y="Score",
        hover_data=["Head", "Neuron", "Direction"],
        labels={"Score": "Congruence"},
        render_mode="webgl",
    )

    st.plotly_chart(fig, use_container_width=True)

    # add search box
    create_search_component(
        activations[["Neuron", "Direction", "Score"]],
        "Head Out Singular Value Projections onto MLP",
    )


def mlp_out_to_svd_in_component(
    _dt, _reading_svd_projection, key="mlp + composition type"
):
    U = _reading_svd_projection
    left_svd_vectors = st.slider(
        "Number of Singular Directions (in)",
        min_value=3,
        max_value=_dt.transformer_config.d_head,
        key="svd out to mlp in" + key,
    )
    U_filtered = U[:, :, :left_svd_vectors, :]

    MLP_out = torch.stack(
        [block.mlp.W_out for block in _dt.transformer.blocks]
    )

    # MLP_out = MLP_out / MLP_out.norm(dim=(-2,-1), keepdim=True)

    activations = einsum(
        "mlp_layer d_model Neuron, head_layer Head d_head_in d_res -> mlp_layer head_layer Neuron Head d_head_in",
        MLP_out,
        U_filtered,
    )

    activations = tensor_to_long_data_frame(
        activations,
        [
            "mlp_layer",
            "head_layer",
            "Neuron",
            "Head",
            "Direction",
        ],
    )

    # since this is neuron out to head in, we want to remove any rows where
    # mlp_layer < head_layer (neurons can only write to heads in later layers)
    activations = activations[
        activations["mlp_layer"] < activations["head_layer"]
    ]

    activations["Head"] = activations["head_layer"].map(
        lambda x: f"L{x}"
    ) + activations["Head"].map(lambda x: f"H{x}")

    activations["Neuron"] = activations["mlp_layer"].map(
        lambda x: f"L{x}"
    ) + activations["Neuron"].map(lambda x: f"N{x}")
    activations["Direction"] = activations["Head"] + activations[
        "Direction"
    ].map(lambda x: f"D{x}")

    # drop head head_layer, mlp_layer, head_out_dim
    activations.drop(["head_layer", "mlp_layer"], axis=1, inplace=True)

    activations = activations.sort_values(
        ["Head", "Direction", "Neuron", "Score"]
    )

    activations.reset_index(drop=True, inplace=True)
    fig = px.scatter(
        activations,
        x=activations.index,
        color="Head",
        y="Score",
        hover_data=["Head", "Neuron", "Direction"],
        labels={"Score": "Congruence"},
        render_mode="webgl",
    )

    st.plotly_chart(fig, use_container_width=True)

    # add search box
    create_search_component(
        activations[["Neuron", "Direction", "Score"]],
        "Head Out Singular Value Projections onto MLP",
        key="search: mlp out to svd in " + key,
    )


def svd_out_to_unembedding_component_top_k_variation(_dt, V_OV, W_U):
    """
    This version of this analysis is based on "The SVD Decomposition is Highly Interpretable"
    Conjecture LessWrong post.

    It doesn't work as nicely when your output dimension is much smaller or not super orthogonal.
    A better alternative here is to use a strip plot, or a scatter plot to
    show all the scores.

    """
    # Unembedding Values
    # shape d_action, d_mod
    activations = einsum("l h d1 d2, a d1 -> l h d2 a", V_OV, W_U)

    # torch.Size([3, 8, 7, 256])
    # Now we want to select a head/layer and plot the imshow of the activations
    # only for the first n activations
    layer, head, k, dims = layer_head_k_selector_ui(_dt, key="ov")

    head_v_projections = activations[layer, head, :dims, :].detach()

    st.write(head_v_projections.shape)
    # get top k activations per column
    topk_values, topk_indices = torch.topk(head_v_projections, k, dim=1)

    # put indices into a heat map then replace with the IDX_TO_ACTION string
    df = pd.DataFrame(topk_indices.T.detach().numpy())
    fig = px.imshow(
        topk_values.T,
        color_continuous_midpoint=0,
        color_continuous_scale="RdBu",
        labels={
            "y": "Output Token Rank",
            "x": "Singular vector",
        },
    )

    fig = fig.update_traces(
        text=df.applymap(lambda x: IDX_TO_ACTION[x]).values,
        texttemplate="%{text}",
        hovertemplate=None,
    )
    st.plotly_chart(fig, use_container_width=True)


def svd_out_to_unembedding_component(_dt, V_OV, W_U):
    right_svd_vectors = st.slider(
        "Number of Singular Directions",
        min_value=3,
        max_value=_dt.transformer_config.d_head,
        key="svd out to unembedding",
    )

    V_OV = V_OV[:, :, :right_svd_vectors, :]

    activations = einsum(
        "l1 h1 d_head_out d_head_ext, a d_head_ext -> l1 h1 d_head_out a",
        V_OV,
        W_U,
    )

    # convert to long df
    activations = tensor_to_long_data_frame(
        activations,
        [
            "Layer",
            "Head",
            "Direction",
            "Action",
        ],
    )
    activations["Head"] = activations["Layer"].map(
        lambda x: f"L{x}"
    ) + activations["Head"].map(lambda x: f"H{x}")

    activations["Direction"] = activations["Head"] + activations[
        "Direction"
    ].map(lambda x: f"D{x}")

    activations["Action"] = activations["Action"].map(
        lambda x: IDX_TO_ACTION[x]
    )

    # remove layer column
    activations.drop(["Layer"], axis=1, inplace=True)

    if st.checkbox("Project into Action space"):
        # pivot the table
        activations = activations.pivot_table(
            index=["Head", "Direction"],
            columns=["Action"],
            values=["Score"],
        )
        activations.columns = activations.columns.droplevel(0)
        activations.reset_index(inplace=True)
        a, b = st.columns(2)
        with a:
            action_1 = st.selectbox(
                "Select Action 1",
                options=IDX_TO_ACTION.values(),
                index=1,
            )
        with b:
            action_2 = st.selectbox(
                "Select Action 2",
                options=IDX_TO_ACTION.values(),
                index=0,
            )

        fig = px.scatter(
            activations,
            x=action_1,
            y=action_2,
            color="Head",
            hover_data=["Head", "Direction", action_1, action_2],
            labels={"Score": "Congruence"},
        )
        # centre plot on 0,0
        y_abs_max = (
            max(
                abs(activations[action_1].min()),
                abs(activations[action_1].max()),
            )
            + 0.1
        )
        x_abs_max = (
            max(
                abs(activations[action_2].min()),
                abs(activations[action_2].max()),
            )
            + 0.1
        )
        fig.update_xaxes(range=[-x_abs_max, x_abs_max])
        fig.update_yaxes(range=[-y_abs_max, y_abs_max])

        st.plotly_chart(fig, use_container_width=True)

        # add search box
        create_search_component(
            activations,
            title="Head Out Singular Value Projections onto Unembedding",
            key="Head Out Singular Value Projections onto Unembedding",
        )
    else:
        fig = px.scatter(
            activations,
            x=activations.index,
            y="Score",
            color="Head",
            # facet_col="layer",
            # opacity=0.5,
            hover_data=["Head", "Direction", "Action"],
            labels={"value": "Congruence"},
        )

        # update x axis to hide the tick labels, and remove the label
        fig.update_xaxes(showticklabels=False, title=None)

        st.plotly_chart(fig, use_container_width=True)

        # add search box
        create_search_component(
            activations[["Head", "Direction", "Action", "Score"]],
            title="Head Out Singular Value Projections onto Unembedding",
            key="Head Out Singular Value Projections onto Unembedding",
        )

    # quick experiment, let's see what happens if we plot two actions against each other.

    # # make the df wide
    # activations_wide = activations.pivot_table(
    #     index = ["Head", "Direction"],
    #     columns = ["Action"],
    #     values = ["Score"]
    # )
    # # remove the multi index
    # activations_wide.columns = activations_wide.columns.droplevel(0)
    # activations_wide.reset_index(inplace=True)
    # st.write(activations_wide.head())

    # fig = px.scatter(
    #     activations_wide,
    #     x="left",
    #     y="right",
    #     color="Head",
    #     # facet_col="layer",
    #     # opacity=0.5,
    #     hover_data=["Head", "Direction"],
    #     labels={"value": "Congruence"},
    # )

    # st.plotly_chart(fig, use_container_width=True)
