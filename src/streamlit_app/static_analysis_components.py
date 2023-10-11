import itertools
import json
import os

import einops
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import torch
from fancy_einsum import einsum
from sklearn.preprocessing import StandardScaler

# create a pyvis network
from pyvis.network import Network

from src.streamlit_app.static_analysis.gridmaps import (
    neuron_projection_gridmap_component,
    qk_gridmap_component,
    pc_df_component,
)

from src.streamlit_app.model_index import model_index

from src.streamlit_app.static_analysis.pca import (
    get_pca,
    get_2d_scatter_plot,
    get_3d_scatter_plot,
    get_scree_plot,
    get_loadings_plot,
)

from src.streamlit_app.static_analysis.svd_projection import (
    embedding_projection_onto_svd_component,
    mlp_out_to_svd_in_component,
    svd_out_to_mlp_in_component,
    svd_out_to_svd_in_component,
    svd_out_to_unembedding_component,
)
from src.streamlit_app.static_analysis.ui import layer_head_channel_selector
from src.streamlit_app.static_analysis.virtual_weights import (
    get_full_qk_state_state,
    get_ov_circuit,
    get_qk_circuit,
    show_ov_action_action_component,
    show_ov_pos_action_component,
    show_ov_rtg_action_component,
    show_ov_state_action_component,
    show_ov_feature_action_component,
)
from src.visualization import (
    get_cosine_sim_df,
    get_param_stats,
    plot_param_stats,
    tensor_cosine_similarity_heatmap,
)

from .components import create_search_component
from .constants import (
    ACTION_NAMES,
    IDX_TO_ACTION,
    POSITION_NAMES,
    SPARSE_CHANNEL_NAMES,
    STATE_EMBEDDING_LABELS,
    get_all_neuron_labels,
    three_channel_schema,
    twenty_idx_format_func,
)
from .utils import get_row_names_from_index_labels, tensor_to_long_data_frame
from .visualizations import plot_heatmap

all_index_labels = [
    SPARSE_CHANNEL_NAMES,
    list(range(7)),
    list(range(7)),
]

indices = list(itertools.product(*all_index_labels))
multi_index = pd.MultiIndex.from_tuples(
    indices,
    names=("x", "y", "z"),  # use labels differently if we have index labels
)

embedding_labels = (
    multi_index.to_series()
    .apply(lambda x: "{0}, ({1},{2})".format(*x))
    .tolist()
)


@st.cache_data(experimental_allow_widgets=True)
def show_param_statistics(_dt):
    with st.expander("Show Parameter Statistics"):
        df = get_param_stats(_dt)
        fig_mean, fig_log_std, fig_norm = plot_param_stats(df)

        st.plotly_chart(fig_mean, use_container_width=True)
        st.plotly_chart(fig_log_std, use_container_width=True)
        st.plotly_chart(fig_norm, use_container_width=True)


# @st.cache_data(experimental_allow_widgets=True)
def show_embeddings(_dt, cache):
    with st.expander("Embeddings"):
        a, b = st.columns(2)

        with a:
            selected_embeddings_idx = st.multiselect(
                "Select Embeddings",
                options=range(len(STATE_EMBEDDING_LABELS)),
                key="embedding",
                format_func=lambda x: STATE_EMBEDDING_LABELS[x],
                default=[263, 312, 258, 307],
            )
            light_mode_friendly = st.checkbox(
                "Make graphs Light Mode friendly"
            )

        with b:
            cluster = st.checkbox("Cluster", value=True)
            centre_embeddings = st.checkbox("Centre Embeddings")

        embeddings = _dt.state_embedding.weight.detach().T

        if not selected_embeddings_idx:
            return

        # filter labels
        selected_embeddings_labels = [
            STATE_EMBEDDING_LABELS[i] for i in selected_embeddings_idx
        ]

        # centre
        if centre_embeddings:
            embeddings = embeddings - embeddings.mean(dim=0)
        # normalise
        embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)

        # filter embeddings
        selected_embeddings = embeddings[selected_embeddings_idx, :]

        df = get_cosine_sim_df(
            selected_embeddings,
            selected_embeddings_labels,
            selected_embeddings_labels,
        )

        pca_df, percent_variance, loadings, fitted_pca = get_pca(
            selected_embeddings, selected_embeddings_labels
        )
        all_embeddings_projection = fitted_pca.transform(embeddings)

        pca_tab, heatmap_tab = st.tabs(["PCA", "Heatmap"])

        with heatmap_tab:
            fig = plot_heatmap(
                df,
                color_continuous_midpoint=0,
                cluster=cluster,
                show_labels=df.shape[0] < 20,
                max_labels=10,
            )
            st.plotly_chart(fig, use_container_width=True)

        with pca_tab:
            (
                scatter_2d_tab,
                scatter_3d_tab,
            ) = st.tabs(["2D-Scatter", "3D-Scatter"])

            with scatter_2d_tab:
                fig = get_2d_scatter_plot(
                    pca_df, percent_variance, light_mode_friendly
                )
                st.plotly_chart(fig, use_container_width=True, theme=None)

            with scatter_3d_tab:
                fig = get_3d_scatter_plot(
                    pca_df, percent_variance, light_mode_friendly
                )
                st.plotly_chart(fig, use_container_width=True)

            (
                scree_plot_tab,
                barchart_tab,
                gridmap_tab,
                save_directions_tab,
            ) = st.tabs(
                ["Scree Plot", "Bar chart", "Gridmap", "Save Directions"]
            )

            with scree_plot_tab:
                fig = get_scree_plot(
                    percent_variance, selected_embeddings_labels
                )
                st.plotly_chart(fig, use_container_width=True)

            with barchart_tab:
                # use fitted pca on all embeddings:

                a, b = st.columns(2)
                with a:
                    pc_selected = st.slider(
                        "Select PC",
                        min_value=0,
                        max_value=all_embeddings_projection.shape[1] - 1,
                    )
                    selected_pc_label = f"PC{pc_selected+1}"

                with b:
                    show_full_distribution = st.checkbox(
                        "Show Full Distribution",
                        value=False,
                    )

                pc_df = pd.DataFrame(
                    data=all_embeddings_projection,
                    columns=[
                        f"PC{i+1}"
                        for i in range(all_embeddings_projection.shape[1])
                    ],
                    index=STATE_EMBEDDING_LABELS,
                )

                if not show_full_distribution:
                    # only include the top 10/bottom 10
                    pc_df_filtered = pc_df.sort_values(
                        by=selected_pc_label, ascending=False
                    )
                    pc_df_filtered = pd.concat(
                        [pc_df_filtered.head(10), pc_df_filtered.tail(10)]
                    ).sort_values(by=selected_pc_label, ascending=False)

                else:
                    pc_df_filtered = pc_df

                # add column to df for if in selected_embeddings_labels
                pc_df_filtered["in_selected"] = pc_df_filtered.index.isin(
                    selected_embeddings_labels
                )
                pc_df_filtered = pc_df_filtered.sort_values(
                    by=selected_pc_label, ascending=False
                )

                fig = px.bar(
                    pc_df_filtered,
                    x=pc_df_filtered.index,
                    y=selected_pc_label,
                    text_auto=True,
                    hover_data=[selected_pc_label, "in_selected"],
                    orientation="v",
                )
                # set color of bar to in_selected without changing order
                # get hex codes for plotly default categorical colors (for dark mode)
                categorical_colour_hexes = px.colors.qualitative.Plotly
                fig.update_traces(
                    marker_color=pc_df_filtered["in_selected"].map(
                        {
                            True: categorical_colour_hexes[0],
                            False: categorical_colour_hexes[1],
                        }
                    )
                )

                st.plotly_chart(fig, use_container_width=True)

            with gridmap_tab:
                pc_df_component(
                    pc_df, all_embeddings_projection, embedding_labels
                )

            with save_directions_tab:
                a, b, c = st.columns(3)
                folder_name = "features"
                with a:
                    save_directions_name = st.text_input("Name of Directions")

                with b:
                    num_selected_embeddings = len(selected_embeddings_idx)
                    num_directions = st.slider(
                        "Number of Directions to Save",
                        1,
                        num_selected_embeddings,
                        num_selected_embeddings,
                    )
                with c:
                    # Will we ever really need more than 16 named directions?
                    named_directions = st.slider(
                        "Number of Directions to Name",
                        1,
                        num_directions,
                        1,
                        key="save_directions_slider",
                    )

                direction_names = []
                for i in range(named_directions):
                    text_input = st.text_input(
                        f"Direction {i+1} Name", placeholder=f"PC{i+1}"
                    )
                    direction_names.append(
                        text_input if text_input else f"PC{i+1}"
                    )

                features = loadings.T[:num_directions, :]

                save_directions_button = st.button("Save Directions")
                if save_directions_button:
                    if not save_directions_name:
                        st.warning(
                            "Please enter a name for the savefile under 'Name of Directions'"
                        )
                    else:
                        pt_filename = os.path.join(
                            folder_name, save_directions_name + ".pt"
                        )
                        json_filename = os.path.join(
                            folder_name, save_directions_name + ".json"
                        )

                        # embeddings = _dt.state_embedding.weight.detach().T
                        # selected_embeddings = embeddings[save_embeddings_idx, :]

                        if not os.path.exists(folder_name):
                            os.makedirs(folder_name)
                        torch.save(features, pt_filename)

                        # save json file
                        json_dict = {
                            "file_name": save_directions_name,
                            "model": model_index[
                                st.session_state.model_selector
                            ],
                            "feature_names": direction_names
                            + [
                                f"PC{i+1}"
                                for i in range(
                                    len(direction_names), num_directions
                                )
                            ],
                            "generated_via": "PCA-Embeddings-Subsets",
                            "embeddings_idx": selected_embeddings_idx,
                            "embeddings_labels": selected_embeddings_labels,
                            "timestamp": "TODO: add timestamp",
                        }

                        with open(json_filename, "w") as f:
                            json.dump(json_dict, f)

                        st.write(
                            "Saved to ",
                            save_directions_name,
                            ".pt, ",
                            save_directions_name,
                            ".json",
                        )

        # with in_action_tab:≠
        #     embedding = _dt.action_embedding[0].weight.detach()[:-1, :]

        #     df = get_cosine_sim_df(embedding)
        #     df.columns = ACTION_NAMES
        #     df.index = ACTION_NAMES

        #     fig = tensor_cosine_similarity_heatmap(
        #         embedding, labels=ACTION_NAMES
        #     )

        #     if st.checkbox("Centre Embeddings", key="centre in action"):
        #         embedding = embedding - embedding.mean(dim=0)

        #     fig = plot_heatmap(
        #         df,
        #         cluster=True,
        #         show_labels=True,
        #     )
        #     st.plotly_chart(fig, use_container_width=True)

        # with out_action_tab:
        #     embedding = _dt.action_predictor.weight.detach()

        #     df = get_cosine_sim_df(embedding)
        #     df.columns = ACTION_NAMES
        #     df.index = ACTION_NAMES

        #     if st.checkbox("Centre Embeddings", key="centre out action"):
        #         embedding = embedding - embedding.mean(dim=0)

        #     fig = plot_heatmap(
        #         df,
        #         cluster=True,
        #         show_labels=True,
        #     )
        #     st.plotly_chart(fig, use_container_width=True)

        # with pca_tab:
        #     state_tab, in_action_tab, out_action_tab = st.tabs(
        #         ["State", "In Action", "Out Action"]
        #     )

        #     with state_tab:
        #         embedding = _dt.state_embedding.weight.detach().T

        #         with st.spinner("Performing PCA..."):
        #             # Normalize the data
        #             normalized_embedding = StandardScaler().fit_transform(
        #                 embedding
        #             )

        #             # Perform PCA
        #             pca = PCA(n_components=2)
        #             pca_results = pca.fit_transform(normalized_embedding)

        #             # st.write(pca_results)
        #             # Create a dataframe for the results
        #             pca_df = pd.DataFrame(
        #                 data=pca_results,
        #                 index=get_row_names_from_index_labels(
        #                     ["State", "x", "y"], all_index_labels
        #                 ),
        #                 columns=["PC1", "PC2"],
        #             )
        #             pca_df.reset_index(inplace=True, names="State")
        #             pca_df["Channel"] = pca_df["State"].apply(
        #                 lambda x: x.split(",")[0]
        #             )

        #         states = set(pca_df["State"].values)
        #         selected_channels = st.multiselect(
        #             "Select Observation Channels",
        #             options=list(states),
        #         )

        #         states_to_filter = [state for state in selected_channels]
        #         if states_to_filter:
        #             pca_df_filtered = pca_df[
        #                 pca_df["State"].isin(states_to_filter)
        #             ]
        #         else:
        #             pca_df_filtered = pca_df

        #         # Create the plot
        #         fig = px.scatter(
        #             pca_df_filtered,
        #             x="PC1",
        #             y="PC2",
        #             title="PCA on Embeddings",
        #             hover_data=["State", "PC1", "PC2"],
        #             color="Channel",
        #         )

        #         st.plotly_chart(fig, use_container_width=True)

        #     with in_action_tab:
        #         embedding = _dt.action_embedding[0].weight.detach()

        #         with st.spinner("Performing PCA..."):
        #             # Normalize the data
        #             normalized_embedding = StandardScaler().fit_transform(
        #                 embedding
        #             )

        #             # Perform PCA
        #             pca = PCA(n_components=2)
        #             pca_results = pca.fit_transform(normalized_embedding)

        #             # st.write(pca_results)
        #             # Create a dataframe for the results
        #             pca_df = pd.DataFrame(
        #                 data=pca_results,
        #                 index=singe_action_index_labels + ["Null"],
        #                 columns=["PC1", "PC2"],
        #             )
        #             pca_df.reset_index(inplace=True, names="Action")

        #         # Create the plot
        #         fig = px.scatter(
        #             pca_df,
        #             x="PC1",
        #             y="PC2",
        #             title="PCA on Embeddings",
        #             hover_data=["Action", "PC1", "PC2"],
        #             color="Action",
        #         )

        #         st.plotly_chart(fig, use_container_width=True)

        #     with out_action_tab:
        #         embedding = _dt.action_predictor.weight.detach()

        #         with st.spinner("Performing PCA..."):
        #             # Normalize the data
        #             normalized_embedding = StandardScaler().fit_transform(
        #                 embedding
        #             )

        #             # Perform PCA
        #             pca = PCA(n_components=2)
        #             pca_results = pca.fit_transform(normalized_embedding)

        #             # st.write(pca_results)
        #             # Create a dataframe for the results
        #             pca_df = pd.DataFrame(
        #                 data=pca_results,
        #                 index=singe_action_index_labels,
        #                 columns=["PC1", "PC2"],
        #             )
        #             pca_df.reset_index(inplace=True, names="Action")

        #         # Create the plot
        #         fig = px.scatter(
        #             pca_df,
        #             x="PC1",
        #             y="PC2",
        #             title="PCA on Embeddings",
        #             hover_data=["Action", "PC1", "PC2"],
        #             color="Action",
        #         )

        #         st.plotly_chart(fig, use_container_width=True)


def show_neuron_directions(_dt):
    with st.expander("Show Neuron In / Out Directions"):
        layers = _dt.transformer_config.n_layers
        all_neuron_labels = get_all_neuron_labels(
            layers, _dt.transformer_config.d_mlp
        )

        MLP_in = (
            torch.concat(
                [block.mlp.W_in for block in _dt.transformer.blocks], dim=1
            )
            .detach()
            .T
        )

        st.write(MLP_in.shape)

        MLP_out = torch.concat(
            [block.mlp.W_out for block in _dt.transformer.blocks],
            dim=0,
        ).detach()

        # not sure what default should be...
        a, b = st.columns(2)
        with a:
            selected_in_neurons_directions = st.multiselect(
                "Select In Neuron Directions",
                options=range(len(all_neuron_labels)),
                key="in neuron directions",
                default=[0],
                format_func=lambda x: all_neuron_labels[x],
            )

        with b:
            selected_out_neurons_directions = st.multiselect(
                "Select Out Neuron Directions",
                options=range(len(all_neuron_labels)),
                key="out neuron directions",
                default=[0],
                format_func=lambda x: all_neuron_labels[x],
            )

        filtered_in_neurons = MLP_in[selected_in_neurons_directions, :].T
        in_labels = [
            all_neuron_labels[i] + "In" for i in selected_in_neurons_directions
        ]

        filtered_out_neurons = MLP_out[selected_out_neurons_directions, :].T
        out_labels = [
            all_neuron_labels[i] + "Out"
            for i in selected_out_neurons_directions
        ]

        all_neurons = torch.concat(
            [filtered_in_neurons, filtered_out_neurons], dim=1
        )

        residual_stream_space_tab, action_space_tab = st.tabs(
            ["Residual Stream Space", "Action Space"]
        )

        with residual_stream_space_tab:
            df = get_cosine_sim_df(all_neurons.T)
            df.columns = in_labels + out_labels
            df.index = in_labels + out_labels

            fig = plot_heatmap(df, cluster=True)
            st.plotly_chart(fig, use_container_width=True)

        with action_space_tab:
            W_U = _dt.action_predictor.weight.detach()
            W_U_normalized = W_U / torch.norm(W_U, dim=1, keepdim=True)
            st.write(W_U_normalized.shape)
            st.write()
            all_neurons = einsum(
                "Action d_model, d_model n_emb -> Action n_emb ",
                W_U_normalized,
                all_neurons.detach(),
            )

            df = get_cosine_sim_df(all_neurons.T)
            df.columns = in_labels + out_labels
            df.index = in_labels + out_labels

            fig = plot_heatmap(df, cluster=True)
            st.plotly_chart(fig, use_container_width=True)

    return


# @st.cache_data(experimental_allow_widgets=True)
def show_qk_circuit(_dt):
    with st.expander("show QK circuit"):
        st.write(
            """
            Usually the QK circuit uses the embedding twice but we have 3 different embeddings so 6 different directed combinations/permutations of 2. 
            """
        )
        st.latex(
            r"""
            QK_{circuit} = W_{E(i)}^T W_Q^T W_K W_{E(j)} \text{ for } i,j \in \{rtg, state\, \text{or} \, action\}
            """
        )

        state_state_tab, state_rtg_tab = st.tabs(
            [
                "QK(state, state)",
                "QK(state, rtg)",
                # "QK(state, action)",
            ]
        )

        # with state_state_tab:
        #     W_QK_full = W_E_state.T @ W_QK @ W_E_state
        #     st.write(W_QK_full.shape)

        #     W_QK_full_reshaped = einops.rearrange(
        #         W_QK_full,
        #         "b (h w c) (h1 w1 c1) -> b h w c h1 w1 c1",
        #         h=7,
        #         w=7,
        #         c=20,
        #         h1=7,
        #         w1=7,
        #         c1=20,
        #     )
        #     st.write(W_QK_full_reshaped.shape)

        height, width, channels = _dt.environment_config.observation_space[
            "image"
        ].shape
        n_heads = _dt.transformer_config.n_heads
        layers = _dt.transformer_config.n_layers

        W_QK = get_qk_circuit(_dt)
        W_E_rtg = _dt.reward_embedding[0].weight
        W_E_state = _dt.state_embedding.weight
        with state_state_tab:
            select_via_k_tab, select_via_q_tab = st.tabs(
                ["Select via K", "Select via Q"]
            )

            with select_via_k_tab:
                selected_key_state_features = st.multiselect(
                    "Select Key Vocabulary Items",
                    options=range(len(STATE_EMBEDDING_LABELS)),
                    format_func=lambda x: STATE_EMBEDDING_LABELS[x],
                    key="key state features",
                    default=[265],
                )

                W_QK_q_filtered_df = get_full_qk_state_state(
                    _dt, k_filter=selected_key_state_features
                )

                # now sum over the selected key state features (So we're seeing how much
                # collective attention is paid to the selected key state features)

                W_QK_q_filtered_df = (
                    W_QK_q_filtered_df.groupby(
                        [
                            "Layer",
                            "Head",
                            "Embedding-Q",
                            "Channel_Q",
                            "X_Q",
                            "Y_Q",
                        ]
                    )
                    .sum()
                    .reset_index()
                )

                # rename Embedding-Q to Embedding
                W_QK_q_filtered_df.rename(
                    columns={
                        "Embedding-Q": "Embedding",
                        "Channel_Q": "Channel",
                        "X_Q": "X",
                        "Y_Q": "Y",
                    },
                    inplace=True,
                )

                # # now make a strip plot
                fig = px.scatter(
                    W_QK_q_filtered_df.sort_values(
                        by=["Layer", "Head", "Channel"]
                    ).reset_index(drop=True),
                    y="Score",
                    color="Head",
                    hover_data=["Head", "Embedding"],
                )

                # # unstructured view
                st.plotly_chart(fig, use_container_width=True)

                # make a gridmap:
                qk_gridmap_component(
                    W_QK_q_filtered_df,
                    facet_col="Channel",
                    key="embeddings Q, QK",
                )

                # # it might be nice to make use of states from the current trajectory later.

            with select_via_q_tab:
                selected_query_state_features = st.multiselect(
                    "Select Query Vocabulary Items",
                    options=range(len(STATE_EMBEDDING_LABELS)),
                    format_func=lambda x: STATE_EMBEDDING_LABELS[x],
                    key="query selection features",
                    default=[258, 335],
                )

                W_QK_k_filtered_df = get_full_qk_state_state(
                    _dt, q_filter=selected_query_state_features
                )

                # now sum over the selected key state features (So we're seeing how much
                # collective attention is paid to the selected key state features)

                W_QK_k_filtered_df = (
                    W_QK_k_filtered_df.groupby(
                        [
                            "Layer",
                            "Head",
                            "Embedding-K",
                            "Channel_K",
                            "X_K",
                            "Y_K",
                        ]
                    )
                    .sum()
                    .reset_index()
                )

                # rename Embedding-Q to Embedding
                W_QK_k_filtered_df.rename(
                    columns={
                        "Embedding-K": "Embedding",
                        "Channel_K": "Channel",
                        "X_K": "X",
                        "Y_K": "Y",
                    },
                    inplace=True,
                )

                # # now make a strip plot
                fig = px.scatter(
                    W_QK_k_filtered_df.sort_values(
                        by=["Layer", "Head", "Channel"]
                    ).reset_index(drop=True),
                    y="Score",
                    color="Head",
                    hover_data=["Head", "Embedding"],
                )

                # # unstructured view
                st.plotly_chart(fig, use_container_width=True)

                # make a gridmap:
                qk_gridmap_component(
                    W_QK_k_filtered_df,
                    facet_col="Channel",
                    key="embeddings K, QK",
                )

                # # it might be nice to make use of states from the current trajectory later.

        with state_rtg_tab:
            # st.write(W_QK.shape)
            W_QK_full = W_E_state.T @ W_QK @ W_E_rtg

            W_QK_full_reshaped = W_QK_full.reshape(
                layers, n_heads, channels, height, width
            )

            if st.checkbox("Show all values"):
                all_scores_df = tensor_to_long_data_frame(
                    W_QK_full_reshaped,
                    dimension_names=["Layer", "Head", "Channel", "X", "Y"],
                )

                # sort by layer, head, action
                all_scores_df = all_scores_df.sort_values(by=["Layer", "Head"])

                # reset index
                all_scores_df = all_scores_df.reset_index(drop=True)
                # channel is categorical
                all_scores_df["Channel"] = all_scores_df["Channel"].astype(
                    "category"
                )
                # order by channe then reset index
                all_scores_df = all_scores_df.sort_values(
                    by=["Channel", "Layer", "Head"]
                )
                all_scores_df.reset_index(inplace=True, drop=True)
                # map indices to channel names
                all_scores_df["Channel"] = all_scores_df["Channel"].map(
                    twenty_idx_format_func
                )
                # make a strip plot
                fig = px.scatter(
                    all_scores_df,
                    x=all_scores_df.index,
                    y="Score",
                    color="Channel",
                    hover_data=["Layer", "Head", "Channel", "X", "Y"],
                    labels={"value": "Congruence"},
                )

                # update x axis to hide the tick labels, and remove the label
                fig.update_xaxes(showticklabels=False, title=None)
                st.plotly_chart(fig, use_container_width=True)

            layer, heads, selected_channels = layer_head_channel_selector(
                _dt, key="srtg"
            )
            abs_max_val = W_QK_full_reshaped.abs().max().item()
            abs_max_val = st.slider(
                "Max Absolute Value Color",
                min_value=abs_max_val / 10,
                max_value=abs_max_val,
                value=abs_max_val,
            )
            columns = st.columns(len(selected_channels))
            for i, channel in enumerate(selected_channels):
                with columns[i]:
                    if channels == 3:
                        st.write(three_channel_schema[channel])
                    elif channels == 20:
                        st.write(twenty_idx_format_func(channel))

            for head in heads:
                st.write("Head", head)
                columns = st.columns(len(selected_channels))
                for i, channel in enumerate(selected_channels):
                    with columns[i]:
                        fig = px.imshow(
                            W_QK_full_reshaped[layer, head, channel]
                            .T.detach()
                            .numpy(),
                            color_continuous_midpoint=0,
                            color_continuous_scale="RdBu",
                            range_color=[-abs_max_val, abs_max_val],
                        )
                        fig.update_layout(
                            coloraxis_showscale=False,
                            margin=dict(l=0, r=0, t=0, b=0),
                        )
                        fig.update_layout(height=180, width=400)
                        st.plotly_chart(
                            fig, use_container_width=True, autosize=True
                        )


# @st.cache_data(experimental_allow_widgets=True)
def show_ov_circuit(_dt):
    with st.expander("Show OV Circuit"):
        st.subheader("OV circuits")

        st.latex(
            r"""
            OV_{circuit} = W_{U(action)} W_O W_V W_{E(i)}  \text{ for } i \in \{rtg, state\, \text{or} \, action\, position\, time\}
            """
        )

        state_tab, action_tab, rtg_tab, pos_tab, feature_tab = st.tabs(
            ["OV(state)", "OV(action)", "OV(rtg)", "OV(pos)", "OV(feature)"]
        )

        with state_tab:
            show_ov_state_action_component(_dt)

        with action_tab:
            show_ov_action_action_component(_dt)

        with rtg_tab:
            show_ov_rtg_action_component(_dt)

        with pos_tab:
            show_ov_pos_action_component(_dt)

        with feature_tab:
            show_ov_feature_action_component(_dt)


@st.cache_data(experimental_allow_widgets=True)
def show_congruence(_dt):
    with st.expander("Show Congruence"):
        W_E_state = _dt.state_embedding.weight
        W_U = _dt.action_predictor.weight
        MLP_in = torch.stack(
            [block.mlp.W_in for block in _dt.transformer.blocks]
        )
        MLP_out = torch.stack(
            [block.mlp.W_out for block in _dt.transformer.blocks]
        )

        a, b = st.columns(2)
        with a:
            selected_writer = st.selectbox(
                "Select Writer",
                options=["Embeddings", "Neurons"],
                key="congruence",
            )
        with b:
            st.write("Select the component writing to the residual stream.")

        if selected_writer == "Embeddings":
            (mlp_in_tab, unembedding_tab) = st.tabs(["MLP_in", "Unembeddings"])

            with mlp_in_tab:
                activations = einsum(
                    "layer d_mlp d_model, d_model n_emb -> layer d_mlp n_emb ",
                    MLP_in,
                    W_E_state,
                )

                df = tensor_to_long_data_frame(
                    activations,
                    ["Layer", "Neuron", "Embedding"],
                )
                df["Layer"] = df["Layer"].map(lambda x: f"L{x}")
                df["Neuron"] = df["Layer"] + df["Neuron"].map(
                    lambda x: f"N{x}"
                )
                df["Embedding"] = df["Embedding"].map(
                    lambda x: embedding_labels[x]
                )
                df["Channel"] = df["Embedding"].map(lambda x: x.split(",")[0])
                df = df.sort_values(by=["Layer", "Channel"])
                df.reset_index(inplace=True, drop=True)

                fig = px.scatter(
                    df,
                    x=df.index,
                    y="Score",
                    color="Channel",
                    hover_data=[
                        "Layer",
                        "Neuron",
                        "Embedding",
                        "Score",
                    ],
                    labels={"Score": "Congruence"},
                )

                # update x axis to hide the tick labels, and remove the label
                fig.update_xaxes(showticklabels=False, title=None)

                st.plotly_chart(fig, use_container_width=True)

                search_tab, visualization_tab = st.tabs(["search", "gridmap"])

                with search_tab:
                    create_search_component(
                        df[["Layer", "Neuron", "Embedding", "Score"]],
                        title="Search MLP to Embeddings",
                        key="mlp to embeddings",
                    )

                with visualization_tab:
                    neuron_projection_gridmap_component(
                        df, key="neuron projection"
                    )

            with unembedding_tab:
                activations = einsum(
                    "Action d_model, d_model n_emb -> Action n_emb ",
                    W_U,
                    W_E_state,
                )

                df = tensor_to_long_data_frame(
                    activations,
                    ["Action", "Embedding"],
                )
                df["Action"] = df["Action"].map(IDX_TO_ACTION)
                df["Embedding"] = df["Embedding"].map(
                    lambda x: embedding_labels[x]
                )
                df["Channel"] = df["Embedding"].map(lambda x: x.split(",")[0])
                df = df.sort_values(by=["Channel", "Action"])
                df.reset_index(inplace=True, drop=True)

                fig = px.scatter(
                    df,
                    x=df.index,
                    y="Score",
                    color="Channel",
                    hover_data=[
                        "Action",
                        "Embedding",
                        "Score",
                    ],
                    labels={"Score": "Congruence"},
                )

                # update x axis to hide the tick labels, and remove the label
                fig.update_xaxes(showticklabels=False, title=None)

                st.plotly_chart(fig, use_container_width=True)

                create_search_component(
                    df,
                    title="Search Unembedding to Embeddings",
                    key="unembedding to embeddings",
                )

        elif selected_writer == "Neurons":
            (
                unembedding_tab,
                mlp_in_tab,
            ) = st.tabs(["MLP to Unembeddings", "MLP to MLP"])

            with unembedding_tab:
                MLP_out_congruence = einsum(
                    "layer d_mlp d_model, d_action d_model -> layer d_mlp d_action",
                    MLP_out,
                    W_U,
                ).detach()

                congruence_df = tensor_to_long_data_frame(
                    MLP_out_congruence, ["Layer", "Neuron", "Action"]
                )

                congruence_df["Layer"] = congruence_df["Layer"].map(
                    lambda x: f"L{x}"
                )
                congruence_df["Neuron"] = congruence_df[
                    "Layer"
                ] + congruence_df["Neuron"].map(lambda x: f"N{x}")

                # sort by Layer and Action
                congruence_df = congruence_df.sort_values(
                    by=["Layer", "Action"]
                ).reset_index(drop=True)
                congruence_df["Action"] = congruence_df["Action"].map(
                    IDX_TO_ACTION
                )

                if st.checkbox("Project into Action space"):
                    # pivot the table
                    congruence_df = congruence_df.pivot_table(
                        index=["Layer", "Neuron"],
                        columns="Action",
                        values="Score",
                    ).reset_index()

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
                        congruence_df,
                        x=action_1,
                        y=action_2,
                        color="Layer",
                        hover_data=["Layer", "Neuron", action_1, action_2],
                        labels={"Score": "Congruence"},
                    )

                    st.plotly_chart(fig, use_container_width=True)

                else:
                    fig = px.scatter(
                        congruence_df,
                        x=congruence_df.index,
                        y="Score",
                        color="Action",
                        hover_data=["Layer", "Action", "Neuron", "Score"],
                        labels={"Score": "Congruence"},
                    )

                    # update x axis to hide the tick labels, and remove the label
                    fig.update_xaxes(showticklabels=False, title=None)

                    st.plotly_chart(fig, use_container_width=True)

                    create_search_component(
                        congruence_df,
                        title="Search MLP to Unembedding",
                        key="mlp to unembedding",
                    )

            with mlp_in_tab:
                mlp_mlp_congruence = einsum(
                    "layer_out d_mlp_out d_model, layer_in d_model d_mlp_in -> layer_in layer_out d_mlp_in d_mlp_out",
                    MLP_out,
                    MLP_in,
                ).detach()

                df = tensor_to_long_data_frame(
                    mlp_mlp_congruence,
                    ["Layer In", "Layer Out", "Neuron In", "Neuron Out"],
                )

                df = df.sort_values(by=["Layer In", "Layer Out"])
                df = df.reset_index(drop=True)

                df["Layer Out"] = df["Layer Out"].map(lambda x: f"L{x}")
                df["Neuron Out"] = df["Layer Out"] + df["Neuron Out"].map(
                    lambda x: f"N{x}Out"
                )
                df["Layer In"] = df["Layer In"].map(lambda x: f"L{x}")
                df["Neuron In"] = df["Layer In"] + df["Neuron In"].map(
                    lambda x: f"N{x}In"
                )

                # remove any rows where the layer out is less than the layer in
                df = df[df["Layer Out"] < df["Layer In"]]
                df = df.reset_index(drop=True)

                fig = px.scatter(
                    df,
                    x=df.index,
                    y="Score",
                    color="Layer In",
                    hover_data=[
                        "Layer Out",
                        "Layer In",
                        "Neuron Out",
                        "Neuron In",
                        "Score",
                    ],
                    labels={"Score": "Congruence"},
                )

                # update x axis to hide the tick labels, and remove the label
                fig.update_xaxes(showticklabels=False, title=None)
                st.plotly_chart(fig, use_container_width=True)

                create_search_component(
                    df,
                    title="Search MLP to MLP",
                    key="mlp to mlp",
                )


# TODO: Add st.cache_data here.
def show_composition_scores(_dt):
    with st.expander("Show Composition Scores"):
        st.markdown(
            "Composition Score calculations per [Mathematical Frameworks for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html#:~:text=The%20above%20diagram%20shows%20Q%2D%2C%20K%2D%2C%20and%20V%2DComposition)"
        )

        q_scores = _dt.transformer.all_composition_scores("Q")
        k_scores = _dt.transformer.all_composition_scores("K")
        v_scores = _dt.transformer.all_composition_scores("V")

        dims = ["L1", "H1", "L2", "H2"]

        q_scores_df = tensor_to_long_data_frame(q_scores, dims)
        q_scores_df["Type"] = "Q"

        k_scores_df = tensor_to_long_data_frame(k_scores, dims)
        k_scores_df["Type"] = "K"

        v_scores_df = tensor_to_long_data_frame(v_scores, dims)
        v_scores_df["Type"] = "V"

        all_scores_df = pd.concat([q_scores_df, k_scores_df, v_scores_df])

        # filter any scores where L2 <= L1
        all_scores_df = all_scores_df[
            all_scores_df["L2"] > all_scores_df["L1"]
        ]

        # concate L1 and H1 to L1H1 and call it "origin"
        all_scores_df["Origin"] = (
            "L"
            + all_scores_df["L1"].astype(str)
            + "H"
            + all_scores_df["H1"].astype(str)
        )

        # concate L2 and H2 to L2H2 and call it "destination"
        all_scores_df["Destination"] = (
            "L"
            + all_scores_df["L2"].astype(str)
            + "H"
            + all_scores_df["H2"].astype(str)
        )

        # sort by type and rewrite the index
        all_scores_df = all_scores_df.sort_values(by="Type")
        all_scores_df.reset_index(inplace=True, drop=True)

        fig = px.scatter(
            all_scores_df,
            x=all_scores_df.index,
            y="Score",
            color="Type",
            hover_data=["Origin", "Destination", "Score", "Type"],
            labels={"value": "Congruence"},
        )

        # update x axis to hide the tick labels, and remove the label
        fig.update_xaxes(showticklabels=False, title=None)
        st.plotly_chart(fig, use_container_width=True)

        topn_tab, search_tab, network_tab = st.tabs(
            ["Top N", "Search", "Network"]
        )

        with topn_tab:
            # let user choose n
            n = st.slider("Top N", value=10, min_value=10, max_value=100)

            # now sort them by Score and return the top 10 scores
            top_n = all_scores_df.sort_values(
                by="Score", ascending=False
            ).head(n)
            # round the score value and show as string
            top_n["Score"] = top_n["Score"].apply(lambda x: f"{x:.2f}")
            # remove L1, L2, H1, H2
            top_n = top_n.drop(columns=["L1", "L2", "H1", "H2"])
            st.write("Top 10 Scores")
            st.write(top_n)

        with search_tab:
            # add a search box so you can find a specific score
            search = st.text_input("Search for a score", value="")
            if search:
                # filter the df by the search term
                search_df = all_scores_df[
                    all_scores_df["Origin"].str.contains(search)
                    | all_scores_df["Destination"].str.contains(search)
                ]
                # sort by score
                search_df = search_df.sort_values(by="Score", ascending=False)
                # round the score value and show as string
                search_df["Score"] = search_df["Score"].apply(
                    lambda x: f"{x:.2f}"
                )
                # remove L1, L2, H1, H2
                search_df = search_df.drop(columns=["L1", "L2", "H1", "H2"])
                st.write("Search Results")
                st.write(search_df)

        with network_tab:
            cutoff = st.slider(
                "Cutoff",
                value=0.15,
                min_value=0.0,
                max_value=max(all_scores_df["Score"]),
            )

            # use network x to create a graph
            # then pass it to pyvis.
            filtered_df = all_scores_df[all_scores_df["Score"] > cutoff]
            # set color based on type, use nice color scheme
            filtered_df["color"] = filtered_df["Type"].apply(
                lambda x: "#FFD700 "
                if x == "Q"
                else "#00FFFF"
                if x == "K"
                else "#8A2BE2"
            )
            filtered_df["weight"] = filtered_df["Score"].apply(
                lambda x: x * 10
            )
            filtered_df["title"] = filtered_df["Score"].apply(
                lambda x: f"{x:.2f}"
            )
            graph = nx.from_pandas_edgelist(
                filtered_df,
                source="Origin",
                target="Destination",
                edge_attr=["Score", "Type", "color"],
            )

            net = Network(notebook=True, bgcolor="#0E1117", font_color="white")
            net.from_nx(graph)

            # write it to disk as a html
            net.write_html("composition_scores.html")

            # show it in streamlit
            HtmlFile = open("composition_scores.html", "r", encoding="utf-8")
            source_code = HtmlFile.read()

            # render using iframe
            components.html(source_code, height=500, width=700)

            # underneath, write the edge list table nicely
            st.write("Edge List")
            # process edge list
            filtered_df = filtered_df.drop(
                columns=["L1", "L2", "H1", "H2", "color"]
            )
            filtered_df["Score"] = filtered_df["Score"].apply(
                lambda x: f"{x:.2f}"
            )

            st.write(filtered_df)

        st.write(
            """
            How much does the query, key or value vector of a second layer head read in information from a given first layer head? 
            """
        )


# @st.cache_data(experimental_allow_widgets=True)
def show_dimensionality_reduction(_dt):
    with st.expander("Dimensionality Reduction"):
        # get head objects.
        W_QK = get_qk_circuit(_dt)
        U_QK, S_QK, V_QK = torch.linalg.svd(W_QK)
        W_OV = get_ov_circuit(_dt)
        U_OV, S_OV, V_OV = torch.linalg.svd(W_OV)
        W_U = _dt.action_predictor.weight

        a, b = st.columns(2)
        with a:
            selected_writer = st.selectbox(
                "Select Writer",
                options=["Embeddings", "Head Output", "Neuron Output"],
                key="svd_virtual_weights",
            )
        with b:
            st.write("Select the component writing to the residual stream.")

        if selected_writer == "Embeddings":
            keys_tab, queries_tab, values_tab = st.tabs(
                ["Keys", "Queries", "Values"]
            )
            with keys_tab:
                embedding_projection_onto_svd_component(_dt, V_QK, key="keys")

            with queries_tab:
                embedding_projection_onto_svd_component(
                    _dt, U_QK, key="queries"
                )

            with values_tab:
                embedding_projection_onto_svd_component(
                    _dt, V_OV, key="values"
                )

        if selected_writer == "Head Output":
            (
                key_composition_tab,
                query_composition_tab,
                value_composition_tab,
                mlp_in_tab,
                unembedding_tab,
            ) = st.tabs(
                [
                    "Keys",
                    "Queries",
                    "Values",
                    "Neuron Activation",
                    "Unembedding",
                ]
            )

            with key_composition_tab:
                V_Q_tmp = V_QK.permute(0, 1, 3, 2)
                svd_out_to_svd_in_component(
                    _dt, U_OV, V_Q_tmp, key="key composition"
                )

            with query_composition_tab:
                svd_out_to_svd_in_component(
                    _dt, U_OV, U_QK, key="query composition"
                )

            with value_composition_tab:
                V_OV_tmp = V_OV.permute(0, 1, 3, 2)
                svd_out_to_svd_in_component(
                    _dt, U_OV, V_OV_tmp, key="value composition"
                )

            with mlp_in_tab:
                svd_out_to_mlp_in_component(_dt, V_OV)

            with unembedding_tab:
                svd_out_to_unembedding_component(_dt, V_OV, W_U)

        if selected_writer == "Neuron Output":
            key_tab, query_tab, value_tab = st.tabs(
                ["Keys", "Queries", "Values"]
            )

            with key_tab:
                V_QK_tmp = V_QK.permute(0, 1, 3, 2)
                mlp_out_to_svd_in_component(_dt, V_QK_tmp, key="key")

            with query_tab:
                mlp_out_to_svd_in_component(_dt, U_QK, key="query")

            with value_tab:
                V_OV_tmp = V_OV.permute(0, 1, 3, 2)
                mlp_out_to_svd_in_component(_dt, V_OV_tmp, key="value")


@st.cache_data(experimental_allow_widgets=True)
def plot_svd_by_head_layer(_dt, S):
    d_head = _dt.transformer_config.d_head
    labels = [
        f"L{i}H{j}"
        for i in range(0, _dt.transformer_config.n_layers)
        for j in range(_dt.transformer_config.n_heads)
    ]
    S = einops.rearrange(S, "l h s -> (l h) s")

    df = pd.DataFrame(S.T.detach().numpy(), columns=labels)
    fig = px.line(
        df,
        range_x=[0, d_head],
        labels={"index": "Singular Value", "value": "Value"},
        title="Singular Value by OV Circuit",
    )
    # add a vertical white dotted line at x = d_head
    fig.add_vline(x=d_head, line_dash="dash", line_color="white")
    st.plotly_chart(fig, use_container_width=True)


# def show_time_embeddings(dt, logit_dir):
#     with st.expander("Show Time Embeddings"):
#         if dt.time_embedding_type == "linear":
#             time_steps = t.arange(100).unsqueeze(0).unsqueeze(-1).to(t.float32)
#             time_embeddings = dt.get_time_embeddings(time_steps).squeeze(0)
#         else:
#             time_embeddings = dt.time_embedding.weight

#         max_timestep = st.slider(
#             "Max timestep",
#             min_value=1,
#             max_value=time_embeddings.shape[0] - 1,
#             value=time_embeddings.shape[0] - 1,
#         )
#         time_embeddings = time_embeddings[: max_timestep + 1]
#         dot_prod = time_embeddings @ logit_dir
#         dot_prod = dot_prod.detach()

#         show_initial = st.checkbox("Show initial time embedding", value=True)
#         fig = px.line(dot_prod)
#         fig.update_layout(
#             title="Time Embedding Dot Product",
#             xaxis_title="Time Step",
#             yaxis_title="Dot Product",
#             legend_title="",
#         )
#         # remove legend
#         fig.update_layout(showlegend=False)
#         if show_initial:
#             fig.add_vline(
#                 x=st.session_state.timesteps[0][-1].item(),
#                 line_dash="dash",
#                 line_color="red",
#                 annotation_text="Current timestep",
#             )
#         st.plotly_chart(fig, use_container_width=True)

#         def calc_cosine_similarity_matrix(matrix: t.Tensor) -> t.Tensor:
#             # Check if the input matrix is square
#             # assert matrix.shape[0] == matrix.shape[1], "The input matrix must be square."

#             # Normalize the column vectors
#             norms = t.norm(
#                 matrix, dim=0
#             )  # Compute the norms of the column vectors
#             normalized_matrix = (
#                 matrix / norms
#             )  # Normalize the column vectors by dividing each element by the corresponding norm

#             # Compute the cosine similarity matrix using matrix multiplication
#             return t.matmul(normalized_matrix.t(), normalized_matrix)

#         similarity_matrix = calc_cosine_similarity_matrix(time_embeddings.T)
#         st.plotly_chart(px.imshow(similarity_matrix.detach().numpy()))


# def show_rtg_embeddings(dt, logit_dir):
#     with st.expander("Show RTG Embeddings"):
#         batch_size = 1028
#         if st.session_state.allow_extrapolation:
#             min_value = -10
#             max_value = 10
#         else:
#             min_value = -1
#             max_value = 1
#         rtg_range = st.slider(
#             "RTG Range",
#             min_value=min_value,
#             max_value=max_value,
#             value=(-1, 1),
#             step=1,
#         )

#         min_rtg = rtg_range[0]
#         max_rtg = rtg_range[1]

#         rtg_range = t.linspace(min_rtg, max_rtg, 100).unsqueeze(-1)

#         rtg_embeddings = dt.reward_embedding(rtg_range).squeeze(0)

#         dot_prod = rtg_embeddings @ logit_dir
#         dot_prod = dot_prod.detach()

#         show_initial = st.checkbox("Show initial RTG embedding", value=True)

#         fig = px.line(x=rtg_range.squeeze(1).detach().numpy(), y=dot_prod)
#         fig.update_layout(
#             title="RTG Embedding Dot Product",
#             xaxis_title="RTG",
#             yaxis_title="Dot Product",
#             legend_title="",
#         )
#         # remove legend
#         fig.update_layout(showlegend=False)
#         if show_initial:
#             fig.add_vline(
#                 x=st.session_state.rtg[0][0].item(),
#                 line_dash="dash",
#                 line_color="red",
#                 annotation_text="Initial RTG",
#             )
#         st.plotly_chart(fig, use_container_width=True)
