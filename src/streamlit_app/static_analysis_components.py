import uuid

import einops
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
import torch
from fancy_einsum import einsum
from .utils import tensor_to_long_data_frame, get_row_names_from_index_labels
from .components import create_search_component
import einops
import itertools

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# create a pyvis network
from pyvis.network import Network
import networkx as nx


from src.visualization import (
    get_param_stats,
    plot_param_stats,
    tensor_cosine_similarity_heatmap,
    get_cosine_sim_df,
)

from .constants import (
    IDX_TO_ACTION,
    IDX_TO_STATE,
    three_channel_schema,
    twenty_idx_format_func,
    SPARSE_CHANNEL_NAMES,
    POSITION_NAMES,
    ACTION_NAMES,
    STATE_EMBEDDING_LABELS,
)
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
def show_embeddings(_dt):
    with st.expander("Embeddings"):
        all_index_labels = [
            SPARSE_CHANNEL_NAMES,
            list(range(7)),
            list(range(7)),
        ]
        position_index_labels = [
            list(range(7)),
            list(range(7)),
        ]

        singe_action_index_labels = [IDX_TO_ACTION[i] for i in range(7)]

        both_action_index_labels = [
            [IDX_TO_ACTION[i] for i in range(7)],
            [IDX_TO_ACTION[i] for i in range(7)],
        ]

        all_embeddings_tab, pca_tab = st.tabs(["Raw", "PCA"])

        with all_embeddings_tab:
            state_tab, in_action_tab, out_action_tab = st.tabs(
                ["State", "In Action", "Out Action"]
            )

            with state_tab:
                a, b, c, d = st.columns(4)
                with a:
                    aggregation_group = st.selectbox(
                        "Select aggregation method",
                        ["None", "Channels", "Positions"],
                    )

                with c:
                    cluster = st.checkbox("Cluster")

                embedding = _dt.state_embedding.weight.detach().T

                with d:
                    centre_embeddings = st.checkbox("Centre Embeddings")
                    if centre_embeddings:
                        embedding = embedding - embedding.mean(dim=0)

                df = get_cosine_sim_df(embedding)
                df.columns = STATE_EMBEDDING_LABELS
                df.index = STATE_EMBEDDING_LABELS

                if aggregation_group == "None":
                    with b:
                        selected_embeddings = st.multiselect(
                            "Select Embeddings",
                            options=STATE_EMBEDDING_LABELS,
                            key="embedding",
                            default=[],
                        )
                        if selected_embeddings:
                            df = df.loc[
                                selected_embeddings, selected_embeddings
                            ]

                    fig = plot_heatmap(
                        df,
                        cluster,
                        show_labels=df.shape[0] < 20,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                if aggregation_group == "Channels":
                    with b:
                        selected_positions = st.multiselect(
                            "Select Positions",
                            options=list(range(len(POSITION_NAMES))),
                            format_func=lambda x: POSITION_NAMES[x],
                            key="position embedding",
                            default=[5, 6],
                        )

                    image_shape = _dt.environment_config.observation_space[
                        "image"
                    ].shape

                    embedding = einops.rearrange(
                        embedding,
                        "(c x y) d -> x y c d",
                        x=image_shape[0],
                        y=image_shape[1],
                        c=image_shape[-1],
                    )

                    if selected_positions:
                        position_index_labels = list(
                            itertools.product(*position_index_labels)
                        )
                        selected_rows = torch.tensor(
                            [
                                position_index_labels[i][0]
                                for i in selected_positions
                            ]
                        )
                        selected_cols = torch.tensor(
                            [
                                position_index_labels[i][1]
                                for i in selected_positions
                            ]
                        )

                        mask = torch.zeros(7, 7)
                        mask[selected_rows, selected_cols] = 1

                        if st.checkbox("Show mask"):
                            st.plotly_chart(px.imshow(mask))

                        embedding = embedding[mask.to(bool)]

                        embedding = einops.reduce(
                            embedding, "p c d -> c d", "sum"
                        )

                    else:
                        embedding = einops.reduce(
                            embedding, "x y c d -> c d", "sum"
                        )

                    fig = tensor_cosine_similarity_heatmap(
                        embedding, labels=SPARSE_CHANNEL_NAMES
                    )
                    st.plotly_chart(fig, use_container_width=True)

                if aggregation_group == "Positions":
                    with b:
                        selected_channels = st.multiselect(
                            "Select Positions",
                            options=list(range(len(SPARSE_CHANNEL_NAMES))),
                            format_func=lambda x: SPARSE_CHANNEL_NAMES[x],
                            key="position embedding",
                            default=[5, 6],
                        )

                    image_shape = _dt.environment_config.observation_space[
                        "image"
                    ].shape

                    embedding = einops.rearrange(
                        embedding,
                        "(c x y) d -> x y c d",
                        x=image_shape[0],
                        y=image_shape[1],
                        c=image_shape[-1],
                    )

                    if selected_channels:
                        embedding = embedding[:, :, selected_channels]
                    embedding = einops.reduce(
                        embedding, "x y c d -> (x y) d", "sum"
                    )

                    fig = tensor_cosine_similarity_heatmap(
                        embedding,
                        labels=["x", "y"],
                        index_labels=position_index_labels,
                    )
                    st.plotly_chart(fig, use_container_width=True)

            with in_action_tab:
                embedding = _dt.action_embedding[0].weight.detach()[:-1, :]

                df = get_cosine_sim_df(embedding)
                df.columns = ACTION_NAMES
                df.index = ACTION_NAMES

                fig = tensor_cosine_similarity_heatmap(
                    embedding, labels=ACTION_NAMES
                )

                if st.checkbox("Centre Embeddings", key="centre in action"):
                    embedding = embedding - embedding.mean(dim=0)

                fig = plot_heatmap(
                    df,
                    cluster=True,
                    show_labels=True,
                )
                st.plotly_chart(fig, use_container_width=True)

            with out_action_tab:
                embedding = _dt.action_predictor.weight.detach()

                df = get_cosine_sim_df(embedding)
                df.columns = ACTION_NAMES
                df.index = ACTION_NAMES

                if st.checkbox("Centre Embeddings", key="centre out action"):
                    embedding = embedding - embedding.mean(dim=0)

                fig = plot_heatmap(
                    df,
                    cluster=True,
                    show_labels=True,
                )
                st.plotly_chart(fig, use_container_width=True)

        with pca_tab:
            state_tab, in_action_tab, out_action_tab = st.tabs(
                ["State", "In Action", "Out Action"]
            )

            with state_tab:
                embedding = _dt.state_embedding.weight.detach().T

                with st.spinner("Performing PCA..."):
                    # Normalize the data
                    normalized_embedding = StandardScaler().fit_transform(
                        embedding
                    )

                    # Perform PCA
                    pca = PCA(n_components=2)
                    pca_results = pca.fit_transform(normalized_embedding)

                    # st.write(pca_results)
                    # Create a dataframe for the results
                    pca_df = pd.DataFrame(
                        data=pca_results,
                        index=get_row_names_from_index_labels(
                            ["State", "x", "y"], all_index_labels
                        ),
                        columns=["PC1", "PC2"],
                    )
                    pca_df.reset_index(inplace=True, names="State")
                    pca_df["Channel"] = pca_df["State"].apply(
                        lambda x: x.split(",")[0]
                    )
                
                states = set(pca_df['State'].values)
                selected_channels = st.multiselect(
                    "Select Observation Channels",
                    options=list(states),
                )

                states_to_filter = [state for state in selected_channels]
                if states_to_filter:
                    pca_df_filtered = pca_df[pca_df['State'].isin(states_to_filter)]
                else:
                    pca_df_filtered = pca_df

                # Create the plot
                fig = px.scatter(
                    pca_df_filtered,
                    x="PC1",
                    y="PC2",
                    title="PCA on Embeddings",
                    hover_data=["State", "PC1", "PC2"],
                    color="Channel",
                )

                st.plotly_chart(fig, use_container_width=True)

            with in_action_tab:
                embedding = _dt.action_embedding[0].weight.detach()

                with st.spinner("Performing PCA..."):
                    # Normalize the data
                    normalized_embedding = StandardScaler().fit_transform(
                        embedding
                    )

                    # Perform PCA
                    pca = PCA(n_components=2)
                    pca_results = pca.fit_transform(normalized_embedding)

                    # st.write(pca_results)
                    # Create a dataframe for the results
                    pca_df = pd.DataFrame(
                        data=pca_results,
                        index=singe_action_index_labels + ["Null"],
                        columns=["PC1", "PC2"],
                    )
                    pca_df.reset_index(inplace=True, names="Action")

                # Create the plot
                fig = px.scatter(
                    pca_df,
                    x="PC1",
                    y="PC2",
                    title="PCA on Embeddings",
                    hover_data=["Action", "PC1", "PC2"],
                    color="Action",
                )

                st.plotly_chart(fig, use_container_width=True)

            with out_action_tab:
                embedding = _dt.action_predictor.weight.detach()

                with st.spinner("Performing PCA..."):
                    # Normalize the data
                    normalized_embedding = StandardScaler().fit_transform(
                        embedding
                    )

                    # Perform PCA
                    pca = PCA(n_components=2)
                    pca_results = pca.fit_transform(normalized_embedding)

                    # st.write(pca_results)
                    # Create a dataframe for the results
                    pca_df = pd.DataFrame(
                        data=pca_results,
                        index=singe_action_index_labels,
                        columns=["PC1", "PC2"],
                    )
                    pca_df.reset_index(inplace=True, names="Action")

                # Create the plot
                fig = px.scatter(
                    pca_df,
                    x="PC1",
                    y="PC2",
                    title="PCA on Embeddings",
                    hover_data=["Action", "PC1", "PC2"],
                    color="Action",
                )

                st.plotly_chart(fig, use_container_width=True)


@st.cache_data(experimental_allow_widgets=True)
def show_neuron_directions(_dt):
    MLP_in = torch.stack(
        [block.mlp.W_in for block in _dt.transformer.blocks]
    ).detach()

    MLP_out = torch.stack(
        [block.mlp.W_out for block in _dt.transformer.blocks]
    ).detach()

    layers = _dt.transformer_config.n_layers

    with st.expander("Show Neuron In / Out Directions"):
        in_tab, out_tab = st.tabs(["In", "Out"])

        with in_tab:
            tabs = st.tabs(["MLP" + str(layer) for layer in range(layers)])
            for i, tab in enumerate(tabs):
                with tab:
                    df = get_cosine_sim_df(MLP_in[i])
                    fig = plot_heatmap(df, cluster=False)
                    st.plotly_chart(fig, use_container_width=True)

        with out_tab:
            tabs = st.tabs(["MLP" + str(layer) for layer in range(layers)])
            for i, tab in enumerate(tabs):
                with tab:
                    df = get_cosine_sim_df(MLP_out[i])
                    fig = plot_heatmap(df, cluster=False)
                    st.plotly_chart(fig, use_container_width=True)
    # dt.transformer.

    return


@st.cache_data(experimental_allow_widgets=True)
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

        st.write(
            "This component is currently in active development. For now we are only showing the Q (State) to K (RTG) visualizations"
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

        # stack the heads
        W_Q = torch.stack([block.attn.W_Q for block in _dt.transformer.blocks])
        W_K = torch.stack([block.attn.W_K for block in _dt.transformer.blocks])
        # inner QK circuits.
        W_QK = einsum(
            "layer head d_model1 d_head, layer head d_model2 d_head -> layer head d_model1 d_model2",
            W_Q,
            W_K,
        )

        W_E_rtg = _dt.reward_embedding[0].weight
        W_E_state = _dt.state_embedding.weight
        with state_state_tab:
            st.write(
                """
                State->State attention is made up of 980*980 coefficients representing each channel\
                and position attending in the key matched up with every channel and position in the\
                query. In order to make this more tractable, let's do two things:

                1. only work on one head at a time.
                2. aggregate first accross channels and then pick only key-query
                    channel combinations that seem important.
                """
            )

            a, b, c = st.columns(3)
            with a:
                layer = st.selectbox("Select a layer", list(range(layers)))
            with b:
                head = st.selectbox("Select a head", list(range(n_heads)))
            with c:
                show_std_channel_channel = st.checkbox(
                    "Show std of coefficient by query-key channel"
                )

            W_QK_full = W_E_state.T @ W_QK[layer, head] @ W_E_state

            W_QK_full_reshaped = W_QK_full.reshape(
                channels, height, width, channels, height, width
            )

            all_scores_df = tensor_to_long_data_frame(
                W_QK_full_reshaped,
                dimension_names=[
                    "Channel-Q",
                    "X-Q",
                    "Y-Q",
                    "Channel-K",
                    "X-K",
                    "Y-K",
                ],
            )

            # order by channe then reset index
            all_scores_df = all_scores_df.sort_values(
                by=["Channel-Q", "Channel-K"],
            )

            # aggregate by channels
            all_scores_df = (
                all_scores_df.groupby(["Channel-Q", "Channel-K"])
                .std()
                .reset_index()
            )

            all_scores_df["Channel-K"] = all_scores_df["Channel-K"].map(
                twenty_idx_format_func
            )

            all_scores_df["Channel-Q"] = all_scores_df["Channel-Q"].map(
                twenty_idx_format_func
            )

            channel_names = [twenty_idx_format_func(i) for i in range(20)]
            if show_std_channel_channel:
                channel_channel_attn_coeff_std = torch.tensor(
                    all_scores_df.Score
                ).reshape(20, 20)

                df = pd.DataFrame(
                    channel_channel_attn_coeff_std,
                    index=channel_names,
                    columns=channel_names,
                )
                fig = px.imshow(
                    df,
                    color_continuous_midpoint=0,
                    color_continuous_scale="RdBu",
                )

                # xticks and yticks are the channel values
                fig.update_xaxes(
                    showgrid=False,
                    ticks="",
                    tickmode="linear",
                    automargin=True,
                    ticktext=channel_names,
                )

                fig.update_yaxes(
                    showgrid=False,
                    ticks="",
                    tickangle=0,
                    tickmode="linear",
                    automargin=True,
                    tickvals=np.arange(len(channel_names)),
                    ticktext=channel_names,
                )

                st.plotly_chart(fig, use_container_width=True)

                st.write(
                    "These charts remind me a lot of a kth rank approximation to a matrix."
                )

            a, b, c = st.columns(3)
            with a:
                query_channel = st.selectbox(
                    "Query Channel",
                    list(range(20)),
                    index=5,
                    format_func=twenty_idx_format_func,
                )
            with b:
                key_channel = st.selectbox(
                    "Key Channel",
                    list(range(20)),
                    index=6,
                    format_func=twenty_idx_format_func,
                )
            with c:
                show_query_filter = st.checkbox("Show Query filter?")

            st.write(
                """
                In order to make this much easier/faster, I've configured the current final state\
                to get the 'active' position in query channel and then we'll sum all the corresponding\
                maps in query values. The result shows us a map of the key channel states which the\
                current state should attend to.
                """
            )

            st.write(st.session_state.obs[0, -1, :, :, query_channel].shape)
            query_filter = st.session_state.obs[0, -1, :, :, query_channel].T
            query_filter = query_filter.to(torch.bool)

            if show_query_filter:
                fig = px.imshow(query_filter)
                st.plotly_chart(fig)

            key_map = W_QK_full_reshaped[
                query_channel, :, :, key_channel, :, :
            ]
            key_map = key_map * query_filter.T[:, :, None, None]
            key_map = key_map.sum(dim=(0, 1)).T.detach()

            fig = px.imshow(
                key_map,
                color_continuous_midpoint=0,
                color_continuous_scale="RdBu",
            )
            st.plotly_chart(fig)

            st.write(
                """
                The intensity of a color here, represents how much the query
                will attend more to a state who's key state for that 
                channel at that position is firing. 

                Note that for now the color range isn't normalized which is not great.

                Last step: We can use channel activation on any given prior state to get a filter,
                apply it to the map and see how much this channel-query, key-query
                contribute to the attention from the current position to the previous position.

                To do: Validate this works by generating the key filter map as well and showing the 
                selected attention contributing terms.
                """
            )

            # indexes = list(range(len(st.session_state.labels)))[1::3]
            # index_to_label = {i:st.session_state.labels[i] for i in indexes}
            # key_pos = st.selectbox(
            #     "Select a previous state",
            #     indexes,
            #     format_func=index_to_label.get)

            # query_filter = st.session_state.obs[0,-1, :,:,query_channel].T

            # previous_pos = st.selectbox()

            # we can the use the can

            # # make a strip plot
            # fig = px.scatter(
            #     all_scores_df,
            #     x=all_scores_df.index,
            #     y="Score",
            #     # facet_col="Channel-Q",
            #     color="Channel-K",
            #     hover_data=["Channel-Q", "Channel-K"],
            #     labels={"value": "Congruence", "Score":"Std of Score in Channel-Channel Group"},
            # )

            # # update x axis to hide the tick labels, and remove the label
            # fig.update_xaxes(showticklabels=False, title=None)
            # st.plotly_chart(fig, use_container_width=True)

            # st.write("Use the above graph to work out which channels to K ")

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


@st.cache_data(experimental_allow_widgets=True)
def show_ov_circuit(_dt):
    with st.expander("Show OV Circuit"):
        st.subheader("OV circuits")

        st.latex(
            r"""
            OV_{circuit} = W_{U(action)} W_O W_V W_{E(State)}
            """
        )

        height, width, channels = _dt.environment_config.observation_space[
            "image"
        ].shape
        n_actions = _dt.environment_config.action_space.n
        n_heads = _dt.transformer_config.n_heads

        if channels == 3:

            def format_func(x):
                return three_channel_schema[x]

        else:
            format_func = twenty_idx_format_func

        selection_columns = st.columns(2)
        with selection_columns[0]:
            layer = st.selectbox(
                "Select Layer",
                options=list(range(_dt.transformer_config.n_layers)),
            )

        with selection_columns[1]:
            heads = st.multiselect(
                "Select Heads",
                options=list(range(n_heads)),
                key="head ov",
                default=[0],
            )
        selection_columns = st.columns(2)
        with selection_columns[0]:
            selected_channels = st.multiselect(
                "Select Observation Channels",
                options=list(range(channels)),
                format_func=format_func,
                key="channels ov",
                default=[5, 6] if channels == 20 else [0, 1, 2],
            )

        with selection_columns[1]:
            selected_actions = st.multiselect(
                "Select Actions",
                options=list(range(n_actions)),
                key="actions ov",
                format_func=lambda x: IDX_TO_ACTION[x],
                default=[0, 1, 2],
            )

        W_U = _dt.action_predictor.weight
        W_O = _dt.transformer.blocks[layer].attn.W_O
        W_V = _dt.transformer.blocks[layer].attn.W_V
        W_E = _dt.state_embedding.weight
        W_OV = W_V @ W_O

        # st.plotly_chart(px.imshow(W_OV.detach().numpy(), facet_col=0), use_container_width=True)
        OV_circuit_full = W_E.T @ W_OV @ W_U.T
        OV_circuit_full_reshaped = OV_circuit_full.reshape(
            n_heads, channels, height, width, n_actions
        ).detach()

        def write_schema():
            columns = st.columns(len(selected_channels))
            for i, channel in enumerate(selected_channels):
                with columns[i]:
                    if channels == 3:
                        st.write(three_channel_schema[channel])
                    elif channels == 20:
                        st.write(twenty_idx_format_func(channel))

        abs_max_val = OV_circuit_full_reshaped.abs().max().item()
        # st.plotly_chart(px.histogram(OV_circuit_full_reshaped.flatten()), use_container_width=True)

        selection_columns = st.columns(2)
        with selection_columns[0]:
            abs_max_val = st.slider(
                "Max Absolute Value Color",
                min_value=abs_max_val / 10,
                max_value=abs_max_val,
                value=abs_max_val,
            )

        with selection_columns[1]:
            st.write("Selecting the color range to control sensitivity.")
            st.write("Defaults to max value in the OV circuit.")
            st.write("This is a pretty high bar for output.")

        search_tab, comparison_tab, minimize = st.tabs(
            ["Unstructured View", "Comparison tab", "Minimize"]
        )

        all_scores_df = tensor_to_long_data_frame(
            OV_circuit_full_reshaped,
            dimension_names=["Head", "Channel", "X", "Y", "Action"],
        )

        with comparison_tab:
            # for one of the heads selected, and a pair of the actins selected,
            # we want a scatter plot of score vs score
            # use a multiselect for each, but them in three  columns

            a, b, c, d = st.columns(4)

            with a:
                head = st.selectbox("Select Head", options=heads)
            with b:
                action_1 = st.selectbox(
                    "Select Action 1",
                    options=selected_actions,
                    format_func=lambda x: IDX_TO_ACTION[x],
                    index=0,
                )
            with c:
                action_2 = st.selectbox(
                    "Select Action 2",
                    options=selected_actions,
                    format_func=lambda x: IDX_TO_ACTION[x],
                    index=1,
                )
            with d:
                # channel selection
                selected_channels_2 = st.multiselect(
                    "Select Channels",
                    options=list(range(channels)),
                    format_func=format_func,
                    key="channels ov comparison",
                    default=[0],
                )

            # filter the dataframe
            filtered_df = all_scores_df[
                (all_scores_df["Head"] == head)
                & (all_scores_df["Action"].isin([action_1, action_2]))
                & (all_scores_df["Channel"].isin(selected_channels_2))
            ]

            # reshape the df so we have the scores of one action in one column and the scores of the other in another
            filtered_df = filtered_df.pivot_table(
                index=["Head", "Channel", "X", "Y"],
                columns="Action",
                values="Score",
            ).reset_index()
            # rename the columns
            filtered_df.columns = [
                "Head",
                "Channel",
                "X",
                "Y",
                IDX_TO_ACTION[action_1],
                IDX_TO_ACTION[action_2],
            ]

            # make a scatter plot of the two scores
            fig = px.scatter(
                filtered_df,
                x=IDX_TO_ACTION[action_1],
                y=IDX_TO_ACTION[action_2],
                hover_data=["Head", "Channel", "X", "Y"],
                labels={
                    "value": "Congruence",
                    "x": IDX_TO_ACTION[action_1],
                    "y": IDX_TO_ACTION[action_2],
                },
            )

            st.plotly_chart(fig, use_container_width=True)

        with search_tab:
            # sort by layer, head, action
            all_scores_df = all_scores_df.sort_values(by=["Head", "Action"])
            # map actions
            all_scores_df["Action"] = all_scores_df["Action"].map(
                IDX_TO_ACTION
            )
            # reset index
            all_scores_df = all_scores_df.reset_index(drop=True)

            # make a strip plot
            fig = px.scatter(
                all_scores_df,
                x=all_scores_df.index,
                y="Score",
                color="Action",
                hover_data=["Head", "Channel", "X", "Y", "Action"],
                labels={"value": "Congruence"},
            )

            # update x axis to hide the tick labels, and remove the label
            fig.update_xaxes(showticklabels=False, title=None)
            st.plotly_chart(fig, use_container_width=True)

        # create streamlit tabs for each head:
        head_tabs = st.tabs([f"L{layer}H{head}" for head in heads])

        for i, head in enumerate(heads):
            with head_tabs[i]:
                write_schema()
                for action in selected_actions:
                    columns = st.columns(len(selected_channels))
                    for i, channel in enumerate(selected_channels):
                        with columns[i]:
                            st.write("Head", head, "-", IDX_TO_ACTION[action])
                            fig = px.imshow(
                                OV_circuit_full_reshaped[
                                    head, channel, :, :, action
                                ].T,
                                color_continuous_midpoint=0,
                                zmax=abs_max_val,
                                zmin=-abs_max_val,
                                color_continuous_scale=px.colors.diverging.RdBu,
                                labels={"x": "X", "y": "Y"},
                            )
                            fig.update_layout(
                                coloraxis_showscale=False,
                                margin=dict(l=0, r=0, t=0, b=0),
                            )
                            fig.update_layout(height=180, width=400)
                            fig.update_xaxes(showgrid=False, ticks="")
                            fig.update_yaxes(showgrid=False, ticks="")
                            st.plotly_chart(
                                fig, use_container_width=True, autosize=True
                            )


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

                create_search_component(
                    df[["Layer", "Neuron", "Embedding", "Score"]],
                    title="Search MLP to Embeddings",
                    key="mlp to embeddings",
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
                    "layer_out d_mlp_out d_model, layer_in d_mlp_in d_model -> layer_in layer_out d_mlp_in d_mlp_out",
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
                    lambda x: f"N{x}"
                )
                df["Layer In"] = df["Layer In"].map(lambda x: f"L{x}")
                df["Neuron In"] = df["Layer In"] + df["Neuron In"].map(
                    lambda x: f"N{x}"
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
            lambda x: embedding_labels[x]
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
            a, b, c = st.columns(3)

            with a:
                directions = st.multiselect(
                    "Select Directions",
                    options=activations["Direction"].unique(),
                    default=["L0H0D0"],
                    key=f"gridmap direction state in, svd {key}",
                )
            with b:
                channels = st.multiselect(
                    "Select Channels",
                    options=SPARSE_CHANNEL_NAMES,
                    default=["key", "ball"],
                    key=f"gridmap channel state in, svd {key}",
                )
            with c:
                abs_col_max = st.slider(
                    "Max Absolute Value Color",
                    min_value=activations.Score.abs().max().item() / 2,
                    max_value=activations.Score.abs().max().item(),
                    value=activations.Score.abs().max().item(),
                )

            directions_tabs = st.tabs(directions)
            for i in range(len(directions)):
                with directions_tabs[i]:
                    columns = st.columns(len(channels))
                    for j in range(len(columns)):
                        with columns[j]:
                            # given some specific head, I want to project onto some channels.
                            fig = plot_gridmap_from_embedding_congruence(
                                activations,
                                directions[i],
                                channels[j],
                                abs_col_max=abs_col_max,
                            )
                            st.plotly_chart(fig, use_container_width=True)

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
            lambda x: embedding_labels[x]
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


def plot_gridmap_from_embedding_congruence(
    activations, direction, channel, abs_col_max
):
    activations_head = activations[activations.Direction == direction]
    activations_head = activations_head[
        activations_head.Embedding.str.contains(channel)
    ]

    scores = torch.tensor(activations_head.Score.values).reshape(7, 7).T

    fig = px.imshow(
        scores,
        color_continuous_midpoint=0,
        color_continuous_scale="RdBu",
        zmin=-abs_col_max,
        zmax=abs_col_max,
    )

    # update hover template
    fig.update_traces(
        hovertemplate=f"{channel}, " + "(%{x},%{y})<br>Congruence: %{z:.2f}"
    )

    # remove color legend
    fig.update_layout(coloraxis_showscale=False)

    return fig


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


@st.cache_data(experimental_allow_widgets=True)
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


@st.cache_data(experimental_allow_widgets=True)
def layer_head_k_selector_ui(_dt, key=""):
    n_actions = _dt.action_predictor.weight.shape[0]
    layer_selection, head_selection, k_selection, d_selection = st.columns(4)

    with layer_selection:
        layer = st.selectbox(
            "Select Layer",
            options=list(range(_dt.transformer_config.n_layers)),
            key="layer" + key,
        )

    with head_selection:
        head = st.selectbox(
            "Select Head",
            options=list(range(_dt.transformer_config.n_heads)),
            key="head" + key,
        )
    with k_selection:
        if n_actions > 3:
            k = st.slider(
                "Select K",
                min_value=3,
                max_value=n_actions,
                value=3,
                step=1,
                key="k" + key,
            )
        else:
            k = 3
    with d_selection:
        if _dt.transformer_config.d_model > 3:
            dims = st.slider(
                "Select Dimensions",
                min_value=3,
                max_value=_dt.transformer_config.d_head,
                value=3,
                step=1,
                key="dims" + key,
            )
        else:
            dims = 3

    return layer, head, k, dims


@st.cache_data(experimental_allow_widgets=True)
def embedding_matrix_selection_ui(_dt):
    embedding_matrix_selection = st.columns(2)
    with embedding_matrix_selection[0]:
        embedding_matrix_1 = st.selectbox(
            "Select Q Embedding Matrix",
            options=["State", "Action", "RTG"],
            key=uuid.uuid4(),
        )
    with embedding_matrix_selection[1]:
        embedding_matrix_2 = st.selectbox(
            "Select K Embedding Matrix",
            options=["State", "Action", "RTG"],
            key=uuid.uuid4(),
        )

    W_E_state = _dt.state_embedding.weight
    W_E_action = _dt.action_embedding[0].weight
    W_E_rtg = _dt.reward_embedding[0].weight

    if embedding_matrix_1 == "State":
        embedding_matrix_1 = W_E_state
    elif embedding_matrix_1 == "Action":
        embedding_matrix_1 = W_E_action
    elif embedding_matrix_1 == "RTG":
        embedding_matrix_1 = W_E_rtg

    if embedding_matrix_2 == "State":
        embedding_matrix_2 = W_E_state
    elif embedding_matrix_2 == "Action":
        embedding_matrix_2 = W_E_action
    elif embedding_matrix_2 == "RTG":
        embedding_matrix_2 = W_E_rtg

    return embedding_matrix_1, embedding_matrix_2


def layer_head_channel_selector(_dt, key=""):
    n_heads = _dt.transformer_config.n_heads
    height, width, channels = _dt.environment_config.observation_space[
        "image"
    ].shape
    layer_selection, head_selection, other_selection = st.columns(3)

    with layer_selection:
        layer = st.selectbox(
            "Select Layer",
            options=list(range(_dt.transformer_config.n_layers)),
            key="layer" + key,
        )

    with head_selection:
        heads = st.multiselect(
            "Select Heads",
            options=list(range(n_heads)),
            key="head qk" + key,
            default=[0],
        )

    if channels == 3:

        def format_func(x):
            return three_channel_schema[x]

    else:
        format_func = twenty_idx_format_func

    with other_selection:
        selected_channels = st.multiselect(
            "Select Observation Channels",
            options=list(range(channels)),
            format_func=format_func,
            key="channels qk" + key,
            default=[0, 1, 2],
        )

    return layer, heads, selected_channels


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
