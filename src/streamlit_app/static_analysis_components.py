import uuid

import einops
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
import torch
from fancy_einsum import einsum
from .utils import tensor_to_long_data_frame
from .components import create_search_component

# create a pyvis network
from pyvis.network import Network
import networkx as nx

from src.visualization import get_param_stats, plot_param_stats

from .constants import (
    IDX_TO_ACTION,
    IDX_TO_STATE,
    three_channel_schema,
    twenty_idx_format_func,
)
from .utils import fancy_histogram, fancy_imshow


def show_param_statistics(dt):
    with st.expander("Show Parameter Statistics"):
        df = get_param_stats(dt)
        fig_mean, fig_log_std, fig_norm = plot_param_stats(df)

        st.plotly_chart(fig_mean, use_container_width=True)
        st.plotly_chart(fig_log_std, use_container_width=True)
        st.plotly_chart(fig_norm, use_container_width=True)


def show_qk_circuit(dt):
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

        height, width, channels = dt.environment_config.observation_space[
            "image"
        ].shape
        n_heads = dt.transformer_config.n_heads
        layers = dt.transformer_config.n_layers

        # stack the heads
        W_Q = torch.stack([block.attn.W_Q for block in dt.transformer.blocks])
        W_K = torch.stack([block.attn.W_K for block in dt.transformer.blocks])
        # inner QK circuits.
        W_QK = einsum(
            "layer head d_model1 d_head, layer head d_model2 d_head -> layer head d_model1 d_model2",
            W_Q,
            W_K,
        )

        W_E_rtg = dt.reward_embedding[0].weight
        W_E_state = dt.state_embedding.weight
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
                head = st.selectbox("Select a layer", list(range(n_heads)))
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
                dt, key="srtg"
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


def show_ov_circuit(dt):
    with st.expander("Show OV Circuit"):
        st.subheader("OV circuits")

        st.latex(
            r"""
            OV_{circuit} = W_{U(action)} W_O W_V W_{E(State)}
            """
        )

        height, width, channels = dt.environment_config.observation_space[
            "image"
        ].shape
        n_actions = dt.environment_config.action_space.n
        n_heads = dt.transformer_config.n_heads

        if channels == 3:

            def format_func(x):
                return three_channel_schema[x]

        else:
            format_func = twenty_idx_format_func

        selection_columns = st.columns(2)
        with selection_columns[0]:
            layer = st.selectbox(
                "Select Layer",
                options=list(range(dt.transformer_config.n_layers)),
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

        W_U = dt.action_predictor.weight
        W_O = dt.transformer.blocks[layer].attn.W_O
        W_V = dt.transformer.blocks[layer].attn.W_V
        W_E = dt.state_embedding.weight
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


def show_congruence(dt):
    with st.expander("Show Congruence"):
        (
            position_tab,
            time_tab,
            w_out_tab,
            mlp_in_tab,
            mlp_out_tab,
        ) = st.tabs(["Position", "Time", "W_O", "MLP_in", "MLP_out"])

    with position_tab:
        st.write(
            """
            We expect position not to have significant congruence with the output logits because
            if it was valuable information, it would be syntactically processed by the transformer and
            not used semantically.
            
            Nevertheless, this tab answers the question: "How much does the position embedding contribute directly
            to the output logits?"
            """
        )

        position_action_mapping = (
            dt.transformer.W_pos @ dt.action_predictor.weight.T
        )
        fig = px.imshow(
            position_action_mapping.T.detach().numpy(),
            labels={"x": "Position", "y": "Action"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with time_tab:
        st.write(
            """
            Like position, we expect time not to have significant congruence with the output logits because
            syntactic processing is likely not hugely important in grid world tasks. However, unlike position,
            it's more plausible in general that it could be important especially if the model is using it to
            memorize behaviors.
            """
        )

        time_action_mapping = (
            dt.time_embedding.weight[:50, :] @ dt.action_predictor.weight.T
        )
        fig = px.imshow(
            time_action_mapping.T.detach().numpy(),
            labels={"x": "Time", "y": "Action"},
        )

        st.plotly_chart(fig, use_container_width=True)

    with w_out_tab:
        W_0 = torch.stack([block.attn.W_O for block in dt.transformer.blocks])

        W_0_congruence = W_0 @ dt.action_predictor.weight.T
        W_0_congruence = W_0_congruence.permute(0, 1, 3, 2)
        W_0_congruence = W_0_congruence.detach()

        W_0_congruence_2d = W_0_congruence.reshape(-1)
        df = pd.DataFrame(W_0_congruence_2d.numpy(), columns=["value"])
        layers, heads, actions, dims = W_0_congruence.shape
        indices = pd.MultiIndex.from_tuples(
            [
                (i, j, k, l)
                for i in range(layers)
                for j in range(heads)
                for k in range(actions)
                for l in range(dims)
            ],
            names=["layer", "head", "action", "dimension"],
        )
        df.index = indices
        df.reset_index(inplace=True)
        # ensure action is interpreted as a categorical variable
        df["action"] = df["action"].map(IDX_TO_ACTION)

        # sort by action
        df = df.sort_values(by="layer")
        fig = px.scatter(
            df,
            x=df.index,
            y="value",
            color="action",
            hover_data=["layer", "head", "action", "dimension", "value"],
        )

        # update x axis to hide the tick labels, and remove the label
        fig.update_xaxes(showticklabels=False, title=None)

        st.plotly_chart(fig, use_container_width=True)

    with mlp_in_tab:
        MLP_in = torch.stack(
            [block.mlp.W_in for block in dt.transformer.blocks]
        )

        state_tab, action_tab, rtg_tab = st.tabs(["State", "Action", "RTG"])

        with state_tab:
            state_in = dt.state_embedding.weight
            MLP_in_state_congruence = MLP_in.permute(0, 2, 1) @ state_in
            MLP_in_state_congruence = MLP_in_state_congruence.reshape(
                MLP_in_state_congruence.shape[0:2]
                + dt.environment_config.observation_space["image"].shape
            )
            MLP_in_state_congruence = MLP_in_state_congruence.permute(
                0, 2, 3, 4, 1
            )
            MLP_in_state_congruence = MLP_in_state_congruence.detach()

            MLP_in_state_congruence_2d = MLP_in_state_congruence.reshape(-1)
            df = pd.DataFrame(
                MLP_in_state_congruence_2d.numpy(), columns=["value"]
            )
            (
                layers,
                height,
                width,
                channels,
                dims,
            ) = MLP_in_state_congruence.shape
            indices = pd.MultiIndex.from_tuples(
                [
                    (i, j, k, l, m)
                    for i in range(layers)
                    for j in range(height)
                    for k in range(width)
                    for l in range(channels)
                    for m in range(dims)
                ],
                names=["layer", "height", "width", "channel", "dimension"],
            )
            df.index = indices
            df.reset_index(inplace=True)
            # ensure action is interpreted as a categorical variable

            if channels == 3:
                df["channel"] = df["channel"].map(IDX_TO_STATE)
            elif channels == 20:
                df["channel"] = df["channel"].map(twenty_idx_format_func)

            # before we filterm store the top 40 rows by value
            top_40 = df.sort_values(by="value", ascending=False).head(40)
            # and the bottom 40
            bottom_40 = df.sort_values(by="value", ascending=True).head(40)

            a, b = st.columns(2)
            with a:
                # create a multiselect to choose the channels of interest
                format_func = (
                    lambda x: three_channel_schema[x]
                    if channels == 3
                    else twenty_idx_format_func(x)
                )
                selected_channels = st.multiselect(
                    "Select Observation Channels",
                    options=list(range(channels)),
                    format_func=format_func,
                    key="channels mlp_in",
                    default=list(range(17)) if channels == 20 else [0, 1, 2],
                )
                mapped_channels = [
                    format_func(channel) for channel in selected_channels
                ]
                df = df[df["channel"].isin(mapped_channels)]

            df = df.sort_values(by="channel")
            df.reset_index(inplace=True, drop=True)

            with b:
                aggregation_level = st.selectbox(
                    "Aggregation Level",
                    options=["X-Y-Channel", "Neuron"],
                    key="aggregation level",
                )
                if aggregation_level == "X-Y-Channel":
                    # st.write(df)
                    df = df.groupby(
                        ["layer", "height", "width", "channel"]
                    ).std()
                    df = df.sort_values(by="channel")
                    df.reset_index(inplace=True)
                    fig = px.scatter(
                        df,
                        x=df.index,
                        y="value",
                        color="channel",
                        # facet_col="layer",
                        hover_data=[
                            "layer",
                            "height",
                            "width",
                            "channel",
                            "value",
                        ],
                        labels={"value": "Std. of Congruence"},
                    )

                    # update x axis to hide the tick labels, and remove the label
                    fig.update_xaxes(showticklabels=False, title=None)

                else:
                    # sort
                    df = df.sort_values(
                        by=["height", "width"], ascending=False
                    )

                    fig = px.scatter(
                        df,
                        x=df.index,
                        y="value",
                        color="channel",
                        # facet_col="layer",
                        hover_data=[
                            "layer",
                            "height",
                            "width",
                            "channel",
                            "dimension",
                            "value",
                        ],
                        labels={"value": "Congruence"},
                    )

                    # update x axis to hide the tick labels, and remove the label
                    fig.update_xaxes(showticklabels=False, title=None)

            st.plotly_chart(fig, use_container_width=True)

            # # group values by height/widt

            # # now show the top 40 and bottom 40
            # st.write("Top 40")
            # st.write(top_40)

            # st.write("Bottom 40")
            # st.write(bottom_40)

        with action_tab:
            action_in = dt.action_embedding[0].weight
            MLP_in_action_congruence = MLP_in.permute(0, 2, 1) @ action_in.T

            MLP_in_action_congruence = MLP_in_action_congruence.permute(
                0, 2, 1
            )
            MLP_in_action_congruence = MLP_in_action_congruence.detach()

            MLP_in_action_congruence_2d = MLP_in_action_congruence.reshape(-1)
            df = pd.DataFrame(
                MLP_in_action_congruence_2d.numpy(), columns=["value"]
            )
            layers, actions, dims = MLP_in_action_congruence.shape
            indices = pd.MultiIndex.from_tuples(
                [
                    (i, k, l)
                    for i in range(layers)
                    for k in range(actions)
                    for l in range(dims)
                ],
                names=["layer", "action", "dimension"],
            )
            df.index = indices
            df.reset_index(inplace=True)
            # ensure action is interpreted as a categorical variable
            df["action"] = df["action"].map(IDX_TO_ACTION)

            fig = px.scatter(
                df,
                x=df.index,
                y="value",
                color="action",
                # facet_col="layer",
                hover_data=["layer", "action", "dimension", "value"],
                labels={"value": "Congruence"},
            )

            # update x axis to hide the tick labels, and remove the label
            fig.update_xaxes(showticklabels=False, title=None)

            st.plotly_chart(fig, use_container_width=True)

        with rtg_tab:
            rtg_in = dt.reward_embedding[0].weight
            MLP_in_rtg_congruence = MLP_in.permute(0, 2, 1) @ rtg_in

            # torch.Size([3, 256, 1])
            MLP_in_rtg_congruence_2d = MLP_in_rtg_congruence.reshape(-1)
            layers, dims = MLP_in_rtg_congruence.squeeze(-1).shape
            indices = pd.MultiIndex.from_tuples(
                [(i, l) for i in range(layers) for l in range(dims)],
                names=["layer", "dimension"],
            )
            df = pd.DataFrame(
                MLP_in_rtg_congruence_2d.detach().numpy(), columns=["value"]
            )
            df.index = indices
            df.reset_index(inplace=True)
            df["layer"] = df["layer"].astype("category")
            fig = px.scatter(
                df,
                x=df.index,
                y="value",
                color="layer",
                hover_data=["layer", "dimension", "value"],
                labels={"value": "Congruence"},
            )

            # update x axis to hide the tick labels, and remove the label
            fig.update_xaxes(showticklabels=False, title=None)

            st.plotly_chart(fig, use_container_width=True)

    with mlp_out_tab:
        MLP_out = torch.stack(
            [block.mlp.W_out for block in dt.transformer.blocks]
        )

        MLP_out_congruence = MLP_out @ dt.action_predictor.weight.T
        MLP_out_congruence = MLP_out_congruence.permute(0, 2, 1)
        MLP_out_congruence = MLP_out_congruence.detach()

        MLP_out_congruence_2d = MLP_out_congruence.reshape(-1)
        df = pd.DataFrame(MLP_out_congruence_2d.numpy(), columns=["value"])
        layers, actions, dims = MLP_out_congruence.shape
        indices = pd.MultiIndex.from_tuples(
            [
                (i, k, l)
                for i in range(layers)
                for k in range(actions)
                for l in range(dims)
            ],
            names=["layer", "action", "dimension"],
        )
        df.index = indices
        df.reset_index(inplace=True)
        # ensure action is interpreted as a categorical variable
        df["action"] = df["action"].map(IDX_TO_ACTION)

        fig = px.scatter(
            df,
            x=df.index,
            y="value",
            color="action",
            # facet_col="layer",
            hover_data=["layer", "action", "dimension", "value"],
            labels={"value": "Congruence"},
        )

        # update x axis to hide the tick labels, and remove the label
        fig.update_xaxes(showticklabels=False, title=None)

        st.plotly_chart(fig, use_container_width=True)


def show_composition_scores(dt):
    with st.expander("Show Composition Scores"):
        st.markdown(
            "Composition Score calculations per [Mathematical Frameworks for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html#:~:text=The%20above%20diagram%20shows%20Q%2D%2C%20K%2D%2C%20and%20V%2DComposition)"
        )

        q_scores = dt.transformer.all_composition_scores("Q")
        k_scores = dt.transformer.all_composition_scores("K")
        v_scores = dt.transformer.all_composition_scores("V")

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


def show_dim_reduction(dt):
    with st.expander("Show Dimensionality Reduction Decomposition"):
        st.write(
            """
            This analysis is heavily based on the [post](https://www.lesswrong.com/posts/mkbGjzxD8d8XqKHzA/the-singular-value-decompositions-of-transformer-weight#Our_SVD_projection_method) by Conjecture on this topic. 
            """
        )

        st.warning("I'm not convinced yet that my implementation is correct.")

        n_layers = dt.transformer_config.n_layers
        d_head = dt.transformer_config.d_head
        n_actions = dt.action_predictor.weight.shape[0]

        svd_tab, eig_tab = st.tabs(["SVD", "Eig"])

        with eig_tab:
            st.write("Not implemented yet")

        with svd_tab:
            qk_tab, ov_tab = st.tabs(["QK", "OV"])

            with qk_tab:
                # stack the heads
                W_Q = torch.stack(
                    [block.attn.W_Q for block in dt.transformer.blocks]
                )
                W_K = torch.stack(
                    [block.attn.W_K for block in dt.transformer.blocks]
                )
                # inner QK circuits.
                W_QK = einsum(
                    "layer head d_model1 d_head, layer head d_model2 d_head -> layer head d_model1 d_model2",
                    W_Q,
                    W_K,
                )

                U, S, V = torch.linalg.svd(W_QK)

                layer, head, k, dims = layer_head_k_selector_ui(dt, key="qk")
                emb1, emb2 = embedding_matrix_selection_ui(dt)

                plot_svd_by_head_layer(dt, S)

            with ov_tab:
                # stack the heads
                W_V = torch.stack(
                    [block.attn.W_V for block in dt.transformer.blocks]
                )
                W_0 = torch.stack(
                    [block.attn.W_O for block in dt.transformer.blocks]
                )

                # inner OV circuits.
                W_OV = torch.einsum("lhmd,lhdn->lhmn", W_V, W_0)

                # Unembedding Values
                W_U = dt.action_predictor.weight

                U, S, V = torch.linalg.svd(W_OV)

                a, b = st.columns(2)
                with a:
                    congruence_with = st.selectbox(
                        "What should we project onto?", ["W_U", "MLP_in"]
                    )

                if congruence_with == "W_U":
                    # shape d_action, d_mod
                    activations = einsum("l h d1 d2, a d1 -> l h d2 a", V, W_U)

                    # torch.Size([3, 8, 7, 256])
                    # Now we want to select a head/layer and plot the imshow of the activations
                    # only for the first n activations
                    with b:
                        layer, head, k, dims = layer_head_k_selector_ui(
                            dt, key="ov"
                        )

                    head_v_projections = activations[
                        layer, head, :dims, :
                    ].detach()
                    # st.write(head_v_projections.shape)
                    # get top k activations per column
                    topk_values, topk_indices = torch.topk(
                        head_v_projections, k, dim=1
                    )

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

                    # fig = px.imshow(
                    #     head_v_projections.T,
                    #     labels={'y': 'Action', 'x': 'Activation'})
                    # st.plotly_chart(fig, use_container_width=True)

                    # now I want to plot the SVD decay for each S.
                    # Let's create labels for each S.

                    plot_svd_by_head_layer(dt, S)

                if congruence_with == "MLP_in":
                    with b:
                        right_svd_vectors = st.slider(
                            "Number of Singular Directions",
                            min_value=10,
                            max_value=dt.transformer_config.d_head,
                        )

                    MLP_in = torch.stack(
                        [block.mlp.W_in for block in dt.transformer.blocks]
                    )
                    V = V[:, :, :right_svd_vectors, :]
                    activations = einsum(
                        "l1 h1 d_head_in d_head_ext, l2 d_head_ext d_mlp_out -> l2 l1 h1 d_head_in d_mlp_out",
                        V,
                        MLP_in,
                    )

                    activations = tensor_to_long_data_frame(
                        activations,
                        [
                            "mlp_layer",
                            "head_layer",
                            "Head",
                            "Singular Direction",
                            "Neuron",
                        ],
                    )

                    # remove rows where mlp_layer < head_layer
                    activations = activations[
                        activations["mlp_layer"] >= activations["head_layer"]
                    ]

                    activations["head_layer"] = activations["head_layer"].map(
                        lambda x: f"L{x}"
                    )
                    activations["Head"] = activations["Head"].map(
                        lambda x: f"H{x}"
                    )
                    activations["Head"] = (
                        activations["head_layer"] + activations["Head"]
                    )

                    activations["mlp_layer"] = activations["mlp_layer"].map(
                        lambda x: f"L{x}"
                    )
                    activations["Neuron"] = activations["Neuron"].map(
                        lambda x: f"N{x}"
                    )
                    activations["Neuron"] = (
                        activations["mlp_layer"] + activations["Neuron"]
                    )

                    # drop head head_layer, mlp_layer, head_out_dim
                    activations.drop(["head_layer", "mlp_layer"], axis=1)

                    activations = activations.sort_values(
                        ["Head", "Singular Direction", "Neuron", "Score"]
                    )
                    activations.reset_index(drop=True, inplace=True)
                    fig = px.scatter(
                        activations,
                        x=activations.index,
                        color="Head",
                        y="Score",
                        hover_data=["Head", "Neuron", "Singular Direction"],
                        labels={"Score": "Congruence"},
                        render_mode="webgl",
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # add search box
                    create_search_component(
                        activations,
                        "Head Out Singular Value Projections onto MLP",
                    )


def plot_svd_by_head_layer(dt, S):
    d_head = dt.transformer_config.d_head
    labels = [
        f"L{i}H{j}"
        for i in range(0, dt.transformer_config.n_layers)
        for j in range(dt.transformer_config.n_heads)
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


def layer_head_k_selector_ui(dt, key=""):
    n_actions = dt.action_predictor.weight.shape[0]
    layer_selection, head_selection, k_selection, d_selection = st.columns(4)

    with layer_selection:
        layer = st.selectbox(
            "Select Layer",
            options=list(range(dt.transformer_config.n_layers)),
            key="layer" + key,
        )

    with head_selection:
        head = st.selectbox(
            "Select Head",
            options=list(range(dt.transformer_config.n_heads)),
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
        if dt.transformer_config.d_model > 3:
            dims = st.slider(
                "Select Dimensions",
                min_value=3,
                max_value=dt.transformer_config.d_head,
                value=3,
                step=1,
                key="dims" + key,
            )
        else:
            dims = 3

    return layer, head, k, dims


def embedding_matrix_selection_ui(dt):
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

    W_E_state = dt.state_embedding.weight
    W_E_action = dt.action_embedding[0].weight
    W_E_rtg = dt.reward_embedding[0].weight

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


def layer_head_channel_selector(dt, key=""):
    n_heads = dt.transformer_config.n_heads
    height, width, channels = dt.environment_config.observation_space[
        "image"
    ].shape
    layer_selection, head_selection, other_selection = st.columns(3)

    with layer_selection:
        layer = st.selectbox(
            "Select Layer",
            options=list(range(dt.transformer_config.n_layers)),
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
