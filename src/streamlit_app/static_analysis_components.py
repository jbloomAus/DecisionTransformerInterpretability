import einops
import pandas as pd
import plotly.express as px
import streamlit as st
import torch
import torch as t
from fancy_einsum import einsum

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

        # let's start with state based ones. These are most important because they produce actions!

        n_heads = dt.transformer_config.n_heads
        height, width, channels = dt.environment_config.observation_space[
            "image"
        ].shape
        layer_selection, head_selection, other_selection = st.columns(3)

        with layer_selection:
            layer = st.selectbox(
                "Select Layer",
                options=list(range(dt.transformer_config.n_layers)),
            )

        with head_selection:
            heads = st.multiselect(
                "Select Heads",
                options=list(range(n_heads)),
                key="head qk",
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
                key="channels qk",
                default=[0, 1, 2],
            )

        state_state_tab, state_rtg_tab, state_action_tab = st.tabs(
            [
                "QK(state, state)",
                "QK(state, rtg)",
                "QK(state, action)",
            ]
        )

        W_E_rtg = dt.reward_embedding[0].weight
        W_E_state = dt.state_embedding.weight
        W_Q = dt.transformer.blocks[layer].attn.W_Q
        W_K = dt.transformer.blocks[layer].attn.W_K

        W_QK = einsum(
            "head d_mod_Q d_head, head d_mod_K d_head -> head d_mod_Q d_mod_K",
            W_Q,
            W_K,
        )

        with state_state_tab:
            W_QK_full = W_E_state.T @ W_QK @ W_E_state
            st.write(W_QK_full.shape)

            W_QK_full_reshaped = einops.rearrange(
                W_QK_full,
                "b (h w c) (h1 w1 c1) -> b h w c h1 w1 c1",
                h=7,
                w=7,
                c=20,
                h1=7,
                w1=7,
                c1=20,
            )
            st.write(W_QK_full_reshaped.shape)

        with state_rtg_tab:
            # st.write(W_QK.shape)
            # W_QK_full = W_E_rtg.T @ W_QK @ W_E_state
            W_QK_full = W_E_state.T @ W_QK @ W_E_rtg

            W_QK_full_reshaped = W_QK_full.reshape(
                n_heads, 1, channels, height, width
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
                        fancy_imshow(
                            W_QK_full_reshaped[head, 0, channel]
                            .T.detach()
                            .numpy(),
                            color_continuous_midpoint=0,
                        )


def show_ov_circuit(dt):
    with st.expander("Show OV Circuit"):
        st.subheader("OV circuits")

        st.latex(
            r"""
            OV_{circuit} = W_{U(action)} W_O W_V W_{E(State)}
            """
        )

        W_U = dt.action_predictor.weight
        W_O = dt.transformer.blocks[0].attn.W_O
        W_V = dt.transformer.blocks[0].attn.W_V
        W_E = dt.state_embedding.weight
        W_OV = W_V @ W_O

        # st.plotly_chart(px.imshow(W_OV.detach().numpy(), facet_col=0), use_container_width=True)
        OV_circuit_full = W_E.T @ W_OV @ W_U.T

        height, width, channels = dt.environment_config.observation_space[
            "image"
        ].shape
        n_actions = W_U.shape[0]
        n_heads = dt.transformer_config.n_heads
        OV_circuit_full_reshaped = OV_circuit_full.reshape(
            n_heads, channels, height, width, n_actions
        )

        if channels == 3:

            def format_func(x):
                return three_channel_schema[x]

        else:
            format_func = twenty_idx_format_func

        selection_columns = st.columns(3)
        with selection_columns[0]:
            heads = st.multiselect(
                "Select Heads",
                options=list(range(n_heads)),
                key="head ov",
                default=[0],
            )

        with selection_columns[1]:
            selected_channels = st.multiselect(
                "Select Observation Channels",
                options=list(range(channels)),
                format_func=format_func,
                key="channels ov",
                default=[0, 1, 2],
            )

        with selection_columns[2]:
            selected_actions = st.multiselect(
                "Select Actions",
                options=list(range(n_actions)),
                key="actions ov",
                format_func=lambda x: IDX_TO_ACTION[x],
                default=[0, 1, 2],
            )

        columns = st.columns(len(selected_channels))
        for i, channel in enumerate(selected_channels):
            with columns[i]:
                if channels == 3:
                    st.write(three_channel_schema[channel])
                elif channels == 20:
                    st.write(twenty_idx_format_func(channel))

        for head in heads:
            for action in selected_actions:
                st.write(f"Head {head} - {IDX_TO_ACTION[action]}")
                columns = st.columns(len(selected_channels))
                for i, channel in enumerate(selected_channels):
                    with columns[i]:
                        fancy_imshow(
                            OV_circuit_full_reshaped[
                                head, channel, :, :, action
                            ]
                            .T.detach()
                            .numpy(),
                            color_continuous_midpoint=0,
                        )


def show_time_embeddings(dt, logit_dir):
    with st.expander("Show Time Embeddings"):
        if dt.time_embedding_type == "linear":
            time_steps = t.arange(100).unsqueeze(0).unsqueeze(-1).to(t.float32)
            time_embeddings = dt.get_time_embeddings(time_steps).squeeze(0)
        else:
            time_embeddings = dt.time_embedding.weight

        max_timestep = st.slider(
            "Max timestep",
            min_value=1,
            max_value=time_embeddings.shape[0] - 1,
            value=time_embeddings.shape[0] - 1,
        )
        time_embeddings = time_embeddings[: max_timestep + 1]
        dot_prod = time_embeddings @ logit_dir
        dot_prod = dot_prod.detach()

        show_initial = st.checkbox("Show initial time embedding", value=True)
        fig = px.line(dot_prod)
        fig.update_layout(
            title="Time Embedding Dot Product",
            xaxis_title="Time Step",
            yaxis_title="Dot Product",
            legend_title="",
        )
        # remove legend
        fig.update_layout(showlegend=False)
        if show_initial:
            fig.add_vline(
                x=st.session_state.timesteps[0][-1].item()
                + st.session_state.timestep_adjustment,
                line_dash="dash",
                line_color="red",
                annotation_text="Current timestep",
            )
        st.plotly_chart(fig, use_container_width=True)

        def calc_cosine_similarity_matrix(matrix: t.Tensor) -> t.Tensor:
            # Check if the input matrix is square
            # assert matrix.shape[0] == matrix.shape[1], "The input matrix must be square."

            # Normalize the column vectors
            norms = t.norm(
                matrix, dim=0
            )  # Compute the norms of the column vectors
            normalized_matrix = (
                matrix / norms
            )  # Normalize the column vectors by dividing each element by the corresponding norm

            # Compute the cosine similarity matrix using matrix multiplication
            return t.matmul(normalized_matrix.t(), normalized_matrix)

        similarity_matrix = calc_cosine_similarity_matrix(time_embeddings.T)
        st.plotly_chart(px.imshow(similarity_matrix.detach().numpy()))


def show_rtg_embeddings(dt, logit_dir):
    with st.expander("Show RTG Embeddings"):
        batch_size = 1028
        if st.session_state.allow_extrapolation:
            min_value = -10
            max_value = 10
        else:
            min_value = -1
            max_value = 1
        rtg_range = st.slider(
            "RTG Range",
            min_value=min_value,
            max_value=max_value,
            value=(-1, 1),
            step=1,
        )

        min_rtg = rtg_range[0]
        max_rtg = rtg_range[1]

        rtg_range = t.linspace(min_rtg, max_rtg, 100).unsqueeze(-1)

        rtg_embeddings = dt.reward_embedding(rtg_range).squeeze(0)

        dot_prod = rtg_embeddings @ logit_dir
        dot_prod = dot_prod.detach()

        show_initial = st.checkbox("Show initial RTG embedding", value=True)

        fig = px.line(x=rtg_range.squeeze(1).detach().numpy(), y=dot_prod)
        fig.update_layout(
            title="RTG Embedding Dot Product",
            xaxis_title="RTG",
            yaxis_title="Dot Product",
            legend_title="",
        )
        # remove legend
        fig.update_layout(showlegend=False)
        if show_initial:
            fig.add_vline(
                x=st.session_state.rtg[0][0].item(),
                line_dash="dash",
                line_color="red",
                annotation_text="Initial RTG",
            )
        st.plotly_chart(fig, use_container_width=True)


def show_dim_reduction(dt):
    with st.expander("Show SVD Decomposition"):
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

                st.write(W_Q.shape)
                st.write(W_K.shape)

                # Embedding Values
                W_E_state = dt.state_embedding.weight
                W_E_action = dt.action_embedding[0].weight
                W_E_rtg = dt.reward_embedding[0].weight

                st.write(W_E_state.shape)
                st.write(W_E_action.shape)
                st.write(W_E_rtg.shape)

                # inner QK circuits.
                W_QK = torch.einsum("lhmd,lhdn->lhmn", W_Q, W_K)

                st.write(W_QK.shape)

                U, S, V = torch.linalg.svd(W_QK)

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

                # shape d_action, d_mod
                activations = einsum("l h d1 d2, a d2 -> l h d1 a", V, W_U)

                # torch.Size([3, 8, 7, 256])
                # Now we want to select a head/layer and plot the imshow of the activations
                # only for the first n activations
                layer_selection, head_selection, k_selection = st.columns(3)

                with layer_selection:
                    layer = st.selectbox(
                        "Select Layer",
                        options=list(range(dt.transformer_config.n_layers)),
                    )

                with head_selection:
                    head = st.selectbox(
                        "Select Head",
                        options=list(range(dt.transformer_config.n_heads)),
                    )
                with k_selection:
                    k = st.slider(
                        "Select K",
                        min_value=3,
                        max_value=n_actions,
                        value=3,
                        step=1,
                    )

                head_v_projections = activations[
                    layer, head, :d_head, :
                ].detach()
                # st.write(head_v_projections.shape)
                # get top k activations per column
                topk_values, topk_indices = torch.topk(
                    head_v_projections, k, dim=1
                )

                # put indices into a heat map then replace with the IDX_TO_ACTION string
                df = pd.DataFrame(topk_indices.T.detach().numpy())
                fig = px.imshow(topk_values.T)
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
                labels = [
                    f"L{i}H{j}"
                    for i in range(0, dt.transformer_config.n_layers)
                    for j in range(dt.transformer_config.n_heads)
                ]
                S = einops.rearrange(S, "l h s -> (l h) s")

                df = pd.DataFrame(S.T.detach().numpy(), columns=labels)
                fig = px.line(
                    df,
                    range_x=[0, d_head + 10],
                    labels={"index": "Singular Value", "value": "Value"},
                    title="Singular Value by OV Circuit",
                )
                # add a vertical white dotted line at x = d_head
                fig.add_vline(x=d_head, line_dash="dash", line_color="white")
                st.plotly_chart(fig, use_container_width=True)
