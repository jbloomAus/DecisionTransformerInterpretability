import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
import torch as t
from einops import rearrange
from fancy_einsum import einsum
from minigrid.core.constants import IDX_TO_COLOR, IDX_TO_OBJECT
import itertools

from src.decision_transformer.utils import get_max_len_from_model_type

from .analysis import get_residual_decomp
from .components import (
    decomp_configuration_ui,
    get_decomp_scan,
    plot_attention_patterns_by_rtg,
    plot_decomp_scan_corr,
    plot_decomp_scan_line,
)
from .constants import (
    IDX_TO_ACTION,
    IDX_TO_STATE,
    three_channel_schema,
    twenty_idx_format_func,
    SPARSE_CHANNEL_NAMES,
)
from .environment import get_action_preds_from_app_state
from .utils import fancy_histogram, fancy_imshow, tensor_to_long_data_frame
from .visualizations import (
    plot_attention_pattern_single,
    plot_heatmap,
    plot_logit_diff,
    plot_logit_scan,
)


RTG_SCAN_BATCH_SIZE = 128

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


def show_attention_pattern(dt, cache, key=""):
    st.latex(
        r"""
        h(x)=\left(A \otimes W_O W_V\right) \cdot x \newline
        """
    )

    st.latex(
        r"""
        A=\operatorname{softmax}\left(x^T W_Q^T W_K x\right)
        """
    )

    visualize_attention_pattern(dt, cache, key)


def visualize_attention_pattern(dt, cache, key=""):
    n_layers = dt.transformer_config.n_layers
    n_heads = dt.transformer_config.n_heads

    (
        a,
        b,
    ) = st.columns(2)
    with a:
        heads = list(range(n_heads))
        layer = st.selectbox(
            "Layer",
            options=list(range(n_layers)),
            key=key + "layer"
        )
        score_or_softmax = st.selectbox(
            "Score or Softmax",
            options=["Score", "Softmax"],
            index=1,
            key = key + "score-or-softmax"
        )
        softmax = score_or_softmax == "Softmax"

    with b:
        scale_by_value = st.selectbox(
            "Scale by value",
            options=[True, False],
            index=0,
            key=key + "scale-by-value"
        )

        if score_or_softmax != "Value Weighted Softmax":
            method = st.selectbox(
                "Select plotting method",
                options=["Plotly", "CircuitsVis"],
                key=key + "plotting-method"
            )
        else:
            method = "Plotly"

    plot_attention_pattern_single(
        cache,
        layer,
        softmax=softmax,
        specific_heads=heads,
        method=method,
        scale_by_value=scale_by_value,
        key=key,
    )


# Attribution / Logit Lens
def show_logit_lens(dt, cache, logit_dir, key=""):
    layertab, componenttab, headtab, neurontab = st.tabs(
        ["Layer", "Component", "Head", "Neuron"]
    )

    with layertab:
        tab1, tab2 = st.tabs(["Layerwise", "Accumulated"])
        with tab1:
            fig = plot_decomposition_dot_product(
                cache, logit_dir, -1, mlp_input=False
            )
            fig.update_yaxes(title_text="Logit Difference")
            st.plotly_chart(fig, use_container_width=True, key=key + "layerwise-logit-diff")

        with tab2:
            fig = plot_decomposition_dot_product(
                cache, logit_dir, -1, mlp_input=False, accumulated=True
            )
            fig.update_yaxes(title_text="Logit Difference")
            st.plotly_chart(fig, use_container_width=True, key=key + "accumulated-logit-diff")

    with componenttab:
        result, labels = cache.get_full_resid_decomposition(
            apply_ln=True, return_labels=True, expand_neurons=False
        )
        attribution = result[:, 0, -1] @ logit_dir
        plot_logit_diff(attribution, labels)

    with headtab:
        result, labels = cache.stack_head_results(
            apply_ln=True, return_labels=True
        )
        heads = dt.transformer_config.n_heads
        attribution = result[:, 0, -1] @ logit_dir
        k = t.topk(attribution, max(5, dt.transformer_config.n_heads))
        attribution = attribution.reshape(-1, heads)
        fig = px.imshow(
            attribution.detach(),
            color_continuous_midpoint=0,
            color_continuous_scale="RdBu",
            title="Logit Difference From Each Head",
            labels={"x": "Head", "y": "Layer"},
        )
        st.plotly_chart(fig, use_container_width=True, key=key + "head-logit-diff")

        st.write(
            f"Top 5 Heads: {', '.join([f'{labels[k.indices[i]]}: {round(k.values[i].item(), 3)}' for i in range(5)])}"
        )

    with neurontab:
        result, labels = cache.get_full_resid_decomposition(
            apply_ln=True, return_labels=True, expand_neurons=True
        )

        attribution = result[:, 0, -1] @ logit_dir

        # use regex to look for the L {number} N {number} pattern in labels
        neuron_attribution_mask = [
            True if re.search(r"L\d+N\d+", label) else False
            for label in labels
        ]

        neuron_attribution = attribution[neuron_attribution_mask]
        neuron_labels = [
            label for label in labels if re.search(r"L\d+N\d+", label)
        ]

        df = pd.DataFrame(
            {
                "Neuron": neuron_labels,
                "Logit Difference": neuron_attribution.detach().numpy(),
            }
        )
        df["Layer"] = df["Neuron"].apply(
            lambda x: int(x.split("L")[1].split("N")[0])
        )

        layertabs = st.tabs(
            ["L" + str(layer) for layer in df["Layer"].unique().tolist()]
        )

        for i, layer in enumerate(df["Layer"].unique().tolist()):
            with layertabs[i]:
                fig = px.scatter(
                    df[df["Layer"] == layer],
                    x="Neuron",
                    y="Logit Difference",
                    hover_data=["Layer"],
                    title="Logit Difference From Each Neuron",
                    color="Logit Difference",
                )
                # color_continuous_scale="RdBu",
                # don't label xtick
                fig.update_xaxes(showticklabels=False)
                st.plotly_chart(fig, use_container_width=True, key=key + f"neuron-logit-diff-{layer}")
    return


def show_neuron_activation_decomposition(dt, cache, logit_dir):
    with st.expander("Neuron Activation Decomposition"):
        a, b = st.columns(2)
        with a:
            neuron_text = st.text_input(
                f"Type the neuron you want", "L0N0", key="neuron_act_analysis"
            )
            # validate neuron
            if not re.search(r"L\d+N\d+", neuron_text):
                st.error("Neuron must be in the format L{number}N{number}")
                return
            # get the neuron index, and the layer
            neuron = int(neuron_text.split("N")[1])
            layer = int(neuron_text.split("L")[1].split("N")[0])

        with b:
            total_neuron_activation = cache[f"blocks.{layer}.mlp.hook_pre"]

        # GET DECOMPOSITION OF THE RESIDUAL STREAM
        decomp_result, labels = cache.get_full_resid_decomposition(
            apply_ln=True,
            return_labels=True,
            expand_neurons=True,
            mlp_input=True,
            layer=layer,
        )  # LN is applied.

        # GET THE MLP IN DIRECTION
        mlp_in = dt.transformer.W_in[layer, :, neuron]
        activation_contributions = decomp_result[:, 0, -1] @ mlp_in

        attribution_df = pd.DataFrame(
            {
                "Component": labels,
                "Activation Difference": activation_contributions.detach().numpy(),
                "Head": [
                    re.search(r"H\d+", label).group()
                    if re.search(r"H\d+", label)
                    else "None"
                    for label in labels
                ],
                "Neuron": [
                    re.search(r"L\d+N\d+", label).group()
                    if re.search(r"L\d+N\d+", label)
                    else "None"
                    for label in labels
                ],
                "Layer": [
                    int(re.search(r"L\d+", label).group().split("L")[1])
                    if re.search(r"L\d+", label)
                    else -1
                    for label in labels
                ],
                "is_neuron": [
                    True if re.search(r"L\d+N\d+", label) else False
                    for label in labels
                ],
                "is_head": [
                    True if re.search(r"H\d+", label) else False
                    for label in labels
                ],
            }
        )

        # 3 tabs, one for head, one for neuron, one for obs token
        lens_tab, obs_tab, head_tab, neuron_tab, debug_tab = st.tabs(
            ["Lens", "Obs Token", "Head", "Neuron", "Debug"]
        )

        with lens_tab:
            tab1, tab2 = st.tabs(["Layerwise", "Accumulated"])
            with tab1:
                fig = plot_decomposition_dot_product(
                    cache, mlp_in, layer, mlp_input=True
                )
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                fig = plot_decomposition_dot_product(
                    cache, mlp_in, layer, mlp_input=True, accumulated=True
                )
                st.plotly_chart(fig, use_container_width=True)

        with obs_tab:
            if dt.transformer_config.state_embedding_type.lower() == "grid":
                # time is technically included in the token
                time = st.session_state.timesteps
                time_embedding = dt.get_time_embedding(time)[0, -1]
                time_attribution = time_embedding @ mlp_in

                # get the state embedding decomposition
                states = st.session_state.obs
                state_mask = (
                    rearrange(
                        states,
                        "batch block height width channel -> (batch block) (channel height width)",
                    )[-1]
                    .numpy()
                    .astype(bool)
                )

                token_decomposition = dt.state_embedding.weight.T
                token_decomposition = cache.apply_ln_to_stack(
                    token_decomposition,
                    layer=layer,
                    pos_slice=-1,
                    has_batch_dim=False,
                    mlp_input=True,
                )
                token_attribution = token_decomposition @ mlp_in
                token_attribution_df = pd.DataFrame(
                    {
                        "Component": embedding_labels,
                        "Activation Difference": token_attribution.detach().numpy(),
                        "Present": state_mask,
                        "Channel": [i.split(",")[0] for i in embedding_labels],
                        "Position": [
                            i.split(",")[1] + "," + i.split(",")[-1]
                            for i in embedding_labels
                        ],
                    }
                )

                # filter using mask
                token_attribution_df = token_attribution_df[
                    token_attribution_df.Present
                ]
                total_activation_contribution_state = token_attribution_df[
                    ["Activation Difference"]
                ].sum()[0]
                st.write(
                    f"Activation Contribution of State Features (cache): {total_activation_contribution_state:.3f}"
                )
                st.write(
                    f"Activation Contribution of Time Embedding: {time_attribution:.3f}"
                )
                st.write(
                    f"Total Activation Contribution (cache): {total_activation_contribution_state + time_attribution:.3f}"
                )

                fig = px.strip(
                    token_attribution_df,
                    x="Channel",
                    y="Activation Difference",
                    color="Channel",
                )
                # fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        with head_tab:
            head_mask = [
                True if re.search(r"H\d+", label) else False
                for label in labels
            ]
            head_attribution = activation_contributions[head_mask]
            n_heads = dt.transformer_config.n_heads
            head_attribution = head_attribution.reshape(-1, n_heads)[
                : (1 + layer)
            ]

            fig = px.imshow(
                head_attribution.detach(),
                color_continuous_midpoint=0,
                color_continuous_scale="RdBu",
                title="Neuron activation by Head",
                labels={"x": "Head", "y": "Layer"},
            )
            st.plotly_chart(fig, use_container_width=True)

        with neuron_tab:
            neuron_attribution = decomp_result[:, 0, -1] @ mlp_in
            # use regex to look for the L {number} N {number} pattern in labels
            neuron_attribution_mask = [
                True if re.search(r"L\d+N\d+", label) else False
                for label in labels
            ]

            neuron_attribution = activation_contributions[
                neuron_attribution_mask
            ]
            neuron_labels = [
                label for label in labels if re.search(r"L\d+N\d+", label)
            ]

            df = pd.DataFrame(
                {
                    "Neuron": neuron_labels,
                    "Activation Difference": neuron_attribution.detach().numpy(),
                }
            )
            df["Layer"] = df["Neuron"].apply(
                lambda x: int(x.split("L")[1].split("N")[0])
            )
            df = df[df["Layer"] < layer]

            if layer > 0:
                layertabs = st.tabs(
                    [
                        "L" + str(layer)
                        for layer in df["Layer"].unique().tolist()
                    ]
                )

                for i, layer in enumerate(df["Layer"].unique().tolist()):
                    with layertabs[i]:
                        fig = px.scatter(
                            df[df["Layer"] == layer],
                            x="Neuron",
                            y="Activation Difference",
                            hover_data=["Layer"],
                            title="Activation Difference From Each Neuron",
                            color="Activation Difference",
                        )
                        # color_continuous_scale="RdBu",
                        # don't label xtick
                        fig.update_xaxes(showticklabels=False)
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No neurons feed into neurons in layer 0.")

        with b:
            st.write(
                f"Total Neuron Activation (cache): {total_neuron_activation[0, -1, neuron].item():.3f}"
            )
            estimate = (
                attribution_df["Activation Difference"].sum()
                + dt.transformer.blocks[layer].mlp.b_in[neuron]
            )
            # estimate = dt.transformer.blocks[layer].mlp.act_fn(estimate).item()
            st.write(f"Total Neuron Activation (estimated) {estimate:.3f}")

        # with debug_tab:

        #     # with aggregate_tab:
        #     activation_sum = (
        #         attribution_df["Activation Difference"].sum().item()
        #     )
        #     st.write("activation_sum", activation_sum)
        #     st.write(
        #         "from heads L0",
        #         attribution_df[attribution_df.Component.str.contains("L0H")][
        #             ["Activation Difference"]
        #         ]
        #         .sum()
        #         .item(),
        #     )
        #     st.write(
        #         "from heads L1",
        #         attribution_df[attribution_df.Component.str.contains("L1H")][
        #             ["Activation Difference"]
        #         ]
        #         .sum()
        #         .item(),
        #     )
        #     st.write(
        #         "from MLP0",
        #         attribution_df[attribution_df.Component.str.contains("L0N")][
        #             ["Activation Difference"]
        #         ]
        #         .sum()
        #         .item(),
        #     )
        #     st.write(
        #         "from MLP1",
        #         attribution_df[attribution_df.Component.str.contains("L1N")][
        #             ["Activation Difference"]
        #         ]
        #         .sum()
        #         .item(),
        #     )
        #     st.write(
        #         "from mlp",
        #         attribution_df[attribution_df.Neuron != "None"][
        #             ["Activation Difference"]
        #         ]
        #         .sum()
        #         .item(),
        #     )
        #     st.write(
        #         "bias: ", dt.transformer.blocks[layer].mlp.b_in[neuron].item()
        #     )
        #     st.write(attribution_df.tail(3))

        #     # check the MLP out
        #     mlp_out = dt.transformer.blocks[layer].mlp.W_out[neuron, :]
        #     st.write(
        #         "expected logit effect, act sum",
        #         attribution_df["Activation Difference"].sum()
        #         * mlp_out
        #         @ logit_dir,
        #     )
        #     st.write(
        #         "expected logit effect, neuron act (no ln)",
        #         total_neuron_activation[0, -1, neuron] * mlp_out @ logit_dir,
        #     )
        #     # st.write("expected logit effect, neuron act (no ln)", total_neuron_activation[0, -1, neuron] * mlp_out ) @ logit_dir)


def plot_decomposition_dot_product(
    cache,
    residual_direction,
    layer,
    mlp_input,
    apply_ln=True,
    accumulated=False,
):
    if accumulated:
        results, labels = cache.accumulated_resid(
            apply_ln=apply_ln,
            return_labels=True,
            layer=layer,
            mlp_input=mlp_input,
            incl_mid=True,
        )
        plot_func = px.line
    else:
        results, labels = cache.decompose_resid(
            apply_ln=apply_ln,
            return_labels=True,
            layer=layer,
            mlp_input=mlp_input,
        )
        plot_func = px.bar

    attribution = results[:, 0, -1] @ residual_direction

    fig = plot_func(
        attribution.detach(),
        labels={"index": "Layer", "value": "Activation"},
    )
    fig.update_layout(
        hovermode="x unified",
        showlegend=False,
        xaxis_tickvals=list(range(len(labels))),
        xaxis_ticktext=labels,
        xaxis_tickangle=45,
    )

    return fig


# from src.visualization import tensor_cosine_similarity_heatmap
# import torch.nn.functional as F
# # Investigate Cache
# def show_cache(dt, cache):

#     with st.expander("Show cache"):

#         head_output_similarity_tab, other_tab = st.tabs(
#             ["Head Output Similarity", "Test"])

#         with head_output_similarity_tab:
#             st.write("Head Output Similarity")


#             result, labels = cache.stack_head_results(
#                 apply_ln=True, return_labels=True, pos_slice = -1
#             )

#             result  = result[:,0].detach()


#             # get cosine similarity matrix
#             # similarities  = F.cosine_similarity(result.unsqueeze(1), result.unsqueeze(0), dim=-1)
#             # st.plotly_chart(px.imshow(similarities))
#             # st.write(similarities.shape)
#             # plot_heatmap(similarities)
#             fig = tensor_cosine_similarity_heatmap(result, labels)
#             st.plotly_chart(fig, use_container_width=True)
def show_cache(dt, cache):
    st.warning("Cache not yet implemented")


def show_residual_stream_projection_onto_component(dt, cache, logit_dir):
    n_layers = dt.transformer_config.n_layers
    n_heads = dt.transformer_config.n_heads
    n_neurons = dt.transformer_config.d_mlp

    heads = []
    for layer in range(n_layers):
        for head in range(n_heads):
            heads.append((layer, head))

    neurons = []
    for layer in range(n_layers):
        for neuron in range(n_neurons):
            neurons.append((layer, neuron))

    decomp_result, labels = cache.get_full_resid_decomposition(
        apply_ln=False,
        return_labels=True,
        expand_neurons=True,
    )  # LN is applied.

    with st.expander("Projection onto Component Output in Residual Stream"):
        # now let's decompose the residual stream

        head_tab, neuron_tab, cluster_tab, help_tab = st.tabs(
            ["Head", "Neuron", "Cluster", "Help"]
        )

        with head_tab:
            head_mask = [
                True if re.search(r"L\d+H\d+", label) else False
                for label in labels
            ]

            head_vectors = decomp_result[head_mask, 0, -1]

            decomp_result, labels = cache.accumulated_resid(
                incl_mid=True,
                apply_ln=False,
                return_labels=True,
            )
            decomp_result = decomp_result[:, 0, -1]
            decomp_norms = t.norm(decomp_result, dim=-1)
            vector_norms = t.norm(head_vectors, dim=-1)
            projection = einsum(
                "layer d_model, head d_model -> layer head",
                decomp_result,
                head_vectors,
            )

            normalize_by = st.selectbox(
                "Normalize by",
                ["Vector Norm", "Component Norm"],
                index=0,
            )
            if normalize_by == "Vector Norm":
                projection = projection / vector_norms.unsqueeze(0)
            elif normalize_by == "Component Norm":
                projection = projection / decomp_norms.T.unsqueeze(-1)

            df = pd.DataFrame(
                projection.detach().numpy(),
                columns=[f"L{layer}H{head}" for layer, head in heads],
            )
            fig = px.line(
                df,
                labels={"index": "Layer", "value": "Activation"},
            )
            fig.update_layout(
                showlegend=True,
                xaxis_tickvals=list(range(len(labels))),
                xaxis_ticktext=labels,
                xaxis_tickangle=45,
            )
            st.plotly_chart(fig, use_container_width=True)

        with neuron_tab:
            decomp_result, labels = cache.get_full_resid_decomposition(
                apply_ln=False,
                return_labels=True,
                expand_neurons=True,
            )  # LN is applied.

            neurons_selected = st.multiselect(
                "Select neurons",
                options=neurons,
                default=[],
                format_func=lambda x: f"L{x[0]}N{x[1]}",
                key="neuron_proj",
            )
            # format selected neurons
            neurons_selected = [
                f"L{layer}N{neuron}" for layer, neuron in neurons_selected
            ]

            neuron_mask = [
                True if label in neurons_selected else False
                for label in labels
            ]

            # get only those neurons in the mask from teh list of labels
            neuron_labels = [
                label for label in labels if label in neurons_selected
            ]

            neuron_vectors = decomp_result[neuron_mask, 0, -1]

            decomp_result, labels = cache.accumulated_resid(
                incl_mid=True,
                apply_ln=False,
                return_labels=True,
            )
            decomp_result = decomp_result[:, 0, -1]
            decomp_norms = t.norm(decomp_result, dim=-1)
            vector_norms = t.norm(neuron_vectors, dim=-1)

            projection = einsum(
                "layer d_model, neuron d_model -> layer neuron",
                decomp_result,
                neuron_vectors,
            )

            normalize_by = st.selectbox(
                "Normalize by",
                ["Vector Norm", "Component Norm"],
                index=0,
                key="asfdsbgfb",
            )
            if normalize_by == "Vector Norm":
                projection = projection / vector_norms.unsqueeze(0)
            elif normalize_by == "Component Norm":
                projection = projection / decomp_norms.T.unsqueeze(-1)

            df = pd.DataFrame(
                projection.detach().numpy(), columns=neuron_labels
            )
            fig = px.line(
                df,
                labels={"index": "Layer", "value": "Activation"},
            )
            fig.update_layout(
                showlegend=True,
                xaxis_tickvals=list(range(len(labels))),
                xaxis_ticktext=labels,
                xaxis_tickangle=45,
            )
            st.plotly_chart(fig, use_container_width=True)

        with cluster_tab:
            decomp_result, labels = cache.get_full_resid_decomposition(
                apply_ln=False,
                return_labels=True,
                expand_neurons=False,
            )  # LN is applied.

            # get final position
            decomp_result = decomp_result[:, 0, -1].detach()

            from src.visualization import get_cosine_sim_df

            df = get_cosine_sim_df(decomp_result)
            df.columns = labels
            df.index = labels

            from src.streamlit_app.visualizations import plot_heatmap

            fig = plot_heatmap(df)
            st.plotly_chart(fig, use_container_width=True)

        with help_tab:
            st.write(
                """
                This is an experimental feature where we look at decompositions of the residual stream and project them onto the output of a head or neuron. 

                This should tell us information is ever getting deleted from the residual stream after being added by a component. 

                The head tab will show how the outputs of heads increase/decrease in the residual stream and the neuron tab does the same. 

                To see how any given component (neuron or head) aligns with each other, I'll do one of those clustergrams I love in the clustergram tab.

                """
            )


# RTG Scan Utilities
def rtg_scan_configuration_ui(dt):
    cola, colb = st.columns(2)
    min_value = -1
    max_value = 1

    with cola:
        rtg_range = st.slider(
            "RTG Range",
            min_value=min_value,
            max_value=max_value,
            value=(0, 1),
            step=1,
        )
        min_rtg = rtg_range[0]
        max_rtg = rtg_range[1]

    with colb:
        max_len = get_max_len_from_model_type(dt.model_type, dt.n_ctx)
        timesteps = st.session_state.timesteps[:, -max_len:]

        if st.checkbox("add timestep noise"):
            # we want to add random integers in the range of a slider to the the timestep, the min/max on slider should be the max timesteps
            if timesteps.max().item() > 0:
                timestep_noise = st.slider(
                    "Timestep Noise",
                    min_value=1.0,
                    max_value=timesteps.max().item(),
                    value=1.0,
                    step=1.0,
                )
                timesteps = timesteps + t.randint(
                    low=int(-1 * timestep_noise),
                    high=int(timestep_noise),
                    size=timesteps.shape,
                    device=timesteps.device,
                )
            else:
                st.info(
                    "Timestep noise only works when we have more than one timestep."
                )
    return min_rtg, max_rtg, max_len, timesteps


def prepare_rtg_scan_tokens(dt, min_rtg, max_rtg, max_len, timesteps):
    batch_size = RTG_SCAN_BATCH_SIZE
    obs = st.session_state.obs[:, -max_len:].repeat(batch_size, 1, 1, 1, 1)
    actions = st.session_state.a[:, -max_len:].repeat(batch_size, 1, 1)
    rtg = st.session_state.rtg[:, -max_len:].repeat(batch_size, 1, 1)
    timesteps = st.session_state.timesteps[:, -max_len:].repeat(
        batch_size, 1, 1
    )
    rtg = (
        t.linspace(min_rtg, max_rtg, batch_size)
        .unsqueeze(-1)
        .unsqueeze(-1)
        .repeat(1, obs.shape[1], 1)
    )

    # duplicate truncation code
    obs = obs[:, -max_len:] if obs.shape[1] > max_len else obs
    if actions is not None:
        actions = (
            actions[:, -(obs.shape[1] - 1) :]
            if (actions.shape[1] > 1 and max_len > 1)
            else None
        )
    timesteps = (
        timesteps[:, -max_len:] if timesteps.shape[1] > max_len else timesteps
    )
    rtg = rtg[:, -max_len:] if rtg.shape[1] > max_len else rtg

    if dt.time_embedding_type == "linear":
        timesteps = timesteps.to(t.float32)
    else:
        timesteps = timesteps.to(t.long)

    # print out shape of each
    tokens = dt.to_tokens(obs, actions, rtg, timesteps)
    return rtg, tokens


def show_rtg_scan(dt, logit_dir):
    with st.expander("Scan Reward-to-Go and Show Residual Contributions"):
        min_rtg, max_rtg, max_len, timesteps = rtg_scan_configuration_ui(dt)
        rtg, tokens = prepare_rtg_scan_tokens(
            dt, min_rtg, max_rtg, max_len, timesteps
        )
        x, cache = dt.transformer.run_with_cache(
            tokens, remove_batch_dim=False
        )
        _, action_preds, _ = dt.get_logits(
            x,
            batch_size=RTG_SCAN_BATCH_SIZE,
            seq_length=max_len,
            no_actions=False,
        )

        (
            logit_tab,
            decomp_tab,
            neuron_tab,
            attention_patterns_by_rtg_tab,
        ) = st.tabs(
            [
                "Logit Scan",
                "Decomposition",
                "Neurons",
                "Attention Patterns By RTG",
            ]
        )

        with logit_tab:
            fig = plot_logit_scan(rtg, action_preds)
            st.plotly_chart(fig, use_container_width=True)

        with decomp_tab:
            decomp_level, cluster, normalize = decomp_configuration_ui()
            df = get_decomp_scan(
                rtg, cache, logit_dir, decomp_level, normalize=normalize
            )
            fig = plot_decomp_scan_line(df)
            st.plotly_chart(fig, use_container_width=True)
            fig2 = plot_decomp_scan_corr(df, cluster)
            st.plotly_chart(fig2, use_container_width=True)

        # with neuron_tab:
        #     st.write(
        #         "We may want to look for families of equivarient neurons."
        #     )

        #     layer = st.selectbox(
        #         "Layer",
        #         options=range(dt.transformer_config.n_layers),
        #         index=dt.transformer_config.n_layers - 1,
        #     )

        #     weights = dt.transformer.blocks[layer].mlp.W_in
        #     activations = cache[f"blocks.{layer}.mlp.hook_pre"][:, -1]

        #     neuron_activations = einsum(
        #         "d_model d_mlp, batch d_model -> batch d_mlp",
        #         weights,
        #         activations,
        #     )

        #     # st.write(neuron_activations.shape)

        #     # cast it to df with columns for each neuron
        #     df = pd.DataFrame(
        #         neuron_activations.detach().cpu().numpy(),
        #         index=rtg[:, 0, 0].detach().cpu().numpy(),
        #         columns=[
        #             f"L{layer}N{i}" for i in range(neuron_activations.shape[1])
        #         ],
        #     )

        #     fig = plot_decomp_scan_line(df, title="")
        #     fig.update_traces(opacity=0.3)
        #     # hide legend
        #     fig.update_layout(showlegend=False)
        #     st.plotly_chart(fig, use_container_width=True)

        # fig2 = plot_decomp_scan_corr(df, True)
        # st.plotly_chart(fig2, use_container_width=True)

        # fig = px.line(
        #     df,
        #     x=df.index,
        #     y=df.columns,
        #     title=f"Layer {layer} Neuron Activations",
        # )

        # # reduce opacity on lines

        # # hide legend
        # fig.update_layout(showlegend=False)
        # st.plotly_chart(fig, use_container_width=True)

        # fig2 = plot_decomp_scan_corr(df, cluster)

        with attention_patterns_by_rtg_tab:
            step_vals = list(
                np.array(
                    [
                        [f"R{i+1}", f"S{i+1}", f"A{i+1}"]
                        for i in range(1 + dt.transformer_config.n_ctx // 3)
                    ]
                ).flatten()
            )[:-1]

            a, b = st.columns(2)
            with a:
                score_or_pattern = st.radio(
                    "Show Attention Score or Pattern",
                    ["Value-Weighted Pattern", "Pattern", "Scores"],
                    index=1,
                    key="score_or_pattern",
                )

            if score_or_pattern == "Scores":
                attention_patterns = torch.stack(
                    [
                        cache[f"blocks.{layer}.attn.hook_attn_scores"]
                        for layer in range(dt.transformer_config.n_layers)
                    ]
                )
            else:
                attention_patterns = torch.stack(
                    [
                        cache[f"blocks.{layer}.attn.hook_pattern"]
                        for layer in range(dt.transformer_config.n_layers)
                    ]
                )

            if score_or_pattern == "Value-Weighted Pattern":
                values = torch.stack(
                    [
                        cache[f"blocks.{layer}.attn.hook_v"]
                        for layer in range(dt.transformer_config.n_layers)
                    ]
                )
                value_norm = torch.norm(values, dim=-1)
                value_norm = value_norm.permute(0, 1, 3, 2)
                st.write("value norm", value_norm.shape)
                st.write("attention patterns", attention_patterns.shape)
                attention_patterns = attention_patterns * value_norm.unsqueeze(
                    -1
                )

            df = tensor_to_long_data_frame(
                attention_patterns,
                dimension_names=["Layer", "RTG", "Head", "Query", "Key"],
            )

            df["RTG"] = df["RTG"].map(lambda x: rtg[x, 0, 0])
            df["Layer"] = df["Layer"].map(lambda x: f"L{x}")
            df["Head"] = df["Layer"] + df["Head"].map(lambda x: f"H{x}")
            df["Key"] = df["Key"].map(lambda x: step_vals[x])
            df["Query"] = df["Query"].map(lambda x: step_vals[x])

            # check that query and key are correct. Sum over key should be 1.
            # st.write(df.groupby(['Layer', 'Head', 'RTG', 'Query']).sum())

            # filter for the last query.
            df = df[df["Query"] == "S9"]
            # drop the Query Column
            df = df.drop(columns=["Query"])

            with b:
                selected_heads = st.multiselect(
                    "Select Heads",
                    options=df["Head"].unique().tolist(),
                    default=["L0H0"],
                    key="head_select",
                )

            head_tabs = st.tabs(
                selected_heads,
            )

            for h in selected_heads:
                with head_tabs[selected_heads.index(h)]:
                    fig = px.line(
                        df[df["Head"] == h], x="RTG", y="Score", color="Key"
                    )

                    st.plotly_chart(fig, use_container_width=True)


# Observation View
def show_observation_view(dt, tokens, logit_dir):
    last_obs = st.session_state.obs[0][-1]

    last_obs_reshaped = rearrange(last_obs, "h w c -> c h w")

    height, width, n_channels = dt.environment_config.observation_space[
        "image"
    ].shape

    weights = dt.state_embedding.weight.detach().cpu()

    weights_reshaped = rearrange(
        weights, "d (c h w) -> c d h w", c=n_channels, h=height, w=width
    )

    embeddings = einsum(
        "c d h w, c h w -> c d",
        weights_reshaped,
        last_obs_reshaped.to(t.float32),
    )

    weight_projections = einsum(
        "d, c d h w -> c h w", logit_dir, weights_reshaped
    )

    activation_projection = weight_projections * last_obs_reshaped

    timesteps = st.session_state.timesteps[0][-1]
    if dt.time_embedding_type == "linear":
        timesteps = timesteps.to(t.float32)
    else:
        timesteps = timesteps.to(t.long)

    time_embedding = dt.time_embedding(timesteps)

    with st.expander("Show observation view"):
        st.subheader("Observation View")
        if n_channels == 3:

            def format_func(x):
                return three_channel_schema[x]

        else:
            format_func = twenty_idx_format_func

        selected_channels = st.multiselect(
            "Select Observation Channels",
            options=list(range(n_channels)),
            format_func=format_func,
            key="channels obs",
            default=[0, 1, 2],
        )
        n_selected_channels = len(selected_channels)

        check_columns = st.columns(4)
        with check_columns[0]:
            contributions_check = st.checkbox("Show contributions", value=True)
        with check_columns[1]:
            input_channel_check = st.checkbox(
                "Show input channels", value=True
            )
        with check_columns[2]:
            weight_proj_check = st.checkbox(
                "Show channel weight proj onto logit dir", value=True
            )
        with check_columns[3]:
            activ_proj_check = st.checkbox(
                "Show channel activation proj onto logit dir", value=True
            )

        if contributions_check:
            contributions = {
                format_func(i): (embeddings[i] @ logit_dir).item()
                for i in selected_channels
            }

            if dt.time_embedding_type == "linear":
                time_contribution = (time_embedding @ logit_dir).item()
            else:
                time_contribution = (time_embedding[0] @ logit_dir).item()

            token_contribution = (tokens[0][-1] @ logit_dir).item()

            contributions = {
                **contributions,
                "time": time_contribution,
                "token": token_contribution,
            }

            fig = px.bar(
                contributions.items(),
                x=0,
                y=1,
                labels={"0": "Channel", "1": "Contribution"},
                text=1,
            )

            # add the value to the bar
            fig.update_traces(texttemplate="%{text:.3f}", textposition="auto")
            fig.update_layout(uniformtext_minsize=8, uniformtext_mode="hide")
            fig.update_yaxes(range=[-8, 8])
            st.plotly_chart(fig, use_container_width=True)

        if input_channel_check:
            columns = st.columns(n_selected_channels)
            for i, channel in enumerate(selected_channels):
                with columns[i]:
                    st.write(format_func(channel))
                    fancy_imshow(last_obs_reshaped[channel].detach().numpy().T)

                    if n_channels == 3:
                        if i == 0:
                            st.write(IDX_TO_OBJECT)
                        elif i == 1:
                            st.write(IDX_TO_COLOR)
                        else:
                            st.write(IDX_TO_STATE)

        if weight_proj_check:
            columns = st.columns(n_selected_channels)
            for i, channel in enumerate(selected_channels):
                with columns[i]:
                    st.write(format_func(channel))
                    fancy_imshow(
                        weight_projections[channel].detach().numpy().T
                    )
                    fancy_histogram(
                        weight_projections[channel].detach().numpy().flatten()
                    )

        if activ_proj_check:
            columns = st.columns(n_selected_channels)
            for i, channel in enumerate(selected_channels):
                with columns[i]:
                    st.write(format_func(channel))
                    fancy_imshow(
                        activation_projection[channel].detach().numpy().T
                    )
                    fancy_histogram(
                        activation_projection[channel]
                        .detach()
                        .numpy()
                        .flatten()
                    )


def project_weights_onto_dir(weights, dir):
    return t.einsum(
        "d, d h w -> h w", dir, weights.reshape(128, 7, 7)
    ).detach()


# Gated MLP
def show_gated_mlp_dynamic(dt, cache):
    with st.expander("Gated MLP"):
        n_layers = dt.transformer_config.n_layers
        n_heads = dt.transformer_config.n_heads

        # want to start by visualizing the mlp activations
        # stack acts/eights
        a_pre = t.stack(
            [
                cache[f"blocks.{layer}.mlp.hook_pre"]
                for layer in range(n_layers)
            ]
        )
        a_pre = dt.transformer.blocks[0].mlp.act_fn(a_pre)
        W_Gate = t.stack([block.mlp.W_gate for block in dt.transformer.blocks])
        W_in = t.stack([block.mlp.W_in for block in dt.transformer.blocks])
        W_0 = torch.stack([block.mlp.W_out for block in dt.transformer.blocks])

        # we know two things
        # 1. what is being gated.
        # 2. for each thing being gated, what the output is.
        # this means we can look at the out congruence by gating!
        # it's a map to the meaning of the each vector in W_O!

        gating_tab, congruence_tab, gating_by_conguence_tab = st.tabs(
            ["Gating", "Conguence", "Gating by Congruence"]
        )

        with gating_tab:
            df = tensor_to_long_data_frame(
                a_pre[:, 0, -1], ["Layer", "Neuron"]
            )
            df["Neuron"] = df["Layer"].map(lambda x: f"L{x}") + df[
                "Neuron"
            ].map(lambda x: f"N{x}")
            df["Layer"] = df["Layer"].astype("category")
            fig = px.scatter(
                df,
                x=df.index,
                y="Score",
                color="Layer",
                hover_data=["Layer", "Neuron"],
            )

            st.plotly_chart(fig, use_container_width=True)

        with congruence_tab:
            # now I want congruence with embed simultaneously.
            W_0_congruence = W_0 @ dt.action_predictor.weight.T
            st.write(W_0_congruence.shape)
            df_congruence = tensor_to_long_data_frame(
                W_0_congruence, ["Layer", "Neuron", "Action"]
            )
            # ensure action is interpreted as a categorical variable
            df_congruence["Action"] = df_congruence["Action"].map(
                IDX_TO_ACTION
            )

            # sort by action
            df_congruence = df_congruence.sort_values(by="Layer")
            fig = px.scatter(
                df_congruence,
                x=df_congruence.index,
                y="Score",
                color="Action",
                hover_data=["Layer", "Action", "Neuron", "Score"],
            )

            # update x axis to hide the tick labels, and remove the label
            fig.update_xaxes(showticklabels=False, title=None)

            st.plotly_chart(fig, use_container_width=True)

        with gating_by_conguence_tab:
            df_grouped = df_congruence.groupby(["Layer", "Neuron"]).apply(
                lambda x: x.loc[x["Score"].idxmax()]
            )

            # Reset the index of the grouped DataFrame
            df_grouped = df_grouped.reset_index(drop=True)

            # Rename the "Action" column to "Highest_Score_Action"
            df_grouped.rename(
                columns={"Action": "CongruentAction"}, inplace=True
            )
            # df_grouped["Layer"] = df_congruence['Layer'].astype('category')
            df_grouped["Neuron"] = df_grouped["Layer"].map(
                lambda x: f"L{x}"
            ) + df_grouped["Neuron"].map(lambda x: f"N{x}")
            df = df.merge(df_grouped, on=["Layer", "Neuron"])
            df.rename(
                columns={
                    "Score_x": "GatingEffect",
                    "Score_y": "MaxCongruence",
                },
                inplace=True,
            )

            fig = px.scatter(
                df,
                x=df.index,
                y="GatingEffect",
                color="CongruentAction",
                # opacity="MaxCongruence",
                hover_data=["Layer", "Neuron", "CongruentAction"],
            )
            # update x axis to hide the tick labels, and remove the label
            fig.update_xaxes(showticklabels=False, title=None)
            st.plotly_chart(fig, use_container_width=True)

            fig = px.box(df, color="CongruentAction", y="GatingEffect")
            st.plotly_chart(fig, use_container_width=True)
