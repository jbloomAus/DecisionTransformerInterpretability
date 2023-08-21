import re

import pandas as pd
import plotly.express as px
import streamlit as st
import torch
import torch as t
from einops import rearrange
from fancy_einsum import einsum
from minigrid.core.constants import IDX_TO_COLOR, IDX_TO_OBJECT

from src.decision_transformer.utils import get_max_len_from_model_type

from .analysis import get_residual_decomp
from .components import (
    decomp_configuration_ui,
    get_decomp_scan,
    plot_decomp_scan_corr,
    plot_decomp_scan_line,
)
from .constants import (
    IDX_TO_ACTION,
    IDX_TO_STATE,
    three_channel_schema,
    twenty_idx_format_func,
)
from .utils import fancy_histogram, fancy_imshow, tensor_to_long_data_frame
from .visualizations import (
    plot_attention_pattern_single,
    plot_heatmap,
    plot_logit_diff,
    plot_logit_scan,
)

RTG_SCAN_BATCH_SIZE = 128


def show_attention_pattern(dt, cache):
    with st.expander("Attention Pattern at at current Reward-to-Go"):
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

        visualize_attention_pattern(dt, cache)


def visualize_attention_pattern(dt, cache):
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
        )
        score_or_softmax = st.selectbox(
            "Score or Softmax",
            options=["Score", "Softmax"],
            index=1,
        )
        softmax = score_or_softmax == "Softmax"

    with b:
        scale_by_value = st.selectbox(
            "Scale by value",
            options=[True, False],
            index=0,
        )

        if score_or_softmax != "Value Weighted Softmax":
            method = st.selectbox(
                "Select plotting method",
                options=["Plotly", "CircuitsVis"],
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
    )


# Attribution / Logit Lens
def show_attributions(dt, cache, logit_dir):
    with st.expander("Show Attributions"):
        layertab, componenttab, headtab, neurontab = st.tabs(
            ["Layer", "Component", "Head", "Neuron"]
        )

        with layertab:
            tab1, tab2 = st.tabs(["Layerwise", "Accumulated"])
            with tab1:
                results, labels = cache.decompose_resid(
                    apply_ln=True, return_labels=True
                )
                attribution = results[:, 0, -1] @ logit_dir
                fig = px.line(
                    attribution.detach(),
                    title="Logit Difference From Residual Stream",
                    labels={"index": "Layer", "value": "Logit Difference"},
                )
                fig.update_layout(
                    hovermode="x unified",
                    showlegend=False,
                    xaxis_tickvals=list(range(len(labels))),
                    xaxis_ticktext=labels,
                    xaxis_tickangle=45,
                )
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                results, labels = cache.accumulated_resid(
                    apply_ln=True, return_labels=True
                )
                attribution = results[:, 0, -1] @ logit_dir
                fig = px.line(
                    attribution.detach(),
                    title="Logit Difference From Accumulated Residual Stream",
                    labels={"index": "Layer", "value": "Logit Difference"},
                )
                fig.update_layout(
                    hovermode="x unified",
                    showlegend=False,
                    xaxis_tickvals=list(range(len(labels))),
                    xaxis_ticktext=labels,
                    xaxis_tickangle=45,
                )
                st.plotly_chart(fig, use_container_width=True)

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
            st.plotly_chart(fig, use_container_width=True)

            st.write(
                f"Top 5 Heads: {', '.join([f'{labels[k.indices[i]]}: {round(k.values[i].item(), 3)}' for i in range(5)])}"
            )

        with neurontab:
            by_neuron_tab, into_neuron_tab = st.tabs(
                ["Attribution by Neuron", "Attribution into Neuron"]
            )
            result, labels = cache.get_full_resid_decomposition(
                apply_ln=True, return_labels=True, expand_neurons=True
            )
            with by_neuron_tab:
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
                            y="Logit Difference",
                            hover_data=["Layer"],
                            title="Logit Difference From Each Neuron",
                            color="Logit Difference",
                        )
                        # color_continuous_scale="RdBu",
                        # don't label xtick
                        fig.update_xaxes(showticklabels=False)
                        st.plotly_chart(fig, use_container_width=True)

            with into_neuron_tab:
                a, b = st.columns(2)
                with a:
                    neuron_text = st.text_input(
                        f"Type the neuron you want", "L0N0"
                    )
                # validate neuron
                if not re.search(r"L\d+N\d+", neuron_text):
                    st.error("Neuron must be in the format L{number}N{number}")
                    return
                # get the neuron index, and the layer
                neuron = int(neuron_text.split("N")[1])
                layer = int(neuron_text.split("L")[1].split("N")[0])

                with b:
                    # total_neuron_activaton = cache["hook_ml"]
                    # st.write(cache.keys())
                    # st.write(f"blocks.{layer}.mlp.hook_mid")
                    total_neuron_activaton = cache[
                        f"blocks.{layer}.mlp.hook_post"
                    ]
                    st.write(total_neuron_activaton[0, -1, neuron])

                # mlp_in = dt.transformer.W_in[layer, neuron, :]
                mlp_in = dt.transformer.W_in[layer, :, neuron]

                attribution = result[:, 0, -1] @ mlp_in

                head_mask = [
                    True if re.search(r"H\d+", label) else False
                    for label in labels
                ]
                head_attribution = attribution[head_mask]

                n_heads = dt.transformer_config.n_heads
                head_attribution = head_attribution.reshape(-1, n_heads)[
                    : (1 + layer)
                ]
                st.write(head_attribution)
                fig = px.imshow(
                    head_attribution.detach(),
                    color_continuous_midpoint=0,
                    color_continuous_scale="RdBu",
                    title="Neuron activation by Head",
                    labels={"x": "Head", "y": "Layer"},
                )
                st.plotly_chart(fig, use_container_width=True)

                neuron_attribution = result[:, 0, -1] @ mlp_in
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

    return


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
            value=(-1, 1),
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

        logit_tab, decomp_tab, neuron_tab = st.tabs(
            ["Logit Scan", "Decomposition", "Neurons"]
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

        with neuron_tab:
            st.write(
                "We may want to look for families of equivarient neurons."
            )

            layer = st.selectbox(
                "Layer",
                options=range(dt.transformer_config.n_layers),
                index=dt.transformer_config.n_layers - 1,
            )

            weights = dt.transformer.blocks[layer].mlp.W_in
            activations = cache[f"blocks.{layer}.mlp.hook_pre"][:, -1]

            neuron_activations = einsum(
                "d_model d_mlp, batch d_model -> batch d_mlp",
                weights,
                activations,
            )

            # st.write(neuron_activations.shape)

            # cast it to df with columns for each neuron
            df = pd.DataFrame(
                neuron_activations.detach().cpu().numpy(),
                index=rtg[:, 0, 0].detach().cpu().numpy(),
                columns=[
                    f"L{layer}N{i}" for i in range(neuron_activations.shape[1])
                ],
            )

            fig = plot_decomp_scan_line(df, title="")
            fig.update_traces(opacity=0.3)
            # hide legend
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

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


# Observation View
def render_observation_view(dt, tokens, logit_dir):
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
