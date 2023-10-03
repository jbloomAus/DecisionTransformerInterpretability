from functools import partial
from typing import Callable, Dict

import re
import pandas as pd
import plotly.express as px
import streamlit as st
import torch
from torch import Tensor
from torch.nn import Module
from torchtyping import TensorType as TT
from tqdm import tqdm
from transformer_lens.hook_points import HookPoint

from src.visualization import (
    get_rendered_obs,
    get_rendered_obss,
    render_minigrid_observation,
    render_minigrid_observations,
)

from .visualizations import plot_logit_scan

from .analysis import get_residual_decomp
from .constants import IDX_TO_ACTION, IDX_TO_OBJECT
from .dynamic_analysis_components import (
    show_attention_pattern,
    show_logit_lens,
)
from .environment import (
    get_action_preds_from_app_state,
    get_action_preds_from_tokens,
    get_modified_tokens_from_app_state,
    get_tokens_from_app_state,
)

from .dynamic_analysis_components import visualize_attention_pattern

ACTION_TO_IDX = {v: k for k, v in IDX_TO_ACTION.items()}
OBJECT_TO_IDX = {v: k for k, v in IDX_TO_OBJECT.items()}

from src.patch_transformer_lens import patching

from .visualizations import (
    plot_action_preds,
    plot_logit_diff,
    plot_single_residual_stream_contributions_comparison,
)

from .components import (
    decomp_configuration_ui,
    get_decomp_scan,
    plot_decomp_scan_line,
    plot_decomp_scan_corr,
)

BATCH_SIZE = 128


# Ablation
def show_ablation(dt, logit_dir, original_cache):
    with st.expander("Ablation Experiment"):
        # make a streamlit form for choosing a component to ablates
        n_layers = dt.transformer_config.n_layers
        n_heads = dt.transformer_config.n_heads

        # make a list of all heads in all layers
        heads = []
        for layer in range(n_layers):
            for head in range(n_heads):
                heads.append((layer, head))

        heads_to_ablate = st.multiselect(
            "Select heads to ablate",
            heads,
            default=[],
            format_func=lambda x: f"L{x[0]}H{x[1]}",
        )

        mlps_to_ablate = st.multiselect(
            "Select MLPs to ablate",
            list(range(n_layers)),
            default=[],
            format_func=lambda x: f"MLP{x}",
        )

        ablate_to_mean = st.checkbox("Ablate to mean", value=True)

        layers_in_heads_to_ablate = set(
            [layer for layer, _ in heads_to_ablate]
        )

        for layer in layers_in_heads_to_ablate:
            heads_in_layer_to_ablate = [
                head for l, head in heads_to_ablate if l == layer
            ]

            ablation_func = get_ablation_function(
                ablate_to_mean, heads_in_layer_to_ablate
            )
            dt.transformer.blocks[layer].attn.hook_z.add_hook(ablation_func)
        for layer in mlps_to_ablate:
            ablation_func = get_ablation_function(
                ablate_to_mean, layer, component="MLP"
            )
            dt.transformer.blocks[layer].hook_mlp_out.add_hook(ablation_func)

        (
            corrupt_action_preds,
            corrupt_x,
            cache,
            tokens,
        ) = get_action_preds_from_app_state(dt)
        dt.transformer.reset_hooks()

        clean_tokens = get_tokens_from_app_state(dt, previous_step=False)
        clean_preds, clean_x, _, _ = get_action_preds_from_tokens(
            dt, clean_tokens
        )

        clean_logit_dif = clean_x[0, -1] @ logit_dir
        corrupted_logit_dif = corrupt_x[0, -1] @ logit_dir

        st.write(
            "Clean Logit Diff: ",
            f"{clean_logit_dif.item():.3f}",
            " Corrupted Logit Diff: ",
            f"{corrupted_logit_dif.item():.3f}",
        )

        prediction_tab, contribution_tab, attention_tab, neuron_tab = st.tabs(
            [
                "Prediction",
                "Attribution",
                "Attention Tab",
                "Neuron Activations",
            ]
        )

        with prediction_tab:
            plot_action_preds(corrupt_action_preds)

        with contribution_tab:
            original_residual_decomp = get_residual_decomp(
                dt, original_cache, logit_dir
            )
            ablation_residual_decomp = get_residual_decomp(
                dt, cache, logit_dir
            )
            plot_single_residual_stream_contributions_comparison(
                original_residual_decomp, ablation_residual_decomp
            )

            heads = dt.transformer_config.n_heads

            result, labels = original_cache.stack_head_results(
                apply_ln=True, return_labels=True
            )

            original_attribution = result[:, 0, -1] @ logit_dir
            original_attribution = original_attribution.reshape(-1, heads)

            result, labels = cache.stack_head_results(
                apply_ln=True, return_labels=True
            )

            attribution = result[:, 0, -1] @ logit_dir
            attribution = attribution.reshape(-1, heads)

            fig = px.imshow(
                attribution.detach() - original_attribution.detach(),
                color_continuous_midpoint=0,
                color_continuous_scale="RdBu",
                title="Change in Logit Difference From Each Head",
                labels={"x": "Head", "y": "Layer"},
            )
            st.plotly_chart(fig, use_container_width=True)

        with attention_tab:
            st.write("Corrupted Cache Attention Patterns")
            visualize_attention_pattern(dt, original_cache)

        with neuron_tab:
            result, labels = original_cache.get_full_resid_decomposition(
                apply_ln=True, return_labels=True, expand_neurons=True
            )
            original_attribution = result[:, 0, -1] @ logit_dir

            result, labels = cache.get_full_resid_decomposition(
                apply_ln=True, return_labels=True, expand_neurons=True
            )
            attribution = result[:, 0, -1] @ logit_dir

            # # use regex to look for the L {number} N {number} pattern in labels
            neuron_attribution_mask = [
                True if re.search(r"L\d+N\d+", label) else False
                for label in labels
            ]

            original_attribution = original_attribution[
                neuron_attribution_mask
            ]
            attribution = attribution[neuron_attribution_mask]

            labels = [
                label for label in labels if re.search(r"L\d+N\d+", label)
            ]

            df = pd.DataFrame(
                {
                    "Neuron": labels,
                    "Original Logit Difference": original_attribution.detach().numpy(),
                    "Ablation Logit Difference": attribution.detach().numpy(),
                    "Change in Logit Difference": (
                        attribution - original_attribution
                    )
                    .detach()
                    .numpy(),
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
                        x="Original Logit Difference",
                        y="Ablation Logit Difference",
                        hover_data=["Layer", "Neuron"],
                        title="Logit Difference From Each Neuron",
                        color="Change in Logit Difference",
                        color_continuous_midpoint=0,
                        color_continuous_scale="RdBu",
                        animation_frame="Layer",
                    )
                    # color_continuous_scale="RdBu",
                    # don't label xtick
                    # fig.update_xaxes(showticklabels=False)
                    # force consistent axis
                    max_val = (
                        max(
                            abs(df["Original Logit Difference"].min()),
                            abs(df["Original Logit Difference"].max()),
                            abs(df["Ablation Logit Difference"].min()),
                            abs(df["Ablation Logit Difference"].max()),
                        )
                        + 0.1
                    )
                    fig.update_layout(
                        xaxis_range=[-max_val, max_val],
                        yaxis_range=[-max_val, max_val],
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # fig = px.scatter(
                    #     df[df["Layer"] == layer],
                    #     x="Neuron",
                    #     y="Change in Logit Difference",
                    #     hover_data=["Layer"],
                    #     title="Change in Logit Difference From Each Neuron",
                    #     color="Change in Logit Difference",
                    #     color_continuous_midpoint=0,
                    #     color_continuous_scale="RdBu",
                    # )
                    # # color_continuous_scale="RdBu",
                    # # don't label xtick
                    # fig.update_xaxes(showticklabels=False)
                    # st.plotly_chart(fig, use_container_width=True)


def get_ablation_function(ablate_to_mean, head_to_ablate, component="HEAD"):
    def head_ablation_hook(
        value: TT["batch", "pos", "head_index", "d_head"],  # noqa: F821
        hook: HookPoint,
    ) -> TT["batch", "pos", "head_index", "d_head"]:  # noqa: F821
        print(f"Shape of the value tensor: {value.shape}")

        if ablate_to_mean:
            value[:, :, head_to_ablate, :] = value[
                :, :, head_to_ablate, :
            ].mean(dim=-1, keepdim=True)
        else:
            value[:, :, head_to_ablate, :] = 0.0
        return value

    def mlp_ablation_hook(
        value: TT["batch", "pos", "d_model"], hook: HookPoint  # noqa: F821
    ) -> TT["batch", "pos", "d_model"]:  # noqa: F821
        print(f"Shape of the value tensor: {value.shape}")

        if ablate_to_mean:
            value[:, :, :] = value[:, :, :].mean(dim=2, keepdim=True)
        else:
            value[:, :, :] = 0  # ablate all but the last token
        return value

    if component == "HEAD":
        return head_ablation_hook
    elif component == "MLP":
        return mlp_ablation_hook


# Activation Patching
def show_activation_patching(dt, logit_dir, original_cache):
    token_labels = st.session_state.labels
    with st.expander("Activation Patching"):
        corrupted_tokens, noise_or_denoise = get_corrupted_tokens_component(
            dt, key="act_patch"
        )
        # look at current state and do a forward pass
        clean_tokens = get_tokens_from_app_state(dt, previous_step=False)
        clean_preds, clean_x, clean_cache, _ = get_action_preds_from_tokens(
            dt, clean_tokens
        )
        (
            corrupt_preds,
            corrupt_x,
            corrupt_cache,
            _,
        ) = get_action_preds_from_tokens(dt, corrupted_tokens)

        if st.checkbox("show corrupted action predictions"):
            plot_action_preds(corrupt_preds)

        if st.checkbox("Show corrupted analyses (slightly expensive)", key="corrupt_analysis"):
            corrupt_attention_pattern_tab, corrupt_logit_lens_tab = st.tabs(["Attention Pattern", "Logit Lens"])
            with corrupt_attention_pattern_tab:
                show_attention_pattern(dt, corrupt_cache, key="corrupt-")
            with corrupt_logit_lens_tab:
                show_logit_lens(dt, corrupt_cache, logit_dir, key="corrupt-")

        clean_logit_dif = clean_x[0, -1] @ logit_dir
        corrupted_logit_dif = corrupt_x[0, -1] @ logit_dir

        st.write(
            "Clean Logit Diff: ",
            f"{clean_logit_dif.item():.3f}," + " Corrupted Logit Diff: ",
            f"{corrupted_logit_dif.item():.3f}",
        )

        if noise_or_denoise.lower() == "noise":
            # we need to flip tokens because denoising works by taking the forward pass of the corrupted tokens
            # and patching in the "correct output" from the clean tokens.
            corrupted_tokens, clean_tokens = clean_tokens, corrupted_tokens
            corrupt_preds, clean_preds = clean_preds, corrupt_preds
            corrupt_x, clean_x = clean_x, corrupt_x
            corrupted_logit_dif, clean_logit_dif = (
                clean_logit_dif,
                corrupted_logit_dif,
            )
            clean_cache, corrupt_cache = corrupt_cache, clean_cache
        (
            restored_logit_difference_tab,
            neuron_activation_difference_tab,
        ) = st.tabs(
            ["Restored Logit Difference", "Neuron Activation Difference"]
        )

        with restored_logit_difference_tab:
            metric_func = logit_diff_recovery_metric
            kwargs = {
                "logit_dir": logit_dir,
                "clean_logit_dif": clean_logit_dif,
                "corrupted_logit_dif": corrupted_logit_dif,
            }

            patching_scan_umbrella_component(
                dt,
                corrupted_tokens,
                clean_cache,
                token_labels,
                metric_func,
                kwargs,
            )

        with neuron_activation_difference_tab:
            a, b = st.columns(2)
            with a:
                neuron_text = st.text_input(
                    f"Type the neuron you want",
                    "L0N0",
                    key="neuron_act_analysis, causal",
                )
                # validate neuron
                if not re.search(r"L\d+N\d+", neuron_text):
                    st.error("Neuron must be in the format L{number}N{number}")
                    return
                # get the neuron index, and the layer
                neuron = int(neuron_text.split("N")[1])
                layer = int(neuron_text.split("L")[1].split("N")[0])
            with b:
                pass

            clean_neuron_activation = clean_cache[
                f"blocks.{layer}.mlp.hook_pre"
            ][0, -1, neuron]
            corrupted_neuron_activation = corrupt_cache[
                f"blocks.{layer}.mlp.hook_pre"
            ][0, -1, neuron]
            with b:
                if noise_or_denoise.lower() == "noise":
                    st.write(f"Clean: {corrupted_neuron_activation:3f}")
                    st.write(f"Corrupted: {clean_neuron_activation:3f}")
                else:
                    st.write(f"Clean: {clean_neuron_activation:3f}")
                    st.write(f"Corrupted: {corrupted_neuron_activation:3f}")

            metric_func = neuron_activation_metric
            kwargs = {
                "layer": layer,
                "neuron": neuron,
                "clean_neuron_activation": clean_neuron_activation,
                "corrupted_neuron_activation": corrupted_neuron_activation,
            }
            patching_scan_umbrella_component(
                dt,
                corrupted_tokens,
                clean_cache,
                token_labels,
                metric_func,
                kwargs,
                apply_metric_to_cache=True,
            )

      
def get_corrupted_tokens_component(dt, key=""):
    a, b, c = st.columns(3)
    with a:
        path_patch_by = st.selectbox(
            "Patch by",
            ["State", "All RTG", "Specific RTG", "Action"],
            key=key + "patch_by_selector",
        )
    with b:
        noise_or_denoise = st.selectbox(
            "Noise or Denoise",
            ["Noise", "Denoise"],
            key=key + "noise_or_denoise_selector",
        )
    with c:
        if path_patch_by == "State":
            number_modifications = st.slider(
                "Number of Modifications",
                min_value=1,
                max_value=3,
                value=1,
                key=key + "number_modifications_slider",
            )

    from src.streamlit_app.environment import (
        preprocess_inputs,
        get_state_history,
    )

    obs, actions, rtg, timesteps = preprocess_inputs(
        **get_state_history(previous_step=False),
    )

    previous_tokens = dt.to_tokens(obs, actions, rtg, timesteps)
    rtg = rtg.clone()  # don't accidentally modify the session state.
    obs = obs.clone()  # don't accidentally modify the session state.

    if path_patch_by == "Specific RTG":
        specific_rtg = st.slider(
            "Corrupted RTG",
            min_value=0.0,
            max_value=st.session_state.rtg[0][0].item(),
            value=0.0,
            key=key + "rtg_slider",
        )
        position = st.slider(
            "Position",
            min_value=0,
            max_value=st.session_state.max_len - 2,
            value=0,
            key=key + "position_slider",
        )

        new_rtg = rtg.clone()
        new_rtg[0][position] = specific_rtg

        st.write("Previous RTG: ", rtg.flatten())
        st.write("Updated RTG: ", new_rtg.flatten())

        corrupted_tokens = dt.to_tokens(obs, actions, new_rtg, timesteps)

        # # at this point I can set the corrupted tokens.
        # corrupted_tokens = get_modified_tokens_from_app_state(
        #     dt, specific_rtg=min_rtg, position=position
        # )

    elif path_patch_by == "All RTG":
        all_rtg = st.slider(
            "Corrupted RTG",
            min_value=0.0,
            max_value=st.session_state.rtg[0][0].item() - 0.01,
            value=0.0,
            key=key + "rtg_slider",
        )

        rtg_dif = rtg[0] - all_rtg
        new_rtg = rtg - rtg_dif

        st.write("Previous RTG: ", rtg.flatten())
        st.write("Updated RTG: ", new_rtg.flatten())

        # at this point I can set the corrupted tokens.
        corrupted_tokens = dt.to_tokens(obs, actions, new_rtg, timesteps)

    elif path_patch_by == "Action":
        new_action = st.selectbox("Choose an action", IDX_TO_ACTION.values())
        action_idx = ACTION_TO_IDX[new_action]
        position = st.slider(
            "Position",
            min_value=0,
            max_value=st.session_state.max_len - 2,
            value=0,
            key=key + "action_selector",
        )

        new_actions = actions.clone()
        new_actions[0][position] = action_idx

        corrupted_tokens = dt.to_tokens(obs, new_actions, rtg, timesteps)

        st.write(actions.squeeze(-1))
        st.write(new_actions.squeeze(-1))

    elif path_patch_by == "State":
        assert (
            dt.environment_config.env_id == "MiniGrid-MemoryS7FixedStart-v0"
        ), "State patching only implemented for MiniGrid-MemoryS7FixedStart-v0"

        max_len = st.session_state.max_len
        env = st.session_state.env
        new_obs = st.session_state.obs[:, -max_len:].clone()

        modification_tabs = st.tabs(
            [f"Modification {i}" for i in range(number_modifications)]
        )

        for i in range(number_modifications):
            with modification_tabs[i]:
                # get instructions
                instructions = get_corrupt_obs_instructions(
                    key=key + f"-modification_{i}"
                )
                position = instructions[0]

                # edit the state
                corrupt_obs = get_updated_obs(
                    new_obs[0], *instructions, key=key + f"-modification_{i}"
                )
                # new_obs = obs.clone().unsqueeze(0)
                new_obs[0][position] = corrupt_obs

        corrupted_tokens = dt.to_tokens(new_obs, actions, rtg, timesteps)

        # make sure the user can see their updates.
        show_update_tick = st.checkbox(
            "Show state update", key=key + "show_update_tick"
        )

        if show_update_tick:
            clean_col, corrupt_col = st.columns(2)
            with clean_col:
                st.write("Clean")
                image = get_rendered_obss(env, obs[0])
                fig = px.imshow(image, animation_frame=0)
                st.plotly_chart(fig, use_container_width=True)
            with corrupt_col:
                st.write("Corrupted")
                image = get_rendered_obss(env, new_obs[0])
                fig = px.imshow(image, animation_frame=0)
                st.plotly_chart(fig, use_container_width=True)

    assert not torch.all(
        corrupted_tokens == previous_tokens
    ), "corrupted tokens are the same!"

    return corrupted_tokens, noise_or_denoise


def get_corrupt_obs_instructions(key=""):
    a, b, c, d = st.columns(4)

    current_len = st.session_state.current_len
    max_len = st.session_state.max_len
    with a:
        if current_len == 1:
            st.write("Can only modify current _state timestep = 1")
            timestep_to_modify = max_len - 1
        else:
            timestep_to_modify = st.selectbox(
                "Timestep",
                range(max_len),
                format_func=lambda x: st.session_state.labels[1::3][x],
                index=max(0, max_len - current_len),
                key=key + "timestep_selector",
            )
    with b:
        x_position_to_update = st.selectbox(
            "X Position (Rel to Agent)",
            range(8),
            format_func=lambda x: f"{x-3}",
            index=2,
            key=key + "x_position_selector",
        )
    with c:
        y_position_to_update = st.selectbox(
            "Y Position (Rel to Agent)",
            range(8),
            format_func=lambda x: f"{6-x}",
            index=6,
            key=key + "y_position_selector",
        )
    with d:
        object_to_update = st.selectbox(
            "Select an object",
            list(IDX_TO_OBJECT.keys()),
            index=OBJECT_TO_IDX["key"],
            format_func=IDX_TO_OBJECT.get,
            key=key + "object_selector",
        )

    return (
        timestep_to_modify,
        x_position_to_update,
        y_position_to_update,
        object_to_update,
    )


def get_updated_obs(
    obs,
    timestep_to_modify,
    x_position_to_update,
    y_position_to_update,
    object_to_update,
    key="",
):
    obs = obs[timestep_to_modify].clone()
    corrupt_obs = obs.detach().clone()
    corrupt_obs[
        x_position_to_update,
        y_position_to_update,
        : len(OBJECT_TO_IDX),
    ] = 0
    corrupt_obs[
        x_position_to_update, y_position_to_update, object_to_update
    ] = 1

    return corrupt_obs


def patching_scan_umbrella_component(
    dt,
    corrupted_tokens,
    clean_cache,
    token_labels,
    metric_func,
    kwargs,
    apply_metric_to_cache=False,
):
    # easy section.
    (
        dummy_tab,
        layer_token_tab,
        residual_stream_by_block_tab,
        head_all_positions_tab,
        head_all_positions_by_component_tab,
        minimize_tab,
    ) = st.tabs(
        [
            "Help",
            "Layer-Token",
            "Layer-Token + Attn/MLP",
            "Head All Positions",
            "Layer-Token-Head",
            "Minimize",
        ]
    )

    # advanced section
    (
        head_by_component_and_position_tab,
        mlp_patching_tab,
    ) = st.tabs(["Layer-Token-Head-Detail", "MLP - Neuron Detail"])

    with dummy_tab:
        st.write(
            """
                ### Activation Patching

                This analysis feature enables the user to perform activation patching experiments.

                You may patch different inputs including observations, actions and RTGs. 

                If "Noise" is selected, then the metric shown is the proportion of restored logit difference
                compared to the original logit difference (between the clean and corrupted logit differences).

                If "Denoise" is selected, then the metric shown is proportion of corrupted logit difference
                compared to the original logit difference (between the clean and corrupted logit differences)
                reversed. That is, how much of the corrupted logit difference is restored.

                Neither of these approaches can be trivially compared to the logit lens, but generally speaking
                noising will make more sense in the context of the current forward pass, since you are patching
                a corrupted node into what is otherwise mostly the same forward pass. Denoising, on the other hand,
                involves patching clean nodes into a corrupted forward pass.

                Complete "recovery" in denoising and complete "corruption" in noising will have values of 1 in the 
                logit difference metric used. This is evident when you look at patching the residual stream before the first
                attention layer of the token which you have edited.

                Note: These tabs automatically run everytime you change the input. 
                If it crashes your browser, let me know. It shouldn't on these models.*
                """
        )

    with layer_token_tab:
        layer_token_patch_component(
            dt,
            corrupted_tokens,
            clean_cache,
            token_labels,
            metric_func,
            kwargs,
            apply_metric_to_cache=apply_metric_to_cache,
        )

    with residual_stream_by_block_tab:
        layer_token_block_patch_component(
            dt,
            corrupted_tokens,
            clean_cache,
            token_labels,
            metric_func,
            kwargs,
            apply_metric_to_cache=apply_metric_to_cache,
        )

    with head_all_positions_tab:
        head_all_positions_patch_component(
            dt,
            corrupted_tokens,
            clean_cache,
            metric_func,
            kwargs,
            apply_metric_to_cache=apply_metric_to_cache,
        )

    with head_all_positions_by_component_tab:
        head_all_positions_by_input_patch_component(
            dt,
            corrupted_tokens,
            clean_cache,
            metric_func,
            kwargs,
            apply_metric_to_cache=apply_metric_to_cache,
        )

    with minimize_tab:
        pass

    with head_by_component_and_position_tab:
        head_by_component_and_position_patch_component(
            dt,
            corrupted_tokens,
            clean_cache,
            metric_func,
            kwargs,
            apply_metric_to_cache=apply_metric_to_cache,
        )

    with mlp_patching_tab:
        mlp_patch_single_neuron_component(
            dt,
            corrupted_tokens,
            clean_cache,
            metric_func,
            kwargs,
            apply_metric_to_cache=apply_metric_to_cache,
        )


def layer_token_patch_component(
    dt,
    corrupted_tokens,
    clean_cache,
    token_labels,
    patching_metric_func,
    patching_metric_kwargs={},
    apply_metric_to_cache=False,
):
    # let's gate until we have a sense for run time.
    patch = patching.get_act_patch_resid_pre(
        dt.transformer,
        corrupted_tokens=corrupted_tokens,
        clean_cache=clean_cache,
        patching_metric=partial(
            patching_metric_func,
            **patching_metric_kwargs,
        ),
        apply_metric_to_cache=apply_metric_to_cache,
    )

    fig = px.imshow(
        patch,
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
        title="Percent Logit Difference Restored From Patched Residual Stream",
        labels={"x": "Sequence Position", "y": "Layer"},
    )

    # set xticks to labels
    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(len(token_labels))),
        ticktext=token_labels,
    )

    st.plotly_chart(fig, use_container_width=True)


def layer_token_block_patch_component(
    dt,
    corrupted_tokens,
    clean_cache,
    token_labels,
    patching_metric_func,
    patching_metric_kwargs={},
    apply_metric_to_cache=False,
):
    # let's gate until we have a sense for run time.
    patch = patching.get_act_patch_block_every(
        dt.transformer,
        corrupted_tokens=corrupted_tokens,
        clean_cache=clean_cache,
        metric=partial(
            patching_metric_func,
            **patching_metric_kwargs,
        ),
        apply_metric_to_cache=apply_metric_to_cache,
    )

    fig = px.imshow(
        patch,
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
        facet_col=0,
        facet_col_wrap=1,
        title="Percent Logit Difference Restored From Patched Residual Stream",
        labels={"x": "Sequence Position", "y": "Layer"},
    )

    # set xticks to labels
    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(len(token_labels))),
        ticktext=token_labels,
    )

    fig.layout.annotations[2]["text"] = "Residual Stream"
    fig.layout.annotations[1]["text"] = "Attention"
    fig.layout.annotations[0]["text"] = "MLP"
    st.plotly_chart(fig, use_container_width=True)


def head_all_positions_patch_component(
    dt,
    corrupted_tokens,
    clean_cache,
    patching_metric_func,
    patching_metric_kwargs={},
    apply_metric_to_cache=False,
):
    patch = patching.get_act_patch_attn_head_out_all_pos(
        dt.transformer,
        corrupted_tokens=corrupted_tokens,
        clean_cache=clean_cache,
        patching_metric=partial(
            patching_metric_func,
            **patching_metric_kwargs,
        ),
        apply_metric_to_cache=apply_metric_to_cache,
    )

    fig = px.imshow(
        patch,
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
        title="Percent Logit Difference Restored From Patched Attn Head Output",
        labels={"x": "Head", "y": "Layer"},
    )

    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(patch.shape[-1])),
    )

    st.plotly_chart(fig, use_container_width=True)


def head_all_positions_by_input_patch_component(
    dt,
    corrupted_tokens,
    clean_cache,
    patching_metric_func,
    patching_metric_kwargs={},
    apply_metric_to_cache=False,
):
    patch = patching.get_act_patch_attn_head_all_pos_every(
        dt.transformer,
        corrupted_tokens=corrupted_tokens,
        clean_cache=clean_cache,
        metric=partial(
            patching_metric_func,
            **patching_metric_kwargs,
        ),
        apply_metric_to_cache=apply_metric_to_cache,
    )

    fig = px.imshow(
        patch,
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
        facet_col=0,
        title="Percent Logit Difference Restored Per Head (All Pos)",
        labels={"x": "Head", "y": "Layer"},
    )

    # remove ticks,
    fig.update_xaxes(showticklabels=False, showgrid=False, ticks="")
    fig.update_yaxes(showticklabels=False, showgrid=False, ticks="")

    facet_labels = ["Output", "Query", "Key", "Value", "Pattern"]
    for i, facet_label in enumerate(facet_labels):
        fig.layout.annotations[i]["text"] = facet_label

    st.plotly_chart(fig, use_container_width=True)


def head_by_component_and_position_patch_component(
    dt,
    corrupted_tokens,
    clean_cache,
    patching_metric_func,
    patching_metric_kwargs={},
    apply_metric_to_cache=False,
):
    if st.checkbox(
        "Run this slightly expensive compute", key=str(apply_metric_to_cache)
    ):
        patch = patching.get_act_patch_attn_head_by_pos_every(
            dt.transformer,
            corrupted_tokens=corrupted_tokens,
            clean_cache=clean_cache,
            metric=partial(
                patching_metric_func,
                **patching_metric_kwargs,
            ),
            apply_metric_to_cache=apply_metric_to_cache,
        )

        fig = px.imshow(
            patch,
            color_continuous_midpoint=0.0,
            color_continuous_scale="RdBu",
            animation_frame=2,
            facet_col=0,
            title="Percent Logit Difference Restored from Patching Per Head (All Pos)",
            labels={"x": "Head", "y": "Layer"},
        )
        # remove ticks,
        fig.update_xaxes(showticklabels=False, showgrid=False, ticks="")
        fig.update_yaxes(showticklabels=False, showgrid=False, ticks="")
        facet_labels = ["Output", "Query", "Key", "Value", "Pattern"]
        for i, facet_label in enumerate(facet_labels):
            fig.layout.annotations[i]["text"] = facet_label

        slider_labels = st.session_state.labels
        st.plotly_chart(fig, use_container_width=True)
        st.write(slider_labels)


def mlp_patch_single_neuron_component(
    dt,
    corrupted_tokens,
    clean_cache,
    patching_metric_func,
    patching_metric_kwargs={},
    apply_metric_to_cache=False,
):
    with st.form(key=str(apply_metric_to_cache)):
        b, c = st.columns(2)
        with b:
            layer = st.selectbox(
                "Layer", list(range(dt.transformer_config.n_layers))
            )
        with c:
            st.write(" ")
            st.write(" ")
            submitted = st.form_submit_button("Patch Neuron!")

        if submitted:
            # let's gate until we have a sense for run time.
            patch = get_act_patch_mlp(
                dt.transformer,
                corrupted_tokens=corrupted_tokens,
                clean_cache=clean_cache,
                metric=partial(
                    patching_metric_func,
                    **patching_metric_kwargs,
                ),
                layer=layer,
                apply_metric_to_cache=apply_metric_to_cache,
            ).detach()

            fig = px.scatter(
                x=list(range(patch.shape[0])),
                y=patch,
                color=patch,
                title="Activation Patching Per Neuron",
                labels={"x": "Position", "y": "Metric"},
            )

            st.plotly_chart(fig, use_container_width=True)


def logit_diff_recovery_metric(
    logits, logit_dir, clean_logit_dif, corrupted_logit_dif
):
    """
    Linear function of logit diff, calibrated so that it equals 0 when performance is
    same as on corrupted input, and 1 when performance is same as on clean input.
    """

    patched_logit_diff = logits[0, -1] @ logit_dir
    result = (patched_logit_diff - corrupted_logit_dif) / (
        clean_logit_dif - corrupted_logit_dif
    )
    return result


def neuron_activation_metric(
    cache, layer, neuron, clean_neuron_activation, corrupted_neuron_activation
):
    """
    Linear function of logit diff, calibrated so that it equals 0 when performance is
    same as on corrupted input, and 1 when performance is same as on clean input.
    """
    patched_neuron_activation = cache[f"blocks.{layer}.mlp.hook_pre"][
        0, -1, neuron
    ]

    result = (patched_neuron_activation - corrupted_neuron_activation) / (
        clean_neuron_activation - corrupted_neuron_activation
    )
    return result


# inspired by: https://clementneo.com/posts/2023/02/11/we-found-an-neuron
def get_act_patch_mlp(
    model: Module,
    corrupted_tokens: Tensor,
    clean_cache: Dict[str, Tensor],
    metric: Callable[[Tensor], Tensor],
    layer: int,
    apply_metric_to_cache: bool = False,
) -> Tensor:
    def patch_neuron_activation(
        corrupted_mlp_act: Tensor, hook, neuron, clean_cache
    ):
        corrupted_mlp_act[:, :, neuron] = clean_cache[hook.name][:, :, neuron]
        return corrupted_mlp_act

    d_mlp = model.cfg.d_mlp
    patched_neurons_normalized_improvement = torch.zeros(
        d_mlp, device=corrupted_tokens.device, dtype=torch.float32
    )

    for neuron in tqdm(range(d_mlp)):
        hook_name = f"blocks.{layer}.mlp.hook_post"
        hook_fn = partial(
            patch_neuron_activation, neuron=neuron, clean_cache=clean_cache
        )
        with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
            patched_neuron_logits, cache = model.run_with_cache(
                corrupted_tokens
            )

        if apply_metric_to_cache:
            patched_neuron_metric = metric(cache)
        else:
            patched_neuron_metric = metric(patched_neuron_logits)

        patched_neurons_normalized_improvement[neuron] = patched_neuron_metric

    return patched_neurons_normalized_improvement


# Activation PAtch with Interpolation.
# based on: https://www.lesswrong.com/posts/5spBue2z2tw4JuDCx/steering-gpt-2-xl-by-adding-an-activation-vector#Evidence_of_generalization
# I think of it as reverse activation patching.
# We're going to try to find a vector that, when added to the activations,
# will make the model predict a certain action.
def show_algebraic_value_editing(dt, logit_dir, original_cache):
    with st.expander("Algebraic Value Editing"):
        st.write(
            """
            
            This technique uses the content of the activations in one forward pass to steer the model in another forward pass.
            
            In order to implement this in the most natural way, 
            I have reuse the activation patching interface.

            1. Select a token modification method, specify your updated tokens
            2. Select the layer at which you would like to insert your activations
            3. Select the coefficient of amplification range you want (to scale the vector.)
            
            You can then see the effect this has on the current logit distribution, 
            and the effect this has on the projections of transformer components
            into the logit direction.

            Please note that:
            - at coefficient = 0, you get the original forward pass.
            - at coefficient = 1, you get the corrupted pass.
            
            """
        )

        # 1. Create a corrupted forward pass using the same essential logic as activation
        # patching.
        corrupted_tokens, noise_or_denoise = get_corrupted_tokens_component(
            dt, key="avec"
        )
        (
            corrupted_preds,
            corrupted_x,
            corrupted_cache,
            _,
        ) = get_action_preds_from_tokens(dt, corrupted_tokens)

        # 2. Select where you want to insert the vector in the forward pass
        # and where you want to do a scan or not.
        a, b, c = st.columns([5, 1, 5])
        with a:
            layer = st.selectbox(
                "Layer",
                list(range(dt.transformer_config.n_layers)),
                key="avec",
            )
            name = f"blocks.{layer}.hook_resid_pre"

        with c:
            coeff_min, coeff_max = st.slider(
                "Coefficient",
                min_value=-2.0,
                max_value=3.0,
                value=[0.0, 1.0],
            )

            # make coeff a vector from min to max of length batch size
            coeff = (
                torch.linspace(coeff_min, coeff_max, BATCH_SIZE)
                .unsqueeze(1)
                .unsqueeze(2)
                .to(corrupted_tokens.device)
            )

        # 3. Run a scan or single forward pass of the original model to get the
        # activations using a hook to insert the chosen value vector.

        act_original = original_cache[name]
        act_corrupted = corrupted_cache[name]

        # theoretically, act_add and act_sub could be a very different situation.
        # but I don't have an easy way to construct that now. So I'm just going to
        # assume we want to semantically retarget the current frame.
        act_sub = act_original
        act_add = act_corrupted
        act_diff = act_add - act_sub

        ave_hook = get_ave_hook("resid_pre", coeff, act_diff)
        editing_hooks = [(f"blocks.{layer}.hook_resid_pre", ave_hook)]

        clean_tokens = get_tokens_from_app_state(dt, previous_step=False)
        clean_tokens = clean_tokens.repeat(BATCH_SIZE, 1, 1)

        with dt.transformer.hooks(fwd_hooks=editing_hooks):
            avec_x, avec_cache = dt.transformer.run_with_cache(
                clean_tokens, remove_batch_dim=False
            )

            _, action_preds, _ = dt.get_logits(
                avec_x,
                batch_size=BATCH_SIZE,
                seq_length=st.session_state.max_len,
                no_actions=False,
            )

        logit_tab, decomp_tab = st.tabs(["Logit Scan", "Decomposition"])

        with logit_tab:
            fig = plot_logit_scan(
                coeff, action_preds, scan_name="Injection Coefficient"
            )
            st.plotly_chart(fig, use_container_width=True)

        with decomp_tab:
            decomp_level, cluster, normalize = decomp_configuration_ui(
                key="avec"
            )
            df = get_decomp_scan(
                coeff, avec_cache, logit_dir, decomp_level, normalize=normalize
            )
            fig = plot_decomp_scan_line(df, "Injection Coefficient")
            st.plotly_chart(fig, use_container_width=True)
            fig2 = plot_decomp_scan_corr(df, cluster, "Injection Coefficient")
            st.plotly_chart(fig2, use_container_width=True)
            if cluster:
                st.write("I know this is a bit janky, will fix later.")

        st.write(
            """
            Read more on this technique [here](https://www.lesswrong.com/posts/5spBue2z2tw4JuDCx/steering-gpt-2-xl-by-adding-an-activation-vector#Evidence_of_generalization)
            """
        )


def get_ave_hook(component, coeff: Tensor, act_diff: Tensor, **kwargs):
    """
    Things I might want to hook:
    - resid_pre


    """

    if component == "resid_pre":

        def ave_hook(resid_pre, hook):
            if resid_pre.shape[1] == 1:
                return  # caching in model.generate for new tokens

            # We only add to the prompt (first call), not the generated tokens.
            # ppos, apos = resid_pre.shape[1], act_diff.shape[1]
            # assert apos <= ppos, f"More mod tokens ({apos}) then prompt tokens ({ppos})!"
            modulated_act_diff = coeff * act_diff

            # add to the beginning (position-wise) of the activations
            resid_pre[:, :, :] += coeff * act_diff

    return ave_hook


# Path Patching using code from Callum McDougall.
from src.patch_transformer_lens.patching import path_patch


def show_path_patching(dt, logit_dir, clean_cache):
    with st.expander("Path Patching"):
        # 1. Create a corrupted forward pass using the same essential logic as activation
        # patching.
        corrupted_tokens, patch_type = get_corrupted_tokens_component(
            dt, key="path_"
        )

        (
            corrupt_preds,
            corrupt_x,
            corrupted_cache,
            _,
        ) = get_action_preds_from_tokens(dt, corrupted_tokens)

        # get clean/corrupt forward passes done.
        clean_tokens = get_tokens_from_app_state(dt, previous_step=False)
        clean_preds, clean_x, _, _ = get_action_preds_from_tokens(
            dt, clean_tokens
        )

        clean_logit_dif = clean_x[0, -1] @ logit_dir
        corrupted_logit_dif = corrupt_x[0, -1] @ logit_dir

        if st.checkbox("show corrupted action predictions", key="path"):
            plot_action_preds(corrupt_preds)

        if st.checkbox("Show corrupted analyses (slightly expensive)", key="corrupt_analysis_path"):
            corrupt_attention_pattern_tab, corrupt_logit_lens_tab = st.tabs(["Attention Pattern", "Logit Lens"])
            with corrupt_attention_pattern_tab:
                show_attention_pattern(dt, corrupted_cache, key="corrupt-path-")
            with corrupt_logit_lens_tab:
                show_logit_lens(dt, corrupted_cache, logit_dir, key="corrupt-path-")

        # rewrite previous line but with nicer formatting
        st.write(
            "Clean Logit Diff: ",
            f"{clean_logit_dif.item():.3f}",
            " Corrupted Logit Diff: ",
            f"{corrupted_logit_dif.item():.3f}",
        )

        logit_diff_metric = partial(
            logit_diff_recovery_metric,
            logit_dir=logit_dir,
            clean_logit_dif=clean_logit_dif,
            corrupted_logit_dif=corrupted_logit_dif,
        )

        (
            help_tab,
            path_patch_block_every_tab,
            path_patch_head_every_tab,
            design_your_own_tab,
        ) = st.tabs(
            [
                "Help",
                "Layer-Token + Attn/MLP",
                "Head All Positions",
                "Design You Own",
            ]
        )

        with help_tab:
            st.write(
                """
                Path patching is a more subtle variation on activation patching which involves 
                specifying nodes in the computational graph and patching the edges between them 
                but fixing everything else. This helps us see things like exactly how 
                two heads compose. 
                
                Path patching from (sender components -> receiver components) involves the following algorithm:
                    (1) Gather all activations on clean and corrupted distributions.
                    (2) Run model on clean with sender components patched from corrupted and all other components frozen. Cache the receiver components.
                    (3) Run model on clean with receiver components patched from previously cached values.

                Result of this algorithm: the model behaves like it does on the clean distribution, but with every direct path from sender -> receiver
                patched (where we define a direct path as one which doesn't go through any other attention heads, only MLPs or skip connections).
                """
            )

        with path_patch_block_every_tab:
            path_patch_block_every = 1 - path_patch(
                dt.transformer,
                clean_tokens=clean_tokens,
                corrupted_tokens=corrupted_tokens,
                clean_cache=clean_cache,
                corrupted_cache=corrupted_cache,
                patching_metric=logit_diff_metric,
                receiver_components=[(-1, "resid_post")],
                receiver_seq_pos="all",
                sender_components="all_blocks",
                sender_seq_pos="each",
                verbose=True,
            )

            fig = px.imshow(
                path_patch_block_every,
                facet_col=0,
                facet_col_wrap=1,
                color_continuous_midpoint=0,
                color_continuous_scale="RdBu",
                title="Logit Difference From Patched Attn Head Output",
                labels={"x": "Sequence Position", "y": "Layer"},
            )

            # set xticks to labels
            token_labels = st.session_state.labels
            fig.update_xaxes(
                showgrid=False,
                ticks="",
                tickmode="linear",
                automargin=True,
                tickvals=list(range(len(token_labels))),
                ticktext=token_labels,
            )

            fig.update_yaxes(
                showgrid=False,
                ticks="",
                tickmode="linear",
                automargin=True,
            )
            fig.layout.annotations[2]["text"] = "Residual Stream"
            fig.layout.annotations[1]["text"] = "Attention"
            fig.layout.annotations[0]["text"] = "MLP"
            st.plotly_chart(fig, use_container_width=True)

        with path_patch_head_every_tab:
            path_patch_head_every = 1 - path_patch(
                dt.transformer,
                clean_tokens=clean_tokens,
                corrupted_tokens=corrupted_tokens,
                clean_cache=clean_cache,
                corrupted_cache=corrupted_cache,
                patching_metric=logit_diff_metric,
                receiver_components=[(-1, "resid_post")],
                receiver_seq_pos="all",
                sender_components="z",
                sender_seq_pos="all",
                verbose=True,
            )

            fig = px.imshow(
                path_patch_head_every,
                color_continuous_midpoint=0,
                color_continuous_scale="RdBu",
                title="Logit Difference From Patched Attn Out (all pos)",
                labels={"x": "Head", "y": "Layer"},
            )

            # set xticks to labels
            fig.update_xaxes(
                showgrid=False,
                ticks="",
                tickmode="linear",
                automargin=True,
            )

            fig.update_yaxes(
                showgrid=False,
                ticks="",
                tickmode="linear",
                automargin=True,
            )

            st.plotly_chart(fig, use_container_width=True)

        with design_your_own_tab:
            layers = list(range(dt.transformer_config.n_layers))
            heads = list(range(dt.transformer_config.n_heads))

            col, a, b, c = st.columns(4)

            with col:
                sender_component = st.selectbox(
                    "Select Sender Component",
                    options=[
                        "z",
                    ],
                )
            with a:
                composition_type = st.selectbox(
                    "Select Receiver Head Component", ["v", "k", "q"]
                )
            head_options = [
                (layer, head, composition_type)
                for layer in layers
                for head in heads
            ]

            mlp_options = [(layer, "mlp_out") for layer in layers]

            with b:
                heads_selected = st.multiselect(
                    label="Select Reciever Heads",
                    options=head_options,
                    format_func=lambda x: f"L{x[0]}H{x[1]}",
                    default=[],
                )

            with c:
                mlp_selected = st.multiselect(
                    label="Select Receiver MLP",
                    options=mlp_options,
                    format_func=lambda x: f"L{x[0]}",
                    default=mlp_options[-1],
                )

            path_patch_head_every = 1 - path_patch(
                dt.transformer,
                clean_tokens=clean_tokens,
                corrupted_tokens=corrupted_tokens,
                clean_cache=clean_cache,
                corrupted_cache=corrupted_cache,
                patching_metric=logit_diff_metric,
                receiver_components=heads_selected + mlp_selected,
                receiver_seq_pos="all",
                sender_components=sender_component,
                sender_seq_pos="all",
                verbose=True,
            )

            if sender_component in ["z"]:
                fig = px.imshow(
                    path_patch_head_every,
                    color_continuous_midpoint=0,
                    color_continuous_scale="RdBu",
                    title="Logit Difference From Patched Attn Out (all pos)",
                    labels={"x": "Head", "y": "Layer"},
                )

                # set xticks to labels
                fig.update_xaxes(
                    showgrid=False,
                    ticks="",
                    tickmode="linear",
                    automargin=True,
                )

                fig.update_yaxes(
                    showgrid=False,
                    ticks="",
                    tickmode="linear",
                    automargin=True,
                )
                st.plotly_chart(fig, use_container_width=True)

            else:
                st.write("Not implemented yet")
