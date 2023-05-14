from functools import partial
from typing import Callable, Dict

import plotly.express as px
import streamlit as st
import torch
from torch import Tensor
from torch.nn import Module
from torchtyping import TensorType as TT
from tqdm import tqdm
from transformer_lens.hook_points import HookPoint

from src.visualization import get_rendered_obs

from .visualizations import plot_logit_scan

from .analysis import get_residual_decomp
from .constants import IDX_TO_ACTION, IDX_TO_OBJECT
from .environment import (
    get_action_preds_from_app_state,
    get_action_preds_from_tokens,
    get_modified_tokens_from_app_state,
    get_tokens_from_app_state,
)

ACTION_TO_IDX = {v: k for k, v in IDX_TO_ACTION.items()}
OBJECT_TO_IDX = {v: k for k, v in IDX_TO_OBJECT.items()}

from src.streamlit_app.patch_transformer_lens import patching

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
        # make a streamlit form for choosing a component to ablate
        n_layers = dt.transformer_config.n_layers
        n_heads = dt.transformer_config.n_heads

        columns = st.columns(4)
        with columns[0]:
            layer = st.selectbox("Layer", list(range(n_layers)))
        with columns[1]:
            component = st.selectbox("Component", ["MLP", "HEAD"], index=1)
        with columns[2]:
            if component == "HEAD":
                head = st.selectbox("Head", list(range(n_heads)))
        with columns[3]:
            ablate_to_mean = st.checkbox("Ablate to mean", value=True)

        if component == "HEAD":
            ablation_func = get_ablation_function(ablate_to_mean, head)
            dt.transformer.blocks[layer].attn.hook_z.add_hook(ablation_func)
        elif component == "MLP":
            ablation_func = get_ablation_function(
                ablate_to_mean, layer, component="MLP"
            )
            dt.transformer.blocks[layer].hook_mlp_out.add_hook(ablation_func)

        action_preds, x, cache, tokens = get_action_preds_from_app_state(dt)
        dt.transformer.reset_hooks()
        if st.checkbox("show action predictions"):
            plot_action_preds(action_preds)
        if st.checkbox("show counterfactual residual contributions"):
            original_residual_decomp = get_residual_decomp(
                dt, original_cache, logit_dir
            )
            ablation_residual_decomp = get_residual_decomp(
                dt, cache, logit_dir
            )
            plot_single_residual_stream_contributions_comparison(
                original_residual_decomp, ablation_residual_decomp
            )

    # then, render a single residual stream contribution with the ablation


def get_ablation_function(ablate_to_mean, head_to_ablate, component="HEAD"):
    def head_ablation_hook(
        value: TT["batch", "pos", "head_index", "d_head"],  # noqa: F821
        hook: HookPoint,
    ) -> TT["batch", "pos", "head_index", "d_head"]:  # noqa: F821
        print(f"Shape of the value tensor: {value.shape}")

        if ablate_to_mean:
            value[:, :, head_to_ablate, :] = value[
                :, :, head_to_ablate, :
            ].mean(dim=2, keepdim=True)
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


def get_corrupted_tokens(dt):
    path_patch_by = st.selectbox(
        "Patch by", ["State", "All RTG", "Specific RTG", "Action"]
    )

    if path_patch_by == "Specific RTG":
        min_rtg = st.slider(
            "Corrupted RTG",
            min_value=0.0,
            max_value=st.session_state.rtg[0][0].item(),
            value=0.0,
        )
        position = st.slider(
            "Position",
            min_value=0,
            max_value=st.session_state.max_len - 2,
            value=0,
        )
        # at this point I can set the corrupted tokens.
        corrupted_tokens = get_modified_tokens_from_app_state(
            dt, specific_rtg=min_rtg, position=position
        )

    elif path_patch_by == "All RTG":
        min_rtg = st.slider(
            "Corrupted RTG",
            min_value=0.0,
            max_value=st.session_state.rtg[0][0].item() - 0.01,
            value=0.0,
        )
        # at this point I can set the corrupted tokens.
        corrupted_tokens = get_modified_tokens_from_app_state(
            dt, all_rtg=min_rtg
        )

    elif path_patch_by == "Action":
        action = st.selectbox("Choose an action", IDX_TO_ACTION.values())
        action_idx = ACTION_TO_IDX[action]
        position = st.slider(
            "Position",
            min_value=0,
            max_value=st.session_state.max_len - 2,
            value=0,
        )
        # at this point I can set the corrupted tokens.
        corrupted_tokens = get_modified_tokens_from_app_state(
            dt, new_action=action_idx, position=position
        )

    elif path_patch_by == "State":
        assert (
            dt.environment_config.env_id == "MiniGrid-MemoryS7FixedStart-v0"
        ), "State patching only implemented for MiniGrid-MemoryS7FixedStart-v0"

        # Let's construct this as a series (starting with one)
        # of changes to any previous state.

        a, b, c, d, f = st.columns(5)

        with a:
            current_len = st.session_state.current_len
            max_len = st.session_state.max_len

            if current_len == 1:
                st.write("Can only modify current _state timestep = 1")
                timestep_to_modify = max_len - 1
            else:
                timestep_to_modify = st.selectbox(
                    "Timestep",
                    range(max_len),
                    format_func=lambda x: st.session_state.labels[1::3][x],
                    index=max(0, max_len - current_len),
                )
        with b:
            x_position_to_update = st.selectbox(
                "X Position (Rel to Agent)",
                range(8),
                format_func=lambda x: f"{x-3}",
                index=2,
            )
        with c:
            y_position_to_update = st.selectbox(
                "Y Position (Rel to Agent)",
                range(8),
                format_func=lambda x: f"{6-x}",
                index=6,
            )
        with d:
            object_to_update = st.selectbox(
                "Select an object",
                list(IDX_TO_OBJECT.keys()),
                index=OBJECT_TO_IDX["key"],
                format_func=IDX_TO_OBJECT.get,
            )
        with f:
            show_update_tick = st.checkbox("Show state update")
        env = st.session_state.env
        obs = st.session_state.obs[0][-max_len:].clone()
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

        if show_update_tick:
            clean_col, corrupt_col = st.columns(2)
            with clean_col:
                st.write("Clean")
                image = get_rendered_obs(env, obs)
                fig = px.imshow(image)
                st.plotly_chart(fig, use_container_width=True)
            with corrupt_col:
                st.write("Corrupted")
                image = get_rendered_obs(env, corrupt_obs)
                fig = px.imshow(image)
                st.plotly_chart(fig, use_container_width=True)

        corrupted_tokens = get_modified_tokens_from_app_state(
            dt, corrupt_obs=corrupt_obs, position=timestep_to_modify
        )

    else:
        st.warning("Not implemented yet")

    return corrupted_tokens


def show_activation_patching(dt, logit_dir, original_cache):
    token_labels = st.session_state.labels
    with st.expander("Activation Patching"):
        corrupted_tokens = get_corrupted_tokens(dt)
        # # look at current state and do a forward pass
        clean_tokens = get_tokens_from_app_state(dt, previous_step=False)
        clean_preds, clean_x, _, _ = get_action_preds_from_tokens(
            dt, clean_tokens
        )
        clean_logit_dif = clean_x[0, -1] @ logit_dir
        corrupt_preds, corrupt_x, _, _ = get_action_preds_from_tokens(
            dt, corrupted_tokens
        )
        corrupted_logit_dif = corrupt_x[0, -1] @ logit_dir

        if st.checkbox("show corrupted action predictions"):
            plot_action_preds(corrupt_preds)

        st.write("Clean Logit Diff: ", clean_logit_dif.item())
        st.write("Corrupted Logit Diff: ", corrupted_logit_dif.item())

        # easy section.
        (
            dummy_tab,
            residual_stream_tab,
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

        with dummy_tab:
            st.write(
                """
                Welcome to Activation Patching! This is a tool for understanding how
                the model's respond when we change the input tokens. 

                You're mission, should you choose to accept it, is to generate hypotheses
                about the algorithm being implemented by the transformer. 

                Use different patching methods and different degrees of granularity to
                refine your hypothesis. 
                
                Keep in mind that this method probably doesn't validate your hypothesis
                unless you use something more advanced like path patching or causal
                scrubbing. Happy patching!

                *PS: These tabs automatically run everytime you change the input. 
                If it crashes your browser, let me know. It shouldn't on these models.*
                """
            )

        with residual_stream_tab:
            # let's gate until we have a sense for run time.
            patch = patching.get_act_patch_resid_pre(
                dt.transformer,
                corrupted_tokens=corrupted_tokens,
                clean_cache=original_cache,
                patching_metric=partial(
                    logit_diff_recovery_metric,
                    logit_dir=logit_dir,
                    clean_logit_dif=clean_logit_dif,
                    corrupted_logit_dif=corrupted_logit_dif,
                ),
            )

            fig = px.imshow(
                patch,
                color_continuous_midpoint=0.0,
                color_continuous_scale="RdBu",
                title="Logit Difference From Patched Residual Stream",
                labels={"x": "Sequence Position", "y": "Layer"},
            )

            # set xticks to labels
            fig.update_xaxes(
                tickmode="array",
                tickvals=list(range(len(token_labels))),
                ticktext=token_labels,
            )

            st.plotly_chart(fig)

        with residual_stream_by_block_tab:
            # let's gate until we have a sense for run time.
            patch = patching.get_act_patch_block_every(
                dt.transformer,
                corrupted_tokens=corrupted_tokens,
                clean_cache=original_cache,
                metric=partial(
                    logit_diff_recovery_metric,
                    logit_dir=logit_dir,
                    clean_logit_dif=clean_logit_dif,
                    corrupted_logit_dif=corrupted_logit_dif,
                ),
            )

            fig = px.imshow(
                patch,
                color_continuous_midpoint=0.0,
                color_continuous_scale="RdBu",
                facet_col=0,
                facet_col_wrap=1,
                title="Logit Difference From Patched Residual Stream",
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
            st.plotly_chart(fig)

        with head_all_positions_tab:
            patch = patching.get_act_patch_attn_head_out_all_pos(
                dt.transformer,
                corrupted_tokens=corrupted_tokens,
                clean_cache=original_cache,
                patching_metric=partial(
                    logit_diff_recovery_metric,
                    logit_dir=logit_dir,
                    clean_logit_dif=clean_logit_dif,
                    corrupted_logit_dif=corrupted_logit_dif,
                ),
            )

            fig = px.imshow(
                patch,
                color_continuous_midpoint=0.0,
                color_continuous_scale="RdBu",
                title="Logit Difference From Patched Attn Head Output",
                labels={"x": "Head", "y": "Layer"},
            )

            fig.update_xaxes(
                tickmode="array",
                tickvals=list(range(patch.shape[-1])),
            )

            st.plotly_chart(fig)

        with head_all_positions_by_component_tab:
            patch = patching.get_act_patch_attn_head_all_pos_every(
                dt.transformer,
                corrupted_tokens=corrupted_tokens,
                clean_cache=original_cache,
                metric=partial(
                    logit_diff_recovery_metric,
                    logit_dir=logit_dir,
                    clean_logit_dif=clean_logit_dif,
                    corrupted_logit_dif=corrupted_logit_dif,
                ),
            )

            st.write(patch.shape)
            fig = px.imshow(
                patch,
                color_continuous_midpoint=0.0,
                color_continuous_scale="RdBu",
                facet_col=0,
                facet_col_wrap=2,
                title="Activation Patching Per Head (All Pos)",
                labels={"x": "Head", "y": "Layer"},
            )

            facet_labels = ["Output", "Query", "Key", "Value", "Pattern"]
            for i, facet_label in enumerate(facet_labels):
                fig.layout.annotations[i]["text"] = facet_label

            st.plotly_chart(fig, use_container_width=True)

        with minimize_tab:
            pass

        # advanced section
        (
            head_by_component_and_position_tab,
            mlp_patching,
        ) = st.tabs(["Layer-Token-Head-Detail", "MLP - Neuron Detail"])

        with head_by_component_and_position_tab:
            if st.checkbox("Run this slightly expensive compute"):
                patch = patching.get_act_patch_attn_head_by_pos_every(
                    dt.transformer,
                    corrupted_tokens=corrupted_tokens,
                    clean_cache=original_cache,
                    metric=partial(
                        logit_diff_recovery_metric,
                        logit_dir=logit_dir,
                        clean_logit_dif=clean_logit_dif,
                        corrupted_logit_dif=corrupted_logit_dif,
                    ),
                )

                fig = px.imshow(
                    patch,
                    color_continuous_midpoint=0.0,
                    color_continuous_scale="RdBu",
                    facet_col=0,
                    facet_col_wrap=2,
                    animation_frame=2,
                    title="Activation Patching Per Head (All Pos)",
                    labels={"x": "Head", "y": "Layer"},
                )

                facet_labels = ["Output", "Query", "Key", "Value", "Pattern"]
                for i, facet_label in enumerate(facet_labels):
                    fig.layout.annotations[i]["text"] = facet_label

                slider_labels = st.session_state.labels
                st.plotly_chart(fig, use_container_width=True)
                st.write(slider_labels)

        with mlp_patching:
            with st.form("MLP Patching"):
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
                        clean_cache=original_cache,
                        metric=partial(
                            logit_diff_recovery_metric,
                            logit_dir=logit_dir,
                            clean_logit_dif=clean_logit_dif,
                            corrupted_logit_dif=corrupted_logit_dif,
                        ),
                        layer=layer,
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


# inspired by: https://clementneo.com/posts/2023/02/11/we-found-an-neuron
def get_act_patch_mlp(
    model: Module,
    corrupted_tokens: Tensor,
    clean_cache: Dict[str, Tensor],
    metric: Callable[[Tensor], Tensor],
    layer: int,
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
        patched_neuron_logits = model.run_with_hooks(
            corrupted_tokens,
            fwd_hooks=[(hook_name, hook_fn)],
            return_type="logits",
        )
        patched_neuron_metric = metric(patched_neuron_logits)
        patched_neurons_normalized_improvement[neuron] = patched_neuron_metric

    return patched_neurons_normalized_improvement


# AVEC!
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
            
            """
        )

        # 1. Create a corrupted forward pass using the same essential logic as activation
        # patching.
        corrupted_tokens = get_corrupted_tokens(dt)
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
                "Layer", list(range(dt.transformer_config.n_layers))
            )
            name = f"blocks.{layer}.hook_resid_pre"

        with c:
            coeff_min, coeff_max = st.slider(
                "Coefficient",
                min_value=-10.0,
                max_value=10.0,
                value=[-10.0, 10.0],
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
        act_diff = act_corrupted - act_original

        def ave_hook(resid_pre, hook):
            if resid_pre.shape[1] == 1:
                return  # caching in model.generate for new tokens

            # We only add to the prompt (first call), not the generated tokens.
            # ppos, apos = resid_pre.shape[1], act_diff.shape[1]
            # assert apos <= ppos, f"More mod tokens ({apos}) then prompt tokens ({ppos})!"
            modulated_act_diff = coeff * act_diff

            # add to the beginning (position-wise) of the activations
            resid_pre[:, :, :] += coeff * act_diff

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
            fig = plot_logit_scan(coeff, action_preds, scan_name="Coefficient")
            st.plotly_chart(fig, use_container_width=True)

        with decomp_tab:
            decomp_level, cluster = decomp_configuration_ui()
            df = get_decomp_scan(coeff, avec_cache, logit_dir, decomp_level)
            fig = plot_decomp_scan_line(df, "Coefficient")
            st.plotly_chart(fig, use_container_width=True)
            fig2 = plot_decomp_scan_corr(df, cluster, "Coefficient")
            st.plotly_chart(fig2, use_container_width=True)
            if cluster:
                st.write("I know this is a bit janky, will fix later.")

        st.write(
            """
            Read more on this technique [here](https://www.lesswrong.com/posts/5spBue2z2tw4JuDCx/steering-gpt-2-xl-by-adding-an-activation-vector#Evidence_of_generalization)
            """
        )
