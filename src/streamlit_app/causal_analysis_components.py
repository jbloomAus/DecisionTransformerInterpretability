import streamlit as st
import plotly.express as px
from torchtyping import TensorType as TT
from transformer_lens.hook_points import HookPoint
from functools import partial
from .analysis import get_residual_decomp
from .environment import (
    get_modified_tokens_from_app_state,
    get_tokens_from_app_state,
    get_action_preds_from_tokens,
    get_action_preds_from_app_state,
)

from .visualizations import (
    plot_action_preds,
    plot_logit_diff,
    plot_single_residual_stream_contributions_comparison,
)

from src.streamlit_app.patch_transformer_lens import patching

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
def show_activation_patching(dt, logit_dir, original_cache):
    with st.expander("Activation Patching"):
        path_patch_by = st.selectbox(
            "Patch by", ["All RTG", "Specific RTG", "Time", "Action", "State"]
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
                max_value=st.session_state.max_len - 1,
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

        else:
            st.warning("Not implemented yet")

        st.write("Not implemented yet")
        (
            residual_stream_tab,
            residual_stream_by_block_tab,
            head_tab,
            head_by_comp,
        ) = st.tabs(
            [
                "Residual Stream",
                "Residual Stream via Attn/MLP",
                "Head",
                "Head via Q,K,O,K,Pattern",
            ]
        )

        # # look at current state and do a forward pass
        clean_tokens = get_tokens_from_app_state(dt, previous_step=True)
        (
            clean_action_preds,
            clean_x,
            clean_cache,
            _,
        ) = get_action_preds_from_tokens(dt, clean_tokens)
        clean_logit_dif = clean_x[0, -1] @ logit_dir
        (
            corrupted_action_preds,
            corrupt_x,
            corrupted_cache,
            _,
        ) = get_action_preds_from_tokens(dt, corrupted_tokens)
        corrupted_logit_dif = corrupt_x[0, -1] @ logit_dir

        with residual_stream_tab:
            # let's gate until we have a sense for run time.
            patch = patching.get_act_patch_resid_pre(
                dt.transformer,
                corrupted_tokens=corrupted_tokens,
                clean_cache=original_cache,
                patching_metric=partial(
                    logit_diff_recovery_metric,
                    logit_dir=logit_dir,
                    clean_logit_diff=clean_logit_dif,
                    corrupted_logit_diff=corrupted_logit_dif,
                ),
            )

            fig = px.imshow(
                patch,
                color_continuous_midpoint=0.0,
                color_continuous_scale="RdBu",
                title="Residual Stream Contributions",
            )
            st.plotly_chart(fig)

    return


def logit_diff_recovery_metric(
    logits, logit_dir, clean_logit_diff, corrupted_logit_diff
):
    """
    Linear function of logit diff, calibrated so that it equals 0 when performance is
    same as on corrupted input, and 1 when performance is same as on clean input.
    """

    patched_logit_diff = logits[0, -1] @ logit_dir
    result = (patched_logit_diff - corrupted_logit_diff) / (
        clean_logit_diff - corrupted_logit_diff
    )
    return result
