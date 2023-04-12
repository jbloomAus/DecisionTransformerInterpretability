import streamlit as st
from torchtyping import TensorType as TT
from transformer_lens.hook_points import HookPoint

from .analysis import get_residual_decomp
from .environment import get_action_preds
from .visualizations import (
    plot_action_preds,
    plot_single_residual_stream_contributions,
    plot_single_residual_stream_contributions_comparison,
)


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

        action_preds, x, cache, tokens = get_action_preds(dt)
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
