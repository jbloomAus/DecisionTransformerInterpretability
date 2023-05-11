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
from .constants import IDX_TO_ACTION, IDX_TO_OBJECT
from src.visualization import preview_state_update

ACTION_TO_IDX = {v: k for k, v in IDX_TO_ACTION.items()}
OBJECT_TO_IDX = {v: k for k, v in IDX_TO_OBJECT.items()}

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
    token_labels = st.session_state.labels
    with st.expander("Activation Patching"):
        path_patch_by = st.selectbox(
            "Patch by", ["All RTG", "Specific RTG", "Action", "State"]
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
                dt.environment_config.env_id
                == "MiniGrid-MemoryS7FixedStart-v0"
            ), "State patching only implemented for MiniGrid-MemoryS7FixedStart-v0"

            # Let's construct this as a series (starting with one)
            # of changes to any previous state.

            a, b, c, d, f = st.columns(5)

            with a:
                current_len = st.session_state.current_len
                max_len = st.session_state.max_len

                if current_len == 1:
                    st.write("No previous states to modify. Timestep = 1")
                    timestep_to_modify = max_len - 1
                else:
                    timestep_to_modify = st.selectbox(
                        "Timestep",
                        range(current_len - max_len - 1, 0),
                        format_func=lambda x: st.session_state.labels[1::3][x],
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
            obs = st.session_state.obs[0][timestep_to_modify].clone()

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
                    image = preview_state_update(env, obs)
                    fig = px.imshow(image)
                    st.plotly_chart(fig, use_container_width=True)
                with corrupt_col:
                    st.write("Corrupted")
                    image = preview_state_update(env, corrupt_obs)
                    fig = px.imshow(image)
                    st.plotly_chart(fig, use_container_width=True)

            corrupted_tokens = get_modified_tokens_from_app_state(
                dt, corrupt_obs=corrupt_obs, position=timestep_to_modify
            )

        else:
            st.warning("Not implemented yet")

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

        (
            residual_stream_tab,
            residual_stream_by_block_tab,
            head_all_positions_tab,
            head_all_positions_by_component_tab,
        ) = st.tabs(
            [
                "Residual Stream",
                "Residual Stream via Attn/MLP",
                "Head All Positions",
                "Head by Component (All Positions)",
            ]
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

    return


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
