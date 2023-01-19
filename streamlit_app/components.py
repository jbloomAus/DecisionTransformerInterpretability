import numpy as np
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
import torch as t
from einops import rearrange

from src.visualization import render_minigrid_observations

from .environment import get_action_preds
from .utils import read_index_html
from .visualizations import (plot_action_preds, plot_attention_pattern,
                             render_env)


def render_game_screen(dt, env):
    columns = st.columns(2)
    with columns[0]:
        action_preds, x, cache, tokens = get_action_preds(dt)
        plot_action_preds(action_preds)
    with columns[1]:
        fig = render_env(env)
        st.pyplot(fig)

    return x, cache, tokens

def render_observation_view(dt, env, tokens, logit_dir):
    
    obs = st.session_state.obs[0]
    last_obs = obs[-1]

    weights = dt.state_encoder.weight.detach().cpu()
    last_obs = st.session_state.obs[0][-1]

    weights_objects = weights[:,:49]#.reshape(128, 7, 7)
    weights_colors = weights[:,49:98]#.reshape(128, 7, 7)
    weights_states = weights[:,98:]#.reshape(128, 7, 7)

    last_obs_reshaped = rearrange(last_obs, "h w c -> (c h w)").to(t.float32).contiguous()
    state_encoding = last_obs_reshaped @  dt.state_encoder.weight.detach().cpu().T
    time_embedding = dt.time_embedding(st.session_state.timesteps[0][-1])

    t.testing.assert_allclose(
        tokens[0][1] - time_embedding[0],
        state_encoding
    )

    last_obs_reshaped = rearrange(last_obs, "h w c -> c h w")
    obj_embedding = weights_objects @ last_obs_reshaped[0].flatten().to(t.float32)
    col_embedding = weights_colors @ last_obs_reshaped[1].flatten().to(t.float32)
    state_embedding = weights_states @ last_obs_reshaped[2].flatten().to(t.float32)

    # ok now we can confirm that the state embedding is the same as the object embedding + color embedding
    t.testing.assert_allclose(
            tokens[0][1] - time_embedding[0],
            obj_embedding + col_embedding + state_embedding
    )


    with st.expander("Show observation view"):
        obj_contribution = (obj_embedding @ logit_dir).item()
        st.write("dot production of object embedding with forward:", obj_contribution) # tokens 

        col_contribution = (col_embedding @ logit_dir).item()
        st.write("dot production of colour embedding with forward:", col_contribution) # tokens 

        time_contribution = (time_embedding[0] @ logit_dir).item()
        st.write("dot production of time embedding with forward:", time_contribution) # tokens 

        state_contribution = (state_embedding @ logit_dir).item()
        st.write("dot production of state embedding with forward:", state_contribution) # tokens

        st.write("Sum of contributions", obj_contribution + col_contribution + time_contribution + state_contribution)

        token_contribution = (tokens[0][1] @ logit_dir).item()
        st.write("dot production of first token embedding with forward:", token_contribution) # tokens 

        def project_weights_onto_dir(weights, dir):
            return t.einsum("d, d h w -> h w", dir, weights.reshape(128,7,7)).detach()

        st.write("projecting weights onto forward direction")

        def plot_weights_obs_and_proj(weights, obs, logit_dir):
            proj = project_weights_onto_dir(weights, logit_dir)
            fig = px.imshow(obs.T)
            fig.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True, autosize=False, width =900)
            fig = px.imshow(proj.T.detach().numpy(), color_continuous_midpoint=0)
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
            weight_proj = proj * obs
            fig = px.imshow(weight_proj.T.detach().numpy(), color_continuous_midpoint=0)
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True, height=0.1, autosize=False)

        a,b,c = st.columns(3)
        with a:
            plot_weights_obs_and_proj(
                weights_objects,
                last_obs_reshaped.reshape(3,7,7)[0].detach().numpy(),
                logit_dir
            )
        with b:
            plot_weights_obs_and_proj(
                weights_colors,
                last_obs_reshaped.reshape(3,7,7)[1].detach().numpy(),
                logit_dir
            )
        with c:
            plot_weights_obs_and_proj(
                weights_states,
                last_obs_reshaped.reshape(3,7,7)[2].detach().numpy(),
                logit_dir
            )

def hyperpar_side_bar():
    with st.sidebar:
        st.subheader("Hyperparameters:")

        initial_rtg = st.slider("Initial RTG", min_value=-1.0, max_value=1.0, value=0.9, step=0.01)
        if "rtg" in st.session_state:
            st.session_state.rtg = initial_rtg - st.session_state.reward

    return initial_rtg

def show_attention_pattern(dt, cache):
    with st.expander("show attention pattern"):
        if dt.n_layers == 1:
            plot_attention_pattern(cache,0)
        else:
            layer = st.slider("Layer", min_value=0, max_value=dt.n_layers-1, value=0, step=1)
            plot_attention_pattern(cache,layer)

def show_residual_stream_contributions(dt, x, cache, tokens, logit_dir):
    with st.expander("Show residual stream contributions:"):
        x_action = x[0][1]
        # st.write(dt.action_embedding.weight.shape)
        st.write("action embedding: ", x_action.shape)
        st.write("dot production of x_action with forward:", (x_action @ logit_dir).item()) # technically  forward over right


        st.latex(
            r'''
            x = s_{original} + r_{mlp} + h_{1.1} + h_{1.2} \newline
            '''
        )
        # x_action = s_{original} + r_{mlp} + h_{1.1} + h_{1.2}
        pos_contribution = (cache["hook_pos_embed"][1] @ logit_dir).item()
        st.write("dot production of pos_embed with forward:", pos_contribution) # pos embed

        token_contribution = (tokens[0][1] @ logit_dir).item()
        st.write("dot production of first token embedding with forward:", token_contribution) # tokens 

        head_0_output = cache["blocks.0.attn.hook_z"][:,0,:] @ dt.transformer.blocks[0].attn.W_O[0]
        head_0_contribution = (head_0_output[1] @ logit_dir).item()
        st.write("dot production of head_0_output with forward:", head_0_contribution)
        head_1_output = cache["blocks.0.attn.hook_z"][:,1,:] @ dt.transformer.blocks[0].attn.W_O[1]
        head_1_contribution = (head_1_output[1] @ logit_dir).item()
        st.write("dot production of head_1_output with forward:", head_1_contribution)
        mlp_output = cache["blocks.0.hook_mlp_out"][1]
        mlp_contribution =   (mlp_output @ logit_dir).item()
        st.write("dot production of mlp_output with forward:", mlp_contribution)

        state_dict = dt.state_dict()

        attn_bias_0_dir = state_dict['transformer.blocks.0.attn.b_O']
        attn_bias_contribution = (attn_bias_0_dir @ logit_dir).item()
        st.write( "dot production of attn_bias_0_dir with forward:", attn_bias_contribution)

        # mlp_bias_0_dir = state_dict['transformer.blocks.0.mlp.b_out']
        # mlp_bias_contribution = (mlp_bias_0_dir @ logit_dir).item()
        # st.write( "dot production of mlp_bias_0_dir with forward:", mlp_bias_contribution)

        sum_of_contributions = token_contribution + head_0_contribution + \
                    head_1_contribution + mlp_contribution + \
                    pos_contribution + attn_bias_contribution
        st.write("sum over contributions:", sum_of_contributions)
        percent_explained = np.abs(sum_of_contributions) / (x_action @ logit_dir).abs().item()
        st.write("percent explained:", percent_explained)

        st.write("This appears to mostly explain how each component of the residual stream contributes to the action prediction.")
        return logit_dir

def render_trajectory_details():
    with st.expander("Trajectory Details"):
        # write out actions, rtgs, rewards, and timesteps
        st.write(f"actions: {st.session_state.a[0].squeeze(-1).tolist()}")
        st.write(f"rtgs: {st.session_state.rtg[0].squeeze(-1).tolist()}")
        st.write(f"rewards: {st.session_state.reward[0].squeeze(-1).tolist()}")
        st.write(f"timesteps: {st.session_state.timesteps[0].squeeze(-1).tolist()}")

def reset_button():
    if st.button("reset"):
        del st.session_state.env
        del st.session_state.dt
        st.experimental_rerun()

def record_keypresses():
    components.html(
        read_index_html(),
        height=0,
        width=0,
    )