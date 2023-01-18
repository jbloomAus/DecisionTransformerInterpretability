import torch as t
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
from .visualizations import plot_attention_pattern, plot_action_preds, render_env
from .utils import read_index_html
from .environment import get_action_preds
from src.visualization import render_minigrid_observations

def render_game_screen(dt, env):
    columns = st.columns(2)
    with columns[0]:
        action_preds, x, cache, tokens = get_action_preds(dt)
        plot_action_preds(action_preds)
    with columns[1]:
        fig = render_env(env)
        st.pyplot(fig)

    return x, cache, tokens

def render_observation_view(dt, env, tokens, forward_dir):
    
    with st.expander("How are observations encoded?"):
        obs = st.session_state.obs[0]
        last_obs = obs[-1]
        st.write(obs.shape)
        st.write(last_obs.shape)

        #  obs is a grid of (OBJECT_IDX, COLOR_IDX, STATE)
        # let's extract the indices of the weights which will match the object_idx in the flattened array
        indices = t.arange(7*7*3)
        object_indices = indices[2::3]

        st.write("how do we encode the observations?")
        # st.write(obs.flatten())

        st.write("what is the agent currently looking at?")
        st.write(f"observation: {obs.shape}")
        result = render_minigrid_observations(env, last_obs.unsqueeze(0))
        st.image(result)

        
        
        obs_selection = st.selectbox("Visualize which obs?", ["none", "all", "object", "state","color"], key = "obs_selection")
        if obs_selection == "object":
        # last_obs = rearrange(last_obs, 'height width channel -> channel height width')
            objects = last_obs.flatten()[::3].reshape(7,7)
            fig = px.imshow(objects.T)
            st.plotly_chart(fig)
        elif obs_selection == "color":
            colors = last_obs.flatten()[1::3].reshape(7,7)
            fig = px.imshow(colors.T)
            st.plotly_chart(fig)
        elif obs_selection == "state":
            states = last_obs.flatten()[2::3].reshape(7,7)
            fig = px.imshow(states.T)
            st.plotly_chart(fig)


        weights = dt.state_encoder.weight.detach().cpu()
        st.write(weights.shape) # weights maps dimensions in the observation (148 flattened) to neurons (128)
        weights_objects_only = t.zeros(weights.shape)
        weights_objects_only[:, ::3] = weights[:, ::3]
        weights_states_only = t.zeros(weights.shape)
        weights_states_only[:, 2::3] = weights[:, 2::3]
        weights_colors_only = t.zeros(weights.shape)
        weights_colors_only[:, 1::3] = weights[:, 1::3]
        activations = weights @ last_obs.flatten().to(t.float32)
        activations_object = weights_objects_only @ last_obs.flatten().to(t.float32)
        activations_state = weights_states_only @ last_obs.flatten().to(t.float32)
        activations_color = weights_colors_only @ last_obs.flatten().to(t.float32)

        st.write('bar chart of activations by neuron')
        fig = px.bar(
            pd.DataFrame(
                dict(
                object = activations_object,
                states = activations_state, 
                color = activations_color)
        ), 
        )

        st.plotly_chart(fig, use_container_width=True)
        st.write(
            "most active neurons: ",  activations.abs().argsort(descending=True)[:10]
        )
        

        weight_selection = st.selectbox("Visualize which weights?", ["none","all", "object", "state","color"])
        if weight_selection == "all":
            st.write("weights - all") 
            st.write(weights.shape)
            # weights = rearrange()
            fig = px.imshow(weights.numpy())
            st.plotly_chart(fig, use_container_width=True)
        elif weight_selection == "object":
            st.write("object weights by position: plotting the sum of all weights associated with objects at each position")
            fig = px.imshow(weights[:,:49].reshape(128,7,7).sum(dim=0).T)
            st.plotly_chart(fig, use_container_width=True)
        elif weight_selection == "state":
            st.write("state weights")
            fig = px.imshow(weights[:,98:].reshape(128,7,7).sum(dim=0).T)
            st.plotly_chart(fig, use_container_width=True)
        elif weight_selection == "color":
            st.write("color weights")
            fig = px.imshow(weights[:,49:98].reshape(128,7,7).sum(dim=0).T)
            st.plotly_chart(fig, use_container_width=True)

        # all activations over trajectory
        all_activations = obs.flatten(1).to(t.float32) @ weights.T
        st.write(all_activations.shape)
        fig = px.line(all_activations)
        # print(all_activations)
        st.plotly_chart(fig, key = 3, use_container_width=True)

    weights = dt.state_encoder.weight.detach().cpu()
    obs = st.session_state.obs[0]
    last_obs = obs[-1]

    with st.expander("let's see if we can map forward dir to obs tokens"):

        show_weights = st.checkbox("show weights")
        d,a,b,c = st.columns(4)
        with d:
            st.write("current partial view")
            result = render_minigrid_observations(env, last_obs.unsqueeze(0))
            st.image(result)
        with a:
            st.write("objects")
            result =t.einsum("d, d h w -> w h", forward_dir, weights[:,:49].reshape(128,7,7))
            if show_weights:
                fig = px.imshow(result.flip(0).detach().numpy(), color_continuous_midpoint=0)
                st.plotly_chart(fig, use_container_width=True)

            objects = last_obs.flatten()[::3].reshape(7,7)
            object_projection_forward = objects*result.flip(0)
            fig = px.imshow(object_projection_forward.detach(), color_continuous_midpoint=0)
            st.plotly_chart(fig, use_container_width=True)

            st.write(object_projection_forward.sum().item())
        with b:
            # this might explain the chirality?
            st.write("color")
            result = t.einsum("d, d h w -> h w", forward_dir, weights[:,49:98].reshape(128,7,7))
            if show_weights:
                fig = px.imshow(result.flip(0).detach().numpy(), color_continuous_midpoint=0)
                st.plotly_chart(fig, use_container_width=True)

            colors = last_obs.flatten()[1::3].reshape(7,7)

            color_projection_forward = colors*result.flip(0)
            fig = px.imshow(color_projection_forward.detach(), color_continuous_midpoint=0)
            st.plotly_chart(fig, use_container_width=True)

            st.write(color_projection_forward.sum().item())
        with c:
            st.write("states")
            result =t.einsum("d, d h w -> h w", forward_dir, weights[:,98:].reshape(128,7,7))
            if show_weights:
                fig = px.imshow(result.detach().numpy(), color_continuous_midpoint=0)
                st.plotly_chart(fig, use_container_width=True)

            states = last_obs.flatten()[2::3].reshape(7,7)
            state_projection_forward = states*result.flip(0)
            fig = px.imshow(state_projection_forward.detach(), color_continuous_midpoint=0)
            st.plotly_chart(fig, use_container_width=True)
            st.write(state_projection_forward.sum().item())

        st.write("Sum of Object, Color, State projections:",
            object_projection_forward.sum() + \
                color_projection_forward.sum())

        token_contribution = (tokens[0][1] @ forward_dir).item()
        st.write("dot production of first token embedding with forward:", token_contribution) # tokens 

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

def show_residual_stream_contributions(dt, x, cache, tokens):
    with st.expander("Show residual stream contributions:"):
        x_action = x[0][1]
        # st.write(dt.action_embedding.weight.shape)
        st.write("action embedding: ", x_action.shape)
        forward_dir = dt.predict_actions.weight[2]-dt.predict_actions.weight[1]
        st.write("dot production of x_action with forward:", (x_action @ forward_dir).item()) # technically  forward over right


        st.latex(
            r'''
            x = s_{original} + r_{mlp} + h_{1.1} + h_{1.2} \newline
            '''
        )
        # x_action = s_{original} + r_{mlp} + h_{1.1} + h_{1.2}
        pos_contribution = (cache["hook_pos_embed"][1] @ forward_dir).item()
        st.write("dot production of pos_embed with forward:", pos_contribution) # pos embed

        token_contribution = (tokens[0][1] @ forward_dir).item()
        st.write("dot production of first token embedding with forward:", token_contribution) # tokens 

        head_0_output = cache["blocks.0.attn.hook_z"][:,0,:] @ dt.transformer.blocks[0].attn.W_O[0]
        head_0_contribution = (head_0_output[1] @ forward_dir).item()
        st.write("dot production of head_0_output with forward:", head_0_contribution)
        head_1_output = cache["blocks.0.attn.hook_z"][:,1,:] @ dt.transformer.blocks[0].attn.W_O[1]
        head_1_contribution = (head_1_output[1] @ forward_dir).item()
        st.write("dot production of head_1_output with forward:", head_1_contribution)
        mlp_output = cache["blocks.0.hook_mlp_out"][1]
        mlp_contribution =   (mlp_output @ forward_dir).item()
        st.write("dot production of mlp_output with forward:", mlp_contribution)

        state_dict = dt.state_dict()

        attn_bias_0_dir = state_dict['transformer.blocks.0.attn.b_O']
        attn_bias_contribution = (attn_bias_0_dir @ forward_dir).item()
        st.write( "dot production of attn_bias_0_dir with forward:", attn_bias_contribution)

        # mlp_bias_0_dir = state_dict['transformer.blocks.0.mlp.b_out']
        # mlp_bias_contribution = (mlp_bias_0_dir @ forward_dir).item()
        # st.write( "dot production of mlp_bias_0_dir with forward:", mlp_bias_contribution)

        sum_of_contributions = token_contribution + head_0_contribution + \
                    head_1_contribution + mlp_contribution + \
                    pos_contribution + attn_bias_contribution
        st.write("sum over contributions:", sum_of_contributions)
        percent_explained = np.abs(sum_of_contributions) / (x_action @ forward_dir).abs().item()
        st.write("percent explained:", percent_explained)

        st.write("This appears to mostly explain how each component of the residual stream contributes to the action prediction.")
        return forward_dir

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