import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
import torch as t
from einops import rearrange
from fancy_einsum import einsum

from .environment import get_action_preds
from .utils import read_index_html
from .visualizations import (plot_action_preds, plot_attention_pattern,
                             render_env)
from .analysis import get_residual_decomp

def render_game_screen(dt, env):
    columns = st.columns(2)
    with columns[0]:
        action_preds, x, cache, tokens = get_action_preds(dt)
        plot_action_preds(action_preds)
    with columns[1]:
        fig = render_env(env)
        st.pyplot(fig)

    return x, cache, tokens

def hyperpar_side_bar():
    with st.sidebar:
        st.subheader("Hyperparameters:")
        allow_extrapolation = st.checkbox("Allow extrapolation")
        if allow_extrapolation:
            initial_rtg = st.slider("Initial RTG", min_value=-10.0, max_value=10.0, value=0.9, step=0.01)
        else:
            initial_rtg = st.slider("Initial RTG", min_value=-1.0, max_value=1.0, value=0.9, step=0.01)
        if "rtg" in st.session_state:
            st.session_state.rtg = initial_rtg - st.session_state.reward

        timestep_adjustment = st.slider("Timestep Adjustment", min_value=-100.0, max_value=100.0, value=0.0, step=1.0)
        st.session_state.timestep_adjustment = timestep_adjustment

    return initial_rtg

def show_attention_pattern(dt, cache):

    
    with st.expander("show attention pattern"):


        st.write(cache.keys())

        # QK_circuit = einsum('head d_mod1 d_mod2, d_mod2 d_mod3, d_mod3 d_mod1 -> head d_mod1 d_mod1', W_QK, W_E_state, W_E_rtg)
        # QK_circuit_full = W_E.T @ W_OV @ W_U.T
        # st.write(OV_circuit_full.shape)

        # a, b = st.columns(2)

        # with a:
        #     st.write("Q")
        #     q_cache = cache['blocks.0.attn.hook_q']
        #     fig = px.line(q_cache[:,0].detach().numpy().T)
        #     st.plotly_chart(fig, use_container_width=True)

        # with b:
        #     st.write("K")
        #     k_cache = cache['blocks.0.attn.hook_k']
        #     fig = px.line(k_cache[:,0].detach().numpy().T)
        #     st.plotly_chart(fig, use_container_width=True)


        st.write('---')

        st.latex(
            r'''
            h(x)=\left(A \otimes W_O W_V\right) \cdot x \newline
            '''
        )

        st.latex(
            r'''
            A=\operatorname{softmax}\left(x^T W_Q^T W_K x\right)
            '''
        )

        softmax = st.checkbox("softmax", value=True)
        heads = st.multiselect("Select Heads", options=list(range(dt.n_heads)), default=list(range(dt.n_heads)), key="heads")

        if dt.n_layers == 1:
            plot_attention_pattern(cache,0, softmax=softmax, specific_heads=heads)
        else:
            layer = st.slider("Layer", min_value=0, max_value=dt.n_layers-1, value=0, step=1)
            plot_attention_pattern(cache,layer, softmax=softmax, specific_heads=heads)


def show_qk_circuit(dt):

    with st.expander("show QK circuit"):

        st.write(
            '''
            Usually the QK circuit uses the embedding twice but since we are interested in Atten to 
            '''
        )
        st.latex(
            r'''
            QK_{circuit} = W_E^T W_Q^T W_K W_E
            '''
        )

        W_E_rtg = dt.reward_embedding[0].weight
        W_E_state = dt.state_encoder.weight
        W_Q = dt.transformer.blocks[0].attn.W_Q
        W_K = dt.transformer.blocks[0].attn.W_K


        W_QK = einsum('head d_mod_Q d_head, head d_mod_K d_head -> head d_mod_Q d_mod_K', W_Q, W_K)
        # st.write(W_QK.shape)

        # W_QK_full = W_E_rtg.T @ W_QK @ W_E_state 
        W_QK_full = W_E_state.T @ W_QK @  W_E_rtg
        # st.write(W_QK_full.shape)

        W_QK_full_reshaped = W_QK_full.reshape(2, 1, 3, 7, 7)
        # st.write(W_QK_full_reshaped.shape)

        heads = st.multiselect("Select Heads", options=list(range(dt.n_heads)), key="head qk")

        a, b, c = st.columns(3)
        with a:
            st.write("Object")
        with b:
            st.write("Color")
        with c:
            st.write("State")
        for head in heads:
            st.write("Head", head)
            with a:
                st.plotly_chart(px.imshow(W_QK_full_reshaped[head,0,0].T.detach().numpy(), color_continuous_midpoint=0), use_container_width=True)
            with b:
                st.plotly_chart(px.imshow(W_QK_full_reshaped[head,0,1].T.detach().numpy(), color_continuous_midpoint=0), use_container_width=True)
            with c:
                st.plotly_chart(px.imshow(W_QK_full_reshaped[head,0,2].T.detach().numpy(), color_continuous_midpoint=0), use_container_width=True)


def show_ov_circuit(dt):

    with st.expander("Show OV Circuit"):
        st.subheader("OV circuits")

        st.latex(
            r'''
            OV_{circuit} = W_U \cdot (W_V \cdot W_O) \cdot W_E
            '''
        )

        W_U = dt.predict_actions.weight
        W_O = dt.transformer.blocks[0].attn.W_O
        W_V = dt.transformer.blocks[0].attn.W_V
        W_E = dt.state_encoder.weight
        W_OV = W_V @ W_O

        # st.plotly_chart(px.imshow(W_OV.detach().numpy(), facet_col=0), use_container_width=True)
        OV_circuit_full = W_E.T @ W_OV @ W_U.T

        #reshape the ov circuit
        OV_circuit_full_reshaped = OV_circuit_full.reshape(2, 3, 7, 7, 3)

    
        heads = st.multiselect("Select Heads", options=list(range(dt.n_heads)), key="head ov")

        for head in heads:
            st.write("Head", head)
            for i in range(dt.env.action_space.n):
                st.write("action: ", i)
                st.plotly_chart(
                    px.imshow(
                        OV_circuit_full_reshaped[head][:,:,:,i].transpose(-1,-2).detach().numpy(), 
                        facet_col=0,
                        color_continuous_midpoint=0
                    ), 
                    use_container_width=True)


def show_residual_stream_contributions_single(dt, cache, logit_dir):
    with st.expander("Show residual stream contributions:"):


        residual_decomp = get_residual_decomp(dt, cache, logit_dir)

        # this plot assumes you only have a single dim
        for key in residual_decomp.keys():
            residual_decomp[key] = residual_decomp[key].squeeze(0)
            st.write(key, residual_decomp[key].shape)

        fig = px.bar(
            pd.DataFrame(residual_decomp, index = [0]).T
        )
        fig.update_layout(
            title="Residual Decomposition",
            xaxis_title="Residual Stream Component",
            yaxis_title="Contribution to Action Prediction",
            legend_title="",
        )
        st.plotly_chart(fig, use_container_width=True)

    return logit_dir

def show_rtg_scan(dt, logit_dir):
    with st.expander("Show RTG Scan"):

        
        batch_size = 1028
        min_rtg = st.slider("Min RTG", min_value=-10, max_value=10, value=-1, step=1)
        max_rtg = st.slider("Max RTG", min_value=-10, max_value=10, value=1, step=1)

        max_len = dt.n_ctx // 3

        if "timestep_adjustment" in st.session_state:
            timesteps = st.session_state.timesteps[:,-max_len:] + st.session_state.timestep_adjustment


        obs = st.session_state.obs[:,-max_len:].repeat(batch_size, 1, 1,1,1)
        a = st.session_state.a[:,-max_len:].repeat(batch_size, 1, 1)
        rtg = st.session_state.rtg[:,-max_len:].repeat(batch_size, 1, 1)
        timesteps = st.session_state.timesteps[:,-max_len:].repeat(batch_size, 1, 1) + st.session_state.timestep_adjustment
        rtg = t.linspace(min_rtg, max_rtg, batch_size).unsqueeze(-1).unsqueeze(-1).repeat(1, max_len, 1)

        tokens = dt.to_tokens(obs, a, rtg, timesteps.to(dtype=t.long))

        x, cache = dt.transformer.run_with_cache(tokens, remove_batch_dim=False)
        state_preds, action_preds, reward_preds = dt.get_logits(x, batch_size=batch_size, seq_length=max_len)

        df = pd.DataFrame({
            "RTG": rtg.squeeze(1).squeeze(1).detach().cpu().numpy(),
            "Left": action_preds[:,0,0].detach().cpu().numpy(),
            "Right": action_preds[:,0,1].detach().cpu().numpy(),
            "Forward": action_preds[:,0,2].detach().cpu().numpy()
        })
        
        # st.write(df.head())

        # draw a line graph with left,right forward over RTG
        fig = px.line(df, x="RTG", y=["Left", "Right", "Forward"], title="Action Prediction vs RTG")

        fig.update_layout(
            xaxis_title="RTG",
            yaxis_title="Action Prediction",
            legend_title="",
        )
        # add vertical dotted lines at RTG = -1, RTG = 0, RTG = 1
        fig.add_vline(x=-1, line_dash="dot", line_width=1, line_color="white")
        fig.add_vline(x=0, line_dash="dot", line_width=1, line_color="white")
        fig.add_vline(x=1, line_dash="dot", line_width=1, line_color="white")

        if st.checkbox("Show RTG Scan"):
            st.plotly_chart(fig, use_container_width=True)

        total_dir = x[:,1,:] @ logit_dir

        # Now let's do the inner product with the logit dir of the components.
        decomp = get_residual_decomp(dt, cache,logit_dir)
        
        df = pd.DataFrame(decomp)
        df["RTG"] = rtg.squeeze(1).squeeze(1).detach().cpu().numpy()
        df["Total Dir"] = total_dir.squeeze(0).detach().cpu().numpy()
        
        # total dir is pretty much accurate
        assert (total_dir.squeeze(0).detach() - df[list(decomp.keys())].sum(axis=1)).mean() < 1e-5
        
        # make a multiselect to choose the decomp keys to compare
        decomp_keys = st.multiselect("Choose components to compare", list(decomp.keys()) + ["Total Dir"], default=list(decomp.keys())+ ["Total Dir"])

        fig = px.line(df, x="RTG", y=decomp_keys, title="Residual Stream Contributions in Directional Analysis")

        fig.add_vline(x=-1, line_dash="dot", line_width=1, line_color="white")
        fig.add_vline(x=0, line_dash="dot", line_width=1, line_color="white")
        fig.add_vline(x=1, line_dash="dot", line_width=1, line_color="white")


        st.plotly_chart(fig, use_container_width=True)

        if st.checkbox("Show Correlation"):
            st.plotly_chart(
                px.imshow(
                    df[set(list(decomp.keys()) + ["Total Dir"]) - {"Positional Embedding", "Attention Bias Layer 0"}].corr(),
                    color_continuous_midpoint=0,
                    title="Correlation between RTG and Residual Stream Components",
                    color_continuous_scale="RdBu"
                ),
                use_container_width=True
            )

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

    if st.session_state.timestep_adjustment == 0: # don't test otherwise, unnecessary
        t.testing.assert_allclose(
            tokens[0][1] - time_embedding[0],
            state_encoding
        )

    last_obs_reshaped = rearrange(last_obs, "h w c -> c h w")
    obj_embedding = weights_objects @ last_obs_reshaped[0].flatten().to(t.float32)
    col_embedding = weights_colors @ last_obs_reshaped[1].flatten().to(t.float32)
    state_embedding = weights_states @ last_obs_reshaped[2].flatten().to(t.float32)

    # ok now we can confirm that the state embedding is the same as the object embedding + color embedding
    if st.session_state.timestep_adjustment == 0: # don't test otherwise, unnecessary
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
        normalize = st.checkbox("Normalize weight_projection", value=True)

        def plot_weights_obs_and_proj(weights, obs, logit_dir, normalize=True):
            proj = project_weights_onto_dir(weights, logit_dir)
            fig = px.imshow(obs.T)
            fig.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True, autosize=False, width =900)
            fig = px.imshow(proj.T.detach().numpy(), color_continuous_midpoint=0)
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
            weight_proj = proj * obs
            if normalize:
                weight_proj = 100*weight_proj / (1e-8+ weight_proj.sum()).abs()
            fig = px.imshow(weight_proj.T.detach().numpy(), color_continuous_midpoint=0)
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True, height=0.1, autosize=False)

        a,b,c = st.columns(3)
        with a:
            plot_weights_obs_and_proj(
                weights_objects,
                last_obs_reshaped.reshape(3,7,7)[0].detach().numpy(),
                logit_dir,
                normalize=normalize
            )
        with b:
            plot_weights_obs_and_proj(
                weights_colors,
                last_obs_reshaped.reshape(3,7,7)[1].detach().numpy(),
                logit_dir,
                normalize=normalize
            )
        with c:
            plot_weights_obs_and_proj(
                weights_states,
                last_obs_reshaped.reshape(3,7,7)[2].detach().numpy(),
                logit_dir,
                normalize=normalize
            )

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