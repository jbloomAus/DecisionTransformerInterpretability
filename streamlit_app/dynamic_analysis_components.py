import pandas as pd
import plotly.express as px
import streamlit as st
import torch as t
from einops import rearrange

from .visualizations import plot_attention_patter_single
from .analysis import get_residual_decomp

def show_attention_pattern(dt, cache):

    with st.expander("Attention Pattern at at current Reward-to-Go"):

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
            plot_attention_patter_single(cache,0, softmax=softmax, specific_heads=heads)
        else:
            layer = st.slider("Layer", min_value=0, max_value=dt.n_layers-1, value=0, step=1)
            plot_attention_patter_single(cache,layer, softmax=softmax, specific_heads=heads)

def show_residual_stream_contributions_single(dt, cache, logit_dir):
    with st.expander("Show Residual Stream Contributions at current Reward-to-Go"):


        residual_decomp = get_residual_decomp(dt, cache, logit_dir)

        # this plot assumes you only have a single dim
        for key in residual_decomp.keys():
            residual_decomp[key] = residual_decomp[key].squeeze(0)
            # st.write(key, residual_decomp[key].shape)

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
    with st.expander("Scan Reward-to-Go and Show Residual Contributions"):

        
        batch_size = 1028
        if st.session_state.allow_extrapolation:
            min_value = -10
            max_value = 10
        else:
            min_value = -1
            max_value = 1
        rtg_range = st.slider(
            "Min RTG", 
            min_value=min_value, 
            max_value=max_value, 
            value=(-1,1), 
            step=1
        )
        min_rtg = rtg_range[0]
        max_rtg = rtg_range[1]
        max_len = dt.n_ctx // 3

        if "timestep_adjustment" in st.session_state:
            timesteps = st.session_state.timesteps[:,-max_len:] + st.session_state.timestep_adjustment

        obs = st.session_state.obs[:,-max_len:].repeat(batch_size, 1, 1,1,1)
        a = st.session_state.a[:,-max_len:].repeat(batch_size, 1, 1)
        rtg = st.session_state.rtg[:,-max_len:].repeat(batch_size, 1, 1)
        timesteps = st.session_state.timesteps[:,-max_len:].repeat(batch_size, 1, 1) + st.session_state.timestep_adjustment
        rtg = t.linspace(min_rtg, max_rtg, batch_size).unsqueeze(-1).unsqueeze(-1).repeat(1, max_len, 1)

        if st.checkbox("add timestep noise"):
            # we want to add random integers in the range of a slider to the the timestep, the min/max on slider should be the max timesteps
            if timesteps.max().item() > 0:
                timestep_noise = st.slider(
                    "Timestep Noise", 
                    min_value=1.0, 
                    max_value=timesteps.max().item(), 
                    value=1.0, 
                    step=1.0
                )
                timesteps = timesteps + t.randint(low = int(-1*timestep_noise), high = int(timestep_noise), size=timesteps.shape, device=timesteps.device)
            else:
                st.info("Timestep noise only works when we have more than one timestep.")

        if dt.time_embedding_type == "linear":
            timesteps = timesteps.to(t.float32)
        else:
            timesteps = timesteps.to(t.long)

        tokens = dt.to_tokens(obs, a, rtg, timesteps)

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

        # add a little more margin to the top
        st.plotly_chart(fig, use_container_width=True)

        columns = st.columns(2)
        with columns[0]:
            attention_pattern = cache["attn_scores", 0, "attn"]
            layer = st.selectbox("Layer", list(range(dt.n_layers)))
        with columns[1]:
            head = st.selectbox("Head", list(range(attention_pattern.shape[1])))

        fig = px.line(
            x = t.linspace(min_rtg, max_rtg, batch_size),
            y = attention_pattern[:,head,1,0],
            title=f"Attention State to RTG for Layer {layer} Head {head}",
            labels={
                "x": "RTG",
                "y": "Attention"
            },
        )
        st.plotly_chart(fig, use_container_width=True)

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
    
    weights = dt.state_encoder.weight.detach().cpu()

    weights_objects = weights[:,:49]#.reshape(128, 7, 7)
    weights_colors = weights[:,49:98]#.reshape(128, 7, 7)
    weights_states = weights[:,98:]#.reshape(128, 7, 7)

    last_obs = st.session_state.obs[0][-1]
    
    last_obs_reshaped = rearrange(last_obs, "h w c -> c h w")
    st.write(last_obs_reshaped.shape
    )

    obj_embedding = weights_objects @ last_obs_reshaped[0].flatten().to(t.float32)
    col_embedding = weights_colors @ last_obs_reshaped[1].flatten().to(t.float32)
    state_embedding = weights_states @ last_obs_reshaped[2].flatten().to(t.float32)


    timesteps = st.session_state.timesteps[0][-1]
    if dt.time_embedding_type == "linear":
        timesteps = timesteps.to(t.float32)
    else:
        timesteps = timesteps.to(t.long)

    time_embedding = dt.time_embedding(timesteps)

    assert_channel_decomposition_valid(
        dt, last_obs, tokens, obj_embedding, col_embedding, state_embedding, time_embedding
    )



    with st.expander("Show observation view"):
        obj_contribution = (obj_embedding @ logit_dir).item()
        st.write("dot production of object embedding with forward:", obj_contribution) # tokens 

        col_contribution = (col_embedding @ logit_dir).item()
        st.write("dot production of colour embedding with forward:", col_contribution) # tokens 

        state_contribution = (state_embedding @ logit_dir).item()
        st.write("dot production of state embedding with forward:", state_contribution) # tokens

        if dt.time_embedding_type == "linear":
            time_contribution = (time_embedding @ logit_dir).item()
        else:
            time_contribution = (time_embedding[0] @ logit_dir).item()
        st.write("dot production of time embedding with forward:", time_contribution) # tokens 

        st.write("Sum of contributions", obj_contribution + col_contribution + time_contribution + state_contribution)

        token_contribution = (tokens[0][1] @ logit_dir).item()
        st.write("dot production of first token embedding with forward:", token_contribution) # tokens 

        def project_weights_onto_dir(weights, dir):
            return t.einsum("d, d h w -> h w", dir, weights.reshape(128,7,7)).detach()

        st.write("projecting weights onto forward direction")
        normalize = st.checkbox("Normalize weight_projection", value=True)

        def plot_weights_obs_and_proj(weights, obs, logit_dir, normalize=True):
            fig = px.imshow(obs.T)
            fig.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True, autosize=False, width =900)
            proj = project_weights_onto_dir(weights, logit_dir)
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

def assert_channel_decomposition_valid(dt, last_obs, tokens, obj_embedding, col_embedding, state_embedding, time_embedding):

    last_obs_reshaped = rearrange(last_obs, "h w c -> (c h w)").to(t.float32).contiguous()
    state_encoding = last_obs_reshaped @  dt.state_encoder.weight.detach().cpu().T

    if st.session_state.timestep_adjustment == 0: # don't test otherwise, unnecessary
        t.testing.assert_allclose(
            tokens[0][1] - time_embedding[0],
            state_encoding
        )


    # ok now we can confirm that the state embedding is the same as the object embedding + color embedding
    if st.session_state.timestep_adjustment == 0: # don't test otherwise, unnecessary
        t.testing.assert_allclose(
                tokens[0][1] - time_embedding[0],
                obj_embedding + col_embedding + state_embedding
        )
