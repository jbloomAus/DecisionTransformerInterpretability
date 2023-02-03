import pandas as pd
import plotly.express as px
import streamlit as st
import torch as t
from einops import rearrange
from fancy_einsum import einsum
from minigrid.core.constants import IDX_TO_OBJECT, IDX_TO_COLOR
from .constants import IDX_TO_STATE, IDX_TO_ACTION, three_channel_schema, twenty_idx_format_func

from .visualizations import plot_attention_pattern_single, plot_single_residual_stream_contributions
from .analysis import get_residual_decomp
from .utils import fancy_histogram, fancy_imshow

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
        heads = st.multiselect("Select Heads", options=list(range(dt.n_heads)), default=list(range(dt.n_heads)), key="heads attention")

        if dt.n_layers == 1:
            plot_attention_pattern_single(cache,0, softmax=softmax, specific_heads=heads)
        else:
            layer = st.slider("Layer", min_value=0, max_value=dt.n_layers-1, value=0, step=1)
            plot_attention_pattern_single(cache,layer, softmax=softmax, specific_heads=heads)

def show_residual_stream_contributions_single(dt, cache, logit_dir):
    with st.expander("Show Residual Stream Contributions at current Reward-to-Go"):


        residual_decomp = get_residual_decomp(dt, cache, logit_dir)

        # this plot assumes you only have a single dim
        plot_single_residual_stream_contributions(residual_decomp)

    return 

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
            "RTG Range", 
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

        if st.checkbox("Show Logit Scan"):
            st.plotly_chart(fig, use_container_width=True)

        total_dir = x[:,1,:] @ logit_dir

        # Now let's do the inner product with the logit dir of the components.
        decomp = get_residual_decomp(dt, cache,logit_dir)
        
        df = pd.DataFrame(decomp)
        df["RTG"] = rtg.squeeze(1).squeeze(1).detach().cpu().numpy()
        df["Total Dir"] = total_dir.squeeze(0).detach().cpu().numpy()
        
        # total dir is pretty much accurate
        assert (total_dir.squeeze(0).detach() - df[list(decomp.keys())].sum(axis=1)).mean() < 1e-5
        
        if st.checkbox("Show component contributions"):
            # make a multiselect to choose the decomp keys to compare
            decomp_keys = st.multiselect("Choose components to compare", list(decomp.keys()) + ["Total Dir"], default=list(decomp.keys())+ ["Total Dir"])

            fig = px.line(df, x="RTG", y=decomp_keys, title="Residual Stream Contributions in Directional Analysis")

            fig.add_vline(x=-1, line_dash="dot", line_width=1, line_color="white")
            fig.add_vline(x=0, line_dash="dot", line_width=1, line_color="white")
            fig.add_vline(x=1, line_dash="dot", line_width=1, line_color="white")

            # add a little more margin to the top
            st.plotly_chart(fig, use_container_width=True)

        if st.checkbox("Show Attention Scan"):
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

def render_observation_view(dt, tokens, logit_dir):

    last_obs = st.session_state.obs[0][-1]
    
    last_obs_reshaped = rearrange(last_obs, "h w c -> c h w")

    n_channels = dt.env.observation_space['image'].shape[-1]
    height = dt.env.observation_space['image'].shape[0]
    width = dt.env.observation_space['image'].shape[1]
    
    weights = dt.state_encoder.weight.detach().cpu()

    weights_reshaped = rearrange(weights, "d (c h w) -> c d h w",c=n_channels, h=height, w=width)

    embeddings = einsum(
        "c d h w, c h w -> c d",
        weights_reshaped,
        last_obs_reshaped.to(t.float32)
    )

    weight_projections = einsum(
        "d, c d h w -> c h w",
        logit_dir,
        weights_reshaped
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
            format_func = lambda x: three_channel_schema[x] 
        else: 
            format_func = twenty_idx_format_func

        selected_channels = st.multiselect(
            "Select Observation Channels", 
            options=list(range(n_channels)), 
            format_func= format_func,
            key="channels obs", 
            default=[0,1,2]
        )
        n_selected_channels = len(selected_channels)

        check_columns = st.columns(4)
        with check_columns[0]:
            contributions_check = st.checkbox("Show contributions", value=True)
        with check_columns[1]:
            input_channel_check = st.checkbox("Show input channels", value=True)
        with check_columns[2]:
            weight_proj_check = st.checkbox("Show channel weight proj onto logit dir", value=True)
        with check_columns[3]:
            activ_proj_check = st.checkbox("Show channel activation proj onto logit dir", value=True)
            
        if contributions_check:

            contributions = {
                format_func(i): (embeddings[i] @ logit_dir).item() for i in selected_channels
            }

            if dt.time_embedding_type == "linear":
                time_contribution = (time_embedding @ logit_dir).item()
            else:
                time_contribution = (time_embedding[0] @ logit_dir).item()

            token_contribution = (tokens[0][1] @ logit_dir).item()

            contributions = {**contributions,  "time": time_contribution, "token": token_contribution}

            fig = px.bar(
                contributions.items(),
                x=0,
                y=1,
                labels={
                    "0": "Channel",
                    "1": "Contribution"
                },
                text=1
            )

            # add the value to the bar
            fig.update_traces(texttemplate='%{text:.3f}', textposition='auto')
            fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
            fig.update_yaxes(range=[-8,8])
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
                    fancy_imshow(weight_projections[channel].detach().numpy().T)
                    fancy_histogram(weight_projections[channel].detach().numpy().flatten())

        if activ_proj_check:
            columns = st.columns(n_selected_channels)
            for i, channel in enumerate(selected_channels):
                with columns[i]:
                    st.write(format_func(channel))
                    fancy_imshow(activation_projection[channel].detach().numpy().T)
                    fancy_histogram(activation_projection[channel].detach().numpy().flatten())

        

def project_weights_onto_dir(weights, dir):
    return t.einsum("d, d h w -> h w", dir, weights.reshape(128,7,7)).detach()

