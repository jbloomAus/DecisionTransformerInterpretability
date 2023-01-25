import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
import torch as t

from .environment import get_action_preds
from .utils import read_index_html
from .visualizations import plot_action_preds, render_env


def render_game_screen(dt, env):
    columns = st.columns(2)
    with columns[0]:
        st.write(f"Reward Objective: {round(st.session_state.rtg[0][-1].item(),2)}")
        action_preds, x, cache, tokens = get_action_preds(dt)
        plot_action_preds(action_preds)
    with columns[1]:
        current_time = st.session_state.timesteps - st.session_state.timestep_adjustment
        st.write(f"Current Time: {int(current_time[0][-1].item())}")
        fig = render_env(env)
        st.pyplot(fig)

    return x, cache, tokens

def hyperpar_side_bar():
    with st.sidebar:
        st.subheader("Hyperparameters")
        allow_extrapolation = st.checkbox("Allow extrapolation")
        st.session_state.allow_extrapolation = allow_extrapolation
        if allow_extrapolation:
            initial_rtg = st.slider("Initial RTG", min_value=-10.0, max_value=10.0, value=0.91, step=0.01)
        else:
            initial_rtg = st.slider("Initial RTG", min_value=-1.0, max_value=1.0, value=0.9, step=0.01)
        if "rtg" in st.session_state:
            st.session_state.rtg = initial_rtg - st.session_state.reward

        timestep_adjustment = st.slider("Timestep Adjustment", min_value=-100.0, max_value=100.0, value=0.0, step=1.0)
        st.session_state.timestep_adjustment = timestep_adjustment

    return initial_rtg

def show_attn_scan(dt):

    with st.expander("Attention Scan"):
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

        st.write(cache.keys())

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