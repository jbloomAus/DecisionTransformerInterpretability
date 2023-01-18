import time
from copy import deepcopy

import gymnasium as gym
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import torch as t

from src.visualization import render_minigrid_observations
from streamlit_app.setup import initialize_playground
from streamlit_app.components import reset_button, show_attention_pattern, hyperpar_side_bar, record_keypresses
from streamlit_app.components import show_residual_stream_contributions, render_trajectory_details, render_game_screen
from streamlit_app.components import render_observation_view
start = time.time()

with st.sidebar:
    st.title("Decision Transformer Interpretability")

model_path = "models/demo_model.pt"
action_string_to_id = {"left": 0, "right": 1, "forward": 2, "pickup": 3, "drop": 4, "toggle": 5, "done": 6}
action_id_to_string = {v: k for k, v in action_string_to_id.items()}

st.session_state.max_len = 1
initial_rtg = hyperpar_side_bar()
env, dt = initialize_playground(model_path, initial_rtg)
x, cache, tokens = render_game_screen(dt, env)
show_attention_pattern(dt, cache)
forward_dir = show_residual_stream_contributions(dt, x, cache, tokens)
render_observation_view(dt, env, tokens, forward_dir)

st.markdown("""---""")

st.session_state.env = env
st.session_state.dt = dt

render_trajectory_details()
reset_button()
end = time.time()
st.write(f"Time taken: {end - start}")
record_keypresses()