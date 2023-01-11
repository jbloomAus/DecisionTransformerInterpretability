import time

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch as t

from src.decision_transformer.model import DecisionTransformer
from src.decision_transformer.offline_dataset import TrajectoryLoader
from src.environments import make_env
import streamlit.components.v1 as components

start = time.time()
st.title("MiniGrid Interpretability Playground")
st.write(
    '''
    To do:
    - new game made: Start from a random environment of the kind your dt was trained on. Observe the state and it's corresponding action. Then, use the dt to predict the next action. You can choose to either take this action, or one of your own choosing. Iterate. 
    '''
)

trajectory_path = "trajectories/MiniGrid-Dynamic-Obstacles-8x8-v0c8c5dccc-b418-492e-bdf8-2c21256cd9f3.pkl"
model_path = "artifacts/MiniGrid-Dynamic-Obstacles-8x8-v0__Dev__1__1673368088:v0/MiniGrid-Dynamic-Obstacles-8x8-v0__Dev__1__1673368088.pt"

action_string_to_id = {"left": 0, "right": 1, "forward": 2, "pickup": 3, "drop": 4, "toggle": 5, "done": 6}
action_id_to_string = {v: k for k, v in action_string_to_id.items()}


@st.cache(allow_output_mutation=True)
def get_env_and_dt(trajectory_path, model_path):
    trajectory_data_set = TrajectoryLoader(trajectory_path, pct_traj=1, device="cpu")
    env_id = trajectory_data_set.metadata['args']['env_id']
    env = make_env(env_id, seed = 1, idx = 0, capture_video=False, run_name = "dev", fully_observed=False, max_steps=30)
    env = env()

    dt = DecisionTransformer(
        env = env, 
        d_model = 128,
        n_heads = 4,
        d_mlp = 256,
        n_layers = 2,
        layer_norm=False,
        state_embedding_type="grid", # hard-coded for now to minigrid.
        max_timestep=1000) # Our DT must have a context window large enough

    dt.load_state_dict(t.load(model_path))
    return env, dt

def render_env(env):
    img = env.render()
    # use matplotlib to render the image
    fig, ax = plt.subplots()
    ax.imshow(img)
    plt.axis('off')
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    return fig


st.subheader("Game Screen")


if "env" not in st.session_state or "dt" not in st.session_state:
    st.write("Loading environment and decision transformer...")
    env, dt = get_env_and_dt(trajectory_path, model_path)
    obs, _ = env.reset()
    fig = render_env(env)
    st.pyplot(fig)
else:
    env = st.session_state.env
    dt = st.session_state.dt
    fig = render_env(env)
    # st.pyplot(fig)
    if "action" in st.session_state:
        action = st.session_state.action
        st.write(f"just took action '{action_id_to_string[st.session_state.action]}'")
        # st.experimental_rerun()
        del action 
        del st.session_state.action



action = st.selectbox("Select an action", ["left", "right", "forward", "pickup", "drop", "toggle", "done"], index = 6)
action = action_string_to_id[action]   
obs, reward, done, trunc, info = env.step(action)
fig = render_env(env)
st.pyplot(fig)
st.write(f"reward: {reward}")
st.write(f"done: {done}")
st.write(f"trunc: {trunc}")
st.write(f"info: {info}")



# render_env(env)

st.session_state.env = env
st.session_state.dt = dt
st.session_state.action = action
st.session_state.obs = done
st.session_state.done = done
st.session_state.trunc = trunc
st.session_state.info = info


def store_trajectory(state, action, obs, reward, done, trunc, info):
    if "trajectories" not in st.session_state:
        st.session_state.trajectories = []
    st.session_state.trajectories.append((state, action, obs, reward, done, trunc, info))

if st.button("reset"):
    obs = env.reset()
    if "action" in st.session_state:
        del st.session_state.action

end = time.time()
st.write(f"Time taken: {end - start}")

print("I was executed")

