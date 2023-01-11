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

def get_action_preds():
    
    state_preds, action_preds, reward_preds = dt.forward(
            states=st.session_state.obs, 
            actions=st.session_state.a, 
            rtgs=st.session_state.rtg, 
            timesteps=st.session_state.timesteps
    )

    # make bar chart of action_preds
    action_preds = action_preds.squeeze(0).squeeze(0)
    action_preds = action_preds.detach().numpy()
    # softmax
    action_preds = np.exp(action_preds) / np.sum(np.exp(action_preds), axis=0)
    st.bar_chart(action_preds)

st.subheader("Game Screen")

initial_rtg = st.slider("Initial RTG", min_value=0.0, max_value=1.0, value=0.5, step=0.01)


def respond_to_action(env, action):
    new_obs, reward, done, trunc, info = env.step(action)

    # append to session state
    st.session_state.obs = t.cat(
                [st.session_state.obs, t.tensor(new_obs['image']).unsqueeze(0).unsqueeze(0)], dim=1)
    # print(t.tensor(action).unsqueeze(0).unsqueeze(0).shape)
    st.session_state.a = t.cat(
                [st.session_state.a, t.tensor([action]).unsqueeze(0).unsqueeze(0)], dim=1)
    st.session_state.reward = t.cat(
                [st.session_state.reward, t.tensor([reward]).unsqueeze(0).unsqueeze(0)], dim=1)

    rtg = initial_rtg - st.session_state.reward.sum()

    st.session_state.rtg = t.cat(
                [st.session_state.rtg, t.tensor([rtg]).unsqueeze(0).unsqueeze(0)], dim=1)
    time = st.session_state.timesteps[-1][-1] + 1
    st.session_state.timesteps = t.cat(
                [st.session_state.timesteps, time.clone().detach().unsqueeze(0).unsqueeze(0)], dim=1)

    st.write(f"reward: {reward}")
    st.write(f"done: {done}")
    st.write(f"trunc: {trunc}")
    st.write(f"info: {info}")

def get_action_from_user(env):

    # create a series of buttons for each action
    button_columns = st.columns(7)
    with button_columns[0]:
        left_button = st.button("Left", key = "left_button")
    with button_columns[1]:
        right_button = st.button("Right", key = "right_button")
    with button_columns[2]:
        forward_button = st.button("Forward", key = "forward_button")
    with button_columns[3]:
        pickup_button = st.button("Pickup", key = "pickup_button")
    with button_columns[4]:
        drop_button = st.button("Drop", key = "drop_button")
    with button_columns[5]:
        toggle_button = st.button("Toggle", key = "toggle_button")
    with button_columns[6]:
        done_button = st.button("Done", key = "done_button")

    # if any of the buttons are pressed, take the corresponding action
    if left_button:
        action = 0
        respond_to_action(env, action)
    elif right_button:
        action = 1
        respond_to_action(env, action)
    elif forward_button:
        action = 2
        respond_to_action(env, action)
    elif pickup_button:
        action = 3
        respond_to_action(env, action)
    elif drop_button:
        action = 4
        respond_to_action(env, action)
    elif toggle_button:
        action = 5
        respond_to_action(env, action)
    elif done_button:
        action = 6
        respond_to_action(env, action)

if "env" not in st.session_state or "dt" not in st.session_state:
    st.write("Loading environment and decision transformer...")
    env, dt = get_env_and_dt(trajectory_path, model_path)
    obs, _ = env.reset()

    # initilize the session state trajectory details
    st.session_state.obs = t.tensor(obs['image']).unsqueeze(0).unsqueeze(0)
    st.session_state.rtg = t.tensor([initial_rtg]).unsqueeze(0).unsqueeze(0)
    st.session_state.reward = t.tensor([0]).unsqueeze(0).unsqueeze(0)
    st.session_state.a = t.tensor([0]).unsqueeze(0).unsqueeze(0)
    st.session_state.timesteps = t.tensor([0]).unsqueeze(0).unsqueeze(0)

    get_action_preds()

else:
    env = st.session_state.env
    dt = st.session_state.dt

if "action" in st.session_state:
    action = st.session_state.action
    if isinstance(action, str):
        action = action_string_to_id[action]
    st.write(f"just took action '{action_id_to_string[st.session_state.action]}'")
    # st.experimental_rerun()
    del action 
    del st.session_state.action
else:
    get_action_from_user(env)



fig = render_env(env)
st.pyplot(fig)

st.session_state.env = env
st.session_state.dt = dt



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

def read_index_html():
    with open("index.html") as f:
        return f.read()

components.html(
    read_index_html(),
    height=0,
    width=0,
)