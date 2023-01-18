import streamlit as st 
import torch as t

from src.utils import load_decision_transformer
from src.environments import make_env

@st.cache(allow_output_mutation=True)
def get_env_and_dt(model_path):
    env_id = 'MiniGrid-Dynamic-Obstacles-8x8-v0'
    env = make_env(env_id, seed = 4200, idx = 0, capture_video=False, run_name = "dev", fully_observed=False, max_steps=30)
    env = env()

    # dt.load_state_dict(t.load(model_path))
    dt = load_decision_transformer(
        model_path, env
    )
    return env, dt

def get_action_preds(dt):
    max_len = dt.n_ctx // 3
    tokens = dt.to_tokens(
        st.session_state.obs[:,-max_len:], 
        st.session_state.a[:,-max_len:],
        st.session_state.rtg[:,-max_len:],
        st.session_state.timesteps[:,-max_len:]
    )

    x, cache = dt.transformer.run_with_cache(tokens, remove_batch_dim=True)
    state_preds, action_preds, reward_preds = dt.get_logits(x, batch_size=1, seq_length=max_len)

    return action_preds, x, cache, tokens

def respond_to_action(env, action, initial_rtg):
    new_obs, reward, done, trunc, info = env.step(action)
    if done:
        st.balloons()
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

def get_action_from_user(env, initial_rtg):

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
        respond_to_action(env, action, initial_rtg)
    elif right_button:
        action = 1
        respond_to_action(env, action, initial_rtg)
    elif forward_button:
        action = 2
        respond_to_action(env, action, initial_rtg)
    elif pickup_button:
        action = 3
        respond_to_action(env, action, initial_rtg)
    elif drop_button:
        action = 4
        respond_to_action(env, action, initial_rtg)
    elif toggle_button:
        action = 5
        respond_to_action(env, action, initial_rtg)
    elif done_button:
        action = 6
        respond_to_action(env, action, initial_rtg)
