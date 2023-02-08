import streamlit as st
import torch as t
from .environment import get_env_and_dt, get_action_from_user

action_string_to_id = {"left": 0, "right": 1, "forward": 2,
                       "pickup": 3, "drop": 4, "toggle": 5, "done": 6}
action_id_to_string = {v: k for k, v in action_string_to_id.items()}


def initialize_playground(model_path, initial_rtg):

    if "env" not in st.session_state or "dt" not in st.session_state:

        env, dt = get_env_and_dt(model_path)
        obs, _ = env.reset()

        # initilize the session state trajectory details
        st.session_state.obs = t.tensor(obs['image']).unsqueeze(0).unsqueeze(0)
        st.session_state.rtg = t.tensor(
            [initial_rtg]).unsqueeze(0).unsqueeze(0)
        st.session_state.reward = t.tensor([0]).unsqueeze(0).unsqueeze(0)
        st.session_state.a = t.tensor([0]).unsqueeze(0).unsqueeze(0)
        st.session_state.timesteps = t.tensor([0]).unsqueeze(0).unsqueeze(0)
        st.session_state.dt = dt
    else:
        env = st.session_state.env
        dt = st.session_state.dt

    if "action" in st.session_state:
        action = st.session_state.action
        if isinstance(action, str):
            action = action_string_to_id[action]
        st.write(
            f"just took action '{action_id_to_string[st.session_state.action]}'")
        del action
        del st.session_state.action
    else:
        get_action_from_user(env, initial_rtg)

    return env, dt
