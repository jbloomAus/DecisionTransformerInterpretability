import streamlit as st
import torch as t

from src.decision_transformer.utils import (
    get_max_len_from_model_type,
    initialize_padding_inputs,
)
from .environment import get_env_and_dt, get_action_from_user

action_string_to_id = {
    "left": 0,
    "right": 1,
    "forward": 2,
    "pickup": 3,
    "drop": 4,
    "toggle": 5,
    "done": 6,
}
action_id_to_string = {v: k for k, v in action_string_to_id.items()}


def initialize_playground(model_path, initial_rtg):
    if "env" not in st.session_state or "dt" not in st.session_state:
        env, dt = get_env_and_dt(model_path)
        obs, _ = env.reset()

        max_len = get_max_len_from_model_type(
            dt.model_type, dt.transformer_config.n_ctx
        )

        action_pad_token = dt.environment_config.action_space.n

        obs, actions, reward, rtg, timesteps, mask = initialize_padding_inputs(
            max_len, obs, initial_rtg, action_pad_token
        )

        rendered_obs = t.from_numpy(env.render()).unsqueeze(0)

        st.session_state.max_len = max_len
        st.session_state.mask = mask
        st.session_state.obs = obs
        st.session_state.rendered_obs = rendered_obs
        st.session_state.reward = reward
        st.session_state.rtg = rtg
        st.session_state.a = actions
        st.session_state.timesteps = timesteps
        st.session_state.dt = dt

    else:
        env = st.session_state.env
        dt = st.session_state.dt

    if "action" in st.session_state:
        action = st.session_state.action
        if isinstance(action, str):
            action = action_string_to_id[action]
        st.write(
            f"just took action '{action_id_to_string[st.session_state.action]}'"
        )
        del action
        del st.session_state.action
    else:
        get_action_from_user(env, initial_rtg)

    return env, dt
