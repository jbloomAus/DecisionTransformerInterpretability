import streamlit as st
import torch as t

from src.decision_transformer.utils import get_max_len_from_model_type
from .environment import get_action_from_user, get_env_and_dt

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
        st.session_state.max_len = max_len
        # initilize the session state trajectory details

        mask = t.concat(
            (
                t.zeros((1, max_len - 1), dtype=t.bool),
                t.ones((1, 1), dtype=t.bool),
            ),
            dim=1,
        )
        obs_dim = obs["image"].shape

        obs = t.concat(
            (
                t.zeros((1, max_len - 1, *obs_dim)),
                t.tensor(obs["image"]).unsqueeze(0).unsqueeze(0),
            ),
            dim=1,
        )

        reward = t.zeros((1, max_len, 1), dtype=t.float)  # no reward yet
        rtg = initial_rtg * t.ones(
            (1, max_len, 1), dtype=t.float
        )  # no reward yet
        timesteps = t.zeros((1, max_len, 1), dtype=t.long)  # no reward yet

        action_pad_token = dt.environment_config.action_space.n
        actions = (
            t.ones(1, max_len - 1, 1, dtype=t.long) * action_pad_token
        )  # done action

        st.session_state.mask = mask
        st.session_state.obs = obs
        st.session_state.reward = reward
        st.session_state.rtg = rtg
        st.session_state.a = actions  # no action taken yet
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
