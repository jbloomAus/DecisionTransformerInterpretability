import torch as t
import streamlit as st
from src.utils import pad_tensor


def get_action_preds_legacy(dt):
    max_len = dt.n_ctx // 3

    if "timestep_adjustment" in st.session_state:
        timesteps = (
            st.session_state.timesteps[:, -max_len:]
            + st.session_state.timestep_adjustment
        )

    obs = st.session_state.obs[:, -max_len:]
    actions = st.session_state.a[:, -max_len:]
    rtg = st.session_state.rtg[:, -max_len:]

    if obs.shape[1] < max_len:
        obs = pad_tensor(obs, max_len)
        actions = pad_tensor(actions, max_len, pad_token=dt.env.action_space.n)
        rtg = pad_tensor(rtg, max_len, pad_token=0)
        timesteps = pad_tensor(timesteps, max_len, pad_token=0)

    if dt.time_embedding_type == "linear":
        timesteps = timesteps.to(dtype=t.float32)
    else:
        timesteps = timesteps.to(dtype=t.long)

    tokens = dt.to_tokens(
        obs,
        actions.to(dtype=t.long),
        rtg,
        timesteps.to(dtype=t.long),
    )

    x, cache = dt.transformer.run_with_cache(tokens, remove_batch_dim=False)
    state_preds, action_preds, reward_preds = dt.get_logits(
        x, batch_size=1, seq_length=max_len
    )

    return action_preds, x, cache, tokens
