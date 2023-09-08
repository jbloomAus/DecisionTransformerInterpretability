import json
import streamlit as st
import torch
from torchinfo import summary
from typing import Optional
import pandas as pd

from src.config import EnvironmentConfig

from src.decision_transformer.utils import (
    load_decision_transformer,
    get_max_len_from_model_type,
)
from src.environments.environments import make_env


@st.cache_data
def get_env_and_dt(model_path):
    # we need to one if the env was one hot encoded. Some tech debt here.
    state_dict = torch.load(model_path)

    env_config = state_dict["environment_config"]
    env_config = EnvironmentConfig(**json.loads(env_config))

    env = make_env(env_config, seed=4200, idx=0, run_name="dev")
    env = env()

    dt = load_decision_transformer(
        model_path, env, tlens_weight_processing=True
    )
    if not hasattr(dt, "n_ctx"):
        dt.n_ctx = dt.transformer_config.n_ctx
    if not hasattr(dt, "time_embedding_type"):
        dt.time_embedding_type = dt.transformer_config.time_embedding_type
    return env, dt


def get_state_history(previous_step=False):
    if previous_step:
        return st.session_state.previous_step_history
    return {
        "dt": st.session_state.dt,
        "obs": st.session_state.obs,
        "rtg": st.session_state.rtg,
        "actions": st.session_state.a,
        "timesteps": st.session_state.timesteps,
    }


def preprocess_inputs(dt, obs, rtg, actions, timesteps):
    max_len = get_max_len_from_model_type(
        dt.model_type,
        dt.transformer_config.n_ctx,
    )

    timesteps = timesteps[:, -max_len:]

    # truncations:
    obs = obs[:, -max_len:] if obs.shape[1] > max_len else obs
    if actions is not None:
        actions = (
            actions[:, -(obs.shape[1] - 1) :]
            if (actions.shape[1] > 1 and max_len > 1)
            else None
        )
    timesteps = (
        timesteps[:, -max_len:] if timesteps.shape[1] > max_len else timesteps
    )
    rtg = rtg[:, -max_len:] if rtg.shape[1] > max_len else rtg

    if dt.time_embedding_type == "linear":
        timesteps = timesteps.to(dtype=torch.float32)
    else:
        timesteps = timesteps.to(dtype=torch.long)

    return obs, actions, rtg, timesteps


def get_tokens_from_app_state(dt, previous_step=False):
    obs, actions, rtg, timesteps = preprocess_inputs(
        **get_state_history(previous_step=previous_step),
    )
    tokens = dt.to_tokens(obs, actions, rtg, timesteps)

    if actions is not None:
        st.session_state.model_summary = summary(
            dt.transformer, input_data=tokens
        )
    return tokens


def get_modified_tokens_from_app_state(
    dt,
    corrupt_obs: Optional[torch.Tensor] = None,
    all_rtg: Optional[float] = None,
    specific_rtg: Optional[float] = None,
    new_action: Optional[int] = None,
    position: Optional[int] = None,
):
    obs, actions, rtg, timesteps = preprocess_inputs(
        **get_state_history(previous_step=False),
    )

    previous_tokens = dt.to_tokens(obs, actions, rtg, timesteps)
    rtg = rtg.clone()  # don't accidentally modify the session state.
    obs = obs.clone()  # don't accidentally modify the session state.
    # now do interventions
    # if all_rtg is not None:
    #     rtg_dif = rtg[0] - all_rtg
    #     new_rtg = rtg - rtg_dif

    #     st.write(rtg.squeeze(-1))
    #     st.write(new_rtg.squeeze(-1))
    #     tokens = dt.to_tokens(obs, actions, new_rtg, timesteps)

    # if specific_rtg is not None:
    #     assert position is not None
    #     # print("specific rtg", specific_rtg)
    #     # make a table showing the rtg at each position before/after
    #     # and then highlight the position that is being changed.

    #     new_rtg = rtg.clone()
    #     new_rtg[0][position] = specific_rtg

    #     st.write(rtg.squeeze(-1))
    #     st.write(new_rtg.squeeze(-1))

    #     tokens = dt.to_tokens(obs, actions, new_rtg, timesteps)

    # if new_action is not None:
    #     assert position is not None
    #     new_actions = actions.clone()
    #     new_actions[0][position] = new_action

    #     tokens = dt.to_tokens(obs, new_actions, rtg, timesteps)

    #     st.write(actions.squeeze(-1))
    #     st.write(new_actions.squeeze(-1))

    if obs is not None:
        assert position is not None
        new_obs = obs.clone()
        new_obs[0][position] = corrupt_obs
        tokens = dt.to_tokens(new_obs, actions, rtg, timesteps)

    # assert at least some of the tokens are different
    assert not torch.all(
        tokens == previous_tokens
    ), "corrupted tokens are the same!"
    return tokens


def get_action_preds_from_tokens(dt, tokens):
    x, cache = dt.transformer.run_with_cache(tokens, remove_batch_dim=False)

    state_preds, action_preds, reward_preds = dt.get_logits(
        x,
        batch_size=1,
        seq_length=st.session_state.max_len,
        no_actions=False,  # we always pad now.
    )
    return action_preds, x, cache, tokens


def get_action_preds_from_app_state(dt):
    # so we can ignore older models when making updates
    tokens = get_tokens_from_app_state(dt)
    x, cache = dt.transformer.run_with_cache(tokens, remove_batch_dim=False)

    state_preds, action_preds, reward_preds = dt.get_logits(
        x,
        batch_size=1,
        seq_length=st.session_state.max_len,
        no_actions=False,  # we always pad now.
    )
    return action_preds, x, cache, tokens


def respond_to_action(env, action, initial_rtg):
    # prior to updating state, store previous inputs for
    # causal analysis
    st.session_state.previous_step_history = get_state_history()

    new_obs, reward, done, trunc, info = env.step(action)
    if done:
        st.error(
            "The agent has just made a game ending move. Please reset the environment."
        )
    # append to session state
    st.session_state.obs = torch.cat(
        [
            st.session_state.obs,
            torch.tensor(new_obs["image"]).unsqueeze(0).unsqueeze(0),
        ],
        dim=1,
    )

    # store the rendered image
    st.session_state.rendered_obs = torch.cat(
        [
            st.session_state.rendered_obs,
            torch.from_numpy(env.render()).unsqueeze(0),
        ],
        dim=0,
    )

    if st.session_state.a is None:
        st.session_state.a = torch.tensor([action]).unsqueeze(0).unsqueeze(0)

    st.session_state.a = torch.cat(
        [st.session_state.a, torch.tensor([action]).unsqueeze(0).unsqueeze(0)],
        dim=1,
    )
    st.session_state.reward = torch.cat(
        [
            st.session_state.reward,
            torch.tensor([reward]).unsqueeze(0).unsqueeze(0),
        ],
        dim=1,
    )

    rtg = initial_rtg - st.session_state.reward.sum()

    st.session_state.rtg = torch.cat(
        [st.session_state.rtg, torch.tensor([rtg]).unsqueeze(0).unsqueeze(0)],
        dim=1,
    )
    time = st.session_state.timesteps[-1][-1] + 1
    st.session_state.timesteps = torch.cat(
        [
            st.session_state.timesteps,
            time.clone().detach().unsqueeze(0).unsqueeze(0),
        ],
        dim=1,
    )
    st.session_state.current_len += 1


def get_action_from_user(env, initial_rtg):
    # create a series of buttons for each action
    button_columns = st.columns(7)
    with button_columns[0]:
        left_button = st.button("Left", key="left_button")
    with button_columns[1]:
        right_button = st.button("Right", key="right_button")
    with button_columns[2]:
        forward_button = st.button("Forward", key="forward_button")
    with button_columns[3]:
        pickup_button = st.button("Pickup", key="pickup_button")
    with button_columns[4]:
        drop_button = st.button("Drop", key="drop_button")
    with button_columns[5]:
        toggle_button = st.button("Toggle", key="toggle_button")
    with button_columns[6]:
        done_button = st.button("Done", key="done_button")

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


def get_token_labels():
    n_timesteps = st.session_state.dt.n_ctx - 2  # assume dt

    n = n_timesteps // 3 + 1
    labels = []

    for i in range(1, n + 1):
        labels.append("R" + str(i))
        labels.append("S" + str(i))
        labels.append("A" + str(i))

    labels.pop()  # remove the last A

    return labels
