import math

import json
import gymnasium as gym
import minigrid
import streamlit as st
import torch as t

from src.config import EnvironmentConfig
from src.models.trajectory_transformer import (
    DecisionTransformer,
    CloneTransformer,
)

from src.decision_transformer.utils import (
    load_decision_transformer,
    get_max_len_from_model_type,
)
from src.environments.environments import make_env
from src.utils import pad_tensor


@st.cache(allow_output_mutation=True)
def get_env_and_dt(model_path):
    # we need to one if the env was one hot encoded. Some tech debt here.
    state_dict = t.load(model_path)
    if "environment_config" in state_dict:
        env_config = state_dict["environment_config"]
        env_config = EnvironmentConfig(**json.loads(env_config))
    else:
        one_hot_encoded = not (
            state_dict["state_encoder.weight"].shape[-1] % 20
        )  # hack for now
        # list all mini grid envs
        minigrid_envs = [
            i for i in gym.envs.registry.keys() if "MiniGrid" in i
        ]
        # find the env id in the path
        env_ids = [i for i in minigrid_envs if i in model_path]
        if len(env_ids) == 0:
            raise ValueError(
                f"Could not find the env id in the model path: {model_path}"
            )
        elif len(env_ids) > 1:
            raise ValueError(
                f"Found more than one env id in the model path: {model_path}"
            )
        else:
            env_id = env_ids[0]
        # env_id = 'MiniGrid-Dynamic-Obstacles-8x8-v0'
        if one_hot_encoded:
            view_size = int(
                math.sqrt(state_dict["state_encoder.weight"].shape[-1] // 20)
            )
        else:
            view_size = int(
                math.sqrt(state_dict["state_encoder.weight"].shape[-1] // 3)
            )

        env_config = EnvironmentConfig(
            env_id=env_id,
            capture_video=False,
            fully_observed=False,
            one_hot_obs=one_hot_encoded,
            view_size=view_size,
            max_steps=30,
        )

    env = make_env(env_config, seed=4200, idx=0, run_name="dev")
    env = env()

    dt = load_decision_transformer(model_path, env)
    if not hasattr(dt, "n_ctx"):
        dt.n_ctx = dt.transformer_config.n_ctx
    if not hasattr(dt, "time_embedding_type"):
        dt.time_embedding_type = dt.transformer_config.time_embedding_type
    return env, dt


def get_action_preds(dt):
    # so we can ignore older models when making updates

    max_len = get_max_len_from_model_type(
        dt.model_type,
        dt.transformer_config.n_ctx,
    )

    if "timestep_adjustment" in st.session_state:
        timesteps = (
            st.session_state.timesteps[:, -max_len:]
            + st.session_state.timestep_adjustment
        )

    obs = st.session_state.obs[:, -max_len:]
    actions = st.session_state.a[:, -max_len:]
    rtg = st.session_state.rtg[:, -max_len:]

    # truncations:
    obs = obs[:, -max_len:] if obs.shape[1] > max_len else obs
    actions = (
        actions[:, -(obs.shape[1] - 1) :]
        if (actions.shape[1] > 1 and max_len > 1)
        else None
    )
    timesteps = (
        timesteps[:, -max_len:] if timesteps.shape[1] > max_len else timesteps
    )
    rtg = rtg[:, -max_len:] if rtg.shape[1] > max_len else rtg

    # st.write("max len: ", max_len)
    # st.write(obs.shape)
    # st.write(actions.shape)
    # st.write(rtg.shape)
    # st.write(timesteps.shape)

    # if obs.shape[1] < max_len:
    #     obs = pad_tensor(obs, max_len)
    #     if actions is not None:
    #         actions = pad_tensor(actions, max_len, pad_token=dt.environment_config.action_space.n)
    #     if rtg is not None:
    #         rtg = pad_tensor(rtg, max_len, pad_token=0)
    #     timesteps = pad_tensor(timesteps, max_len, pad_token=0)

    if dt.time_embedding_type == "linear":
        timesteps = timesteps.to(dtype=t.float32)
    else:
        timesteps = timesteps.to(dtype=t.long)

    tokens = dt.to_tokens(
        obs,
        actions,
        rtg,
        timesteps.to(dtype=t.long),
    )

    x, cache = dt.transformer.run_with_cache(tokens, remove_batch_dim=False)

    state_preds, action_preds, reward_preds = dt.get_logits(
        x, batch_size=1, seq_length=obs.shape[1], no_actions=actions is None
    )

    return action_preds, x, cache, tokens


def respond_to_action(env, action, initial_rtg):
    new_obs, reward, done, trunc, info = env.step(action)
    if done:
        st.error(
            "The agent has just made a game ending move. Please reset the environment."
        )
    # append to session state
    st.session_state.obs = t.cat(
        [
            st.session_state.obs,
            t.tensor(new_obs["image"]).unsqueeze(0).unsqueeze(0),
        ],
        dim=1,
    )
    # print(t.tensor(action).unsqueeze(0).unsqueeze(0).shape)
    st.session_state.a = t.cat(
        [st.session_state.a, t.tensor([action]).unsqueeze(0).unsqueeze(0)],
        dim=1,
    )
    st.session_state.reward = t.cat(
        [
            st.session_state.reward,
            t.tensor([reward]).unsqueeze(0).unsqueeze(0),
        ],
        dim=1,
    )

    rtg = initial_rtg - st.session_state.reward.sum()

    st.session_state.rtg = t.cat(
        [st.session_state.rtg, t.tensor([rtg]).unsqueeze(0).unsqueeze(0)],
        dim=1,
    )
    time = st.session_state.timesteps[-1][-1] + 1
    st.session_state.timesteps = t.cat(
        [
            st.session_state.timesteps,
            time.clone().detach().unsqueeze(0).unsqueeze(0),
        ],
        dim=1,
    )


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
