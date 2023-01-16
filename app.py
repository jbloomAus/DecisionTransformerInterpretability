import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
import torch as t

from src.environments import make_env
from src.utils import load_decision_transformer

start = time.time()
st.title("MiniGrid Interpretability Playground")
st.write(
    '''
    To do:
    - new game made: Start from a random environment of the kind your dt was trained on. Observe the state and it's corresponding action. Then, use the dt to predict the next action. You can choose to either take this action, or one of your own choosing. Iterate. 
    '''
)

model_path = "models/demo_model.pt"
action_string_to_id = {"left": 0, "right": 1, "forward": 2, "pickup": 3, "drop": 4, "toggle": 5, "done": 6}
action_id_to_string = {v: k for k, v in action_string_to_id.items()}

# env_id = st.selectbox("Select the environment",
#         gym.envs.registry.keys(),
#         index=0)
        

st.session_state.max_len = 1
@st.cache(allow_output_mutation=True)
def get_env_and_dt(model_path):
    env_id = 'MiniGrid-Dynamic-Obstacles-8x8-v0'
    env = make_env(env_id, seed = 4200, idx = 0, capture_video=False, run_name = "dev", fully_observed=False, max_steps=30)
    env = env()

    # n_ctx = 3
    # max_len = n_ctx // 3
    # st.session_state.max_len = max_len
    # dt = DecisionTransformer(
    #     env = env, 
    #     d_model = 128,
    #     n_heads = 2,
    #     d_mlp = 256,
    #     n_layers = 1,
    #     n_ctx=n_ctx,
    #     layer_norm=False,
    #     state_embedding_type="grid", # hard-coded for now to minigrid.
    #     max_timestep=300) # Our DT must have a context window large enough

    # dt.load_state_dict(t.load(model_path))
    dt = load_decision_transformer(
        model_path, env
    )

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

def plot_action_preds(action_preds):
     # make bar chart of action_preds
    action_preds = action_preds[-1][-1]
    action_preds = action_preds.detach().numpy()
    # softmax
    action_preds = np.exp(action_preds) / np.sum(np.exp(action_preds), axis=0)
    action_preds = pd.DataFrame(
        action_preds, 
        index=list(action_id_to_string.values())[:3]
        )
    st.bar_chart(action_preds)

def plot_attention_pattern(cache, layer):
    n_tokens = st.session_state.dt.n_ctx - 1
    attention_pattern = cache["pattern", layer, "attn"]
    fig = px.imshow(
        attention_pattern[:,:n_tokens,:n_tokens], 
        facet_col=0, range_color=[0,1])
    st.plotly_chart(fig)

def respond_to_action(env, action):
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

with st.sidebar:
    st.subheader("Game Screen")

    initial_rtg = st.slider("Initial RTG", min_value=-10.0, max_value=1.0, value=0.5, step=0.01)
    if "rtg" in st.session_state:
        # generate rtg vector as initial rtg - cumulative reward
        st.session_state.rtg = initial_rtg - st.session_state.reward
        # st.session_state.rtg = t.tensor([initial_rtg]).unsqueeze(0).unsqueeze(0)


if "env" not in st.session_state or "dt" not in st.session_state:
    st.write("Loading environment and decision transformer...")
    env, dt = get_env_and_dt(model_path)
    obs, _ = env.reset()

    # initilize the session state trajectory details
    st.session_state.obs = t.tensor(obs['image']).unsqueeze(0).unsqueeze(0)
    st.session_state.rtg = t.tensor([initial_rtg]).unsqueeze(0).unsqueeze(0)
    st.session_state.reward = t.tensor([0]).unsqueeze(0).unsqueeze(0)
    st.session_state.a = t.tensor([0]).unsqueeze(0).unsqueeze(0)
    st.session_state.timesteps = t.tensor([0]).unsqueeze(0).unsqueeze(0)

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

columns = st.columns(2)

with columns[0]:
    action_preds, x, cache, tokens = get_action_preds()
    plot_action_preds(action_preds)
with columns[1]:
    fig = render_env(env)
    st.pyplot(fig)

with st.expander("show attention pattern"):
    if dt.n_layers == 1:
        plot_attention_pattern(cache,0)
    else:
        layer = st.slider("Layer", min_value=0, max_value=dt.n_layers-1, value=0, step=1)
        plot_attention_pattern(cache,layer)
        # timesteps_b = st.slider("Number of Tokens", min_value=1, max_value=, value=dt.n_tokens, step=1)

with st.expander("Show residual stream contributions:"):
    x_action = x[0][1]
    # st.write(dt.action_embedding.weight.shape)
    st.write("action embedding: ", x_action.shape)
    forward_dir = dt.predict_actions.weight[2]-dt.predict_actions.weight[1]
    st.write("dot production of x_action with forward:", (x_action @ forward_dir).item()) # technically  forward over right


    st.latex(
        r'''
        x = s_{original} + r_{mlp} + h_{1.1} + h_{1.2} \newline
        '''
    )
    # x_action = s_{original} + r_{mlp} + h_{1.1} + h_{1.2}
    pos_contribution = (cache["hook_pos_embed"][1] @ forward_dir).item()
    st.write("dot production of pos_embed with forward:", pos_contribution) # pos embed

    token_contribution = (tokens[0][1] @ forward_dir).item()
    st.write("dot production of first token embedding with forward:", token_contribution) # tokens 

    head_0_output = cache["blocks.0.attn.hook_z"][:,0,:] @ dt.transformer.blocks[0].attn.W_O[0]
    head_0_contribution = (head_0_output[1] @ forward_dir).item()
    st.write("dot production of head_0_output with forward:", head_0_contribution)
    head_1_output = cache["blocks.0.attn.hook_z"][:,1,:] @ dt.transformer.blocks[0].attn.W_O[1]
    head_1_contribution = (head_1_output[1] @ forward_dir).item()
    st.write("dot production of head_1_output with forward:", head_1_contribution)
    mlp_output = cache["blocks.0.hook_mlp_out"][1]
    mlp_contribution =   (mlp_output @ forward_dir).item()
    st.write("dot production of mlp_output with forward:", mlp_contribution)

    state_dict = dt.state_dict()

    attn_bias_0_dir = state_dict['transformer.blocks.0.attn.b_O']
    attn_bias_contribution = (attn_bias_0_dir @ forward_dir).item()
    st.write( "dot production of attn_bias_0_dir with forward:", attn_bias_contribution)

    # mlp_bias_0_dir = state_dict['transformer.blocks.0.mlp.b_out']
    # mlp_bias_contribution = (mlp_bias_0_dir @ forward_dir).item()
    # st.write( "dot production of mlp_bias_0_dir with forward:", mlp_bias_contribution)

    sum_of_contributions = token_contribution + head_0_contribution + \
                head_1_contribution + mlp_contribution + \
                pos_contribution + attn_bias_contribution
    st.write("sum over contributions:", sum_of_contributions)
    percent_explained = np.abs(sum_of_contributions) / (x_action @ forward_dir).abs().item()
    st.write("percent explained:", percent_explained)

    st.write("This appears to mostly explain how each component of the residual stream contributes to the action prediction.")
    
    
    # st.write(state_dict.keys())
    # st.write(state_dict['transformer.blocks.0.attn.b_O'].shape)
    # # let's see the contribution


 
st.markdown("""---""")

st.session_state.env = env
st.session_state.dt = dt

with st.expander("Trajectory Details"):
    # write out actions, rtgs, rewards, and timesteps
    st.write(f"actions: {st.session_state.a[0].squeeze(-1).tolist()}")
    st.write(f"rtgs: {st.session_state.rtg[0].squeeze(-1).tolist()}")
    st.write(f"rewards: {st.session_state.reward[0].squeeze(-1).tolist()}")
    st.write(f"timesteps: {st.session_state.timesteps[0].squeeze(-1).tolist()}")


def store_trajectory(state, action, obs, reward, done, trunc, info):
    if "trajectories" not in st.session_state:
        st.session_state.trajectories = []
    st.session_state.trajectories.append((state, action, obs, reward, done, trunc, info))

if st.button("reset"):
    del st.session_state.env
    del st.session_state.dt
    st.experimental_rerun()
    

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