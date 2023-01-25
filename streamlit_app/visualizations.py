
import plotly.express as px 
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List

action_string_to_id = {"left": 0, "right": 1, "forward": 2, "pickup": 3, "drop": 4, "toggle": 5, "done": 6}
action_id_to_string = {v: k for k, v in action_string_to_id.items()}


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
    fig = px.bar(action_preds, orientation='h',
        labels={"index": "", "value": "Probability"},
        height=320,
        width= 320,
        # labels={"index": "Action", "value": "Probability"},
        text = action_preds[0].astype(float).round(2),
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=20),
        showlegend=False,
        font = dict(size=18)
    )
    # fig.update_xaxes(
    #     range=[0,1]
    # )
    fig.update_yaxes(
        # ticktext=action_preds,
        tickfont=dict(size=18, color = "white"),
        ticklabelposition="inside",
        automargin = True
    )

    st.plotly_chart(fig, use_container_width=True)

def plot_attention_pattern(cache, layer, softmax=True, specific_heads: List = None):

    n_tokens = st.session_state.dt.n_ctx - 1
    if softmax:
        if cache["pattern", layer, "attn"].shape[0] == 1:
            attention_pattern = cache["pattern", layer, "attn"][0]
    else: 
        if cache["pattern", layer, "attn"].shape[0] == 1:
            attention_pattern = cache["attn_scores", layer, "attn"][0]

    attention_pattern = attention_pattern[:,:n_tokens,:n_tokens]
    if specific_heads is not None:
        attention_pattern = attention_pattern[specific_heads]

    fig = px.imshow(
        attention_pattern,
        facet_col=0, 
        range_color=[0,1],
        x = ["RTG","State"],
        y = ["RTG","State"]
    )
    # fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)

    st.plotly_chart(fig, use_container_width=True)

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
