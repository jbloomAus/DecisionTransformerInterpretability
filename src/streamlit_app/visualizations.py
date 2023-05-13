from typing import List
import streamlit.components.v1 as components
import circuitsvis as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

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


def plot_action_preds(action_preds):
    # make bar chart of action_preds
    action_preds = action_preds[-1][-1]
    action_preds = action_preds.detach().numpy()
    # softmax
    action_preds = np.exp(action_preds) / np.sum(np.exp(action_preds), axis=0)

    # get the entropy of the action preds
    entropy = -np.sum(action_preds * np.log(action_preds), axis=0)

    n_actions = len(action_preds)
    action_preds = pd.DataFrame(
        action_preds, index=list(action_id_to_string.values())[:n_actions]
    )
    fig = px.bar(
        action_preds,
        orientation="h",
        labels={"index": "", "value": "Probability"},
        height=320,
        width=320,
        # labels={"index": "Action", "value": "Probability"},
        text=action_preds[0].astype(float).round(2),
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=20), showlegend=False, font=dict(size=18)
    )
    # fig.update_xaxes(
    #     range=[0,1]
    # )
    fig.update_yaxes(
        # ticktext=action_preds,
        tickfont=dict(size=18),
        ticklabelposition="inside",
        automargin=True,
    )

    st.plotly_chart(fig, use_container_width=True)
    st.write("Current Entropy: ", str(round(entropy, 3)))


def plot_attention_pattern_single(
    cache, layer, softmax=True, specific_heads: List = None
):
    labels = st.session_state.labels
    if softmax:
        if cache["pattern", layer, "attn"].shape[0] == 1:
            attention_pattern = cache["pattern", layer, "attn"][0]
            if specific_heads is not None:
                attention_pattern = attention_pattern[specific_heads]

            st.write(attention_pattern.shape)
            st.write(len(labels))
            result = cv.attention.attention_patterns(
                attention=attention_pattern, tokens=labels
            )
            components.html(str(result), width=500, height=600)
        else:
            st.write("Not implemented yet")

    else:
        if cache["pattern", layer, "attn"].shape[0] == 1:
            attention_pattern = cache["attn_scores", layer, "attn"][0]

            if specific_heads is not None:
                attention_pattern = attention_pattern[specific_heads]

            result = cv.attention.attention_heads(
                attention=attention_pattern, tokens=labels
            )
            components.html(str(result), width=500, height=800)
        else:
            st.write("Not implemented yet")


def render_env(env):
    img = env.render()
    # use matplotlib to render the image
    fig, ax = plt.subplots()
    ax.imshow(img)
    plt.axis("off")
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    return fig


def plot_logit_diff(logit_diff, labels):
    # this plot assumes you only have a single dim

    fig = px.bar(
        x=labels,
        y=logit_diff.detach(),
        text=logit_diff.detach(),
    )

    fig.update_layout(
        title="Residual Decomposition",
        xaxis_title="Transformer Component",
        yaxis_title="Logit Difference",
        legend_title="",
    )

    # fig.update_yaxes(range=[-13, 13])
    fig.update_traces(texttemplate="%{text:.3f}", textposition="auto")
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode="hide")
    st.plotly_chart(fig, use_container_width=True)


def plot_single_residual_stream_contributions_comparison(
    residual_decomp_1, residual_decomp_2
):
    # this plot assumes you only have a single dim
    for key in residual_decomp_1.keys():
        residual_decomp_1[key] = residual_decomp_1[key].squeeze(0)
        residual_decomp_2[key] = residual_decomp_2[key].squeeze(0)
        # st.write(key, residual_decomp[key].shape)

    # make a df out of both dicts, one column each
    df1 = pd.DataFrame(residual_decomp_1, index=[0]).T
    df2 = pd.DataFrame(residual_decomp_2, index=[0]).T
    # rename df
    df1.columns = ["Original"]
    df2.columns = ["Ablation"]
    df = pd.concat([df1, df2], axis=1)

    texts = [df1.values, df2.values]
    fig = px.bar(df, barmode="group")

    for i, t in enumerate(texts):
        fig.data[i].text = t
        fig.data[i].textposition = "auto"

    fig.update_layout(
        title="Residual Decomposition",
        xaxis_title="Residual Stream Component",
        yaxis_title="Contribution to Action Prediction",
        legend_title="",
    )
    fig.update_yaxes(range=[-13, 13])
    fig.update_traces(texttemplate="%{text:.3f}", textposition="auto")
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode="hide")
    st.plotly_chart(fig, use_container_width=True)


import plotly.graph_objects as go
import plotly.figure_factory as ff
from scipy.spatial.distance import pdist, squareform


def plot_dendrogram_heatmap(
    df, color_continuous_midpoint=0, color_continuous_scale="RdBu"
):
    # Convert dataframe to numpy array
    data_array = df.to_numpy()
    labels = df.columns

    # Initialize figure by creating upper dendrogram
    fig = ff.create_dendrogram(data_array, orientation="bottom", labels=labels)
    for i in range(len(fig["data"])):
        fig["data"][i]["yaxis"] = "y2"

    # Create Side Dendrogram
    dendro_side = ff.create_dendrogram(data_array, orientation="right")
    for i in range(len(dendro_side["data"])):
        dendro_side["data"][i]["xaxis"] = "x2"

    # Add Side Dendrogram Data to Figure
    for data in dendro_side["data"]:
        fig.add_trace(data)

    # Create Heatmap
    dendro_leaves = dendro_side["layout"]["yaxis"]["ticktext"]
    dendro_leaves = list(map(int, dendro_leaves))
    data_dist = pdist(data_array)
    heat_data = squareform(data_dist)
    heat_data = heat_data[dendro_leaves, :]
    heat_data = heat_data[:, dendro_leaves]

    heatmap = [
        go.Heatmap(
            x=dendro_leaves,
            y=dendro_leaves,
            z=heat_data,
            colorscale=color_continuous_scale,
            zmid=color_continuous_midpoint,
        )
    ]

    heatmap[0]["x"] = fig["layout"]["xaxis"]["tickvals"]
    heatmap[0]["y"] = dendro_side["layout"]["yaxis"]["tickvals"]

    # Add Heatmap Data to Figure
    for data in heatmap:
        fig.add_trace(data)

    # Edit Layout
    fig.update_layout(
        {
            "width": 800,
            "height": 800,
            "showlegend": False,
            "hovermode": "closest",
        }
    )
    # Edit xaxis
    fig.update_layout(
        xaxis={
            "domain": [0.15, 1],
            "mirror": False,
            "showgrid": False,
            "showline": False,
            "zeroline": False,
            "ticks": "",
        }
    )
    # Edit xaxis2
    fig.update_layout(
        xaxis2={
            "domain": [0, 0.15],
            "mirror": False,
            "showgrid": False,
            "showline": False,
            "zeroline": False,
            "showticklabels": False,
            "ticks": "",
        }
    )

    # Edit yaxis
    fig.update_layout(
        yaxis={
            "domain": [0, 0.85],
            "mirror": False,
            "showgrid": False,
            "showline": False,
            "zeroline": False,
            "showticklabels": False,
            "ticks": "",
        }
    )
    # Edit yaxis2
    fig.update_layout(
        yaxis2={
            "domain": [0.825, 0.975],
            "mirror": False,
            "showgrid": False,
            "showline": False,
            "zeroline": False,
            "showticklabels": False,
            "ticks": "",
        }
    )

    # Plot!
    return fig
