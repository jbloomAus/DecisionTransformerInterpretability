from typing import List

import torch
import circuitsvis as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from scipy.cluster import hierarchy

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
    cache,
    layer,
    softmax=True,
    specific_heads: List = None,
    method="Plotly",
    scale_by_value=False,
    key="",
):
    labels = st.session_state.labels

    if method == "Plotly":
        attn_norm = torch.norm(cache[f"blocks.{layer}.attn.hook_v"], dim=-1)
        if softmax:
            attention_pattern = cache["pattern", layer, "attn"][0]
            col_arg = {"color_continuous_midpoint": 0}
        else:
            attention_pattern = cache["attn_scores", layer, "attn"][0]
            # -attention_pattern.max().item(),-attention_pattern.max().item(),
            col_arg = {"range_color": [-20, 20]}
        attention_pattern = attention_pattern[specific_heads]

        if scale_by_value:
            attention_pattern = attention_pattern * attn_norm[0].T.unsqueeze(
                -1
            )
        head_tabs = st.tabs([f"L{layer}H{head}" for head in specific_heads])
        for head in range(len(specific_heads)):
            with head_tabs[head]:
                df = pd.DataFrame(
                    attention_pattern[head], index=labels, columns=labels
                )
                fig = px.imshow(
                    df,
                    # color_continuous_midpoint=0,
                    color_continuous_scale="RdBu",
                    height=600,
                    width=600,
                    # range_color=color_range,
                    **col_arg,
                )

                # remove ticks and colorbar, rotate labels and make sure every one is shown, reduce font size
                fig.update_xaxes(
                    showgrid=False,
                    ticks="",
                    tickmode="linear",
                    automargin=True,
                    ticktext=labels,
                )

                fig.update_yaxes(
                    showgrid=False,
                    ticks="",
                    tickangle=0,
                    tickmode="linear",
                    automargin=True,
                    tickvals=np.arange(len(labels)),
                    ticktext=labels,
                )

                # use labels as tick text
                fig.update_xaxes(ticktext=labels)
                fig.update_yaxes(ticktext=labels)

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    key=key + "attention-pattern",
                )

        return

    # this is very cursed. I'm sorry.
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


def plot_heatmap(
    df,
    color_continuous_midpoint=0,
    color_continuous_scale="RdBu",
    cluster=True,
    show_labels=True,
    max_labels=0,
):
    # Convert dataframe to numpy array
    data_array = df.to_numpy()

    if cluster:
        linkage = hierarchy.linkage(data_array)
        dendrogram = hierarchy.dendrogram(
            linkage, no_plot=True, color_threshold=-np.inf
        )
        reordered_ind = dendrogram["leaves"]
        # reorder df by ind
        df = df.iloc[reordered_ind, reordered_ind]
        data_array = df.to_numpy()

    fig = px.imshow(
        df,
        color_continuous_midpoint=color_continuous_midpoint,
        color_continuous_scale=color_continuous_scale,
        height=600,
        width=600,
        text_auto=".2f" if max_labels >= data_array.shape[0] else False,
    )

    # make text auto font size larger
    fig.update_traces(textfont_size=18)

    # remove ticks and colorbar, rotate labels and make sure every one is shown, reduce font size
    fig.update_xaxes(
        showgrid=False,
        ticks="",
        tickangle=45,
        tickmode="linear",
        automargin=True,
    )
    fig.update_yaxes(
        showgrid=False,
        ticks="",
        tickangle=0,
        tickmode="linear",
        automargin=True,
    )

    # hide the colorbar
    fig.update_layout(coloraxis_showscale=False)

    # make the x and y axis tick font larger
    fig.update_xaxes(tickfont=dict(size=18))
    fig.update_yaxes(tickfont=dict(size=18))

    if not show_labels:
        fig.update_xaxes(
            visible=False,
        )
        fig.update_yaxes(
            visible=False,
        )

    return fig


def plot_logit_scan(scan_values, action_preds, position=-1, scan_name="RTG"):
    preds_over_rtg = {
        scan_name: scan_values[:, position, 0].detach().cpu().numpy(),
        "Left": action_preds[:, position, 0].detach().cpu().numpy(),
        "Right": action_preds[:, position, 1].detach().cpu().numpy(),
        "Forward": action_preds[:, position, 2].detach().cpu().numpy(),
    }

    if action_preds.shape[-1] == 7:
        preds_over_rtg["Pickup"] = (
            action_preds[:, position, 3].detach().cpu().numpy()
        )
        preds_over_rtg["Drop"] = (
            action_preds[:, position, 4].detach().cpu().numpy()
        )
        preds_over_rtg["Toggle"] = (
            action_preds[:, position, 5].detach().cpu().numpy()
        )
        preds_over_rtg["Done"] = (
            action_preds[:, position, 6].detach().cpu().numpy()
        )

    df = pd.DataFrame(preds_over_rtg)

    # draw a line graph with left,right forward over RTG
    if action_preds.shape[-1] == 7:
        fig = px.line(
            df,
            x=scan_name,
            y=[
                "Left",
                "Right",
                "Forward",
                "Pickup",
                "Drop",
                "Toggle",
                "Done",
            ],
            # title="Action Prediction vs " + scan_name,
        )
    else:
        fig = px.line(
            df,
            x="RTG",
            y=["Left", "Right", "Forward"],
            title="Action Prediction vs RTG",
        )

    fig.update_layout(
        xaxis_title=scan_name,
        yaxis_title="Action Prediction",
        legend_title="",
    )
    # add vertical gridlines
    fig.update_xaxes(showgrid=True, gridwidth=1)
    # add vertical dotted lines at RTG = -1, RTG = 0, RTG = 1
    # fig.add_vline(x=-1, line_dash="dot", line_width=1, line_color="white")
    # fig.add_vline(x=0, line_dash="dot", line_width=1, line_color="white")
    # fig.add_vline(x=1, line_dash="dot", line_width=1, line_color="white")

    return fig
