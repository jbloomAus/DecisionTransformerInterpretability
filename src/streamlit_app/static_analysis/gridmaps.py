import streamlit as st
import torch
import plotly.express as px

from src.streamlit_app.constants import (
    IDX_TO_ACTION,
    IDX_TO_STATE,
    three_channel_schema,
    twenty_idx_format_func,
    SPARSE_CHANNEL_NAMES,
    POSITION_NAMES,
    ACTION_NAMES,
    STATE_EMBEDDING_LABELS,
    get_all_neuron_labels,
)

# gridmap variants


def ov_gridmap_component(activations, key="embeddings"):
    a, b, c, d = st.columns(4)

    with a:
        heads = st.multiselect(
            "Select Head",
            options=activations["Head"].unique(),
            default=["L0H0"],
            key=f"gridmap direction state in, ov {key}",
        )
    with b:
        channels = st.multiselect(
            "Select Channels",
            options=SPARSE_CHANNEL_NAMES,
            default=["key", "ball"],
            key=f"gridmap channel state in, ov {key}",
        )
    with c:
        selected_actions = st.multiselect(
            "Select Actions",
            options=activations.Action.unique(),
            default=["left", "right"],
            key=f"gridmap action state in, ov {key}",
        )

    with d:
        abs_col_max = st.slider(
            "Max Absolute Value Color",
            min_value=activations.Score.abs().max().item() / 2,
            max_value=activations.Score.abs().max().item(),
            value=activations.Score.abs().max().item(),
        )

    head_tabs = st.tabs(heads)
    for i in range(len(heads)):
        with head_tabs[i]:
            for j in range(len(channels)):
                # given some specific head, I want to project onto some channels.
                activations_tmp = activations[activations.Head == heads[i]]
                activations_tmp = activations_tmp[
                    activations_tmp.Action.isin(selected_actions)
                ]
                activations_tmp = activations_tmp[
                    activations_tmp.Embedding.str.contains(channels[j])
                ]
                fig = plot_gridmap_from_embedding_congruence(
                    activations_tmp,
                    abs_col_max=abs_col_max,
                    facet_col="Action",
                )
                # add channel to title
                fig.update_layout(title=f"Channel {channels[j]}")

                st.plotly_chart(fig, use_container_width=True)


def qk_gridmap_component(activations, facet_col="Channel", key="embeddings"):
    a, b, d = st.columns(3)

    with a:
        heads = st.multiselect(
            "Select Head",
            options=activations["Head"].unique(),
            default=["L0H0"],
            key=f"gridmap direction state in, ov {key}",
        )
    with b:
        channels = st.multiselect(
            "Select Channels",
            options=SPARSE_CHANNEL_NAMES,
            default=["key", "ball"],
            key=f"gridmap channel state in, ov {key}",
        )

    with d:
        abs_col_max = st.slider(
            "Max Absolute Value Color",
            min_value=activations.Score.abs().max().item() / 2,
            max_value=activations.Score.abs().max().item(),
            value=activations.Score.abs().max().item(),
        )

    head_tabs = st.tabs(heads)
    for i in range(len(heads)):
        with head_tabs[i]:
            # given some specific head, I want to project onto some channels.
            activations_tmp = activations[activations.Head == heads[i]]
            activations_tmp = activations_tmp[
                activations_tmp.Channel.isin(channels)
            ]
            fig = plot_gridmap_from_embedding_congruence(
                activations_tmp,
                abs_col_max=abs_col_max,
                facet_col=facet_col,
            )

            st.plotly_chart(fig, use_container_width=True)


def neuron_projection_gridmap_component(activations, key="neuron projection"):
    a, b, c = st.columns(3)

    with a:
        neurons = st.multiselect(
            "Neuron",
            options=activations.Neuron.unique(),
            default=["L0N0"],
            key=f"gridmap neuron {key}",
        )
    with b:
        channels = st.multiselect(
            "Select Channels",
            options=SPARSE_CHANNEL_NAMES,
            default=["key", "ball"],
            key=f"gridmap channel state in, svd {key}",
        )
    with c:
        abs_col_max = st.slider(
            "Max Absolute Value Color",
            min_value=activations.Score.abs().max().item() / 2,
            max_value=activations.Score.abs().max().item(),
            value=activations.Score.abs().max().item(),
        )

    directions_tabs = st.tabs(neurons)
    for i in range(len(neurons)):
        with directions_tabs[i]:
            columns = st.columns(len(channels))
            for j in range(len(columns)):
                with columns[j]:
                    # given some specific head, I want to project onto some channels.
                    fig = plot_gridmap_from_embedding_congruence(
                        activations[(activations.Neuron == neurons[i]) & (activations.Channel == channels[j])],
                        abs_col_max=abs_col_max,
                    )
                    st.plotly_chart(fig, use_container_width=True)


def svd_projection_gridmap_component(activations, key="embeddings"):
    a, b, c = st.columns(3)

    with a:
        directions = st.multiselect(
            "Select Directions",
            options=activations["Direction"].unique(),
            default=["L0H0D0"],
            key=f"gridmap direction state in, svd {key}",
        )
    with b:
        channels = st.multiselect(
            "Select Channels",
            options=SPARSE_CHANNEL_NAMES,
            default=["key", "ball"],
            key=f"gridmap channel state in, svd {key}",
        )
    with c:
        abs_col_max = st.slider(
            "Max Absolute Value Color",
            min_value=activations.Score.abs().max().item() / 2,
            max_value=activations.Score.abs().max().item(),
            value=activations.Score.abs().max().item(),
        )

    directions_tabs = st.tabs(directions)
    for i in range(len(directions)):
        with directions_tabs[i]:
            columns = st.columns(len(channels))
            for j in range(len(columns)):
                with columns[j]:
                    # given some specific head, I want to project onto some channels.
                    fig = plot_gridmap_from_embedding_congruence(
                        activations[(activations.Direction == directions[i]) & (activations.Channel == channels[j])],
                        abs_col_max=abs_col_max,
                    )
                    st.plotly_chart(fig, use_container_width=True)


def plot_gridmap_from_embedding_congruence(
    activations, abs_col_max, facet_col=None
):
    if not facet_col:
        channel = activations.Embedding.values[0].split(",")[0]
        scores = torch.tensor(activations.Score.values).reshape(7, 7).T
        fig = px.imshow(
            scores,
            color_continuous_midpoint=0,
            color_continuous_scale="RdBu",
            zmin=-abs_col_max,
            zmax=abs_col_max,
        )

        # update hover template
        fig.update_traces(
            hovertemplate=f"{channel}, "
            + "(%{x},%{y})<br>Congruence: %{z:.2f}"
        )

        # remove color legend
        fig.update_layout(coloraxis_showscale=False)

    else:
        n_facets = len(activations[facet_col].unique())
        activations = activations.sort_values([facet_col, "Y", "X"])
        scores = torch.tensor(activations.Score.values).reshape(n_facets, 7, 7)

        fig = px.imshow(
            scores,
            color_continuous_midpoint=0,
            color_continuous_scale="RdBu",
            facet_col=0,
            zmin=-abs_col_max,
            zmax=abs_col_max,
        )

        # update hover template for each facet with channel
        # fig.update_traces(
        #     hovertemplate="(%{x},%{y})<br>Congruence: %{z:.2f}<br>",
        # )

        # rename facet titles to be the facet col value
        actions = activations[facet_col].unique()
        fig.for_each_annotation(
            lambda a: a.update(text=actions[int(a.text.split("=")[1])])
        )

        fig.for_each_trace(
            lambda trace: trace.update(
                hovertemplate=f"{actions[int(trace.name)]}, (%{{x}},%{{y}})<br>Score: %{{z:.2f}}<br>"
            )
        )

        # make x-ticks at every value, remove the tick but keep text
        fig.update_xaxes(
            tickmode="array",
            tickvals=list(range(7)),
            ticktext=list(range(7)),
            showticklabels=True,
            title=None,
        )

        # do same for y but only for first facet
        fig.update_yaxes(
            tickmode="array",
            tickvals=list(range(7)),
            ticktext=list(range(7)),
            showticklabels=True,
            title=None,
            row=1,
            col=1,
        )

        # remove color legend
        fig.update_layout(coloraxis_showscale=False)

    return fig
