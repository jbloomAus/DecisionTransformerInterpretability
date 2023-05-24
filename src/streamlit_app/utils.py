import plotly.express as px
import streamlit as st
import os
import pandas as pd
import numpy as np
import itertools


def read_index_html():
    with open("index.html") as f:
        return f.read()


def fancy_imshow(img, color_continuous_midpoint=0):
    fig = px.imshow(
        img,
        color_continuous_midpoint=color_continuous_midpoint,
        color_continuous_scale=px.colors.diverging.RdBu,
    )
    fig.update_layout(
        coloraxis_showscale=False, margin=dict(l=0, r=0, t=0, b=0)
    )
    fig.update_layout(height=180, width=400)
    st.plotly_chart(fig, use_container_width=True, autosize=True)


def fancy_histogram(vector):
    fig = px.histogram(vector)
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def list_models(path):
    model_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".pt"):
                model_list.append(os.path.join(root, file))
    return model_list


def tensor_to_long_data_frame(tensor_result, dimension_names):
    assert len(tensor_result.shape) == len(
        dimension_names
    ), "The number of dimension names must match the number of dimensions in the tensor"

    tensor_2d = tensor_result.reshape(-1)
    df = pd.DataFrame(tensor_2d.detach().numpy(), columns=["Score"])

    indices = pd.MultiIndex.from_tuples(
        list(np.ndindex(tensor_result.shape)),
        names=dimension_names,
    )
    df.index = indices
    df.reset_index(inplace=True)
    return df


def get_row_names_from_index_labels(names, index_labels):
    indices = list(itertools.product(*index_labels))
    multi_index = pd.MultiIndex.from_tuples(
        indices,
        names=names,  # use labels differently if we have index labels
    )
    if len(names) == 3:
        multi_index = multi_index.to_series().apply(
            lambda x: "{0}, ({1},{2})".format(*x)
        )

    elif names == 2:
        multi_index = multi_index.to_series().apply(
            lambda x: "({0},{1})".format(*x)
        )
    else:
        raise ("Index labels must be 2 or 3 dimensional")

    return multi_index
