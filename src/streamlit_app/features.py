import os
import json
import torch

import pandas as pd
import streamlit as st
import plotly.express as px

DEFAULT_FEATURE_DIR = "features"


def show_saved_features():
    with st.expander("Saved Features"):
        features, feature_metadata = load_features()
        feature_names = feature_metadata["feature_idx"].tolist()

        a, b = st.columns([0.8, 0.2])
        with a:
            st.subheader(f"Saved Features ({len(feature_names)})")
        with b:
            if st.button("Delete All"):
                delete_features()
                return

        st.write(feature_metadata, use_container_width=True)

        # generate similarity plot.
        similarities = features @ features.T

        fig = px.imshow(
            similarities,
            color_continuous_midpoint=0,
            color_continuous_scale="RdBu",
        )
        # make the xtick labels the feature names

        fig.update_layout(
            xaxis=dict(
                tickmode="array",
                tickvals=list(range(len(feature_names))),
                ticktext=feature_names,
            ),
            yaxis=dict(
                tickmode="array",
                tickvals=list(range(len(feature_names))),
                ticktext=feature_names,
            ),
        )

        st.subheader("Feature Similarity")
        st.plotly_chart(fig, use_container_width=True)


def delete_features():
    feature_jsons = [
        os.path.join(DEFAULT_FEATURE_DIR, f)
        for f in os.listdir(DEFAULT_FEATURE_DIR)
        if f.endswith(".json")
    ]
    feature_pts = [
        os.path.join(DEFAULT_FEATURE_DIR, f)
        for f in os.listdir(DEFAULT_FEATURE_DIR)
        if f.endswith(".pt")
    ]

    for f in feature_jsons:
        os.remove(f)
    for f in feature_pts:
        os.remove(f)


def load_features(feature_path=DEFAULT_FEATURE_DIR):
    feature_jsons = [
        os.path.join(feature_path, f)
        for f in os.listdir(feature_path)
        if f.endswith(".json")
    ]
    feature_pts = [
        os.path.join(feature_path, f)
        for f in os.listdir(feature_path)
        if f.endswith(".pt")
    ]

    # assert that the number of jsons and pts are the same
    assert len(feature_jsons) == len(
        feature_pts
    ), "Number of jsons and pts must be the same"

    # load each of the feature jsons
    jsons = []
    for f in feature_jsons:
        with open(f, "r") as f:
            jsons.append(json.load(f))

    # load each of the feature pts
    pts = []
    for f in feature_pts:
        pts.append(torch.load(f))

    if len(pts) == 0:
        st.write("No saved features")
        return
    features = torch.cat(pts, dim=0)

    feature_names = [f["feature_names"] for f in jsons]
    feature_names = [item for sublist in feature_names for item in sublist]

    feature_metadata = pd.DataFrame.from_dict(jsons)
    feature_metadata = feature_metadata.explode("feature_names")

    # assert feature names are unique
    # get the idx of features without uniuqe names
    idx = [
        i for i, x in enumerate(feature_names) if feature_names.count(x) > 1
    ]
    # for any features which aren't unique, add the file name
    for i in idx:
        feature_names[
            i
        ] = f"{feature_names[i]}_{feature_metadata.iloc[i]['file_name']}"
    # add the feature names to the metadata
    feature_metadata["feature_idx"] = feature_names
    feature_metadata.reset_index(inplace=True, drop=True)

    return features, feature_metadata
