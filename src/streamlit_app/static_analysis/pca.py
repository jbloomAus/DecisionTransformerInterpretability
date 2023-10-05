import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def get_pca(data, labels, n_components=None):
    # Normalize the data
    normalized_embedding = torch.tensor(data.T, dtype=torch.float32)

    # Perform PCA
    if not n_components:
        n_components = len(labels)

    pca = PCA(n_components=n_components)
    # pca_results = pca.fit_transform(normalized_embedding)
    fitted_pca = pca.fit(normalized_embedding.T)
    pca_results = fitted_pca.transform(normalized_embedding.T)

    # Create a dataframe for the results
    pca_df = pd.DataFrame(
        data=pca_results,
        index=labels,
        columns=[f"PC{i+1}" for i in range(n_components)],
    )
    pca_df.reset_index(inplace=True, names="State")
    pca_df["Channel"] = pca_df["State"].apply(lambda x: x.split(",")[0])

    # get the percent variance explained
    percent_variance = pca.explained_variance_ratio_ * 100

    loadings = torch.tensor(fitted_pca.components_.T, dtype=torch.float32)

    return pca_df, percent_variance, loadings, fitted_pca


def get_2d_scatter_plot(pca_df, percent_variance, light_mode):
    # Create the plot
    fig = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        hover_data=["State", "PC1", "PC2", "Channel"],
        color="Channel",
        opacity=0.9,
        text="State",
        labels={
            "PC1": "PC1 ({:.2f}%)".format(percent_variance[0]),
            "PC2": "PC2 ({:.2f}%)".format(percent_variance[1]),
        },
    )

    # if light mode use simple white theme
    if light_mode:
        fig.update_layout(template="plotly")

    # set range to be -1.2 to 1.2 for x and y
    fig.update_xaxes(range=[-1.2, 1.2])
    fig.update_yaxes(range=[-1.2, 1.2])

    # remove horizontal and vertical guides, but make axis lines thicker and dashed
    fig.update_xaxes(showgrid=False, zeroline=True, zerolinewidth=3)
    fig.update_yaxes(showgrid=False, zeroline=True, zerolinewidth=3)

    # remove legend
    fig.update_layout(showlegend=False)

    # move text up
    fig.update_traces(textposition="top center")

    # increase point size
    fig.update_traces(marker=dict(size=10))

    for _, row in pca_df.iterrows():
        fig.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=row["PC1"],
            y1=row["PC2"],
            line=dict(color="black" if light_mode else "white", width=2),
            layer="below",
        )

    # increase font size
    fig.update_layout(
        font=dict(
            size=20,
        ),
    )
    # increase font size for axis labels
    fig.update_xaxes(title_font=dict(size=24))
    fig.update_yaxes(title_font=dict(size=24))

    fig.update_layout(height=550, width=550)

    return fig


def get_3d_scatter_plot(pca_df, percent_variance, light_mode):
    fig = px.scatter_3d(
        pca_df,
        x="PC1",
        y="PC2",
        z="PC3",
        hover_data=["State", "PC1", "PC2", "PC3", "State"],
        color="Channel",
        text="State",
        labels={
            "PC1": "PC1 ({:.2f}%)".format(percent_variance[0]),
            "PC2": "PC2 ({:.2f}%)".format(percent_variance[1]),
            "PC3": "PC3 ({:.2f}%)".format(percent_variance[2]),
        },
        opacity=0.9,
    )

    # increase point size
    fig.update_traces(marker=dict(size=10))

    for _, row in pca_df.iterrows():
        trace = go.Scatter3d(
            x=[0, row["PC1"]],
            y=[0, row["PC2"]],
            z=[0, row["PC3"]],
            mode="lines",
            line=dict(color="black" if light_mode else "white", width=3),
            # remove from legend
            showlegend=False,
            # remove hover
            hoverinfo="skip",
        )
        fig.add_trace(trace)

    # increase height
    # Modify the font size in the layout
    fig.update_layout(
        font=dict(
            size=18,
        ),
        height=800,
    )

    return fig


def get_scree_plot(percent_variance, labels):
    fig = px.bar(
        x=[f"PC{i+1}" for i in range(len(labels))],
        y=percent_variance,
        title="Scree Plot",
        labels={
            "x": "Principal Components",
            "y": "Percent Variance Explained",
        },
        text=[f"{p:.2f}%" for p in percent_variance],
    )

    fig.update_layout(
        font=dict(
            size=18,
        ),
    )

    return fig


def get_loadings_plot(loadings, labels):
    # normalize row norm
    loadings = loadings / loadings.norm(dim=1, keepdim=True)

    loadings_df = pd.DataFrame(
        data=loadings,
        # index=labels,
        # columns=[f"PC{i+1}" for i in range(len(labels))],
    )
    fig = px.imshow(
        loadings_df,
        color_continuous_midpoint=0,
        color_continuous_scale="RdBu",
        text_auto=True,
    )
    fig.update_layout(
        coloraxis_showscale=False,
        margin=dict(l=0, r=0, t=0, b=0),
    )
    fig.update_layout(height=800, width=800)
    return fig
