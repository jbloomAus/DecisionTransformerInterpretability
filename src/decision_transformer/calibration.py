import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm

from .train import evaluate_dt_agent


def calibration_statistics(
    dt,
    env_id,
    env_func,
    initial_rtg_range=np.linspace(-1, 1, 21),
    trajectories=100,
    num_envs=8,
):
    statistics = []
    pbar = tqdm(initial_rtg_range, desc="initial_rtg")
    for initial_rtg in pbar:
        statistics.append(
            evaluate_dt_agent(
                env_id=env_id,
                model=dt,
                env_func=env_func,
                track=False,
                batch_number=0,
                initial_rtg=initial_rtg,
                trajectories=trajectories,
                use_tqdm=False,
                num_envs=num_envs,
            )
        )
        pbar.set_description(f"initial_rtg: {initial_rtg}")
    return statistics


def plot_calibration_statistics(statistics, show_spread=False, CI=0.95):
    df = pd.DataFrame(statistics)

    # add line to legend
    fig = px.line(
        df,
        x="initial_rtg",
        y="mean_reward",
        title="Calibration Plot",
        template="plotly_white",
    )
    fig.update_layout(
        xaxis_title="Initial RTG",
        yaxis_title="Reward",
        legend_title="",
    )
    fig.add_shape(
        type="line",
        x0=df.initial_rtg.min(),
        y0=df.initial_rtg.min(),
        x1=df.initial_rtg.max(),
        y1=df.initial_rtg.max(),
        line=dict(
            color="LightSeaGreen",
            width=4,
            dash="dashdot",
        ),
    )

    if show_spread:
        upper = (1 - CI) / 2
        lower = 1 - upper
        # add 95% CI
        # also show 97.5 percentile and 2.5 percentile
        df["percentile_975"] = df.rewards.apply(
            lambda x: np.percentile(x, 100 * upper)
        )
        df["percentile_025"] = df.rewards.apply(
            lambda x: np.percentile(x, 100 * lower)
        )
        # shade between 97.5 and 2.5 percentile
        fig.add_trace(
            go.Scatter(
                x=df.initial_rtg,
                y=df.percentile_975,
                mode="lines",
                name=f"{(int(CI*100))}% CI",
                showlegend=False,
                # then use light blue fill color
                line=dict(color="rgba(0,0,0,0)", width=0.5),
                fillcolor="rgba(0,100,80,0.2)",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df.initial_rtg,
                y=df.percentile_025,
                mode="lines",
                name=f"{(int(CI*100))}% CI",
                # then use light blue fill color
                line=dict(color="rgba(0,0,0,0)", width=0.5),
                fillcolor="rgba(0,100,80,0.2)",
                fill="tonexty",
            )
        )

    return fig
