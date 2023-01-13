import pandas as pd 
import plotly.express as px
from tqdm import tqdm
import numpy as np
from .train import evaluate_dt_agent

def calibration_statistics(dt, env_id, make_env, initial_rtg_range = np.linspace(-1,1,21), trajectories=100):
    statistics = []
    pbar = tqdm(initial_rtg_range, desc="initial_rtg")
    for initial_rtg in pbar:
        statistics.append(evaluate_dt_agent(
            env_id = env_id,
            dt = dt,
            make_env=make_env,
            track = False,
            batch_number = 0,
            initial_rtg=initial_rtg,
            trajectories=trajectories,
            use_tqdm=False))
        pbar.set_description(f"initial_rtg: {initial_rtg}")
    return statistics

def plot_calibration_statistics(statistics):

    df = pd.DataFrame(statistics)

    fig = px.line(df, x="initial_rtg", y="mean_reward", title="Calibration Plot",
        template="plotly_white")
    fig.update_layout(
        xaxis_title="Initial RTG",
        yaxis_title="Mean Reward",
        legend_title="Proportion Completed",
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
        )
    )

    # fig.show()
    return fig 