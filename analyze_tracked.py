import pandas as pd
import plotly.express as px
import numpy as np 
import plotly.graph_objects as go
import os
import json
from plotly.subplots import make_subplots

day = "2026-08-08"
run_name = "kovasznay tracked (ANALYZE AND INCLUDE)"
multirun_folder = f"multirun/{day}/{run_name}"

dfs = []

for filename in os.listdir(multirun_folder):

    # skip non-folder
    if ".yaml" in filename:
        continue

    df = None
    read_json = None

    for artifact_file in os.listdir(f"{multirun_folder}/{filename}"):

        full_artifact_path = f"{multirun_folder}/{filename}/{artifact_file}"

        if ".json" in artifact_file:
            
            with open(full_artifact_path, "r") as f:
                read_json = json.load(f)
    
        elif artifact_file == "tracked.csv":

            df = pd.read_csv(full_artifact_path)

    df["seed"] = read_json["seed"]
    df["best_mse"] = read_json["best_mse"]

    dfs.append(df)

smoothed_percentage_updated_figure = go.Figure()
val_mse_figure = go.Figure()

for df in dfs:

    # smooth the signal
    df["smoothed_updated_percentage"] = df["updated_percentage"].rolling(window=25).mean()

    seed_val = df["seed"].iloc[0]

    smoothed_percentage_updated_figure.add_trace(go.Scatter(
        x=df["epoch"],
        y=df["smoothed_updated_percentage"],
        mode='lines',
        name=f"Seed {seed_val}"
    ))

    val_mse_figure.add_trace(go.Scatter(
        x = df["epoch"],
        y = df["val_mse"],
        mode="lines+markers",
        name=f"Seed {seed_val}",
        connectgaps=True,
    ))

smoothed_percentage_updated_figure.update_layout(
    title="Percentage Params Updated (smoothed)",
    xaxis_title="Epoch",
    yaxis_title="Smoothed Updated Percentage"
)

val_mse_figure.update_yaxes(type="log")

val_mse_figure.update_layout(
    title="Val MSE per epoch",
    xaxis_title="Epoch",
    yaxis_title="Val MSE"
)

smoothed_percentage_updated_figure.show()
val_mse_figure.show()

COLOR_MAP = {
    "init": "#1f77b4",   # blue
    "bound": "#ff7f0e",  # orange
    "pde": "#2ca02c"     # green
}

# show cosine similarity for different seeds
for df in dfs:

    # g_config, adam_out, update_dir
    # 3 rows
    cosine_sim_fig = make_subplots(
        rows = 3, cols=1,
        subplot_titles=("Cosine Sim 'g_config'", "Cosine Sim 'adam_out'", "Cosine Sim 'update_dir'"),
        shared_xaxes=True
    )

    for loss_name in ["init", "bound", "pde"]:
        for v, vec_name in enumerate(["g_config", "adam_out", "update_dir"]):

            smoothed_cosine_sim = df[f"{vec_name}_vs_g_{loss_name}"].rolling(window=50, min_periods=1).mean()

            cosine_sim_fig.add_trace(
                go.Scatter(
                    x = df["epoch"],
                    y = smoothed_cosine_sim,
                    mode = "lines",
                    name=f"vs. g_{loss_name}",
                    legendgroup=loss_name,
                    showlegend=(v == 0),
                    line=dict(color=COLOR_MAP[loss_name])
                ),
                row = (v + 1), col = 1
            )

    cosine_sim_fig.show()

# win percentage (adam output vs update dir)
# regular config has win rate 0 since adam out = update dir
fig_win_update_dir_over_adam_out = go.Figure()
fig_average_cosine_sim_update_dir = go.Figure()

seeds = [f"Seed {df['seed'].iloc[0]}" for df in dfs]

for loss_name in ["init", "bound", "pde"]:

    # colect values for all seeds
    y_values = []

    avg_cosine_sim_update_dir = []
    
    for df in dfs:
        
        adam_out_cosine_sim = df[f"adam_out_vs_g_{loss_name}"].to_numpy()
        update_dir_cosine_sim = df[f"update_dir_vs_g_{loss_name}"].to_numpy()

        win_rate_on_loss = (update_dir_cosine_sim > adam_out_cosine_sim).mean()

        #print(f"adam out cosine sim: #nans = {np.sum(np.isnan(adam_out_cosine_sim))}")
        #print(f"update dir cosine sim: #nans = {np.sum(np.isnan(update_dir_cosine_sim))}")
        
        y_values.append(win_rate_on_loss)
        avg_cosine_sim_update_dir.append(np.mean(update_dir_cosine_sim[1:]))

    #print(avg_cosine_sim_update_dir)

    fig_win_update_dir_over_adam_out.add_trace(go.Bar(
        x=seeds,
        y=y_values,
        name=f"g_{loss_name}",
        marker_color=COLOR_MAP[loss_name]
    ))

    fig_average_cosine_sim_update_dir.add_trace(go.Bar(
        x=seeds,
        y=avg_cosine_sim_update_dir,
        name=f"g_{loss_name}",
        marker_color=COLOR_MAP[loss_name]
    ))

fig_win_update_dir_over_adam_out.update_layout(
    title="Win Rate (update dir over adam out) across Seeds",
    xaxis_title="Seeds",
    yaxis_title="Win Rate",
    barmode='group'
)

fig_average_cosine_sim_update_dir.update_layout(
    title="Avg. Cosine Sim (update dir over adam out) across Seeds",
    xaxis_title="Seeds",
    yaxis_title="Avg. Cosine Sim.",
    barmode='group'
)



fig_win_update_dir_over_adam_out.show()
fig_average_cosine_sim_update_dir.show()