"""
Author: Wenyu Ouyang
Date: 2023-10-19 21:34:29
LastEditTime: 2023-10-19 21:59:34
LastEditors: Wenyu Ouyang
Description: SHAP methods for deep learning models
FilePath: /torchhydro/torchhydro/explainers/shap.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""
import torch
import numpy as np
import shap


def plot_summary_shap_values(shap_values: torch.tensor, columns):
    mean_shap_values = shap_values.mean(axis=["preds", "batches"])

    fig = go.Figure()
    bar_plot = go.Bar(
        y=columns, x=mean_shap_values.abs().mean(axis="observations"), orientation="h"
    )
    fig.add_trace(bar_plot)
    fig.update_layout(yaxis={"categoryorder": "array", "categoryarray": columns[::-1]})

    return fig


def plot_summary_shap_values_over_time_series(shap_values: torch.tensor, columns):
    abs_mean_shap_values = shap_values.mean(axis=["batches"]).abs()
    multi_shap_values = abs_mean_shap_values.mean(axis="observations")

    fig = go.Figure()
    for i, pred_shap_values in enumerate(multi_shap_values.align_to("preds", ...)):
        fig.add_trace(
            go.Bar(
                y=columns, x=pred_shap_values, name=f"time-step {i}", orientation="h"
            )
        )
    fig.update_layout(
        barmode="stack",
        yaxis={"categoryorder": "array", "categoryarray": columns[::-1]},
    )
    return fig


def plot_shap_values_from_history(shap_values: torch.tensor, history: torch.tensor):
    mean_shap_values = shap_values.mean(axis=["preds", "batches"])
    mean_history_values = history.mean(axis="batches")

    figs: List[go.Figure] = []
    for feature_history, feature_shap_values in zip(
        mean_history_values.align_to("features", ...),
        mean_shap_values.align_to("features", ...),
    ):
        fig = go.Figure()
        scatter = go.Scatter(
            y=jitter(feature_shap_values),
            x=feature_shap_values,
            mode="markers",
            marker=dict(
                color=feature_history,
                colorbar=dict(title=dict(side="right", text="feature values")),
                colorscale=px.colors.sequential.Bluered,
            ),
        )
        fig.add_trace(scatter)
        fig.update_yaxes(range=[-0.05, 0.05])
        fig.update_xaxes(title_text="shap value")
        fig.update_layout(showlegend=False)
        figs.append(fig)
    return figs


def deep_explain_model_summary_plot(deep_hydro, test_loader) -> None:
    """Generate feature summary plot for trained deep learning models

    Parameters
    ----------
        model (object): trained model
        test_loader (TestLoader): test data loader
    """
    deep_hydro.model.eval()

    # background shape (L, N, M)
    # L - batch size, N - history length, M - feature size
    s_values_list = []
    if isinstance(history, list):
        deep_hydro.model = deep_hydro.model.to("cpu")
        deep_explainer = shap.DeepExplainer(deep_hydro.model, history)
        shap_values = deep_explainer.shap_values(history)
        s_values_list.append(shap_values)
    else:
        deep_explainer = shap.DeepExplainer(deep_hydro.model, background_tensor)
        shap_values = deep_explainer.shap_values(background_tensor)
    shap_values = np.stack(shap_values)
    # shap_values needs to be 4-dimensional
    if len(shap_values.shape) != 4:
        shap_values = np.expand_dims(shap_values, axis=0)
    shap_values = torch.tensor(
        shap_values, names=["preds", "batches", "observations", "features"]
    )

    # summary plot shows overall feature ranking
    # by average absolute shap values
    fig = plot_summary_shap_values(shap_values, test_loader.df.columns)
    abs_mean_shap_values = shap_values.mean(axis=["preds", "batches"])
    multi_shap_values = abs_mean_shap_values.mean(axis="observations")

    # summary plot for multi-step outputs
    # multi_shap_values = shap_values.apply_along_axis(np.mean, 'batches')
    fig = plot_summary_shap_values_over_time_series(shap_values, test_loader.df.columns)

    # summary plot for one prediction at datetime_start
    if isinstance(history, list):
        hist = history[0]
    else:
        hist = history

    history_numpy = torch.tensor(
        hist.cpu().numpy(), names=["batches", "observations", "features"]
    )

    shap_values = deep_explainer.shap_values(history)
    shap_values = np.stack(shap_values)
    if len(shap_values.shape) != 4:
        shap_values = np.expand_dims(shap_values, axis=0)
    shap_values = torch.tensor(
        shap_values, names=["preds", "batches", "observations", "features"]
    )

    figs = plot_shap_values_from_history(shap_values, history_numpy)


def plot_shap_value_heatmaps(shap_values: torch.tensor):
    average_shap_value_over_batches = shap_values.mean(axis="batches")

    x = [i for i in range(shap_values.align_to("observations", ...).shape[0])]
    y = [i for i in range(shap_values.align_to("preds", ...).shape[0])]

    figs: List[go.Figure] = []
    for shap_values_features in average_shap_value_over_batches.align_to(
        "features", ...
    ):
        fig = go.Figure()
        heatmap = go.Heatmap(
            z=shap_values_features,
            x=x,
            y=y,
            colorbar=dict(title=dict(side="right", text="feature values")),
            colorscale=px.colors.sequential.Bluered,
        )
        fig.add_trace(heatmap)
        fig.update_xaxes(title_text="sequence history steps")
        fig.update_yaxes(title_text="prediction steps")
        figs.append(fig)
    return figs


def deep_explain_model_heatmap(deep_hydro, test_loader) -> None:
    """Generate feature heatmap for prediction at a start time
    Args:
        model ([type]): trained model
        test_loader ([TestLoader]): test data loader
    Returns:
        None
    """
    deep_hydro.model.eval()

    # background shape (L, N, M)
    # L - batch size, N - history length, M - feature size
    # for each element in each N x M batch in L,
    # attribute to each prediction in forecast len
    s_values_list = []
    if isinstance(history, list):
        deep_explainer = shap.DeepExplainer(deep_hydro.model, history)
        shap_values = deep_explainer.shap_values(history)
        s_values_list.append(shap_values)
    else:
        deep_explainer = shap.DeepExplainer(deep_hydro.model, background_tensor)
        shap_values = deep_explainer.shap_values(background_tensor)
    shap_values = np.stack(shap_values)  # forecast_len x N x L x M
    if len(shap_values.shape) != 4:
        shap_values = np.expand_dims(shap_values, axis=0)
    shap_values = torch.tensor(
        shap_values, names=["preds", "batches", "observations", "features"]
    )
    figs = plot_shap_value_heatmaps(shap_values)

    # heatmap one prediction sequence at datetime_start
    # (seq_len*forecast_len) per fop feature
    to_explain = history
    shap_values = deep_explainer.shap_values(to_explain)
    shap_values = np.stack(shap_values)
    if len(shap_values.shape) != 4:
        shap_values = np.expand_dims(shap_values, axis=0)
    shap_values = torch.tensor(
        shap_values, names=["preds", "batches", "observations", "features"]
    )  # no fake ballo t
    figs = plot_shap_value_heatmaps(shap_values)
