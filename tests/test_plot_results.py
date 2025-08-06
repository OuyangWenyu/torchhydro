import logging

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

logging.basicConfig(level=logging.WARNING)


def plot_rainfall_runoff_wu(
    t=None,
    p=None,
    qs=None,
    fig_size=(8, 6),
    qs_c_lst="rbkgcmy",
    p_c_lst="rbkgcmy",
    qs_leg_lst=None,
    p_leg_lst=None,
    dash_lines=None,
    title=None,
    xlabel=None,
    y_qs_label=None,
    y_p_label=None,
    linewidth=1,
    bar_or_plot=False,
    save_path=None,
):
    fig, ax = plt.subplots(figsize=fig_size)
    if dash_lines is not None:
        assert type(dash_lines) == list
    else:
        dash_lines = np.full(len(qs), False).tolist()
    for k in range(len(qs)):
        q = qs[k]
        leg_str = None
        if qs_leg_lst is not None:
            leg_str = qs_leg_lst[k]
        (line_i,) = ax.plot(t, q, color=qs_c_lst[k], label=leg_str, linewidth=linewidth)
        if dash_lines[k]:
            line_i.set_dashes([2, 2, 10, 2])

    ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] * 1.2)

    if title is not None:
        ax.set_title(title, loc="center", fontdict={"fontsize": 15})
    if y_qs_label is not None:
        ax.set_ylabel(y_qs_label, fontsize=15)

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=15)

    lines_t, labels_t = ax.get_legend_handles_labels()

    if p is not None:
        ax2 = ax.twinx()
        ax2.invert_yaxis()

        for k in range(len(p)):
            leg_str = None
            if p_leg_lst is not None:
                leg_str = p_leg_lst[k]
            if bar_or_plot:
                ax2.bar(t, p[k], label=leg_str, color=p_c_lst[k])
            else:
                ax2.plot(t, p[k], label=leg_str, color=p_c_lst[k])
        if y_p_label is not None:
            ax2.set_ylabel(y_p_label, fontsize=15)

        lines_p, labels_p = ax2.get_legend_handles_labels()
        ax.legend(
            lines_t + lines_p,
            labels_t + labels_p,
            bbox_to_anchor=(0.01, 0.9),
            loc="upper left",
            fontsize=15,
        )
    else:
        ax.legend(
            lines_t,
            labels_t,
            bbox_to_anchor=(0.01, 0.9),
            loc="upper left",
            fontsize=15,
        )

    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    if save_path is not None:
        plt.ioff()
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def aggregate_dataset(ds: xr.Dataset, gap, basin_id):
    if gap == "3h":
        gap = 3
        start_times = [2, 5, 8, 11, 14, 17, 20, 23]
        end_times = [1, 4, 7, 10, 13, 16, 19, 22]
        time_index = ds.indexes["time"]
        # 修剪开始时间
        while time_index[0].hour not in start_times:
            ds = ds.isel(time=slice(1, None))
            time_index = ds.indexes["time"]
        # 修剪结束时间
        while time_index[-1].hour not in end_times:
            ds = ds.isel(time=slice(None, -1))
            time_index = ds.indexes["time"]
    df_res = ds.to_dataframe().reset_index()
    df_res.set_index("time", inplace=True)
    numeric_cols = df_res.select_dtypes(include=[np.number]).columns
    aggregated_data = {}
    sm_surface_data = []
    sm_rootzone_data = []
    for col in numeric_cols:
        if col in ["sm_surface", "sm_rootzone"]:
            continue
        data = df_res[col].values
        aggregated_values = []
        for start in range(0, len(data), gap):
            chunk = data[start : start + gap]
            if np.isnan(chunk).any():
                aggregated_values.append(np.nan)
            else:
                aggregated_values.append(np.sum(chunk))
        aggregated_times = df_res.index[gap - 1 :: gap][: len(aggregated_values)]
        aggregated_data[col] = xr.DataArray(
            np.array(aggregated_values).reshape(-1, 1),
            dims=["time", "basin"],
            coords={"time": aggregated_times, "basin": [basin_id]},
        )
    # 处理 sm_surface 和 sm_rootzone 变量
    if "sm_surface" in df_res.columns:
        sm_surface_data = df_res["sm_surface"].iloc[gap - 1 :: gap].values
        aggregated_data["sm_surface"] = xr.DataArray(
            sm_surface_data.reshape(-1, 1),
            dims=["time", "basin"],
            coords={"time": aggregated_times, "basin": [basin_id]},
        )
    if "sm_rootzone" in df_res.columns:
        sm_rootzone_data = df_res["sm_rootzone"].iloc[gap - 1 :: gap].values
        aggregated_data["sm_rootzone"] = xr.DataArray(
            sm_rootzone_data.reshape(-1, 1),
            dims=["time", "basin"],
            coords={"time": aggregated_times, "basin": [basin_id]},
        )
    if "total_evaporation_hourly" in df_res.columns:
        aggregated_data["total_precipitation_hourly"] *= 1000
        aggregated_data["total_evaporation_hourly"] *= -1000
    result_ds = xr.Dataset(
        aggregated_data,
        coords={"time": aggregated_times, "basin": [basin_id]},
    )
    result_ds = result_ds.transpose("basin", "time")
    return result_ds


import os
from hydrodatasource.reader import access_fs
import pandas as pd
import xarray as xr


def test_plot_result():
    show = pd.read_csv("/home/wangyang1/torchhydro/data/basin_id(498+41).csv", dtype={"id": str})
    gage_id = show["id"].values.tolist()
    pred_path = "/home/jiaxuwu/IdeaProjects/torchhydro/experiments/results/train_with_ear5land/ex1_539_basins/epochbestflow_pred.nc"
    obs_path = "/home/jiaxuwu/IdeaProjects/torchhydro/experiments/results/train_with_ear5land/ex1_539_basins/epochbestflow_obs.nc"
    pred = xr.open_dataset(pred_path)
    obs = xr.open_dataset(obs_path)
    time = pred["streamflow"].coords["time"].values
    rainfall_name = 'total_precipitation_hourly'
    for basin in gage_id:
        if os.path.exists(f"figure_0701_539basins_era5land_cn_1h/{basin}.jpg"):
            continue
        try:
            prcp_path = f"basins-origin/hour_data/1h/mean_data/data_forcing_era5land_streamflow/data_forcing_streamflow_{basin}.nc"
            rainfall = access_fs.spec_path(prcp_path, head="minio")
            rainfall = aggregate_dataset(ds=rainfall, gap="3h", basin_id=basin)
            rainfall = np.ravel(rainfall[rainfall_name].sel(time=time).values)
            plot_rainfall_runoff_wu(
                time,
                p=None if rainfall is None else [rainfall],
                qs=[
                    obs["streamflow"].sel(basin=basin).values,
                    pred["streamflow"].sel(basin=basin).values,
                ],
                fig_size=(12, 6),
                p_c_lst=None if rainfall is None else ["tab:red"],
                qs_c_lst=["tab:blue", "tab:orange"],
                qs_leg_lst=["Observation", "Prediction"],
                p_leg_lst=None if rainfall is None else ["GPM"],
                title=(
                    f"Comparison of Observed and Predicted Streamflow({basin})"
                    if rainfall is None
                    else f"Comparison of GPM,Observed and Predicted Streamflow({basin})"
                ),
                xlabel="time(h)",
                y_qs_label="streamflow(mm/h)",
                y_p_label=None if rainfall is None else "precipitation(mm/h)",
                linewidth=2,
                bar_or_plot=False,
                save_path=f"figure_0701_539basins_era5land_cn_1h/{basin}.jpg"
            )
        except KeyError:
            print(basin)
