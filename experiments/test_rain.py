import os
import json
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional

warnings.filterwarnings("ignore")

# ===================== 核心配置 ======================
# 1. 时间周期配置（仅保留验证期）
VALID_START = pd.to_datetime("2023-01-01 00:00:00")
VALID_END = pd.to_datetime("2023-12-31 23:00:00")

# 2. 文件路径配置
flood_events_excel = "/Volumes/Untitled/223_FloodEvents/场次筛选记录 -40.xlsx"
basin_data_dir = "/Volumes/Untitled/223_FloodEvents/timeseries/1h"

# NC文件路径（用户指定的验证期文件）
valid_nc_obs_path = "/Users/cylenlc/work/torchhydro/results/newbasin_223_FloodEvents_40_ori_nopeak_3_test/mtslstm_DapengScaler_hidden_128_128_lr_5e-04_batch_128_epoch_1_lrf_0.9/epochbest_model.pthflow_obs.nc"
valid_nc_pred_path = "/Users/cylenlc/work/torchhydro/results/newbasin_223_FloodEvents_40_ori_nopeak_3_test/mtslstm_DapengScaler_hidden_128_128_lr_5e-04_batch_128_epoch_1_lrf_0.9/epochbest_model.pthflow_pred.nc"

# 3. 保存路径配置
save_root = "results/newbasin_223_FloodEvents_40_ori_nopeak_3_test/"
valid_plot_dir = os.path.join(save_root, "valid_plots")

# ===================== 样式配置 ======================
precip_style = {
    "color": "#4169E1",
    "alpha": 0.7,
    "label": "Hourly Precipitation",
    "width": 0.8 / 24,
    "yaxis_label": "Precipitation (mm/h)",
    "yaxis_rotation": 270,
    "yaxis_labelpad": 20,
}

streamflow_style = {
    "obs": {
        "color": "black",
        "linestyle": "-",
        "marker": ".",
        "markersize": 2,
        "linewidth": 2.5,
        "label": "Observed Streamflow",
    },
    "pred": {
        "color": "#FF5733",
        "linestyle": "--",
        "linewidth": 2,
        "label": "Predicted Streamflow (MTSLSTM)",
    },
    "yaxis_label": "Streamflow (mm/h)",
    "yaxis_color": "black",
}

plot_config = {
    "figsize": (14, 6),
    "xlabel": "Time",
    "fontsize_label": 12,
    "fontsize_title": 14,
    "fontsize_legend": 10,
    "grid_alpha": 0.5,
    "grid_linestyle": "--",
    "time_format_short": "%Y-%m-%d",
    "time_format_hour": "%H:%M",
    "day_interval": 1,
    "hour_interval": 6,
}


# ===================== 工具函数 ======================
def ensure_dir(path):
    """确保文件夹存在"""
    Path(path).mkdir(parents=True, exist_ok=True)


def read_flood_events_from_excel():
    """从Excel读取验证期场次划分"""
    try:
        df = pd.read_excel(flood_events_excel)
        required_cols = ["basin_id", "start_time", "end_time"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Excel缺少列：{col}")

        df["start_time"] = pd.to_datetime(df["start_time"])
        df["end_time"] = pd.to_datetime(df["end_time"])

        flood_events = {}
        for basin_id, group in df.groupby("basin_id"):
            flood_events[basin_id] = [
                (idx + 1, row["start_time"], row["end_time"])
                for idx, row in group.iterrows()
            ]

        # 仅保留验证期场次
        valid_flood_events = {}
        for basin_id, events in flood_events.items():
            valid_events = [
                (fid, st, et)
                for fid, st, et in events
                if VALID_START <= st <= VALID_END
            ]
            if valid_events:
                valid_flood_events[basin_id] = valid_events

        print(f"\n✅ 场次读取完成：")
        print(
            f"- 验证期：{sum(len(v) for v in valid_flood_events.values())} 场次（{len(valid_flood_events)} 流域）"
        )
        print(f"- 验证期流域列表：{list(valid_flood_events.keys())}")
        return valid_flood_events
    except Exception as e:
        print(f"读取场次Excel失败：{e}")
        raise


def get_basin_id_to_nc_index_mapping(ds_obs, valid_basin_list):
    """
    核心修复：读取NC文件的`basin`变量，建立流域ID到NC索引的精准映射
    自动适配格式：Excel中是sanxia_60613300，NC中可能是60613300（纯数字）
    """
    basin_id_to_index = {}

    # 关键修改：读取NC文件中的`basin`变量（而非basin_id）
    if "basin" in ds_obs.coords or "basin" in ds_obs.variables:
        # 读取NC中的所有basin值（兼容coords和variables两种存储方式）
        if "basin" in ds_obs.coords:
            nc_basin_vals = ds_obs["basin"].values
        else:
            nc_basin_vals = ds_obs["basin"].values

        # 预处理NC的basin值：转字符串、去空格、统一格式
        nc_basin_strs = []
        for val in nc_basin_vals:
            # 处理数字/字符串混合的情况
            if isinstance(val, (int, np.integer, float, np.floating)):
                # 数字转字符串（去掉小数点）
                nc_str = str(int(val)) if val.is_integer() else str(val)
            else:
                nc_str = str(val).strip()
            nc_basin_strs.append(nc_str)

        # 建立双向匹配：适配sanxia_前缀
        for excel_basin_id in valid_basin_list:
            # 提取Excel流域ID的纯数字部分（比如sanxia_60613300 → 60613300）
            excel_basin_num = excel_basin_id.replace("sanxia_", "").strip()

            # 匹配逻辑：优先完全匹配，再匹配纯数字
            if excel_basin_id in nc_basin_strs:
                idx = nc_basin_strs.index(excel_basin_id)
                basin_id_to_index[excel_basin_id] = idx
            elif excel_basin_num in nc_basin_strs:
                idx = nc_basin_strs.index(excel_basin_num)
                basin_id_to_index[excel_basin_id] = idx

        # 打印映射结果（便于核对）
        print(f"\n✅ 从NC文件读取到basin映射（共{len(basin_id_to_index)}个匹配）：")
        for excel_id, nc_idx in basin_id_to_index.items():
            nc_val = nc_basin_strs[nc_idx]
            print(f"  Excel流域：{excel_id} → NC流域值：{nc_val} → NC索引：{nc_idx}")

        # 提示未匹配的流域
        unmatched = [bid for bid in valid_basin_list if bid not in basin_id_to_index]
        if unmatched:
            print(f"\n⚠️ 以下流域未在NC文件中匹配到：{unmatched}")
            print(f"   NC文件中的basin值列表：{nc_basin_strs}")
    else:
        # 如果NC文件没有basin字段，打印详细信息供手动配置
        print(f"\n❌ NC文件中未找到'basin'变量！")
        print(f"   NC文件所有coords：{list(ds_obs.coords.keys())}")
        print(f"   NC文件所有variables：{list(ds_obs.variables.keys())}")

    return basin_id_to_index


def read_basin_data(gage_id):
    """读取流域实测降雨数据（mm/h）"""
    gage_id_simple = gage_id.replace("sanxia_", "") if "sanxia_" in gage_id else gage_id
    basin_csv_path = os.path.join(basin_data_dir, f"{gage_id_simple}.csv")
    if not os.path.exists(basin_csv_path):
        basin_csv_path = os.path.join(basin_data_dir, f"{gage_id}.csv")
        if not os.path.exists(basin_csv_path):
            raise FileNotFoundError(f"流域数据文件不存在：{gage_id}")

    try:
        df_basin = pd.read_csv(basin_csv_path, parse_dates=["time"], index_col="time")
        required_cols = ["precipitation_obs"]
        for col in required_cols:
            if col not in df_basin.columns:
                raise ValueError(f"缺少列 '{col}' in {basin_csv_path}")

        # 仅筛选验证期数据
        precip_series = (
            df_basin["precipitation_obs"].loc[VALID_START:VALID_END].dropna()
        )
        print(f"加载{gage_id}(valid)降雨数据：{len(precip_series)} 条")
        return precip_series
    except Exception as e:
        print(f"读取流域{gage_id}数据失败：{e}")
        raise


def load_streamflow_dataset():
    """加载验证期NC文件"""
    try:
        ds_obs = xr.open_dataset(valid_nc_obs_path)
        ds_pred = xr.open_dataset(valid_nc_pred_path)
        # 打印NC文件关键信息（便于调试）
        print(f"\n✅ 加载NC文件成功：")
        print(f"   观测NC维度：{ds_obs.dims}")
        print(f"   预测NC维度：{ds_pred.dims}")
        print(f"   观测NC coords：{list(ds_obs.coords.keys())}")
        print(f"   观测NC variables：{list(ds_obs.variables.keys())}")
        return ds_obs, ds_pred
    except Exception as e:
        print(f"加载NC文件失败：{e}")
        raise


def create_nc_time_index(nc_time_length):
    """创建NC时间索引（匹配验证期）"""
    start, end = VALID_START, VALID_END

    full_time_index = pd.date_range(start=start, end=end, freq="H")
    if len(full_time_index) >= nc_time_length:
        nc_time_index = full_time_index[:nc_time_length]
    else:
        extend_length = nc_time_length - len(full_time_index)
        extend_index = pd.date_range(
            start=full_time_index[-1] + pd.Timedelta(hours=1),
            periods=extend_length,
            freq="H",
        )
        nc_time_index = pd.concat([full_time_index, extend_index])

    print(f"\n创建valid期NC时间索引：长度{len(nc_time_index)}")
    return nc_time_index


def read_streamflow_by_index(ds_obs, ds_pred, basin_index, nc_time_index):
    """读取验证期NC径流数据"""
    try:
        max_basin_index = ds_obs.sizes["basin"] - 1
        if basin_index < 0 or basin_index > max_basin_index:
            raise ValueError(
                f"流域索引{basin_index}超出NC文件范围（0~{max_basin_index}）"
            )

        # 读取观测数据（使用正确的streamflow变量）
        obs_raw = ds_obs["streamflow"].isel(basin=basin_index)
        obs_raw = obs_raw.values.flatten()
        obs_raw = obs_raw[: len(nc_time_index)]
        obs_raw = np.pad(
            obs_raw, (0, len(nc_time_index) - len(obs_raw)), constant_values=np.nan
        )

        # 读取预测数据
        pred_raw = ds_pred["streamflow"].isel(basin=basin_index)
        pred_raw = pred_raw.values.flatten()
        pred_raw = pred_raw[: len(nc_time_index)]
        pred_raw = np.pad(
            pred_raw, (0, len(nc_time_index) - len(pred_raw)), constant_values=np.nan
        )

        # 转换为Series并过滤验证期
        streamflow_obs = (
            pd.Series(obs_raw, index=nc_time_index, name="streamflow")
            .loc[VALID_START:VALID_END]
            .dropna()
        )
        streamflow_pred = (
            pd.Series(pred_raw, index=nc_time_index, name="streamflow")
            .loc[VALID_START:VALID_END]
            .dropna()
        )

        print(
            f"✅ 流域索引{basin_index}数据读取完成：观测{len(streamflow_obs)}条，预测{len(streamflow_pred)}条"
        )
        return streamflow_obs, streamflow_pred
    except Exception as e:
        print(f"读取径流数据失败：{e}")
        raise


def slice_data_by_flood(
    precip_series,
    streamflow_obs,
    streamflow_pred,
    start_time,
    end_time,
    gage_id,
    flood_id,
):
    """截取场次数据并对齐时间"""
    try:
        precip_slice = precip_series[
            (precip_series.index >= start_time) & (precip_series.index <= end_time)
        ]
        obs_slice = streamflow_obs[
            (streamflow_obs.index >= start_time) & (streamflow_obs.index <= end_time)
        ]
        pred_slice = streamflow_pred[
            (streamflow_pred.index >= start_time) & (streamflow_pred.index <= end_time)
        ]

        min_data_points = 10
        if (
            len(precip_slice) < min_data_points
            or len(obs_slice) < min_data_points
            or len(pred_slice) < min_data_points
        ):
            raise ValueError(
                f"数据点不足（{len(precip_slice)}/{len(obs_slice)}/{len(pred_slice)} < {min_data_points}）"
            )

        common_idx = precip_slice.index.intersection(obs_slice.index).intersection(
            pred_slice.index
        )
        if len(common_idx) < min_data_points:
            raise ValueError(f"共同时间点不足（{len(common_idx)} < {min_data_points}）")

        # 调试：打印场次数据范围
        print(f"  场次{flood_id}数据范围：")
        print(
            f"    降雨：{precip_slice.index.min()} ~ {precip_slice.index.max()}（{len(precip_slice)}条）"
        )
        print(
            f"    实测径流：{obs_slice.index.min()} ~ {obs_slice.index.max()}（{len(obs_slice)}条）"
        )
        print(
            f"    预测径流：{pred_slice.index.min()} ~ {pred_slice.index.max()}（{len(pred_slice)}条）"
        )

        return (
            precip_slice.loc[common_idx],
            obs_slice.loc[common_idx],
            pred_slice.loc[common_idx],
        )
    except Exception as e:
        print(f"跳过场次{flood_id}：{e}")
        return None, None, None


# ===================== 绘图函数 ======================
def plot_flood_hydrograph(
    gage_id, flood_id, precip, streamflow_obs, streamflow_pred, start_time, end_time
):
    """绘制场次降雨径流图"""
    # 确定保存路径
    basin_save_dir = os.path.join(valid_plot_dir, gage_id)
    ensure_dir(basin_save_dir)

    # 文件名
    start_time_str = start_time.strftime("%Y%m%d%H%M%S")
    save_file = os.path.join(
        basin_save_dir, f"{gage_id}_{start_time_str}_flood{flood_id}_rainflow.png"
    )

    # 绘图
    fig, ax1 = plt.subplots(figsize=plot_config["figsize"])

    # 径流曲线
    ax1.plot(streamflow_obs.index, streamflow_obs.values, **streamflow_style["obs"])
    ax1.plot(streamflow_pred.index, streamflow_pred.values, **streamflow_style["pred"])
    ax1.set_ylabel(
        streamflow_style["yaxis_label"],
        fontsize=plot_config["fontsize_label"],
        fontweight="bold",
    )
    ax1.set_xlabel(
        plot_config["xlabel"], fontsize=plot_config["fontsize_label"], fontweight="bold"
    )
    ax1.set_ylim(bottom=0)
    ax1.grid(alpha=plot_config["grid_alpha"], linestyle=plot_config["grid_linestyle"])

    # 降雨柱状图
    ax2 = ax1.twinx()
    ax2.bar(
        precip.index,
        precip.values,
        **{
            k: v
            for k, v in precip_style.items()
            if k not in ["yaxis_label", "yaxis_rotation", "yaxis_labelpad"]
        },
    )
    max_precip = precip.max() if precip.max() > 0 else 1
    ax2.set_ylim(0, max_precip * 3)
    ax2.invert_yaxis()
    ax2.set_ylabel(
        precip_style["yaxis_label"],
        fontsize=plot_config["fontsize_label"],
        fontweight="bold",
        rotation=precip_style["yaxis_rotation"],
        labelpad=precip_style["yaxis_labelpad"],
    )

    # 时间轴配置
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=plot_config["day_interval"]))
    ax1.xaxis.set_major_formatter(
        mdates.DateFormatter(plot_config["time_format_short"])
    )
    ax1.xaxis.set_minor_locator(
        mdates.HourLocator(interval=plot_config["hour_interval"])
    )
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=9)
    plt.setp(ax1.xaxis.get_minorticklabels(), rotation=0, ha="right", fontsize=8)

    # 图例和标题
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper left",
        fontsize=plot_config["fontsize_legend"],
    )
    plt.title(
        f"Basin {gage_id} - Flood Event {flood_id}\n({start_time.strftime('%Y-%m-%d')} ~ {end_time.strftime('%Y-%m-%d')})",
        fontsize=plot_config["fontsize_title"],
        fontweight="bold",
        pad=20,
    )

    # 保存
    plt.tight_layout()
    plt.savefig(save_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ 图保存完成：{save_file}")
    return save_file


# ===================== 主处理函数 ======================
def process_valid_flood_events():
    """仅处理验证期场次：读取数据+绘图"""
    success_plots = 0

    # 初始化文件夹
    ensure_dir(valid_plot_dir)

    # 1. 读取验证期场次
    valid_flood_events = read_flood_events_from_excel()
    valid_basin_list = list(valid_flood_events.keys())

    # 2. 加载NC文件
    ds_obs, ds_pred = load_streamflow_dataset()
    nc_time_length = ds_obs.sizes["time"]
    nc_time_index = create_nc_time_index(nc_time_length)

    # 3. 核心修复：读取NC的`basin`变量，建立精准映射（传入valid_basin_list适配格式）
    basin_id_to_index = get_basin_id_to_nc_index_mapping(ds_obs, valid_basin_list)

    # 4. 无映射则终止，提示手动配置
    if not basin_id_to_index:
        print("\n❌ 无法自动建立流域映射，请手动核对NC文件的basin值和Excel的basin_id！")
        print(f"   验证期需要处理的流域：{valid_basin_list}")
        print(f"   NC文件basin维度长度：{ds_obs.sizes['basin']}")
        return

    # 5. 遍历流域（仅处理有映射的流域）
    for basin_id, events in valid_flood_events.items():
        print(f"\n===== 处理流域 {basin_id} (valid) ======")

        # 从精准映射中获取NC索引
        if basin_id not in basin_id_to_index:
            print(f"❌ 流域{basin_id}未在NC映射中找到，跳过！")
            continue
        basin_index = basin_id_to_index[basin_id]
        print(f"✅ 流域{basin_id} → NC索引{basin_index}（精准匹配）")

        try:
            # 读取流域降雨数据
            precip_series = read_basin_data(basin_id)

            # 读取径流数据（使用精准匹配的NC索引）
            streamflow_obs, streamflow_pred = read_streamflow_by_index(
                ds_obs, ds_pred, basin_index, nc_time_index
            )

            if len(streamflow_obs) == 0 or len(streamflow_pred) == 0:
                print(f"❌ 无有效径流数据，跳过")
                continue

            # 遍历场次
            for flood_id, start_time, end_time in events:
                print(f"\n处理场次{flood_id}：{start_time} ~ {end_time}")
                # 截取场次数据
                precip_slice, obs_slice, pred_slice = slice_data_by_flood(
                    precip_series,
                    streamflow_obs,
                    streamflow_pred,
                    start_time,
                    end_time,
                    basin_id,
                    flood_id,
                )

                if precip_slice is None:
                    continue

                # 绘图
                plot_flood_hydrograph(
                    basin_id,
                    flood_id,
                    precip_slice,
                    obs_slice,
                    pred_slice,
                    start_time,
                    end_time,
                )
                success_plots += 1

        except Exception as e:
            print(f"❌ 处理流域{basin_id}失败：{e}")
            continue

    # 关闭NC文件
    ds_obs.close()
    ds_pred.close()

    # 统计结果
    print("\n===== 批量处理完成 =====")
    print(f"验证期：成功绘图{success_plots}张")
    print(f"结果路径：{valid_plot_dir}")


def main():
    # 仅执行验证期绘图流程
    process_valid_flood_events()


if __name__ == "__main__":
    main()
