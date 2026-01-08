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
# 1. 时间周期配置
TRAIN_START = pd.to_datetime("2014-01-01 00:00:00")
TRAIN_END = pd.to_datetime("2023-07-31 23:00:00")

# 验证期配置（与训练脚本中的设置保持一致）
VALID_START_CONFIG = pd.to_datetime("2023-08-01 00:00:00")  # 训练脚本中设置的验证期起始时间
VALID_END_CONFIG = pd.to_datetime("2024-12-31 23:00:00")    # 训练脚本中设置的验证期结束时间

# 2. Warmup时间偏移配置（重要！）
# 由于训练时设置了warmup_length，导致NC文件的时间标签与实际数据错位
# 如果训练时 warmup_length=720小时（30天），则验证期NC文件时间标签向后偏移30天
# 需要在读取NC数据后，把时间索引向前偏移相应天数
VALID_WARMUP_OFFSET_DAYS = 30  # 验证期NC文件的时间偏移天数（根据训练时的warmup_length调整）
                                # warmup_length=0 → 偏移0天
                                # warmup_length=240小时 → 偏移10天
                                # warmup_length=720小时 → 偏移30天

# 自动计算NC文件中的实际时间范围（向后偏移）
VALID_START = VALID_START_CONFIG + pd.Timedelta(days=VALID_WARMUP_OFFSET_DAYS)
VALID_END = VALID_END_CONFIG

print(f"验证期配置:")
print(f"  训练脚本设置: {VALID_START_CONFIG} 到 {VALID_END_CONFIG}")
print(f"  Warmup偏移: {VALID_WARMUP_OFFSET_DAYS} 天")
print(f"  NC文件实际时间范围: {VALID_START} 到 {VALID_END}")

# 3. 文件路径配置
flood_events_excel = "Temporary_file/副本场次筛选汇总20260106.xlsx"
basin_data_dir = "/home/litianli/Dataset/223_FloodEvents_106/timeseries/1h/"

# NC文件路径
train_nc_obs_path = "results/6_newbasin_223_FloodEvents_form_106/mtslstm_DapengScaler_hidden_128_128_lr_1e-03_batch_64_epoch_3_lrf_0.95/epochbest_model.pthflow_obs.nc"
train_nc_pred_path = "results/6_newbasin_223_FloodEvents_form_106/mtslstm_DapengScaler_hidden_128_128_lr_1e-03_batch_64_epoch_3_lrf_0.95/epochbest_model.pthflow_pred.nc"
valid_nc_obs_path = "results/newbasin_223_FloodEvents_form_106/mtslstm_DapengScaler_hidden_128_128_lr_1e-03_batch_64_epoch_20_lrf_0.95/epochbest_model.pthflow_obs.nc"
valid_nc_pred_path = "results/newbasin_223_FloodEvents_form_106/mtslstm_DapengScaler_hidden_128_128_lr_1e-03_batch_64_epoch_20_lrf_0.95/epochbest_model.pthflow_pred.nc"

# 4. 保存路径配置
save_root = "results/6_223_FloodEvents_106/"
train_plot_dir = os.path.join(save_root, "train_plots")
valid_plot_dir = os.path.join(save_root, "valid_plots")
metrics_save_dir = os.path.join(save_root, "metrics_results")  # 重命名更通用

# ===================== 样式配置（全英文） ======================
precip_style = {
    "color": "#4169E1",       
    "alpha": 0.7,             
    "label": "Hourly Precipitation",
    "width": 0.8/24,          
    "yaxis_label": "Precipitation (mm/h)",
    "yaxis_rotation": 270,    
    "yaxis_labelpad": 20      
}

streamflow_style = {
    "obs": {
        "color": "black",
        "linestyle": "-",
        "marker": ".",
        "markersize": 2,
        "linewidth": 2.5,      
        "label": "Observed Streamflow"
    },
    "pred": {
        "color": "#FF5733",    
        "linestyle": "--",
        "linewidth": 2,        
        "label": "Predicted Streamflow (MTSLSTM)"
    },
    "yaxis_label": "Streamflow (mm/h)",
    "yaxis_color": "black"
}

plot_config = {
    "figsize": (14, 8),       # 调整画布高度以容纳更多指标标注
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
    "metric_fontsize": 10,    # 微调字体大小适配更多指标
    "metric_bg_alpha": 0.8,
}

# ===================== 工具函数 ======================
def ensure_dir(path):
    """Ensure directory exists"""
    Path(path).mkdir(parents=True, exist_ok=True)

def read_flood_events_from_excel():
    """Read flood events from Excel and split into train/valid periods"""
    try:
        df = pd.read_excel(flood_events_excel)
        required_cols = ["basin_id", "start_time", "end_time"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Excel missing column: {col}")
        
        # 处理可能只包含时间的单元格
        df["start_time"] = pd.to_datetime(df["start_time"], errors='coerce')
        df["end_time"] = pd.to_datetime(df["end_time"], errors='coerce')
        
        # 删除无法转换的行
        invalid_rows = df[df["start_time"].isna() | df["end_time"].isna()]
        if not invalid_rows.empty:
            print(f"⚠️  警告: 发现 {len(invalid_rows)} 行无效的时间数据，已跳过:")
            print(invalid_rows[["basin_id", "start_time", "end_time"]])
            df = df.dropna(subset=["start_time", "end_time"])
        
        flood_events = {}
        for basin_id, group in df.groupby("basin_id"):
            flood_events[basin_id] = [(idx+1, row["start_time"], row["end_time"]) 
                                     for idx, row in group.iterrows()]
        
        train_flood_events = {}
        valid_flood_events = {}
        for basin_id, events in flood_events.items():
            train_events = [(fid, st, et) for fid, st, et in events if TRAIN_START <= st <= TRAIN_END]
            valid_events = [(fid, st, et) for fid, st, et in events if VALID_START <= st <= VALID_END]
            if train_events:
                train_flood_events[basin_id] = train_events
            if valid_events:
                valid_flood_events[basin_id] = valid_events
        
        print(f"\n✅ Flood events loaded successfully：")
        print(f"- Training period: {sum(len(v) for v in train_flood_events.values())} events ({len(train_flood_events)} basins)")
        print(f"- Validation period: {sum(len(v) for v in valid_flood_events.values())} events ({len(valid_flood_events)} basins)")
        return train_flood_events, valid_flood_events
    except Exception as e:
        print(f"Failed to read flood events Excel: {e}")
        raise

def read_basin_data(gage_id, time_period):
    """Read basin observed precipitation data (mm/h)"""
    gage_id_simple = gage_id.replace("sanxia_", "") if "sanxia_" in gage_id else gage_id
    basin_csv_path = os.path.join(basin_data_dir, f"{gage_id_simple}.csv")
    if not os.path.exists(basin_csv_path):
        basin_csv_path = os.path.join(basin_data_dir, f"{gage_id}.csv")
        if not os.path.exists(basin_csv_path):
            raise FileNotFoundError(f"Basin data file not found: {gage_id}")
    
    try:
        df_basin = pd.read_csv(
            basin_csv_path,
            parse_dates=["time"],
            index_col="time"
        )
        required_cols = ["precipitation_obs"]
        for col in required_cols:
            if col not in df_basin.columns:
                raise ValueError(f"Missing column '{col}' in {basin_csv_path}")
        
        if time_period == "train":
            start, end = TRAIN_START, TRAIN_END
        else:
            start, end = VALID_START, VALID_END
        
        precip_series = df_basin["precipitation_obs"].loc[start:end].dropna()
        print(f"Loaded {gage_id} ({time_period}) precipitation data: {len(precip_series)} records")
        return precip_series
    except Exception as e:
        print(f"Failed to read basin {gage_id} data: {e}")
        raise

def load_streamflow_dataset(nc_obs_path, nc_pred_path):
    """Load NC files (only open, no data reading)"""
    try:
        ds_obs = xr.open_dataset(nc_obs_path)
        ds_pred = xr.open_dataset(nc_pred_path)
        if "streamflow" not in ds_obs.variables:
            raise ValueError(f"Observation NC file missing 'streamflow' variable. Existing variables: {list(ds_obs.variables.keys())}")
        if "streamflow" not in ds_pred.variables:
            raise ValueError(f"Prediction NC file missing 'streamflow' variable. Existing variables: {list(ds_pred.variables.keys())}")
        print(f"NC files loaded successfully: Observation dimensions {ds_obs.dims}, Prediction dimensions {ds_pred.dims}")
        return ds_obs, ds_pred
    except Exception as e:
        print(f"Failed to load NC files: {e}")
        raise

def read_streamflow_by_basin_name(ds_obs, ds_pred, basin_id, time_period):
    """Read streamflow data by basin name + time range"""
    try:
        if time_period == "train":
            time_slice = slice(TRAIN_START, TRAIN_END)
        else:
            time_slice = slice(VALID_START, VALID_END)
        
        obs_selected = ds_obs["streamflow"].sel(basin=basin_id, time=time_slice)
        pred_selected = ds_pred["streamflow"].sel(basin=basin_id, time=time_slice)
        
        streamflow_obs = obs_selected.to_series().dropna()
        streamflow_pred = pred_selected.to_series().dropna()
        
        # ⚠️ 验证期时间修正：根据配置的VALID_WARMUP_OFFSET_DAYS进行时间偏移
        if time_period == "valid" and VALID_WARMUP_OFFSET_DAYS > 0:
            streamflow_obs.index = streamflow_obs.index - pd.Timedelta(days=VALID_WARMUP_OFFSET_DAYS)
            streamflow_pred.index = streamflow_pred.index - pd.Timedelta(days=VALID_WARMUP_OFFSET_DAYS)
            print(f"  ⚠️  验证期时间已修正：NC标签向前偏移{VALID_WARMUP_OFFSET_DAYS}天以对齐实际数据")
        
        streamflow_obs.name = "streamflow"
        streamflow_pred.name = "streamflow"
        streamflow_obs.index.name = "time"
        streamflow_pred.index.name = "time"
        
        print(f"Basin {basin_id} data loaded: Observation {len(streamflow_obs)} records, Prediction {len(streamflow_pred)} records")
        return streamflow_obs, streamflow_pred
    except KeyError as e:
        raise ValueError(f"Basin {basin_id} not found in NC files: {e}")
    except Exception as e:
        print(f"Failed to read streamflow data: {e}")
        raise

def slice_data_by_flood(precip_series, streamflow_obs, streamflow_pred, 
                        start_time, end_time, gage_id, flood_id):
    """Slice event data and align time"""
    try:
        precip_slice = precip_series[(precip_series.index >= start_time) & (precip_series.index <= end_time)]
        obs_slice = streamflow_obs[(streamflow_obs.index >= start_time) & (streamflow_obs.index <= end_time)]
        pred_slice = streamflow_pred[(streamflow_pred.index >= start_time) & (streamflow_pred.index <= end_time)]
        
        min_data_points = 10
        if len(precip_slice) < min_data_points or len(obs_slice) < min_data_points or len(pred_slice) < min_data_points:
            raise ValueError(f"Insufficient data points ({len(precip_slice)}/{len(obs_slice)}/{len(pred_slice)} < {min_data_points})")
        
        common_idx = precip_slice.index.intersection(obs_slice.index).intersection(pred_slice.index)
        if len(common_idx) < min_data_points:
            raise ValueError(f"Insufficient common time points ({len(common_idx)} < {min_data_points})")
        
        return precip_slice.loc[common_idx], obs_slice.loc[common_idx], pred_slice.loc[common_idx]
    except Exception as e:
        print(f"Skip flood event {flood_id}: {e}")
        return None, None, None

def calculate_nse(obs, pred):
    """Calculate Nash-Sutcliffe Efficiency (NSE)"""
    mask = ~np.isnan(obs) & ~np.isnan(pred)
    if not np.any(mask):
        return np.nan
    obs_clean = obs[mask]
    pred_clean = pred[mask]
    
    obs_mean = np.mean(obs_clean)
    if np.all(obs_clean == obs_mean):
        return np.nan
    
    numerator = np.sum((obs_clean - pred_clean) ** 2)
    denominator = np.sum((obs_clean - obs_mean) ** 2)
    nse = 1 - (numerator / denominator)
    return round(nse, 8)

def calculate_runoff_coefficient(precip_data, runoff_data):
    """
    Calculate runoff coefficient: Runoff Coefficient = Total Runoff Depth / Total Precipitation
    - precip_data: Hourly precipitation (mm/h)
    - runoff_data: Hourly runoff (mm/h) (obs or pred)
    - Returns: Runoff coefficient (4 decimal places) or NaN
    """
    try:
        mask = ~np.isnan(precip_data) & ~np.isnan(runoff_data)
        if not np.any(mask):
            return np.nan
        
        precip_clean = precip_data[mask]
        runoff_clean = runoff_data[mask]
        
        total_precip = np.sum(precip_clean)
        if total_precip <= 1e-6:
            return np.nan
        
        total_runoff = np.sum(runoff_clean)
        runoff_coeff = total_runoff / total_precip
        runoff_coeff = np.clip(runoff_coeff, 0, 2)
        return round(runoff_coeff, 4)
    except Exception as e:
        print(f"Failed to calculate runoff coefficient: {e}")
        return np.nan

def calculate_peak_deviation(obs, pred):
    """Calculate peak flow deviation (%)"""
    mask = ~np.isnan(obs) & ~np.isnan(pred)
    if not np.any(mask):
        return np.nan
    obs_clean = obs[mask]
    pred_clean = pred[mask]
    
    obs_peak = np.max(obs_clean)
    if obs_peak <= 1e-6:
        return np.nan
    pred_peak = np.max(pred_clean)
    
    peak_deviation = (pred_peak - obs_peak) / obs_peak * 100
    return round(peak_deviation, 2)

def calculate_volume_deviation(obs, pred):
    """Calculate runoff volume deviation (%)"""
    mask = ~np.isnan(obs) & ~np.isnan(pred)
    if not np.any(mask):
        return np.nan
    obs_clean = obs[mask]
    pred_clean = pred[mask]
    
    obs_volume = np.sum(obs_clean)
    if obs_volume <= 1e-6:
        return np.nan
    pred_volume = np.sum(pred_clean)
    
    volume_deviation = (pred_volume - obs_volume) / obs_volume * 100
    return round(volume_deviation, 2)

# ===================== 绘图函数（显示所有指标，全英文）=====================
def plot_flood_hydrograph(gage_id, flood_id, precip, streamflow_obs, streamflow_pred, 
                          start_time, end_time, time_period, nse_value, 
                          obs_runoff_coeff, pred_runoff_coeff, peak_deviation, volume_deviation):
    """Plot flood hydrograph with all metrics (English only)"""
    if time_period == "train":
        save_dir = train_plot_dir
    else:
        save_dir = valid_plot_dir
    
    basin_save_dir = os.path.join(save_dir, gage_id)
    ensure_dir(basin_save_dir)
    
    start_time_str = start_time.strftime("%Y%m%d%H%M%S")
    save_file = os.path.join(basin_save_dir, f"{gage_id}_{start_time_str}_flood{flood_id}_rainflow.png")
    
    # Create figure
    fig, ax1 = plt.subplots(figsize=plot_config["figsize"])
    
    # Plot streamflow curves
    ax1.plot(streamflow_obs.index, streamflow_obs.values, **streamflow_style["obs"])
    ax1.plot(streamflow_pred.index, streamflow_pred.values, **streamflow_style["pred"])
    ax1.set_ylabel(streamflow_style["yaxis_label"], fontsize=plot_config["fontsize_label"], fontweight="bold")
    ax1.set_xlabel(plot_config["xlabel"], fontsize=plot_config["fontsize_label"], fontweight="bold")
    ax1.set_ylim(bottom=0)
    ax1.grid(alpha=plot_config["grid_alpha"], linestyle=plot_config["grid_linestyle"])
    
    # Plot precipitation bar chart
    ax2 = ax1.twinx()
    ax2.bar(precip.index, precip.values, **{k: v for k, v in precip_style.items() if k not in ['yaxis_label', 'yaxis_rotation', 'yaxis_labelpad']})
    max_precip = precip.max() if precip.max() > 0 else 1
    ax2.set_ylim(0, max_precip * 3)
    ax2.invert_yaxis()
    ax2.set_ylabel(
        precip_style["yaxis_label"],
        fontsize=plot_config["fontsize_label"],
        fontweight="bold",
        rotation=precip_style["yaxis_rotation"],
        labelpad=precip_style["yaxis_labelpad"]
    )
    
    # Time axis configuration
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=plot_config["day_interval"]))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter(plot_config["time_format_short"]))
    ax1.xaxis.set_minor_locator(mdates.HourLocator(interval=plot_config["hour_interval"]))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=9)
    plt.setp(ax1.xaxis.get_minorticklabels(), rotation=0, ha="right", fontsize=8)
    
    # Legend and title (English only)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=plot_config["fontsize_legend"])
    plt.title(
        f"Basin {gage_id} - Flood Event {flood_id}\n({start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')})",
        fontsize=plot_config["fontsize_title"],
        fontweight="bold",
        pad=20
    )
    
    # ========== 标注所有指标（全英文） ==========
    # Format metric displays
    nse_display = "NSE: NaN" if np.isnan(nse_value) else f"NSE: {nse_value:.4f}"
    peak_display = "Peak Dev: NaN" if np.isnan(peak_deviation) else f"Peak Dev: {peak_deviation:.2f}%"
    vol_display = "Vol Dev: NaN" if np.isnan(volume_deviation) else f"Vol Dev: {volume_deviation:.2f}%"
    obs_runoff_display = "Obs Runoff Coeff: NaN" if np.isnan(obs_runoff_coeff) else f"Obs Runoff Coeff: {obs_runoff_coeff:.4f}"
    pred_runoff_display = "Pred Runoff Coeff: NaN" if np.isnan(pred_runoff_coeff) else f"Pred Runoff Coeff: {pred_runoff_coeff:.4f}"
    
    # Combine annotation text (5 lines, top-right corner)
    annotation_text = f"{nse_display}\n{peak_display}\n{vol_display}\n{obs_runoff_display}\n{pred_runoff_display}"
    ax1.text(
        x=0.98, y=0.95,
        s=annotation_text,
        transform=ax1.transAxes,
        fontsize=plot_config["metric_fontsize"],
        fontweight="bold",
        ha="right",
        va="top",
        bbox=dict(
            boxstyle="round,pad=0.6",  # 增大padding容纳更多文字
            facecolor="white",
            alpha=plot_config["metric_bg_alpha"],
            edgecolor="gray"
        )
    )
    
    # Save figure
    plt.tight_layout()
    plt.savefig(save_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Figure saved: {save_file}")
    return save_file

# ===================== 统计函数（场次级+流域级） ======================
def calculate_basin_metrics(event_metrics_df):
    """
    Calculate basin-level metrics from event-level data
    - Group by basin_id, compute statistics for all metrics
    """
    # Define statistics to calculate
    stats_funcs = {
        "NSE": ["count", "mean", "median", "min", "max"],
        "Obs_Runoff_Coeff": ["count", "mean", "median", "min", "max"],
        "Pred_Runoff_Coeff": ["count", "mean", "median", "min", "max"],
        "Peak_Deviation": ["count", "mean", "median", "min", "max"],
        "Volume_Deviation": ["count", "mean", "median", "min", "max"]
    }
    
    # Rename columns for consistent grouping
    rename_cols = {}
    for col in event_metrics_df.columns:
        if "NSE_" in col:
            rename_cols[col] = "NSE"
        elif "Obs_Runoff_Coeff_" in col:
            rename_cols[col] = "Obs_Runoff_Coeff"
        elif "Pred_Runoff_Coeff_" in col:
            rename_cols[col] = "Pred_Runoff_Coeff"
        elif "Peak_Deviation_" in col:
            rename_cols[col] = "Peak_Deviation"
        elif "Volume_Deviation_" in col:
            rename_cols[col] = "Volume_Deviation"
    
    df_temp = event_metrics_df.rename(columns=rename_cols)
    
    # Calculate basin-level statistics
    basin_metrics = df_temp.groupby("basin_id").agg({
        "NSE": stats_funcs["NSE"],
        "Obs_Runoff_Coeff": stats_funcs["Obs_Runoff_Coeff"],
        "Pred_Runoff_Coeff": stats_funcs["Pred_Runoff_Coeff"],
        "Peak_Deviation": stats_funcs["Peak_Deviation"],
        "Volume_Deviation": stats_funcs["Volume_Deviation"]
    })
    
    # Flatten multi-level columns
    basin_metrics.columns = ['_'.join(col).strip() for col in basin_metrics.columns.values]
    basin_metrics = basin_metrics.reset_index()
    
    # Rename columns to add period suffix (if needed)
    period_suffix = "_TRAIN" if "NSE_TRAIN" in event_metrics_df.columns else "_VALID"
    rename_final = {}
    for col in basin_metrics.columns:
        if col != "basin_id" and not col.endswith(period_suffix):
            rename_final[col] = f"{col}{period_suffix}"
    basin_metrics = basin_metrics.rename(columns=rename_final)
    
    return basin_metrics

# ===================== 主处理函数 ======================
def process_flood_events(time_period, flood_events, nc_obs_path, nc_pred_path):
    """Process flood events: read data + calculate metrics + plot + save results (event + basin level)"""
    event_metrics_results = []
    success_plots = 0
    
    # Load NC files
    ds_obs, ds_pred = load_streamflow_dataset(nc_obs_path, nc_pred_path)
    
    # Iterate over basins
    for basin_id, events in flood_events.items():
        print(f"\n===== Processing Basin {basin_id} ({time_period}) ======")
        try:
            # Read basin precipitation data
            precip_series = read_basin_data(basin_id, time_period)
            
            # Read streamflow data by basin name
            streamflow_obs, streamflow_pred = read_streamflow_by_basin_name(
                ds_obs, ds_pred, basin_id, time_period
            )
            
            if len(streamflow_obs) == 0 or len(streamflow_pred) == 0:
                print(f"No valid streamflow data, skip")
                continue
            
            # Iterate over flood events
            for flood_id, start_time, end_time in events:
                print(f"\nProcessing Flood Event {flood_id}: {start_time} to {end_time}")
                # Slice event data
                precip_slice, obs_slice, pred_slice = slice_data_by_flood(
                    precip_series, streamflow_obs, streamflow_pred,
                    start_time, end_time, basin_id, flood_id
                )
                
                if precip_slice is None:
                    continue
                
                # Calculate all metrics
                nse_value = calculate_nse(obs_slice.values, pred_slice.values)
                obs_runoff_coeff = calculate_runoff_coefficient(precip_slice.values, obs_slice.values)
                pred_runoff_coeff = calculate_runoff_coefficient(precip_slice.values, pred_slice.values)
                peak_deviation = calculate_peak_deviation(obs_slice.values, pred_slice.values)
                volume_deviation = calculate_volume_deviation(obs_slice.values, pred_slice.values)
                
                print(f"Flood Event {flood_id} - NSE: {nse_value} | Obs Runoff Coeff: {obs_runoff_coeff} | Pred Runoff Coeff: {pred_runoff_coeff} | Peak Dev: {peak_deviation}% | Vol Dev: {volume_deviation}%")
                
                # Plot (show all metrics)
                plot_flood_hydrograph(
                    basin_id, flood_id, precip_slice, obs_slice, pred_slice,
                    start_time, end_time, time_period, nse_value,
                    obs_runoff_coeff, pred_runoff_coeff, peak_deviation, volume_deviation
                )
                success_plots += 1
                
                # Save event-level metrics
                event_metrics_results.append({
                    "basin_id": basin_id,
                    "flood_event_id": flood_id,
                    "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
                    f"NSE_{time_period.upper()}": nse_value,
                    f"Obs_Runoff_Coeff_{time_period.upper()}": obs_runoff_coeff,
                    f"Pred_Runoff_Coeff_{time_period.upper()}": pred_runoff_coeff,
                    f"Peak_Deviation_{time_period.upper()}": peak_deviation,
                    f"Volume_Deviation_{time_period.upper()}": volume_deviation
                })
        
        except Exception as e:
            print(f"Failed to process Basin {basin_id}: {e}")
            continue
    
    # Close NC files
    ds_obs.close()
    ds_pred.close()
    
    # Convert event results to DataFrame
    event_metrics_df = pd.DataFrame(event_metrics_results)
    
    # Calculate basin-level metrics
    basin_metrics_df = calculate_basin_metrics(event_metrics_df)
    
    # Save results to CSV
    ensure_dir(metrics_save_dir)
    
    # 1. Event-level CSV
    event_save_path = os.path.join(metrics_save_dir, f"{time_period}_event_level_metrics.csv")
    event_metrics_df.to_csv(event_save_path, index=False, encoding="utf-8")
    print(f"\n{time_period} period event-level metrics saved to: {event_save_path}")
    
    # 2. Basin-level CSV
    basin_save_path = os.path.join(metrics_save_dir, f"{time_period}_basin_level_metrics.csv")
    basin_metrics_df.to_csv(basin_save_path, index=False, encoding="utf-8")
    print(f"{time_period} period basin-level metrics saved to: {basin_save_path}")
    
    return success_plots, len(event_metrics_results)

# ===================== 主函数 ======================
def main():
    # 1. Initialize directories
    ensure_dir(train_plot_dir)
    ensure_dir(valid_plot_dir)
    ensure_dir(metrics_save_dir)
    
    # 2. Read flood event division
    train_flood_events, valid_flood_events = read_flood_events_from_excel()
    
    # 3. Process training period
    print("\n========== Processing Training Period ==========")
    train_plots, train_event_count = process_flood_events(
        "train", train_flood_events, train_nc_obs_path, train_nc_pred_path
    )
    
    # 4. Process validation period
    print("\n========== Processing Validation Period ==========")
    valid_plots, valid_event_count = process_flood_events(
        "valid", valid_flood_events, valid_nc_obs_path, valid_nc_pred_path
    )
    
    # 5. Print summary
    print("\n===== Batch Processing Completed =====")
    print(f"Training Period: {train_plots} successful plots, {train_event_count} event-level records")
    print(f"Validation Period: {valid_plots} successful plots, {valid_event_count} event-level records")
    print(f"Result Paths:")
    print(f"- Training Plots: {train_plot_dir}")
    print(f"- Validation Plots: {valid_plot_dir}")
    print(f"- Metrics Results (Event + Basin Level): {metrics_save_dir}")

if __name__ == "__main__":
    main()