"""
按洪水事件绘制 Baseline vs WDNE 对比图（去归一化，带上降雨）

基于 plot_flood_events_by_year.py 的逻辑，但：
- 使用 nc 文件中的去归一化数据
- 同时对比 Baseline 和 WDNE 两个模型
- 叠加降雨柱状图
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from pathlib import Path
import os, warnings
warnings.filterwarnings('ignore')

# ---- 配置 ----
BASELINE_DIR = Path("/home/lizilin/code/torchhydro/results/baseline_lstm_gnn/songliao_3h_normal")
WDNE_DIR = Path("/home/lizilin/code/torchhydro/results/wdne_adaptive/songliao_3h_normal")
RAIN_DIR = "/mnt/data/ClassC/songliaorrevent/timeseries/3h"
ATTR_FILE = "/mnt/data/ClassC/songliaorrevent/attributes/attributes.csv"
OUTPUT_ROOT = Path("/home/lizilin/code/torchhydro/results/comparison_plots/flood_events")

# 预加载流域面积 (km²)
import pandas as pd
_attr_df = pd.read_csv(ATTR_FILE)
_attr_df["basin_id_full"] = "songliao_" + _attr_df["basin_id"].astype(str)
BASIN_AREA = dict(zip(_attr_df["basin_id"].astype(str), _attr_df["area"]))
print(f"已加载 {len(BASIN_AREA)} 个流域的面积数据")

def mm3h_to_m3s(flow_mm3h, area_km2):
    """mm/3h → m³/s:  flow * area / (3*3.6)"""
    return flow_mm3h * area_km2 / 10.8

BASIN_IDS = [
    "songliao_20800900", "songliao_20810200", "songliao_21100150",
    "songliao_21110150", "songliao_21401050", "songliao_21401550"
]

# ---- 数据加载 ----
def load_nc_data(nc_path, basin_id, var="inflow"):
    """读 nc 文件，返回 (去归一化流量数组, 时间datetime数组, flood_event数组)"""
    ds = xr.open_dataset(nc_path)
    da_inflow = ds.sel(basin=basin_id)["inflow"]
    flow = da_inflow.values
    time = pd.to_datetime(da_inflow.time.values)

    flood = None
    if "flood_event" in ds.data_vars:
        flood = ds.sel(basin=basin_id)["flood_event"].values

    ds.close()
    return flow, time, flood


def load_rainfall(basin_id, time_array):
    """读 CSV 降雨数据，对齐到时间"""
    csv_path = os.path.join(RAIN_DIR, f"{basin_id}.csv")
    if not os.path.exists(csv_path):
        return np.zeros(len(time_array))
    df = pd.read_csv(csv_path, parse_dates=["time"])
    df = df.set_index("time")
    if "rain" not in df.columns:
        return np.zeros(len(time_array))
    rain = df["rain"].reindex(time_array).fillna(0).values
    return rain


def identify_flood_events(flood_array, time_array, min_gap=6):
    """识别洪水事件（连续的 flood_event>0），min_gap 是合并间隔"""
    events = []
    in_event = False
    start = None
    for i, v in enumerate(flood_array):
        is_flood = not np.isnan(v) and v > 0
        if is_flood and not in_event:
            in_event = True
            start = i
        elif not is_flood and in_event:
            in_event = False
            events.append((start, i - 1))
    if in_event:
        events.append((start, len(flood_array) - 1))

    # 合并间隔小于 min_gap 的相邻事件
    merged = []
    for s, e in events:
        if merged and s - merged[-1][1] <= min_gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged


def nse(obs, pred):
    mask = ~(np.isnan(obs) | np.isnan(pred))
    if mask.sum() < 2:
        return np.nan
    o, p = obs[mask], pred[mask]
    return 1 - np.sum((o - p)**2) / np.sum((o - o.mean())**2)


def plot_event(basin_id, event_idx, start, end, obs_flow, bl_flow, wd_flow,
               rain, time, output_dir):
    """绘制单场洪水事件的降雨+观测+双模型对比图"""
    # 扩展窗口：前后各加一些点看上下文
    ctx = 12
    s = max(0, start - ctx)
    e = min(len(obs_flow) - 1, end + ctx)

    t = time[s:e+1]
    o = obs_flow[s:e+1]
    b = bl_flow[s:e+1]
    w = wd_flow[s:e+1]
    r = rain[s:e+1]

    # 计算洪水段内的 NSE
    bl_nse = nse(obs_flow[start:end+1], bl_flow[start:end+1])
    wd_nse = nse(obs_flow[start:end+1], wd_flow[start:end+1])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), sharex=True,
                                    gridspec_kw={'height_ratios': [1, 3]})

    # 上：降雨柱状图
    ax1.bar(t, r, width=0.04, color='steelblue', alpha=0.6)
    ax1.set_ylabel('Precip (mm/3h)', fontsize=11)
    ax1.set_ylim(max(1, np.nanmax(r)*1.5), 0)  # 倒置
    ax1.axvspan(t[start-s], t[end-s], alpha=0.1, color='red')  # 洪水段标记

    # 单位换算 mm/3h → m³/s
    area = BASIN_AREA.get(basin_id, 1.0)
    o_m3s = mm3h_to_m3s(o, area)
    b_m3s = mm3h_to_m3s(b, area)
    w_m3s = mm3h_to_m3s(w, area)

    # 下：流量 (m³/s)
    ax2.plot(t, o_m3s, 'k-', linewidth=1.8, label='Observed')
    ax2.plot(t, b_m3s, color='steelblue', linestyle='--', linewidth=1.3,
             label=f'Baseline LSTM-GNN (NSE={bl_nse:.3f})')
    ax2.plot(t, w_m3s, color='darkorange', linestyle='-.', linewidth=1.3,
             label=f'WDNE Adaptive (NSE={wd_nse:.3f})')
    ax2.axvspan(t[start-s], t[end-s], alpha=0.1, color='red')
    ax2.set_ylabel('Inflow (m³/s)', fontsize=11)
    ax2.legend(fontsize=10, loc='upper left')
    ax2.grid(True, alpha=0.3)

    t0_str = pd.Timestamp(t[start-s]).strftime('%Y-%m-%d %H:%M')
    t1_str = pd.Timestamp(t[end-s]).strftime('%Y-%m-%d %H:%M')
    ax2.set_title(
        f'{basin_id} — Flood Event #{event_idx+1} ({t0_str} ~ {t1_str})',
        fontsize=13, fontweight='bold')

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"event_{event_idx+1:03d}.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return bl_nse, wd_nse


def main(target_basin=None):
    """主函数：target_basin 为 None 则处理所有流域"""
    basins = [target_basin] if target_basin else BASIN_IDS

    obs_nc = BASELINE_DIR / "epochbest_model.pthflow_obs.nc"
    baseline_pred_nc = BASELINE_DIR / "epochbest_model.pthflow_pred.nc"
    wdne_pred_nc = WDNE_DIR / "epochbest_model.pthflow_pred.nc"

    for bid in basins:
        print(f"\n{'='*60}")
        print(f"处理流域: {bid}")
        print(f"{'='*60}")

        # 加载数据
        obs_flow, time_obs, flood_obs = load_nc_data(obs_nc, bid)
        bl_flow, time_bl, _ = load_nc_data(baseline_pred_nc, bid)
        wd_flow, time_wd, _ = load_nc_data(wdne_pred_nc, bid)
        rain = load_rainfall(bid, time_obs)

        # 用观测的 flood_event 识别事件
        flood = flood_obs if flood_obs is not None else np.zeros_like(obs_flow)

        events = identify_flood_events(flood, time_obs)
        print(f"  识别到 {len(events)} 场洪水事件")

        if len(events) == 0:
            continue

        output_dir = OUTPUT_ROOT / bid.replace("songliao_", "")
        all_metrics = []

        for i, (s, e) in enumerate(events):
            bl_nse, wd_nse = plot_event(
                bid, i, s, e, obs_flow, bl_flow, wd_flow,
                rain, time_obs, output_dir)
            all_metrics.append({
                'event': i+1, 'start': s, 'end': e,
                'duration': e-s+1,
                'baseline_nse': bl_nse, 'wdne_nse': wd_nse
            })
            if i < 5 or i >= len(events) - 2:  # 只打印前5和后2
                print(f"  Event #{i+1}: dur={e-s+1}, "
                      f"BL_NSE={bl_nse:.3f}, WD_NSE={wd_nse:.3f}")

        # 汇总
        df = pd.DataFrame(all_metrics)
        df.to_csv(os.path.join(output_dir, "events_summary.csv"), index=False)
        valid = df.dropna(subset=['baseline_nse', 'wdne_nse'])
        if len(valid) > 0:
            print(f"\n  流域 {bid} 汇总 ({len(valid)} 场有效):")
            print(f"  Baseline 平均 NSE: {valid['baseline_nse'].mean():.3f}")
            print(f"  WDNE    平均 NSE: {valid['wdne_nse'].mean():.3f}")

    print(f"\n所有图片保存在: {OUTPUT_ROOT}")


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else None
    main(target)
