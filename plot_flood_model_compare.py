"""
三模型洪水事件对比图：LSTM-only vs LSTM+GNN vs WDNE
分别绘制训练集和测试集的洪水事件
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from pathlib import Path
import os, warnings
warnings.filterwarnings('ignore')

# ---- 配置 ----
RESULTS = {
    "Basin LSTM": "/home/lizilin/code/torchhydro/results/basin_lstm/test",
    "LSTM-only":  "/home/lizilin/code/torchhydro/results/baseline_lstm_only/songliao_3h_normal",
    "LSTM+GNN":   "/home/lizilin/code/torchhydro/results/baseline_lstm_gnn/songliao_3h_normal",
    "WDNE":       "/home/lizilin/code/torchhydro/results/wdne_adaptive/songliao_3h_normal",
}
RAIN_DIR = "/mnt/data/ClassC/songliaorrevent/timeseries/3h"
ATTR_FILE = "/mnt/data/ClassC/songliaorrevent/attributes/attributes.csv"
OUTPUT_ROOT = Path("/home/lizilin/code/torchhydro/results/comparison_plots/three_models")

MODEL_COLORS = {"Basin LSTM": "#2ca02c", "LSTM-only": "green", "LSTM+GNN": "steelblue", "WDNE": "darkorange"}
MODEL_STYLES = {"Basin LSTM": ":", "LSTM-only": "-.", "LSTM+GNN": "--", "WDNE": "-"}

TRAIN_END = pd.Timestamp("2014-12-31 23:00:00")
TEST_START = pd.Timestamp("2019-01-01 02:00:00")

BASIN_IDS = [
    "songliao_20800900", "songliao_20810200", "songliao_21100150",
    "songliao_21110150", "songliao_21401050", "songliao_21401550"
]

# 流域面积
_attr_df = pd.read_csv(ATTR_FILE)
_attr_df["basin_id_full"] = "songliao_" + _attr_df["basin_id"].astype(str)
BASIN_AREA = dict(zip(_attr_df["basin_id_full"], _attr_df["area"]))

def mm3h_to_m3s(flow, area):
    return flow * area / 10.8


def load_data(result_dir, basin_id):
    """加载观测+预测流量 + 时间"""
    obs_nc = os.path.join(result_dir, "epochbest_model.pthflow_obs.nc")
    pred_nc = os.path.join(result_dir, "epochbest_model.pthflow_pred.nc")

    ds_obs = xr.open_dataset(obs_nc)
    obs_flow = ds_obs.sel(basin=basin_id)["inflow"].values
    time = pd.to_datetime(ds_obs.sel(basin=basin_id)["inflow"].time.values)
    flood = ds_obs.sel(basin=basin_id)["flood_event"].values if "flood_event" in ds_obs else None
    ds_obs.close()

    ds_pred = xr.open_dataset(pred_nc)
    pred_flow = ds_pred.sel(basin=basin_id)["inflow"].values
    ds_pred.close()

    return obs_flow, pred_flow, time, flood


def load_rainfall(basin_id, time_array):
    csv_path = os.path.join(RAIN_DIR, f"{basin_id}.csv")
    if not os.path.exists(csv_path):
        return np.zeros(len(time_array))
    df = pd.read_csv(csv_path, parse_dates=["time"]).set_index("time")
    if "rain" not in df.columns:
        return np.zeros(len(time_array))
    return df["rain"].reindex(time_array).fillna(0).values


def identify_flood_events(flood_array):
    """识别洪水事件"""
    events = []
    in_event, start = False, None
    for i, v in enumerate(flood_array):
        is_flood = not np.isnan(v) and v > 0
        if is_flood and not in_event:
            in_event, start = True, i
        elif not is_flood and in_event:
            in_event = False
            events.append((start, i - 1))
    if in_event:
        events.append((start, len(flood_array) - 1))
    # 合并间隔≤6的事件
    merged = []
    for s, e in events:
        if merged and s - merged[-1][1] <= 6:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged


def nse(o, p):
    mask = ~(np.isnan(o) | np.isnan(p))
    if mask.sum() < 2: return np.nan
    o, p = o[mask], p[mask]
    return 1 - np.sum((o - p)**2) / np.sum((o - o.mean())**2)


def plot_three_model_event(basin_id, event_idx, start, end, all_models_data,
                           rain, time, output_dir, period_label):
    """绘制三个模型对比的单场洪水事件图"""
    ctx = 12
    s = max(0, start - ctx)
    e = min(len(time) - 1, end + ctx)

    t = time[s:e+1]
    r = rain[s:e+1]
    area = BASIN_AREA.get(basin_id, 1.0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), sharex=True,
                                    gridspec_kw={'height_ratios': [1, 3]})

    # 降雨
    ax1.bar(t, r, width=0.04, color='steelblue', alpha=0.5)
    ax1.set_ylabel('Precip (mm/3h)', fontsize=11)
    ax1.set_ylim(max(1, np.nanmax(r)*1.5), 0)
    ax1.axvspan(t[start-s], t[end-s], alpha=0.08, color='red')

    # 流量
    nse_results = {}
    for model_name, (obs_flow, pred_flow) in all_models_data.items():
        o = obs_flow[s:e+1]
        p = pred_flow[s:e+1]
        o_m3s = mm3h_to_m3s(o, area)
        p_m3s = mm3h_to_m3s(p, area)

        nse_val = nse(obs_flow[start:end+1], pred_flow[start:end+1])
        nse_results[model_name] = nse_val

        label = f"{model_name} (NSE={nse_val:.3f})"
        if model_name == "LSTM-only":
            ax2.plot(t, o_m3s, 'k-', linewidth=1.8, label='Observed')
            ax2.plot(t, p_m3s, color=MODEL_COLORS[model_name],
                     linestyle=MODEL_STYLES[model_name], linewidth=1.2, label=label)
        else:
            ax2.plot(t, p_m3s, color=MODEL_COLORS[model_name],
                     linestyle=MODEL_STYLES[model_name], linewidth=1.2, label=label)

    ax2.axvspan(t[start-s], t[end-s], alpha=0.08, color='red')
    ax2.set_ylabel('Inflow (m³/s)', fontsize=11)
    ax2.legend(fontsize=9, loc='upper left', ncol=2)
    ax2.grid(True, alpha=0.3)

    t0_str = pd.Timestamp(t[start-s]).strftime('%Y-%m-%d %H:%M')
    t1_str = pd.Timestamp(t[end-s]).strftime('%Y-%m-%d %H:%M')
    ax2.set_title(f'{basin_id} — {period_label} Event #{event_idx+1} ({t0_str} ~ {t1_str})',
                  fontsize=13, fontweight='bold')

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, f"event_{event_idx+1:03d}.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    return nse_results


def main(target_basin=None):
    basins = [target_basin] if target_basin else BASIN_IDS

    for bid in basins:
        print(f"\n{'='*70}")
        print(f"流域: {bid}")
        print(f"{'='*70}")

        # 加载观测（用于识别事件和时间）
        obs_nc = os.path.join(RESULTS["LSTM+GNN"], "epochbest_model.pthflow_obs.nc")
        _, _, time_all, flood_all = load_data(RESULTS["LSTM+GNN"], bid)

        # 加载各模型预测
        all_models = {}
        for name, d in RESULTS.items():
            obs_f, pred_f, _, _ = load_data(d, bid)
            all_models[name] = (obs_f, pred_f)

        rain = load_rainfall(bid, time_all)
        events = identify_flood_events(flood_all if flood_all is not None else np.zeros_like(time_all))
        print(f"  共 {len(events)} 场洪水事件")

        # 分训练集/测试集
        train_events = [(i, s, e) for i, (s, e) in enumerate(events) if time_all[e] <= TRAIN_END]
        test_events = [(i, s, e) for i, (s, e) in enumerate(events) if time_all[s] >= TEST_START]

        print(f"  训练集: {len(train_events)} 场, 测试集: {len(test_events)} 场")

        for period_name, evt_list in [("train", train_events), ("test", test_events)]:
            if not evt_list:
                continue
            out_dir = OUTPUT_ROOT / bid.replace("songliao_", "") / period_name
            metrics = []

            for i, s, e in evt_list:
                nse_r = plot_three_model_event(bid, i, s, e, all_models, rain, time_all,
                                               out_dir, period_name.upper())
                metrics.append({"event": i+1, "duration": e-s+1, **nse_r})

            # 汇总
            df = pd.DataFrame(metrics)
            df.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
            if len(df) > 0:
                print(f"  [{period_name.upper()}] {len(df)} 场:")
                for m in ["LSTM-only", "LSTM+GNN", "WDNE"]:
                    valid = df[m].dropna()
                    print(f"    {m}: mean NSE={valid.mean():.3f}, median={valid.median():.3f}")

    print(f"\n图片保存在: {OUTPUT_ROOT}")


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else None
    main(target)
