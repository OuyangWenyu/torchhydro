"""
四模型洪水事件对比: Basin LSTM / LSTM-only / LSTM+GNN / WDNE
"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt, xarray as xr, os
from pathlib import Path
import warnings; warnings.filterwarnings('ignore')

RESULTS = {
    "Basin LSTM": "/home/lizilin/code/torchhydro/results/basin_lstm/test",
    "LSTM-only":  "/home/lizilin/code/torchhydro/results/baseline_lstm_only/songliao_3h_normal",
    "LSTM+GNN":   "/home/lizilin/code/torchhydro/results/baseline_lstm_gnn/songliao_3h_normal",
    "WDNE":       "/home/lizilin/code/torchhydro/results/wdne_adaptive/songliao_3h_normal",
}
COLORS = {"Basin LSTM":"#2ca02c","LSTM-only":"green","LSTM+GNN":"steelblue","WDNE":"darkorange"}
STYLES = {"Basin LSTM":":","LSTM-only":"-.","LSTM+GNN":"--","WDNE":"-"}

RAIN_DIR = "/mnt/data/ClassC/songliaorrevent/timeseries/3h"
ATTR_FILE = "/mnt/data/ClassC/songliaorrevent/attributes/attributes.csv"
OUT = Path("/home/lizilin/code/torchhydro/results/comparison_plots/four_models")

_attr = pd.read_csv(ATTR_FILE)
# basin_id 已有 songliao_ 前缀，不需要再加
AREA = dict(zip(_attr["basin_id"].astype(str), _attr["area"]))

def mm3h_to_m3s(f, a): return f * a / 10.8

def load_by_time(nc, basin_id, time_target):
    """按目标时间数组提取数据"""
    ds = xr.open_dataset(nc)
    da = ds.sel(basin=basin_id)["inflow"]
    # 对齐到目标时间
    da_aligned = da.reindex(time=time_target, method="nearest")
    vals = da_aligned.values
    ds.close()
    return vals

def nse(o, p):
    mask = ~(np.isnan(o) & np.isnan(p))
    if mask.sum()<2: return np.nan
    o,p=o[mask],p[mask]
    return 1-np.sum((o-p)**2)/np.sum((o-o.mean())**2)

BASINS = ["songliao_20800900","songliao_20810200","songliao_21100150",
          "songliao_21110150","songliao_21401050","songliao_21401550"]

for bid in BASINS:
    print(f"\n{'='*60}")
    print(f"流域: {bid}")

    # 从 LSTM+GNN 获取时间轴和 flood_event
    obs_nc = os.path.join(RESULTS["LSTM+GNN"], "epochbest_model.pthflow_obs.nc")
    ds_obs = xr.open_dataset(obs_nc)
    time_all = pd.to_datetime(ds_obs.sel(basin=bid)["inflow"].time.values)
    flood = ds_obs.sel(basin=bid)["flood_event"].values
    ds_obs.close()

    # 加载各模型预测（按时间对齐）
    preds = {}
    for name, d in RESULTS.items():
        pred_nc = os.path.join(d, "epochbest_model.pthflow_pred.nc")
        preds[name] = load_by_time(pred_nc, bid, time_all)
    obs_flow = load_by_time(obs_nc, bid, time_all)

    # 识别洪水事件
    events = []
    in_ev, start = False, None
    for i, v in enumerate(flood):
        is_f = not np.isnan(v) and v > 0
        if is_f and not in_ev: in_ev, start = True, i
        elif not is_f and in_ev: in_ev = False; events.append((start, i-1))
    if in_ev: events.append((start, len(flood)-1))
    # 合并间隔≤6
    merged = []
    for s, e in events:
        if merged and s - merged[-1][1] <= 6: merged[-1] = (merged[-1][0], e)
        else: merged.append((s, e))
    events = merged
    print(f"  共 {len(events)} 场洪水事件")

    # 加载降雨
    rain = np.zeros(len(time_all))
    csv_p = os.path.join(RAIN_DIR, f"{bid}.csv")
    if os.path.exists(csv_p):
        df = pd.read_csv(csv_p, parse_dates=["time"]).set_index("time")
        if "rain" in df.columns:
            rain = df["rain"].reindex(time_all).fillna(0).values

    area = AREA.get(bid, 1.0)

    for ev_idx, (start, end) in enumerate(events):
        ctx = 12
        s = max(0, start - ctx)
        e = min(len(time_all)-1, end + ctx)

        t = time_all[s:e+1]
        r = rain[s:e+1]
        o = obs_flow[s:e+1]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), sharex=True,
                                        gridspec_kw={'height_ratios':[1,3]})
        ax1.bar(t, r, width=0.04, color='steelblue', alpha=0.5)
        ax1.set_ylabel('Precip (mm/3h)', fontsize=11)
        ax1.set_ylim(max(1, np.nanmax(r)*1.5), 0)

        o_m3s = mm3h_to_m3s(o, area)
        ax2.plot(t, o_m3s, 'k-', linewidth=2.0, label='Observed', zorder=10)

        nses = {}
        for name, p_all in preds.items():
            p = p_all[s:e+1]
            p_m3s = mm3h_to_m3s(p, area)
            ns = nse(obs_flow[start:end+1], p_all[start:end+1])
            nses[name] = ns
            ax2.plot(t, p_m3s, color=COLORS[name], linestyle=STYLES[name],
                     linewidth=1.2, label=f'{name} (NSE={ns:.3f})')

        ax2.axvspan(t[start-s], t[end-s], alpha=0.08, color='red')
        ax2.set_ylabel('Inflow (m³/s)', fontsize=11)
        ax2.legend(fontsize=8, ncol=2, loc='upper left')
        ax2.grid(True, alpha=0.3)

        t0 = pd.Timestamp(t[start-s]).strftime('%Y-%m-%d %H:%M')
        t1 = pd.Timestamp(t[end-s]).strftime('%Y-%m-%d %H:%M')
        ax2.set_title(f'{bid} — Flood Event #{ev_idx+1} ({t0} ~ {t1})',
                      fontsize=13, fontweight='bold')
        fig.tight_layout()

        out_d = OUT / bid.replace("songliao_","")
        os.makedirs(out_d, exist_ok=True)
        fig.savefig(out_d / f"event_{ev_idx+1:03d}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

    # 汇总
    all_nses = {m: [] for m in RESULTS}
    for ev_idx, (start, end) in enumerate(events):
        for name in RESULTS:
            all_nses[name].append(nse(obs_flow[start:end+1], preds[name][start:end+1]))

    print(f"  逐事件平均 NSE:")
    for name in RESULTS:
        vals = [v for v in all_nses[name] if not np.isnan(v)]
        if vals:
            print(f"    {name:<12}: mean={np.mean(vals):.3f}  median={np.median(vals):.3f}")

print(f"\n完成: {OUT}")
