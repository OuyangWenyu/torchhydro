"""
四模型洪水事件对比 — 按时间对齐，正确版
"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt, xarray as xr, os
from pathlib import Path
import warnings; warnings.filterwarnings('ignore')

MODELS_KEYS = ["Basin LSTM","LSTM-only","LSTM+GNN","WDNE"]
MODEL_PATHS = {
    "Basin LSTM": ("/home/lizilin/code/torchhydro/results/basin_lstm/test/epochbest_model.pthflow_obs.nc",
                   "/home/lizilin/code/torchhydro/results/basin_lstm/test/epochbest_model.pthflow_pred.nc"),
    "LSTM-only":  ("/home/lizilin/code/torchhydro/results/baseline_lstm_only/songliao_3h_normal/epochbest_model.pthflow_obs.nc",
                   "/home/lizilin/code/torchhydro/results/baseline_lstm_only/songliao_3h_normal/epochbest_model.pthflow_pred.nc"),
    "LSTM+GNN":   ("/home/lizilin/code/torchhydro/results/baseline_lstm_gnn/songliao_3h_normal/epochbest_model.pthflow_obs.nc",
                   "/home/lizilin/code/torchhydro/results/baseline_lstm_gnn/songliao_3h_normal/epochbest_model.pthflow_pred.nc"),
    "WDNE":       ("/home/lizilin/code/torchhydro/results/wdne_adaptive/songliao_3h_normal/epochbest_model.pthflow_obs.nc",
                   "/home/lizilin/code/torchhydro/results/wdne_adaptive/songliao_3h_normal/epochbest_model.pthflow_pred.nc"),
}
COLORS = {"Basin LSTM":"#2ca02c","LSTM-only":"green","LSTM+GNN":"steelblue","WDNE":"darkorange"}
STYLES = {"Basin LSTM":":","LSTM-only":"-.","LSTM+GNN":"--","WDNE":"-"}

RAIN_DIR = "/mnt/data/ClassC/songliaorrevent/timeseries/3h"
ATTR_FILE = "/mnt/data/ClassC/songliaorrevent/attributes/attributes.csv"
OUT = Path("/home/lizilin/code/torchhydro/results/comparison_plots/with_static")

_attr = pd.read_csv(ATTR_FILE)
AREA = dict(zip(_attr["basin_id"].astype(str), _attr["area"]))

BASINS = ["songliao_20800900","songliao_20810200","songliao_21100150",
          "songliao_21110150","songliao_21401050","songliao_21401550"]

def mm3h_to_m3s(f, a): return f * a / 10.8

def nse(o, p):
    mask = ~np.isnan(o) & ~np.isnan(p)
    if mask.sum()<2: return np.nan
    o,p=o[mask],p[mask]
    return 1-np.sum((o-p)**2)/np.sum((o-o.mean())**2)

def load_series(nc, bid):
    ds = xr.open_dataset(nc)
    s = pd.Series(ds.sel(basin=bid)["inflow"].values,
                  index=pd.to_datetime(ds.sel(basin=bid)["inflow"].time.values))
    ds.close()
    return s

for bid in BASINS:
    print(f"\n{'='*60}\n{bid}")

    # 加载各模型
    all_obs, all_pred = {}, {}
    for name in MODELS_KEYS:
        obs_nc, pred_nc = MODEL_PATHS[name]
        all_obs[name] = load_series(obs_nc, bid)
        all_pred[name] = load_series(pred_nc, bid)

    # 识别洪水事件
    ds_f = xr.open_dataset(MODEL_PATHS["LSTM+GNN"][0])
    flood = ds_f.sel(basin=bid)["flood_event"].values
    t_ref = pd.to_datetime(ds_f.sel(basin=bid)["inflow"].time.values)
    ds_f.close()

    events_t = []
    in_ev, start_i = False, None
    for i, v in enumerate(flood):
        if not np.isnan(v) and v > 0 and not in_ev: in_ev, start_i = True, i
        elif (np.isnan(v) or v == 0) and in_ev: in_ev = False; events_t.append((pd.Timestamp(t_ref[start_i]), pd.Timestamp(t_ref[i-1])))
    if in_ev: events_t.append((pd.Timestamp(t_ref[start_i]), pd.Timestamp(t_ref[-1])))
    # 合并间隔≤36h
    merged = []
    for s, e in events_t:
        if merged and (s - merged[-1][1]) <= pd.Timedelta(hours=36): merged[-1] = (merged[-1][0], e)
        else: merged.append((s, e))
    events_t = merged

    # 降雨
    rain_s = pd.Series(0.0, index=t_ref)
    csv_p = os.path.join(RAIN_DIR, f"{bid}.csv")
    if os.path.exists(csv_p):
        df = pd.read_csv(csv_p, parse_dates=["time"]).set_index("time")
        if "rain" in df.columns:
            rain_s = df["rain"].reindex(t_ref).fillna(0)

    area = AREA.get(bid, 1.0)

    for ev_idx, (t_start, t_end) in enumerate(events_t):
        ctx = pd.Timedelta(hours=36)
        t0, t1 = t_start - ctx, t_end + ctx

        ref_slice = all_obs["LSTM+GNN"][t0:t1]
        if len(ref_slice) == 0: continue
        t_plot = ref_slice.index.values

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), sharex=True,
                                        gridspec_kw={'height_ratios':[1,3]})

        r = rain_s[t0:t1].reindex(ref_slice.index).fillna(0).values
        ax1.bar(t_plot, r, width=0.04, color='steelblue', alpha=0.5)
        ax1.set_ylabel('Precip (mm/3h)', fontsize=11)
        ymax = max(1, np.nanmax(r)*1.5) if len(r)>0 and not np.all(np.isnan(r)) else 1
        ax1.set_ylim(ymax, 0)

        nses = {}
        for name in MODELS_KEYS:
            o_slice = all_obs[name][t0:t1].reindex(ref_slice.index)
            p_slice = all_pred[name][t0:t1].reindex(ref_slice.index)
            o_m3s = mm3h_to_m3s(o_slice.values, area)
            p_m3s = mm3h_to_m3s(p_slice.values, area)

            ns = nse(all_obs[name][t_start:t_end].values, all_pred[name][t_start:t_end].values)
            nses[name] = ns

            if name == "Basin LSTM":
                ax2.plot(t_plot, o_m3s, 'k-', linewidth=2.0, label='Observed', zorder=10)
            ax2.plot(t_plot, p_m3s, color=COLORS[name], linestyle=STYLES[name], linewidth=1.2,
                     label=f'{name} (NSE={ns:.3f})')

        ax2.axvspan(t_start, t_end, alpha=0.08, color='red')
        ax2.set_ylabel('Inflow (m³/s)', fontsize=11)
        ax2.legend(fontsize=8, ncol=2, loc='upper left')
        ax2.grid(True, alpha=0.3)
        ax2.set_title(f'{bid} Event #{ev_idx+1} ({t_start.strftime("%Y-%m-%d %H:%M")} ~ {t_end.strftime("%Y-%m-%d %H:%M")})', fontsize=13, fontweight='bold')
        fig.tight_layout()
        out_d = OUT / bid.replace("songliao_","")
        os.makedirs(out_d, exist_ok=True)
        fig.savefig(out_d / f"event_{ev_idx+1:03d}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

    # 汇总
    print(f"  {len(events_t)} events:")
    for name in MODELS_KEYS:
        ns_list = [nse(all_obs[name][s:e].values, all_pred[name][s:e].values) for s,e in events_t]
        valid = [v for v in ns_list if not np.isnan(v)]
        if valid: print(f"    {name:<12}: mean={np.mean(valid):.3f}  med={np.median(valid):.3f}")

print(f"\nDone: {OUT}")
