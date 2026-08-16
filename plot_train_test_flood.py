"""
三模型对比 — 分训练期/测试期逐洪水事件画图
使用 eval_full 的全时段预测结果
"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt, xarray as xr, os
from pathlib import Path
import warnings; warnings.filterwarnings('ignore')

TRAIN_END = pd.Timestamp("2014-12-31 23:00:00")
TEST_START = pd.Timestamp("2019-01-01 02:00:00")

MODELS = {
    "LSTM-only": "/home/lizilin/code/torchhydro/results/eval_full/lstm_only",
    "LSTM+GNN":  "/home/lizilin/code/torchhydro/results/eval_full/lstm_gnn",
    "WDNE":      "/home/lizilin/code/torchhydro/results/eval_full/wdne",
}
COLORS = {"LSTM-only":"green","LSTM+GNN":"steelblue","WDNE":"darkorange"}
STYLES = {"LSTM-only":"-.","LSTM+GNN":"--","WDNE":"-"}

RAIN_DIR = "/mnt/data/ClassC/songliaorrevent/timeseries/3h"
ATTR_FILE = "/mnt/data/ClassC/songliaorrevent/attributes/attributes.csv"
OUT = Path("/home/lizilin/code/torchhydro/results/comparison_plots/train_test_flood")

_attr = pd.read_csv(ATTR_FILE)
AREA = dict(zip(_attr["basin_id"].astype(str), _attr["area"]))

BASINS = ["songliao_20800900","songliao_20810200","songliao_21100150",
          "songliao_21110150","songliao_21401050","songliao_21401550"]

def mm3h_to_m3s(f, a): return f * a / 10.8

def event_metrics(o, p):
    """计算单场洪水的综合指标: NSE, KGE, RMSE, 峰值误差%, 洪量偏差%"""
    mask = ~np.isnan(o) & ~np.isnan(p)
    if mask.sum() < 2:
        return {"NSE": np.nan, "KGE": np.nan, "RMSE": np.nan,
                "peak_err%": np.nan, "vol_bias%": np.nan}
    o, p = o[mask], p[mask]
    mu_o, mu_p = o.mean(), p.mean()
    # NSE
    nse_val = 1 - np.sum((o-p)**2) / np.sum((o-mu_o)**2)
    # KGE
    r = np.corrcoef(o, p)[0,1] if len(o) > 2 else 0
    alpha = p.std() / o.std() if o.std() > 0 else 1
    beta = mu_p / mu_o if mu_o != 0 else 1
    kge_val = 1 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2)
    # RMSE
    rmse_val = np.sqrt(np.mean((o-p)**2))
    # Peak error %
    o_peak, p_peak = o.max(), p.max()
    peak_err = abs(p_peak - o_peak) / o_peak * 100 if o_peak > 0 else np.nan
    # Volume bias %
    vol_bias = (p.sum() - o.sum()) / o.sum() * 100 if o.sum() > 0 else np.nan
    return {"NSE": nse_val, "KGE": kge_val, "RMSE": rmse_val,
            "peak_err%": peak_err, "vol_bias%": vol_bias}

def nse(o, p):
    mask = ~np.isnan(o) & ~np.isnan(p)
    if mask.sum()<2: return np.nan
    o,p=o[mask],p[mask]
    return 1-np.sum((o-p)**2)/np.sum((o-o.mean())**2)

for bid in BASINS:
    print(f"\n{bid}")
    area = AREA.get(bid, 1.0)

    # 加载各模型全时段预测
    all_pred = {}; all_obs = {}
    for name, d in MODELS.items():
        ds = xr.open_dataset(f"{d}/epochbest_model.pthflow_pred.nc")
        all_pred[name] = pd.Series(ds.sel(basin=bid)["inflow"].values,
                                    index=pd.to_datetime(ds.sel(basin=bid)["inflow"].time.values))
        ds.close()
        ds_o = xr.open_dataset(f"{d}/epochbest_model.pthflow_obs.nc")
        all_obs[name] = pd.Series(ds_o.sel(basin=bid)["inflow"].values,
                                   index=pd.to_datetime(ds_o.sel(basin=bid)["inflow"].time.values))
        ds_o.close()

    # 用 LSTM+GNN 识别洪水事件
    t_ref = all_obs["LSTM+GNN"].index.values
    ds_f = xr.open_dataset(f"{MODELS['LSTM+GNN']}/epochbest_model.pthflow_obs.nc")
    flood = ds_f.sel(basin=bid)["flood_event"].values
    ds_f.close()

    # 识别事件
    events_t = []
    in_ev, start_i = False, None
    for i, v in enumerate(flood):
        if not np.isnan(v) and v > 0 and not in_ev: in_ev, start_i = True, i
        elif (np.isnan(v) or v == 0) and in_ev:
            in_ev = False
            events_t.append((pd.Timestamp(t_ref[start_i]), pd.Timestamp(t_ref[i-1])))
    if in_ev: events_t.append((pd.Timestamp(t_ref[start_i]), pd.Timestamp(t_ref[-1])))
    # 合并间隔≤36h
    merged = []
    for s, e in events_t:
        if merged and (s - merged[-1][1]) <= pd.Timedelta(hours=36): merged[-1] = (merged[-1][0], e)
        else: merged.append((s, e))
    events_t = merged

    # 分训练期/测试期
    train_events = [(s,e) for s,e in events_t if e <= TRAIN_END]
    test_events = [(s,e) for s,e in events_t if s >= TEST_START]

    # 降雨
    rain_s = pd.Series(0.0, index=t_ref)
    csv_p = os.path.join(RAIN_DIR, f"{bid}.csv")
    if os.path.exists(csv_p):
        df = pd.read_csv(csv_p, parse_dates=["time"]).set_index("time")
        if "rain" in df.columns: rain_s = df["rain"].reindex(t_ref).fillna(0)

    for period_name, evt_list, period_label in [
        ("train", train_events, "TRAINING"),
        ("test", test_events, "TEST")
    ]:
        if not evt_list: continue
        print(f"  {period_label}: {len(evt_list)} events")

        for ev_idx, (t_start, t_end) in enumerate(evt_list):
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
            for name in MODELS:
                o_slice = all_obs[name][t0:t1].reindex(ref_slice.index)
                p_slice = all_pred[name][t0:t1].reindex(ref_slice.index)
                o_m3s = mm3h_to_m3s(o_slice.values, area)
                p_m3s = mm3h_to_m3s(p_slice.values, area)

                ns = nse(all_obs[name][t_start:t_end].values, all_pred[name][t_start:t_end].values)
                nses[name] = ns

                if name == "LSTM-only":
                    ax2.plot(t_plot, o_m3s, 'k-', linewidth=2.0, label='Observed', zorder=10)
                ax2.plot(t_plot, p_m3s, color=COLORS[name], linestyle=STYLES[name],
                         linewidth=1.2, label=f'{name} (NSE={ns:.3f})')

            ax2.axvspan(t_start, t_end, alpha=0.08, color='red')
            ax2.set_ylabel('Inflow (m³/s)', fontsize=11)
            ax2.legend(fontsize=8, ncol=3, loc='upper left')
            ax2.grid(True, alpha=0.3)
            ax2.set_title(f'{bid} [{period_label}] Event #{ev_idx+1} ({t_start.strftime("%Y-%m-%d %H:%M")} ~ {t_end.strftime("%Y-%m-%d %H:%M")})',
                          fontsize=13, fontweight='bold')
            fig.tight_layout()
            out_d = OUT / period_label.lower() / bid.replace("songliao_","")
            os.makedirs(out_d, exist_ok=True)
            fig.savefig(out_d / f"event_{ev_idx+1:03d}.png", dpi=150, bbox_inches='tight')
            plt.close(fig)

        # 汇总（扩充指标）
        for name in MODELS:
            all_m = [event_metrics(all_obs[name][s:e].values, all_pred[name][s:e].values) for s,e in evt_list]
            for metric in ["NSE","KGE","RMSE","peak_err%","vol_bias%"]:
                vals = [m[metric] for m in all_m if not np.isnan(m[metric])]
                if vals:
                    print(f"    {name:<12} {metric:<10}: mean={np.mean(vals):.3f}  med={np.median(vals):.3f}")

print(f"\nDone: {OUT}")
