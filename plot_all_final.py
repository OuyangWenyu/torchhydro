"""
全部图表重做 — 仅展示修复后正确结果
1. 实验1: 四模型×多指标对比
2. 实验2: 静态属性
3. 实验3: 单流域
4. 实验4: 空间敏感性(全时段放大)
5. 全部洪水事件图
"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt, xarray as xr, os
from pathlib import Path
import warnings; warnings.filterwarnings('ignore')

# ===== 配置 =====
OUT = Path("/home/lizilin/code/torchhydro/results/report/figures")
BASINS = ['20800900','20810200','21100150','21110150','21401050','21401550']
BASINS_5 = ['20800900','20810200','21110150','21401050','21401550']
ATTR_FILE = "/mnt/data/ClassC/songliaorrevent/attributes/attributes.csv"
RAIN_DIR = "/mnt/data/ClassC/songliaorrevent/timeseries/3h"
_attr = pd.read_csv(ATTR_FILE)
AREA = dict(zip(_attr["basin_id"].astype(str), _attr["area"]))

MODEL_PATHS = {
    "Basin LSTM": "/home/lizilin/code/torchhydro/results/basin_lstm/test",
    "LSTM-only":  "/home/lizilin/code/torchhydro/results/baseline_lstm_only/songliao_3h_normal",
    "LSTM+GNN":   "/home/lizilin/code/torchhydro/results/baseline_lstm_gnn/songliao_3h_normal",
    "WDNE":       "/home/lizilin/code/torchhydro/results/wdne_adaptive/songliao_3h_normal",
}
COLORS = {"Basin LSTM":"#2ca02c","LSTM-only":"green","LSTM+GNN":"steelblue","WDNE":"darkorange"}

def mm3h_to_m3s(f,a): return f*a/10.8
def nse(o,p):
    mask=~np.isnan(o)&~np.isnan(p)
    if mask.sum()<2: return np.nan
    o,p=o[mask],p[mask]; return 1-np.sum((o-p)**2)/np.sum((o-o.mean())**2)
def kge(o,p):
    mask=~np.isnan(o)&~np.isnan(p)
    if mask.sum()<2: return np.nan
    o,p=o[mask],p[mask]
    r=np.corrcoef(o,p)[0,1]; a=p.std()/o.std() if o.std()>0 else 1
    b=p.mean()/o.mean() if o.mean()!=0 else 1
    return 1-np.sqrt((r-1)**2+(a-1)**2+(b-1)**2)
def peak_err(o,p):
    mask=~np.isnan(o)&~np.isnan(p)
    if mask.sum()<2: return np.nan
    return abs(p[mask].max()-o[mask].max())/o[mask].max()*100

# ===== 读取所有模型指标 =====
all_metrics = {}
for name, d in MODEL_PATHS.items():
    csv_path = f"{d}/metric_inflow.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        all_metrics[name] = dict(zip(df['basin_id'].str.replace('songliao_',''), df['NSE']))

# ===== 图1: 实验1 四模型NSE柱状图 =====
fig, ax = plt.subplots(figsize=(14, 7))
x = np.arange(len(BASINS)); w = 0.2
for i, name in enumerate(["Basin LSTM","LSTM-only","LSTM+GNN","WDNE"]):
    vals = [all_metrics[name].get(b, np.nan) for b in BASINS]
    bars = ax.bar(x + i*w - 0.3, vals, w, label=name, color=COLORS[name], alpha=0.85)
    for j, (bar, v) in enumerate(zip(bars, vals)):
        if not np.isnan(v):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02 if v>0 else -0.08,
                    f'{v:.2f}', ha='center', fontsize=7, rotation=90)
ax.set_ylabel('NSE', fontsize=13); ax.set_xticks(x); ax.set_xticklabels(BASINS, fontsize=11)
ax.legend(fontsize=11, loc='lower left'); ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
avg_text = "  ".join([f"{n}:{np.nanmean(list(all_metrics[n].get(b,np.nan) for b in BASINS)):.2f}" for n in ["Basin LSTM","LSTM-only","LSTM+GNN","WDNE"]])
ax.set_title(f'Exp 1: NSE by Basin (Pure Rainfall)\nAvg NSE — {avg_text}', fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3); fig.tight_layout()
fig.savefig(OUT / '01_exp1_nse.png', dpi=150); plt.close()

# ===== 图2: 多指标雷达对比 =====
# 取所有流域的中位数指标
def load_series(nc,bid):
    ds=xr.open_dataset(nc); s=pd.Series(ds.sel(basin=f'songliao_{bid}')['inflow'].values, index=pd.to_datetime(ds.sel(basin=f'songliao_{bid}')['inflow'].time.values)); ds.close(); return s

# 计算每个事件的多指标
from plot_train_test_flood import event_metrics as em

# ===== 图3: 实验2 静态属性 NSE =====
static_paths = {
    "LSTM-only+static": "/home/lizilin/code/torchhydro/results/static_5basins_lstm_only/songliao_3h_normal",
    "LSTM+GNN+static":  "/home/lizilin/code/torchhydro/results/static_5basins_lstm_gnn/songliao_3h_normal",
    "WDNE+static":      "/home/lizilin/code/torchhydro/results/static_5basins_wdne/songliao_3h_normal",
}
static_nse = {}
for name, d in static_paths.items():
    csv = f"{d}/metric_inflow.csv"
    if os.path.exists(csv):
        df = pd.read_csv(csv)
        static_nse[name] = dict(zip(df['basin_id'].str.replace('songliao_',''), df['NSE']))

fig, ax = plt.subplots(figsize=(12, 7))
x = np.arange(len(BASINS_5)); w = 0.25
for i, name in enumerate(static_nse):
    vals = [static_nse[name].get(b, np.nan) for b in BASINS_5]
    bars = ax.bar(x + i*w - 0.25, vals, w, label=name, alpha=0.85)
    for j, (bar, v) in enumerate(zip(bars, vals)):
        if not np.isnan(v):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02 if v>0 else -0.08,
                    f'{v:.2f}', ha='center', fontsize=7, rotation=90)
ax.set_ylabel('NSE', fontsize=13); ax.set_xticks(x); ax.set_xticklabels(BASINS_5, fontsize=11)
ax.legend(fontsize=10, loc='lower left'); ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
avg_text = "  ".join([f"{n}:{np.nanmean(list(static_nse[n].values())):.2f}" for n in static_nse])
ax.set_title(f'Exp 2: NSE with Static Attributes (5 Basins)\nAvg — {avg_text}', fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3); fig.tight_layout()
fig.savefig(OUT / '02_exp2_static.png', dpi=150); plt.close()

# ===== 图4: 全时段逐事件多指标表（仅21401550作为示例）=====
# 这个需要全时段eval，先跳过。改为画洪水事件图

# ===== 图5: WDNE 演进 =====
fig, ax = plt.subplots(figsize=(14, 6))
wdne_stages = {
    'Broken edges (old)':  [0.57, 0.19, -0.61, 0.43, 0.86, 0.84],
    'Fixed edges steps=2': [0.27, 0.46,  0.45, 0.56, 0.36, 0.45],
    'Fixed edges steps=4': [0.24, 0.71, -0.33, 0.37, 0.71, 0.71],
}
x = np.arange(6); w = 0.25
for i, (name, vals) in enumerate(wdne_stages.items()):
    ax.bar(x + i*w - 0.25, vals, w, label=name, alpha=0.8)
ax.set_xticks(x); ax.set_xticklabels(BASINS, fontsize=11)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5); ax.legend(fontsize=10)
ax.set_ylabel('NSE', fontsize=13)
ax.set_title('WDNE Evolution: Broken Edges → Fixed edges → Tuned steps=4', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / '03_wdne_evolution.png', dpi=150); plt.close()

# ===== 图6: 空间敏感性 =====
print("Drawing spatial sensitivity...")
spatial_dir = "/home/lizilin/code/torchhydro/results/spatial_test"
if os.path.exists(spatial_dir):
    bid = 'songliao_21401550'
    scenarios = {'Normal':'normal','Upstream x4':'sim1','Midstream x4':'sim2','Downstream x4':'sim3'}
    colors_s = {'Normal':'black','Upstream x4':'blue','Midstream x4':'orange','Downstream x4':'red'}
    styles_s = {'Normal':'-','Upstream x4':'--','Midstream x4':'-.','Downstream x4':':'}

    # 加载所有预测
    try:
        obs_nc = f'{spatial_dir}/normal/epochbest_model.pthflow_obs.nc'
        ds_obs = xr.open_dataset(obs_nc)
        obs = ds_obs.sel(basin=bid)['inflow']; t = pd.to_datetime(obs.time.values)
        fe = ds_obs.sel(basin=bid)['flood_event'].values; ds_obs.close()
        area = AREA.get('21401550', 1.0)

        # 找最大事件
        events = []; in_ev, start = False, None
        for i, v in enumerate(fe):
            if not np.isnan(v) and v>0 and not in_ev: in_ev,start=True,i
            elif (np.isnan(v) or v==0) and in_ev: in_ev=False; events.append((start,i-1))
        if in_ev: events.append((start, len(fe)-1))
        events.sort(key=lambda x: np.nanmax(obs.values[x[0]:x[1]+1]), reverse=True)

        for ev_idx in range(min(2, len(events))):
            s,e = events[ev_idx]; ctx=36
            s1,e1 = max(0,s-ctx), min(len(t)-1,e+ctx)
            preds = {}
            for name, mode in scenarios.items():
                ds = xr.open_dataset(f'{spatial_dir}/{mode}/epochbest_model.pthflow_pred.nc')
                preds[name] = ds.sel(basin=bid)['inflow'].values; ds.close()

            rain = np.zeros(len(t))
            csv_p = os.path.join(RAIN_DIR, f'{bid}.csv')
            if os.path.exists(csv_p):
                df=pd.read_csv(csv_p,parse_dates=['time']).set_index('time')
                if 'rain' in df.columns: rain = df['rain'].reindex(t).fillna(0).values

            t_plot = t[s1:e1+1]; r = rain[s1:e1+1]
            fig,(ax1,ax2)=plt.subplots(2,1,figsize=(18,10),sharex=True,gridspec_kw={'height_ratios':[1,3]})
            ax1.bar(t_plot, r, width=0.04, color='steelblue', alpha=0.5)
            ax1.set_ylabel('Precip (mm/3h)', fontsize=11)
            ax1.set_ylim(max(1,np.nanmax(r)*1.5),0)
            o_plot = obs.values[s1:e1+1]*area/10.8
            ax2.plot(t_plot, o_plot, 'k-', linewidth=2.5, label='Observed', zorder=10)
            for name in scenarios:
                p = preds[name][s1:e1+1]*area/10.8
                ax2.plot(t_plot, p, color=colors_s[name], linestyle=styles_s[name], linewidth=1.5, label=name)
            ax2.axvspan(t[s],t[e],alpha=0.08,color='red')
            ax2.set_ylabel('Inflow (m³/s)',fontsize=11); ax2.legend(fontsize=9); ax2.grid(True,alpha=0.3)
            ax2.set_title(f'{bid} Spatial Sensitivity Event #{ev_idx+1}',fontsize=13,fontweight='bold')
            fig.tight_layout(); fig.savefig(OUT/f'04_spatial_event{ev_idx+1}.png',dpi=150); plt.close()
        print("Spatial plots done")
    except Exception as e: print(f"Spatial error: {e}")
else:
    print("No spatial test data")

# ===== 洪水事件图（test set, 4 models）=====
print("\nDrawing flood events...")
flood_out = OUT / "flood_events"
for bid_long in [f"songliao_{b}" for b in BASINS]:
    bid_short = bid_long.replace('songliao_','')
    bid_dir = flood_out / bid_short; os.makedirs(bid_dir, exist_ok=True)

    # 用 LSTM+GNN 识别人事件
    try:
        obs_nc = f"{MODEL_PATHS['LSTM+GNN']}/epochbest_model.pthflow_obs.nc"
        ds_f = xr.open_dataset(obs_nc)
        flood = ds_f.sel(basin=bid_long)['flood_event'].values
        t_ref = pd.to_datetime(ds_f.sel(basin=bid_long)['inflow'].time.values)
        ds_f.close()

        all_obs_s = {}; all_pred_s = {}
        for name, d in MODEL_PATHS.items():
            try:
                all_obs_s[name] = load_series(f"{d}/epochbest_model.pthflow_obs.nc", bid_short)
                all_pred_s[name] = load_series(f"{d}/epochbest_model.pthflow_pred.nc", bid_short)
            except: pass

        events_t = []; in_ev, start_i = False, None
        for i, v in enumerate(flood):
            if not np.isnan(v) and v>0 and not in_ev: in_ev,start_i=True,i
            elif (np.isnan(v) or v==0) and in_ev: in_ev=False; events_t.append((pd.Timestamp(t_ref[start_i]),pd.Timestamp(t_ref[i-1])))
        if in_ev: events_t.append((pd.Timestamp(t_ref[start_i]),pd.Timestamp(t_ref[-1])))
        merged = []
        for s,e in events_t:
            if merged and (s-merged[-1][1])<=pd.Timedelta(hours=36): merged[-1]=(merged[-1][0],e)
            else: merged.append((s,e))
        events_t = merged

        area = AREA.get(bid_long, 1.0)
        rain_s = pd.Series(0.0, index=t_ref)
        csv_p = os.path.join(RAIN_DIR, f'{bid_long}.csv')
        if os.path.exists(csv_p):
            df=pd.read_csv(csv_p,parse_dates=['time']).set_index('time')
            if 'rain' in df.columns: rain_s=df['rain'].reindex(t_ref).fillna(0)

        for ev_idx,(t_start,t_end) in enumerate(events_t):
            ctx=pd.Timedelta(hours=36); t0,t1=t_start-ctx,t_end+ctx
            ref_slice = all_obs_s.get('LSTM+GNN', pd.Series(dtype=float))
            if hasattr(ref_slice,'index') and len(ref_slice[t0:t1])==0: continue

            fig,(ax1,ax2)=plt.subplots(2,1,figsize=(18,10),sharex=True,gridspec_kw={'height_ratios':[1,3]})
            r=rain_s[t0:t1].values; t_plot=t_ref[(t_ref>=t0)&(t_ref<=t1)]
            if len(r)!=len(t_plot): r=r[:len(t_plot)]
            ax1.bar(t_plot, r, width=0.04, color='steelblue', alpha=0.5)
            ax1.set_ylabel('Precip (mm/3h)',fontsize=11)
            ax1.set_ylim(max(1,np.nanmax(r)*1.5) if len(r)>0 else 1,0)

            nses={}
            for name in ["Basin LSTM","LSTM-only","LSTM+GNN","WDNE"]:
                if name not in all_pred_s: continue
                o_slice = all_obs_s[name][t0:t1].reindex(t_plot).values
                p_slice = all_pred_s[name][t0:t1].reindex(t_plot).values
                o_m3s=mm3h_to_m3s(o_slice,area); p_m3s=mm3h_to_m3s(p_slice,area)
                ns=nse(all_obs_s[name][t_start:t_end].values, all_pred_s[name][t_start:t_end].values)
                nses[name]=ns
                if name=="Basin LSTM": ax2.plot(t_plot,o_m3s,'k-',linewidth=2.0,label='Observed',zorder=10)
                ax2.plot(t_plot,p_m3s,color=COLORS[name],linestyle={'Basin LSTM':':','LSTM-only':'-.','LSTM+GNN':'--','WDNE':'-'}[name],linewidth=1.2,label=f'{name} (NSE={ns:.3f})')
            ax2.axvspan(t_start,t_end,alpha=0.08,color='red')
            ax2.set_ylabel('Inflow (m³/s)',fontsize=11); ax2.legend(fontsize=7,ncol=2,loc='upper left'); ax2.grid(True,alpha=0.3)
            ax2.set_title(f'{bid_long} Event #{ev_idx+1}',fontsize=13,fontweight='bold')
            fig.tight_layout(); fig.savefig(bid_dir/f'event_{ev_idx+1:03d}.png',dpi=150); plt.close()
        print(f"  {bid_short}: {len(events_t)} events plotted")
    except Exception as e:
        print(f"  {bid_short}: ERROR - {e}")

print(f"\nAll done: {OUT}")
