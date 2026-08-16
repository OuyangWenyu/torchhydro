"""PPT汇报图 — 最终版本 (edge fix + route_steps=4)"""
import matplotlib.pyplot as plt, numpy as np, os
os.makedirs('/home/lizilin/code/torchhydro/results/summary/plots', exist_ok=True)
basins = ['20800900','20810200','21100150','21110150','21401050','21401550']
basins_5 = ['20800900','20810200','21110150','21401050','21401550']

# ===== 图1: 实验1 NSE柱状 =====
exp1 = {
    'Basin LSTM': [-0.05, 0.60, 0.33, 0.35, 0.60, 0.79],
    'LSTM-only':  [0.69, 0.33, 0.51, 0.47, 0.49, 0.14],
    'LSTM+GNN':   [0.50, 0.47, 0.48, 0.50, 0.81, 0.76],
    'WDNE(steps=4)': [0.24, 0.71, -0.33, 0.37, 0.71, 0.71],
}
colors = {'Basin LSTM':'#2ca02c','LSTM-only':'green','LSTM+GNN':'steelblue','WDNE(steps=4)':'darkorange'}
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(basins)); w = 0.2
for i, (name, vals) in enumerate(exp1.items()):
    ax.bar(x + i*w - 0.3, vals, w, label=name, color=colors.get(name), alpha=0.85)
ax.set_ylabel('NSE', fontsize=13); ax.set_xticks(x); ax.set_xticklabels(basins, fontsize=11)
ax.legend(fontsize=10, loc='lower left'); ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_title('Exp 1: NSE by Basin (Pure Rainfall, Edge Fixed + steps=4)', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
fig.tight_layout(); fig.savefig('/home/lizilin/code/torchhydro/results/summary/plots/01_exp1_nse_barchart.png', dpi=150); plt.close()

# ===== 图2: 实验2 静态属性 =====
exp2 = {
    'LSTM-only+static':  [0.64, 0.31, 0.24, 0.81, 0.85],
    'LSTM+GNN+static':   [0.51, 0.35, 0.60, 0.69, 0.79],
    'WDNE+static':       [0.66, 0.31, 0.53, 0.80, 0.89],
}
colors2 = {'LSTM-only+static':'green','LSTM+GNN+static':'steelblue','WDNE+static':'darkorange'}
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(basins_5)); w = 0.25
for i, (name, vals) in enumerate(exp2.items()):
    ax.bar(x + i*w - 0.25, vals, w, label=name, color=colors2.get(name), alpha=0.85)
    for j, v in enumerate(vals):
        ax.text(j + i*w - 0.25, v+0.02, f'{v:.2f}', ha='center', fontsize=7, rotation=90)
ax.set_ylabel('NSE', fontsize=13); ax.set_xticks(x); ax.set_xticklabels(basins_5, fontsize=11)
ax.legend(fontsize=10, loc='lower left'); ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_title('Exp 2: Static Attributes (5 Basins, Edge Fixed + steps=4)', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
fig.tight_layout(); fig.savefig('/home/lizilin/code/torchhydro/results/summary/plots/02_exp2_static.png', dpi=150); plt.close()

# ===== 图3: WDNE修复前后 + 调优 =====
wdne_old_broken = [0.57, 0.19, -0.61, 0.43, 0.86, 0.84]
wdne_fixed_s2   = [0.27, 0.46, 0.45, 0.56, 0.36, 0.45]
wdne_fixed_s4   = [0.24, 0.71, -0.33, 0.37, 0.71, 0.71]
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(6)
ax.bar(x - 0.25, wdne_old_broken, 0.22, label='Before (broken edges)', color='lightcoral', alpha=0.8)
ax.bar(x,        wdne_fixed_s2,   0.22, label='After fix (steps=2)',  color='steelblue', alpha=0.8)
ax.bar(x + 0.25, wdne_fixed_s4,   0.22, label='After fix (steps=4)',  color='darkorange', alpha=0.8)
ax.set_xticks(x); ax.set_xticklabels(basins, fontsize=11)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5); ax.legend(fontsize=10)
ax.set_ylabel('NSE', fontsize=13)
ax.set_title('WDNE Evolution: Broken Edges -> Fixed Edges -> Tuned steps=4', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.annotate('21100150: broken=-0.61, fixed=+0.45',
            xy=(2, 0.45), xytext=(3.5, 0.7),
            arrowprops=dict(arrowstyle='->', color='red'), fontsize=10, color='red', fontweight='bold')
fig.tight_layout(); fig.savefig('/home/lizilin/code/torchhydro/results/summary/plots/03_wdne_evolution.png', dpi=150); plt.close()

# ===== 图4: 静态属性下三模型平均NSE =====
avgs = [0.57, 0.59, 0.64]
fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(['LSTM-only','LSTM+GNN','WDNE (steps=4)'], avgs, color=['green','steelblue','darkorange'], alpha=0.85)
for i, v in enumerate(avgs):
    ax.text(i, v+0.02, f'{v:.2f}', ha='center', fontsize=14, fontweight='bold')
ax.set_ylabel('Average NSE', fontsize=13)
ax.set_title('Exp 2: Average NSE with Static Attributes', fontsize=14, fontweight='bold')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5); ax.grid(axis='y', alpha=0.3)
fig.tight_layout(); fig.savefig('/home/lizilin/code/torchhydro/results/summary/plots/04_static_avg.png', dpi=150); plt.close()

# ===== 图5: 逐流域最优模型标注 =====
fig, ax = plt.subplots(figsize=(14, 6))
best_model = ['LSTM-only','WDNE','LSTM-only','LSTM+GNN','LSTM+GNN','LSTM+GNN']
best_color  = ['green','darkorange','green','steelblue','steelblue','steelblue']
all_nse = {
    'LSTM-only (no graph)': [0.69, 0.33, 0.51, 0.47, 0.49, 0.14],
    'LSTM+GNN':             [0.50, 0.47, 0.48, 0.50, 0.81, 0.76],
    'WDNE (steps=4)':       [0.24, 0.71, -0.33, 0.37, 0.71, 0.71],
}
for i, (name, vals) in enumerate(all_nse.items()):
    ax.plot(basins, vals, 'o-', linewidth=2, markersize=8, label=name)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5); ax.legend(fontsize=11)
ax.set_ylabel('NSE', fontsize=13); ax.set_xlabel('Basin', fontsize=13)
ax.set_title('Exp 1: NSE by Basin — Different Models Excel at Different Basins', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
fig.tight_layout(); fig.savefig('/home/lizilin/code/torchhydro/results/summary/plots/05_per_basin_lines.png', dpi=150); plt.close()

print('All PPT figures regenerated (edge fix + steps=4 final)')
