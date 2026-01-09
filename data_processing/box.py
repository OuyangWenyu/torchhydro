import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置字体（无中文）
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# 读取CSV文件并提取NSE列，过滤掉小于0的值
file_path = 'results/7_84/metrics_results/valid_event_level_metrics.csv'
df = pd.read_csv(file_path)
nse_data = df['NSE_VALID'].dropna()  # 去除缺失值
nse_data = nse_data[nse_data >= 0]  # 过滤掉小于0的数据

# 创建图形
fig, ax = plt.subplots(figsize=(8, 6))

# 绘制箱型图（仅基于过滤后的数据）
box_plot = ax.boxplot(nse_data, patch_artist=True, 
                      boxprops=dict(facecolor='lightblue', alpha=0.7),
                      medianprops=dict(color='red', linewidth=2),
                      whiskerprops=dict(color='black', linewidth=1.5),
                      capprops=dict(color='black', linewidth=1.5),
                      flierprops=dict(marker='o', markerfacecolor='orange', alpha=0.6, markersize=4))

# 绘制每个数据点（仅显示>=0的点）
np.random.seed(42)  # 固定随机种子使抖动一致
x = np.random.normal(1, 0.04, size=len(nse_data))  # 水平方向轻微抖动
ax.scatter(x, nse_data, alpha=0.6, s=30, c='darkblue', edgecolors='black', linewidth=0.5)

# 标出中位数（基于过滤后的数据）
median_value = nse_data.median()
ax.text(1.15, median_value, f'Median: {median_value:.3f}', 
        fontsize=12, fontweight='bold', 
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

# 设置图形属性
ax.set_ylabel('NSE', fontsize=14, fontweight='bold')
ax.set_title('NSE Metric Distribution (≥ 0)', fontsize=16, fontweight='bold')  # 标题注明过滤条件
ax.set_xticklabels(['NSE'], fontsize=12)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# 调整y轴范围（可选，使图形更美观）
if len(nse_data) > 0:
    y_min = max(0, nse_data.min() - 0.05)
    y_max = nse_data.max() + 0.05
    ax.set_ylim(y_min, y_max)

# 调整布局
plt.tight_layout()

# 保存图片
plt.savefig('7_84.png', dpi=300, bbox_inches='tight')
plt.show()