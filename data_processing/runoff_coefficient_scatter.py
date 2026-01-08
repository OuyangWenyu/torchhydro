import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# -------------------------- 1. 路径设置 --------------------------
train_csv_path = r"results/6_106/metrics_results/train_event_level_metrics.csv"
valid_csv_path = r"results/6_106/metrics_results/valid_event_level_metrics.csv"
output_main_folder = r"results/6_106"
output_per_basin_folder = os.path.join(output_main_folder, "per_basin_plots")

# -------------------------- 2. 创建文件夹 --------------------------
if not os.path.exists(output_main_folder):
    os.makedirs(output_main_folder)
    print(f"Created main folder: {output_main_folder}")
if not os.path.exists(output_per_basin_folder):
    os.makedirs(output_per_basin_folder)
    print(f"Created per-basin folder: {output_per_basin_folder}")

# -------------------------- 3. 数据读取与预处理 --------------------------
# 读取训练集数据
df_train = pd.read_csv(train_csv_path)
df_train["dataset"] = "train"

# 读取验证集数据
df_valid = pd.read_csv(valid_csv_path)
df_valid["dataset"] = "valid"

# 合并数据
df = pd.concat([df_train, df_valid], ignore_index=True)

# 重命名列以匹配绘图需求
df = df.rename(columns={
    "Obs_Runoff_Coeff_TRAIN": "runoff_coefficient_obs",
    "Pred_Runoff_Coeff_TRAIN": "runoff_coefficient_pred"
})

# 如果验证集列名不同，也进行处理
if "Obs_Runoff_Coeff_VALID" in df.columns:
    df.loc[df["dataset"] == "valid", "runoff_coefficient_obs"] = df.loc[df["dataset"] == "valid", "Obs_Runoff_Coeff_VALID"]
    df.loc[df["dataset"] == "valid", "runoff_coefficient_pred"] = df.loc[df["dataset"] == "valid", "Pred_Runoff_Coeff_VALID"]

# 数据清洗：删除缺失值
df = df.dropna(subset=["runoff_coefficient_obs", "runoff_coefficient_pred"])
df["basin_id"] = df["basin_id"].astype(str)
df["flood_event_id"] = df["flood_event_id"].astype(str)

# 定义函数：清洗流域ID（用于文件命名）
def clean_basin_id(raw_id):
    cleaned = ''.join([c for c in raw_id if c.isdigit() or c == '_']).split("_")[-1]
    return cleaned

# 生成清洗后的流域ID
df["cleaned_basin_id"] = df["basin_id"].apply(clean_basin_id)

# 获取所有唯一流域
unique_basins = df["basin_id"].unique()
print(f"Successfully read data: {len(unique_basins)} basins, {len(df)} valid records")
print(f"Train records: {len(df[df['dataset']=='train'])}, Valid records: {len(df[df['dataset']=='valid'])}")

# -------------------------- 4. 颜色配置 --------------------------
cmap = plt.colormaps.get_cmap('tab10')
basin_color_map = {basin: cmap(i % 10) for i, basin in enumerate(unique_basins)}

# -------------------------- 5. 绘制总览图（所有流域，区分训练集和验证集） --------------------------
plt.rcParams["figure.dpi"] = 300
plt.rcParams["figure.figsize"] = (12, 8)
fig, ax = plt.subplots()

# 绘制训练集散点
df_train_plot = df[df["dataset"] == "train"]
ax.scatter(
    df_train_plot["runoff_coefficient_obs"],
    df_train_plot["runoff_coefficient_pred"],
    color="#1f77b4",
    s=40,
    alpha=0.7,
    edgecolors="white",
    linewidth=0.5,
    marker='o',
    label="Train"
)

# 绘制验证集散点
df_valid_plot = df[df["dataset"] == "valid"]
ax.scatter(
    df_valid_plot["runoff_coefficient_obs"],
    df_valid_plot["runoff_coefficient_pred"],
    color="#ff7f0e",
    s=40,
    alpha=0.7,
    edgecolors="white",
    linewidth=0.5,
    marker='^',
    label="Valid"
)

# 添加1:1参考线
ax.plot([0, 2], [0, 2], color="red", linestyle="--", linewidth=1.5, alpha=0.8, label="1:1 Line")

# 总览图样式
ax.set_title("Runoff Coefficient: Observed vs Predicted (All Basins)", fontsize=16, fontweight="bold")
ax.set_xlabel("Observed Runoff Coefficient", fontsize=12)
ax.set_ylabel("Predicted Runoff Coefficient", fontsize=12)
ax.set_xlim(0, 2)
ax.set_ylim(0, 2)
ax.set_aspect('equal', adjustable='box')  # 设置横纵比一致
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(loc="upper left", fontsize=10)

# 保存总览图
total_plot_path = os.path.join(output_main_folder, "all_basins_obs_vs_pred.png")
plt.tight_layout()
plt.savefig(total_plot_path, bbox_inches="tight")
plt.close()
print(f"Saved total overview plot: {total_plot_path}")

# -------------------------- 6. 绘制逐流域图（1:1线+场次编号标注） --------------------------
label_offset = (df["runoff_coefficient_pred"].max() - df["runoff_coefficient_pred"].min()) * 0.01

for basin in unique_basins:
    basin_data = df[df["basin_id"] == basin]
    cleaned_id = basin_data["cleaned_basin_id"].iloc[0]
    
    # 创建画布
    plt.rcParams["figure.figsize"] = (10, 6)
    fig, ax = plt.subplots()
    
    # 分别绘制训练集和验证集
    train_data = basin_data[basin_data["dataset"] == "train"]
    valid_data = basin_data[basin_data["dataset"] == "valid"]
    
    # 绘制训练集点
    if len(train_data) > 0:
        ax.scatter(
            train_data["runoff_coefficient_obs"],
            train_data["runoff_coefficient_pred"],
            color="#1f77b4",
            s=50,
            alpha=0.8,
            edgecolors="white",
            linewidth=0.5,
            marker='o',
            label="Train"
        )
    
    # 绘制验证集点
    if len(valid_data) > 0:
        ax.scatter(
            valid_data["runoff_coefficient_obs"],
            valid_data["runoff_coefficient_pred"],
            color="#ff7f0e",
            s=50,
            alpha=0.8,
            edgecolors="white",
            linewidth=0.5,
            marker='^',
            label="Valid"
        )

    # 为每个点标注场次开始时间
    for idx, row in basin_data.iterrows():
        text_y = row["runoff_coefficient_pred"] + label_offset
        # 提取开始时间并格式化（只显示日期部分）
        start_time = pd.to_datetime(row["start_time"]).strftime("%Y-%m-%d")
        ax.text(
            row["runoff_coefficient_obs"],
            text_y,
            start_time,
            fontsize=6,
            ha="center",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.7)
        )

    # 添加1:1参考线
    ax.plot([0, 2], [0, 2], color="red", linestyle="--", linewidth=1.5, alpha=0.8, label="1:1 Line")

    # 逐流域图样式
    ax.set_title(f"Runoff Coefficient - Basin {cleaned_id}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Observed Runoff Coefficient", fontsize=12)
    ax.set_ylabel("Predicted Runoff Coefficient", fontsize=12)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_aspect('equal', adjustable='box')  # 设置横纵比一致
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", fontsize=10)

    # 保存逐流域图
    per_basin_path = os.path.join(output_per_basin_folder, f"{cleaned_id}.png")
    plt.tight_layout()
    plt.savefig(per_basin_path, bbox_inches="tight")
    plt.close()
    print(f"Saved per-basin plot: {per_basin_path}")

print("All plots generated successfully!")
