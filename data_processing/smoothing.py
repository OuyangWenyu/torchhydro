import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import os
import warnings
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")

# ===================== 核心配置（最上方，方便修改） =====================
# 1. Excel配置文件路径
excel_config_path = "Temporary_file/floodevents_train_pick.xlsx"

# 2. 路径配置
data_dir = "/home/litianli/Dataset/223_FloodEvents/timeseries/1h/"
# 输出路径改为Temporary_file下的子文件夹
output_root_dir = "Temporary_file"
output_data_dir = os.path.join(output_root_dir, "smoothing_csv")  # 平滑后数据保存路径
output_plot_dir = os.path.join(output_root_dir, "rainflow_comparison_plots")  # 对比图保存路径

# 3. 平滑参数
window_length = 20
polyorder = 2

# 4. 绘图样式配置
plot_config = {
    "figsize": (14, 6),  # 调整画布尺寸适配单图
    "xlabel": "Date",
    "fontsize_label": 12,
    "fontsize_title": 14,
    "fontsize_legend": 10,
    "grid_alpha": 0.5,
    "grid_linestyle": "--",
    "time_format": "%Y-%m-%d %H",
    "month_interval": 1
}

precip_style = {
    "color": "#4169E1",
    "alpha": 0.7,
    "label": "Hourly Precipitation (mm)",
    "width": 0.8/24
}

streamflow_style = {
    "original": {
        "color": "black",
        "linestyle": "-",
        "linewidth": 2.5,
        "marker": ".",
        "markersize": 2,
        "label": "Original Streamflow (mm/h)"
    },
    "smoothed": {
        "color": "#FF6347",
        "linestyle": "-",
        "linewidth": 2.5,
        "marker": ".",
        "markersize": 2,
        "label": "Smoothed Streamflow (mm/h)"
    },
    "yaxis_label": "Streamflow (mm/h)"
}

# ===================== 工具函数 =====================
def ensure_dir(path):
    """确保目录存在，不存在则创建"""
    if not os.path.exists(path):
        os.makedirs(path)

def get_time_column(df):
    """自动检测时间列"""
    time_cols = ['date', 'time', 'datetime', 'Date', 'Time', 'Datetime', 'DATE', 'DATETIME']
    for col in time_cols:
        if col in df.columns:
            return col
    return None

def format_time_for_filename(dt):
    """格式化时间为文件名友好格式（去除空格、冒号等特殊字符）"""
    return dt.strftime("%Y%m%d_%H%M%S")

# ===================== 数据平滑处理函数 =====================
def process_basin(basin_name, start_time, end_time):
    """
    处理单个流域：
    1. 读取数据
    2. 对指定时间段的streamflow进行平滑
    3. 返回原始数据、平滑后数据（仅指定时间段）
    """
    input_file = os.path.join(data_dir, f"{basin_name}.csv")
    
    # 读取数据
    try:
        df = pd.read_csv(input_file)
        print(f"ℹ️  处理 {basin_name}：原始数据形状 {df.shape}")
    except FileNotFoundError:
        print(f"⚠️  警告：文件 {input_file} 不存在，跳过")
        return None, None
    except Exception as e:
        print(f"❌ {basin_name} 读取失败：{e}，跳过")
        return None, None
    
    # 必要列检查
    required_cols = ['streamflow', 'precipitation_obs']
    for col in required_cols:
        if col not in df.columns:
            print(f"❌ {basin_name} 缺少 '{col}' 列，跳过")
            return None, None
    
    # 时间列处理
    time_col = get_time_column(df)
    if not time_col:
        print(f"❌ {basin_name} 未找到时间列，跳过")
        return None, None
    
    # 转换时间列并筛选指定时间段
    try:
        df['datetime'] = pd.to_datetime(df[time_col], errors='coerce')
        df = df.dropna(subset=['datetime'])
        # 转换起止时间格式
        start_dt = pd.to_datetime(start_time)
        end_dt = pd.to_datetime(end_time)
        # 筛选指定时间段数据
        mask_time = (df['datetime'] >= start_dt) & (df['datetime'] <= end_dt)
        df_time_filtered = df[mask_time].copy()
        if len(df_time_filtered) == 0:
            print(f"❌ {basin_name} 无指定时间段({start_time}~{end_time})数据，跳过")
            return None, None
    except Exception as e:
        print(f"❌ {basin_name} 时间处理失败：{e}，跳过")
        return None, None
    
    # 空值统计
    null_count = df_time_filtered['streamflow'].isnull().sum()
    total_count = len(df_time_filtered)
    print(f"ℹ️ {basin_name} 指定时间段：总数据 {total_count} 条，空值 {null_count} 条（占比 {null_count/total_count:.2%}）")
    
    # 复制原始数据用于平滑
    df_smoothed = df.copy()
    smoothed_streamflow = df['streamflow'].copy()
    
    # 对指定时间段的非空径流数据进行平滑
    non_null_mask = df_time_filtered['streamflow'].notna()
    non_null_data = df_time_filtered.loc[non_null_mask, 'streamflow'].values
    period_length = len(non_null_data)
    
    if period_length >= window_length:
        # 执行S-G平滑
        smoothed_vals = savgol_filter(non_null_data, window_length=window_length, polyorder=polyorder)
        # 将平滑后数据写回原索引
        valid_indices = df_time_filtered.loc[non_null_mask].index
        smoothed_streamflow.iloc[valid_indices] = smoothed_vals
        print(f"✅ {basin_name} 指定时间段数据已平滑（有效数据{period_length}条）")
    elif period_length > 0:
        print(f"⚠️ {basin_name} 指定时间段有效数据{period_length}条 < 窗口长度{window_length}（不平滑）")
    else:
        print(f"ℹ️ {basin_name} 指定时间段无有效径流数据")
    
    # 更新平滑后径流列
    df_smoothed['streamflow'] = smoothed_streamflow
    
    # 保存平滑后数据
    ensure_dir(output_data_dir)
    output_file = os.path.join(output_data_dir, f"{basin_name}.csv")
    df_smoothed.to_csv(output_file, index=False)
    print(f"✅ {basin_name} 平滑后数据已保存：{output_file}")
    
    # 返回指定时间段的原始/平滑数据 + 格式化后的开始时间（用于绘图命名）
    plot_data_original = df[mask_time][['datetime', 'streamflow', 'precipitation_obs']].copy()
    plot_data_smoothed = df_smoothed[mask_time][['datetime', 'streamflow', 'precipitation_obs']].copy()
    
    return plot_data_original, plot_data_smoothed, start_dt

# ===================== 绘图函数（单图双曲线） =====================
def plot_rainflow_comparison(basin_name, df_original, df_smoothed, start_time, end_time, formatted_start_time):
    """绘制单图对比：同一图中展示原始/平滑径流 + 降雨"""
    ensure_dir(output_plot_dir)
    # 新的文件名规则：流域ID_开始时间_smooth_comparison.png
    plot_filename = f"{basin_name}_{formatted_start_time}_smooth_comparison.png"
    save_path = os.path.join(output_plot_dir, plot_filename)
    
    # 创建单个子图（核心修改：去掉双列子图）
    fig, ax1 = plt.subplots(figsize=plot_config["figsize"])
    fig.suptitle(f"{basin_name} Rainfall-Runoff Comparison ({start_time} ~ {end_time})", 
                 fontsize=plot_config["fontsize_title"], fontweight="bold", y=1.02)

    # ---------------------- 绘制径流曲线（原始+平滑） ----------------------
    # 原始径流
    ax1.plot(
        df_original['datetime'], df_original['streamflow'],
        color=streamflow_style["original"]["color"],
        linestyle=streamflow_style["original"]["linestyle"],
        linewidth=streamflow_style["original"]["linewidth"],
        marker=streamflow_style["original"]["marker"],
        markersize=streamflow_style["original"]["markersize"],
        label=streamflow_style["original"]["label"]
    )
    # 平滑径流（新增：在同一轴上绘制）
    ax1.plot(
        df_smoothed['datetime'], df_smoothed['streamflow'],
        color=streamflow_style["smoothed"]["color"],
        linestyle=streamflow_style["smoothed"]["linestyle"],
        linewidth=streamflow_style["smoothed"]["linewidth"],
        marker=streamflow_style["smoothed"]["marker"],
        markersize=streamflow_style["smoothed"]["markersize"],
        label=streamflow_style["smoothed"]["label"]
    )
    
    # 径流轴配置
    ax1.set_ylabel(streamflow_style["yaxis_label"], fontsize=plot_config["fontsize_label"], fontweight="bold")
    ax1.set_xlabel(plot_config["xlabel"], fontsize=plot_config["fontsize_label"], fontweight="bold")
    ax1.set_ylim(bottom=0)
    ax1.grid(alpha=plot_config["grid_alpha"], linestyle=plot_config["grid_linestyle"])
    
    # ---------------------- 绘制降雨柱状图（双轴） ----------------------
    ax2 = ax1.twinx()
    ax2.bar(
        df_original['datetime'], df_original['precipitation_obs'],  # 降雨数据原始/平滑一致，用原始即可
        color=precip_style["color"],
        alpha=precip_style["alpha"],
        width=precip_style["width"],
        label=precip_style["label"]
    )
    max_precip = df_original['precipitation_obs'].max() if df_original['precipitation_obs'].max() > 0 else 1
    ax2.set_ylim(0, max_precip * 3)
    ax2.invert_yaxis()
    ax2.set_ylabel("Precipitation (mm)", fontsize=plot_config["fontsize_label"], fontweight="bold", 
                   rotation=270, labelpad=20)
    
    # 时间轴格式化
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=12))  # 每12小时标注一次
    ax1.xaxis.set_major_formatter(mdates.DateFormatter(plot_config["time_format"]))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")
    
    # 整合图例（原始径流+平滑径流+降雨）
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=plot_config["fontsize_legend"])

    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ {basin_name} 对比图已保存：{save_path}\n")

# ===================== 批量处理主函数 =====================
def main():
    # 1. 读取Excel配置文件
    try:
        config_df = pd.read_excel(excel_config_path)
        print(f"✅ 成功读取配置文件：{excel_config_path}")
        print(f"ℹ️  配置文件共 {len(config_df)} 条记录")
    except FileNotFoundError:
        print(f"❌ 配置文件 {excel_config_path} 不存在，程序退出")
        return
    except Exception as e:
        print(f"❌ 读取配置文件失败：{e}，程序退出")
        return
    
    # 2. 检查必要列是否存在
    required_config_cols = ['basin_id', 'smoothing', 'smoothing_start_time', 'smoothing_end_time']
    missing_cols = [col for col in required_config_cols if col not in config_df.columns]
    if missing_cols:
        print(f"❌ 配置文件缺少必要列：{missing_cols}，程序退出")
        return
    
    # 3. 筛选需要平滑的流域（smoothing=1）
    smooth_df = config_df[config_df['smoothing'] == 1].copy()
    if len(smooth_df) == 0:
        print("ℹ️  配置文件中无需要平滑的流域（smoothing=1），程序退出")
        return
    print(f"ℹ️  筛选出 {len(smooth_df)} 个需要平滑的流域")
    
    # 4. 初始化输出目录
    ensure_dir(output_root_dir)
    ensure_dir(output_data_dir)
    ensure_dir(output_plot_dir)
    
    print(f"\n🚀 开始处理 {len(smooth_df)} 个流域")
    print(f"ℹ️  平滑参数：窗口长度{window_length}，多项式阶数{polyorder}")
    print("-" * 80)
    
    # 5. 遍历需要平滑的流域
    for idx, row in smooth_df.iterrows():
        basin_name = row['basin_id']
        start_time = row['smoothing_start_time']
        end_time = row['smoothing_end_time']
        
        # 检查时间是否为空
        if pd.isna(start_time) or pd.isna(end_time):
            print(f"\n⚠️  流域 {basin_name} 时间为空，跳过")
            continue
        
        print(f"\n===== 处理流域：{basin_name} =====")
        # 数据平滑处理（新增返回格式化的开始时间）
        df_original, df_smoothed, start_dt = process_basin(basin_name, start_time, end_time)
        # 绘制对比图
        if df_original is not None and df_smoothed is not None:
            # 格式化开始时间为文件名友好格式
            formatted_start = format_time_for_filename(start_dt)
            plot_rainflow_comparison(basin_name, df_original, df_smoothed, start_time, end_time, formatted_start)
    
    print("🎉 所有流域处理完成！")
    print(f"📁 平滑后数据路径：{output_data_dir}")
    print(f"📁 对比图保存路径：{output_plot_dir}")

if __name__ == "__main__":
    main()