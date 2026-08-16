#!/usr/bin/env python3
"""
按年份和洪水事件序号绘制洪水事件分析图
每个流域单独文件夹，每场洪水单独图片
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import h5py
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def load_data_with_time(obs_file, pred_file):
    """加载数据并解析时间信息"""
    obs_data = {}
    pred_data = {}
    
    try:
        # 加载观测数据
        with h5py.File(obs_file, 'r') as f:
            for var_name in f.keys():
                obs_data[var_name] = f[var_name][:]
                print(f"Obs {var_name}: shape {obs_data[var_name].shape}")
        
        # 加载预测数据  
        with h5py.File(pred_file, 'r') as f:
            for var_name in f.keys():
                pred_data[var_name] = f[var_name][:]
                print(f"Pred {var_name}: shape {pred_data[var_name].shape}")
                
        return obs_data, pred_data
        
    except Exception as e:
        print(f"数据加载失败: {e}")
        return None, None

def convert_time_to_datetime(time_array):
    """将时间数组转换为datetime对象"""
    # 假设时间是从1980-01-01开始的3小时间隔
    base_time = datetime(1980, 1, 1)
    datetime_array = []
    
    for time_val in time_array:
        # 假设time_val是从base_time开始的3小时数
        dt = base_time + timedelta(hours=time_val * 3)
        datetime_array.append(dt)
    
    return np.array(datetime_array)

def identify_flood_events_with_year(flood_event_array, time_array, basin_id, min_duration=3):
    """识别洪水事件并标注年份和序号"""
    events = []
    flood_event_series = np.array(flood_event_array).flatten()
    
    # 转换时间
    try:
        datetime_array = convert_time_to_datetime(time_array)
    except:
        # 如果时间转换失败，使用索引代替
        print(f"警告: 时间转换失败，使用索引代替")
        datetime_array = np.arange(len(time_array))
    
    in_event = False
    event_start = None
    
    # 按年份统计事件序号
    year_event_count = {}
    
    for i, event_val in enumerate(flood_event_series):
        # 检查是否为洪水事件 (> 0 且非NaN)
        is_flood_event = not np.isnan(event_val) and event_val > 0
        
        if is_flood_event and not in_event:
            # 洪水事件开始
            in_event = True
            event_start = i
        elif not is_flood_event and in_event:
            # 洪水事件结束
            in_event = False
            duration = i - event_start
            if duration >= min_duration:
                # 获取事件开始时间的年份
                if isinstance(datetime_array[event_start], datetime):
                    year = datetime_array[event_start].year
                else:
                    # 如果是索引，估算年份 (假设从1980年开始，每年8*365.25/3个时间步)
                    year = 1980 + int(event_start / (8 * 365.25))
                
                # 统计该年份的事件序号
                if year not in year_event_count:
                    year_event_count[year] = 0
                year_event_count[year] += 1
                
                events.append({
                    'basin_id': basin_id,
                    'year': year,
                    'event_number': year_event_count[year],
                    'start_idx': event_start,
                    'end_idx': i - 1,
                    'duration': duration,
                    'start_time': datetime_array[event_start],
                    'end_time': datetime_array[i-1]
                })
    
    # 处理持续到最后的洪水事件
    if in_event:
        duration = len(flood_event_series) - event_start
        if duration >= min_duration:
            if isinstance(datetime_array[event_start], datetime):
                year = datetime_array[event_start].year
            else:
                year = 1980 + int(event_start / (8 * 365.25))
            
            if year not in year_event_count:
                year_event_count[year] = 0
            year_event_count[year] += 1
            
            events.append({
                'basin_id': basin_id,
                'year': year,
                'event_number': year_event_count[year],
                'start_idx': event_start,
                'end_idx': len(flood_event_series) - 1,
                'duration': duration,
                'start_time': datetime_array[event_start],
                'end_time': datetime_array[-1]
            })
    
    return events

def calculate_metrics(obs, pred):
    """计算评估指标"""
    obs = np.array(obs).flatten()
    pred = np.array(pred).flatten()
    
    # 去除NaN值
    valid_mask = ~(np.isnan(obs) | np.isnan(pred))
    obs_valid = obs[valid_mask]
    pred_valid = pred[valid_mask]
    
    if len(obs_valid) == 0:
        return {"error": "No valid data points"}
    
    metrics = {}
    
    # NSE
    obs_mean = np.mean(obs_valid)
    sse = np.sum((obs_valid - pred_valid) ** 2)
    sst = np.sum((obs_valid - obs_mean) ** 2)
    metrics['NSE'] = 1 - (sse / sst) if sst != 0 else float('-inf')
    
    # R²
    corr_matrix = np.corrcoef(obs_valid, pred_valid)
    metrics['R2'] = corr_matrix[0, 1] ** 2 if not np.isnan(corr_matrix[0, 1]) else 0
    
    # RMSE
    metrics['RMSE'] = np.sqrt(np.mean((obs_valid - pred_valid) ** 2))
    
    # MAE
    metrics['MAE'] = np.mean(np.abs(obs_valid - pred_valid))
    
    return metrics

def plot_single_flood_event(obs_inflow, pred_inflow, event, basin_name, save_dir):
    """绘制单个洪水事件的对比图"""
    
    start_idx = event['start_idx']
    end_idx = event['end_idx']
    year = event['year']
    event_number = event['event_number']
    
    # 提取事件数据
    obs_event = obs_inflow[start_idx:end_idx+1]
    pred_event = pred_inflow[start_idx:end_idx+1]
    
    # 计算指标
    metrics = calculate_metrics(obs_event, pred_event)
    
    if 'error' in metrics:
        print(f"  警告: {basin_name} {year}年第{event_number}场洪水数据无效")
        return
    
    # 创建图片
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 绘制时间序列
    time_steps = range(len(obs_event))
    ax.plot(time_steps, obs_event, 'b-', linewidth=2.5, label='Observed', alpha=0.8)
    ax.plot(time_steps, pred_event, 'r--', linewidth=2.5, label='Predicted', alpha=0.8)
    
    # 设置标题和标签
    title = f'{basin_name} - {year} Flood Event #{event_number}'
    title += f'\nDuration: {event["duration"]} timesteps (3-hour intervals)'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.set_xlabel('Time Steps (3-hour intervals)', fontsize=12)
    ax.set_ylabel('Inflow', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 添加指标信息
    info_text = f'NSE: {metrics["NSE"]:.3f}\n'
    info_text += f'R²: {metrics["R2"]:.3f}\n' 
    info_text += f'RMSE: {metrics["RMSE"]:.3f}\n'
    info_text += f'MAE: {metrics["MAE"]:.3f}'
    
    # 添加峰值信息
    obs_peak = np.max(obs_event)
    pred_peak = np.max(pred_event)
    bias = pred_peak - obs_peak
    
    peak_text = f'Obs Peak: {obs_peak:.2f}\n'
    peak_text += f'Pred Peak: {pred_peak:.2f}\n'
    peak_text += f'Peak Bias: {bias:+.2f}'
    
    # 将信息框放在图的左上角和右上角
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
           verticalalignment='top', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    ax.text(0.98, 0.98, peak_text, transform=ax.transAxes, 
           verticalalignment='top', horizontalalignment='right', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    
    # 保存图片 (文件名格式: 年份_第几场洪水.png)
    filename = f"{year}_{event_number}.png"
    save_path = save_dir / filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    已保存: {filename} (NSE: {metrics['NSE']:.3f})")
    
    return metrics

def plot_flood_events_by_basin(obs_data, pred_data, save_dir):
    """按流域绘制所有洪水事件"""
    
    obs_inflow = obs_data.get('inflow')
    pred_inflow = pred_data.get('inflow')
    flood_event_var = obs_data.get('flood_event')
    time_array = obs_data.get('time')
    basin_names = obs_data.get('basin', None)
    
    if obs_inflow is None or pred_inflow is None or flood_event_var is None:
        print("缺少必要数据！")
        return
    
    # 处理流域名称
    if basin_names is not None:
        try:
            basin_names = [name.decode('utf-8') if isinstance(name, bytes) else str(name) 
                          for name in basin_names]
        except:
            basin_names = [f"Basin_{i}" for i in range(obs_inflow.shape[0])]
    else:
        basin_names = [f"Basin_{i}" for i in range(obs_inflow.shape[0])]
    
    print(f"处理 {len(basin_names)} 个流域:")
    for i, name in enumerate(basin_names):
        print(f"  {i}: {name}")
    
    # 创建主绘图文件夹
    plots_dir = save_dir / "flood_events_plots"
    plots_dir.mkdir(exist_ok=True)
    
    all_metrics = []
    
    # 处理每个流域
    for basin_id in range(len(basin_names)):
        basin_name = basin_names[basin_id]
        print(f"\n处理流域: {basin_name}")
        
        # 创建流域子文件夹
        basin_dir = plots_dir / basin_name.replace(' ', '_')
        basin_dir.mkdir(exist_ok=True)
        
        # 识别该流域的洪水事件
        basin_flood_events = flood_event_var[basin_id, :]
        flood_events = identify_flood_events_with_year(
            basin_flood_events, time_array, basin_id
        )
        
        print(f"  发现 {len(flood_events)} 场洪水事件")
        
        if len(flood_events) == 0:
            print(f"  {basin_name} 没有洪水事件")
            continue
        
        # 按年份分组显示
        year_groups = {}
        for event in flood_events:
            year = event['year']
            if year not in year_groups:
                year_groups[year] = 0
            year_groups[year] += 1
        
        print(f"  洪水事件分布:")
        for year, count in sorted(year_groups.items()):
            print(f"    {year}年: {count}场")
        
        # 绘制每场洪水事件
        basin_metrics = []
        for event in flood_events:
            metrics = plot_single_flood_event(
                obs_inflow[basin_id, :], 
                pred_inflow[basin_id, :], 
                event, 
                basin_name, 
                basin_dir
            )
            
            if metrics and 'error' not in metrics:
                metrics['basin'] = basin_name
                metrics['year'] = event['year']
                metrics['event_number'] = event['event_number']
                metrics['duration'] = event['duration']
                basin_metrics.append(metrics)
                all_metrics.append(metrics)
        
        # 保存该流域的指标统计
        if basin_metrics:
            df = pd.DataFrame(basin_metrics)
            csv_path = basin_dir / f"{basin_name}_metrics.csv"
            df.to_csv(csv_path, index=False)
            
            # 打印流域总结
            avg_nse = df['NSE'].mean()
            avg_r2 = df['R2'].mean() 
            avg_rmse = df['RMSE'].mean()
            print(f"  流域平均指标: NSE={avg_nse:.3f}, R²={avg_r2:.3f}, RMSE={avg_rmse:.3f}")
    
    # 保存总体指标统计
    if all_metrics:
        total_df = pd.DataFrame(all_metrics)
        total_csv_path = plots_dir / "all_basins_metrics.csv"
        total_df.to_csv(total_csv_path, index=False)
        
        print(f"\n=== 总体统计 ===")
        print(f"总共分析了 {len(all_metrics)} 场洪水事件")
        print(f"平均NSE: {total_df['NSE'].mean():.3f}")
        print(f"平均R²: {total_df['R2'].mean():.3f}")
        print(f"平均RMSE: {total_df['RMSE'].mean():.3f}")
        
        # 按年份统计
        print(f"\n按年份统计:")
        year_stats = total_df.groupby('year').agg({
            'NSE': ['count', 'mean'],
            'R2': 'mean',
            'RMSE': 'mean'
        }).round(3)
        print(year_stats)
    
    print(f"\n所有图片已保存到: {plots_dir}")

def main():
    """主函数"""
    # 文件路径
    #result_dir = Path("/home/lizilin/code/torchhydro/results/events_train_3h/regional_weighted_mse_loss")
    result_dir = Path("/home/lizilin/code/torchhydro/results/gnn_experiment/songliao_3h_test_")
    obs_file = result_dir / "epoch60flow_obs.nc"
    pred_file = result_dir / "epoch60flow_pred.nc"
    
    print("=== 按年份和洪水序号绘制洪水事件分析图 ===")
    print(f"观测文件: {obs_file}")
    print(f"预测文件: {pred_file}")
    
    if not obs_file.exists():
        print(f"观测文件不存在: {obs_file}")
        return
    
    if not pred_file.exists():
        print(f"预测文件不存在: {pred_file}")
        return
    
    # 加载数据
    print("\n1. 加载数据...")
    obs_data, pred_data = load_data_with_time(obs_file, pred_file)
    
    if obs_data is None or pred_data is None:
        print("数据加载失败")
        return
    
    print("数据加载成功!")
    
    # 绘制洪水事件
    print("\n2. 分析和绘制洪水事件...")
    plot_flood_events_by_basin(obs_data, pred_data, result_dir)
    
    print("\n分析完成!")

if __name__ == "__main__":
    main()
