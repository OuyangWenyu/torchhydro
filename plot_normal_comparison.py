#!/usr/bin/env python3
"""
Normal mode comparison plotting script with denormalized data and rainfall
Compare observed values with normal mode predictions using original physical units

Paths:
- Observations: /home/lizilin/code/torchhydro/results/.../songliao_3h_test_normal/epoch60flow_obs.nc
- Predictions: /home/lizilin/code/torchhydro/results/.../songliao_3h_test_normal/epoch60flow_pred.nc
- Output: /home/lizilin/code/torchhydro/results/.../songliao_3h_test_normal/normal_comparison_plots/

Author: Assistant
Date: 2025-09-16
Updated: Added denormalization and rainfall-runoff plotting functionality
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import flood event analysis utilities
from flood_utils import (
    load_songliao_timeseries_data,
    load_basin_attributes,
    convert_flow_units_batch,
    find_flood_events_from_labels,
    align_flood_events_to_observations,
    extract_continuous_flood_periods,
    extract_period_statistics,
    assign_year_and_event_number,
    load_netcdf_data,
    calculate_metrics,
    find_flood_events_threshold_based
)

# Import rainfall-runoff plotting functionality
try:
    from hydroutils.hydro_plot import plot_rainfall_runoff
    HYDROUTILS_AVAILABLE = True
except ImportError:
    print("Warning: hydroutils not available, using simplified plotting")
    HYDROUTILS_AVAILABLE = False

try:
    from hydrodatasource.reader.data_source import SelfMadeHydroDataset
    HYDRODATASOURCE_AVAILABLE = True
except ImportError:
    print("Warning: hydrodatasource not available, rainfall data loading will be limited")
    HYDRODATASOURCE_AVAILABLE = False

# 禁用matplotlib的字体警告
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']

# Set font and figure styles - 使用默认字体避免警告
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def load_normal_comparison_data(base_dir, epoch="30", songliao_data_path="/mnt/data/ClassC/songliaorrevent"):
    """Load normal mode comparison data with optional songliao data"""
    base_dir = Path(base_dir)
    
    data_dir = base_dir / "songliao_3h_test_normal"

    # Prefer exact filenames when possible, but allow flexible epoch strings
    obs_path = data_dir / f"epoch{epoch}flow_obs.nc"
    pred_path = data_dir / f"epoch{epoch}flow_pred.nc"

    files = {
        'obs': obs_path,
        'normal': pred_path
    }

    # If exact files are missing, try to find files that contain the epoch string
    for name, path in list(files.items()):
        if not path.exists():
            try:
                if name == 'obs':
                    matches = list(data_dir.glob(f"*{epoch}*flow_obs*.nc"))
                else:
                    matches = list(data_dir.glob(f"*{epoch}*flow_pred*.nc"))
                if not matches:
                    # Fallback to any file containing the epoch substring
                    matches = list(data_dir.glob(f"*{epoch}*.nc"))
                if matches:
                    files[name] = matches[0]
            except Exception:
                pass
     
    # Check if files exist
    missing_files = []
    existing_files = []
    for name, path in files.items():
        if not path.exists():
            missing_files.append(f"{name}: {path}")
        else:
            existing_files.append(f"{name}: {path}")
    
    print("File status:")
    for file in existing_files:
        print(f"  ✓ {file}")
    
    if missing_files:
        print("Missing files:")
        for file in missing_files:
            print(f"  ✗ {file}")
    
    # Load data
    data = {}
    for name, path in files.items():
        if path.exists():
            print(f"Loading {name} data: {path}")
            data[name] = load_netcdf_data(path)
            if data[name] is None:
                print(f"Failed to load {name} data")
            else:
                # Print data structure for debugging
                print(f"  {name} variables: {list(data[name].keys())}")
                for var_name, var_data in data[name].items():
                    if hasattr(var_data, 'shape'):
                        print(f"    {var_name}: shape {var_data.shape}")
        else:
            print(f"Skipping missing file: {name}")
            data[name] = None
    
    # 设置松辽数据集路径
    data['songliao_data_path'] = songliao_data_path
    
    return data

def plot_rainfall_runoff_with_flood_events(time_data, rainfall_data, obs_flow_data, pred_flow_data, 
                                          flood_events_mask=None, fig_size=(16, 10), title="", 
                                          ylabel="Flow", prcp_ylabel="Precipitation", 
                                          obs_label="Observed", pred_label="Predicted"):
    """
    绘制降雨-径流图，包含洪水事件标识
    
    Parameters:
    -----------
    time_data : array-like
        时间序列
    rainfall_data : array-like
        降雨数据
    obs_flow_data : array-like
        观测径流数据
    pred_flow_data : array-like
        预测径流数据
    flood_events_mask : array-like, optional
        洪水事件掩码（True表示洪水事件时段）
    """
    fig, (ax_rain, ax_flow) = plt.subplots(2, 1, figsize=fig_size, 
                                          gridspec_kw={'height_ratios': [1, 3]}, 
                                          sharex=True)
    
    # 绘制降雨 (上图，倒置)
    if rainfall_data is not None:
        # 自动计算合适的柱子宽度，避免重叠
        if len(time_data) > 1:
            try:
                # 计算时间间隔
                if hasattr(time_data[0], 'timestamp'):  # datetime对象
                    time_diff = (time_data[1] - time_data[0]).total_seconds() / 3600  # 转换为小时
                    bar_width = time_diff * 0.8 / 24  # 调整为matplotlib的时间单位
                else:
                    # 假设时间间隔均匀，使用更小的宽度避免重叠
                    time_diff = time_data[1] - time_data[0] 
                    bar_width = time_diff * 0.6  # 60%宽度，避免重叠
            except:
                bar_width = 0.6  # 默认较小宽度
        else:
            bar_width = 0.6  # 默认较小宽度
        
        print(f"Using bar width: {bar_width}")
        
        # 分别绘制洪水期间和非洪水期间的降雨，避免重叠
        if flood_events_mask is not None and len(flood_events_mask) == len(rainfall_data):
            # 创建颜色数组，根据洪水事件标记使用不同颜色
            colors = ['orange' if is_flood else 'skyblue' for is_flood in flood_events_mask]
            
            # 单次绘制，避免重叠
            bars = ax_rain.bar(time_data, rainfall_data, color=colors, alpha=0.7, 
                              width=bar_width, edgecolor='none')
            
            # 添加图例
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='skyblue', alpha=0.7, label='Normal Period'),
                Patch(facecolor='orange', alpha=0.7, label='Flood Event Period')
            ]
            ax_rain.legend(handles=legend_elements, loc='upper right', fontsize=12)
        else:
            # 没有洪水标识时，全部用浅蓝色
            ax_rain.bar(time_data, rainfall_data, color='skyblue', alpha=0.7, 
                       width=bar_width, edgecolor='none', label='Precipitation')
            ax_rain.legend(loc='upper right', fontsize=12)
        
        ax_rain.set_ylabel(prcp_ylabel, fontsize=14)  # 增大Y轴标签字号
        ax_rain.invert_yaxis()  # 倒置Y轴
        ax_rain.grid(True, alpha=0.3)
        ax_rain.tick_params(axis='both', which='major', labelsize=12)  # 增大刻度标签字号
    
    # 绘制径流 (下图)
    # 观测径流
    if obs_flow_data is not None:
        ax_flow.plot(time_data, obs_flow_data, color='black', linestyle='-', 
                    linewidth=2, label=obs_label, alpha=0.8)
    
    # 预测径流
    if pred_flow_data is not None:
        ax_flow.plot(time_data, pred_flow_data, color='blue', linestyle='--', 
                    linewidth=2, label=pred_label, alpha=0.8)
    
    # 高亮洪水事件期间
    if flood_events_mask is not None and len(flood_events_mask) == len(time_data):
        # 找到连续的洪水事件区间
        flood_periods = []
        start_idx = None
        for i, is_flood in enumerate(flood_events_mask):
            if is_flood and start_idx is None:
                start_idx = i
            elif not is_flood and start_idx is not None:
                flood_periods.append((start_idx, i-1))
                start_idx = None
        if start_idx is not None:  # 处理到末尾的情况
            flood_periods.append((start_idx, len(flood_events_mask)-1))
        
        # 绘制洪水事件背景 - 只在径流图中绘制一次
        for start, end in flood_periods:
            if start < len(time_data) and end < len(time_data):
                ax_flow.axvspan(time_data[start], time_data[end], 
                               alpha=0.2, color='red', label='Flood Event' if start == flood_periods[0][0] else "")
    
    ax_flow.set_ylabel(ylabel, fontsize=14)  # 增大Y轴标签字号
    ax_flow.set_xlabel('Time', fontsize=14)  # 增大X轴标签字号
    ax_flow.legend(loc='upper right', fontsize=12)  # 增大图例字号
    ax_flow.grid(True, alpha=0.3)
    ax_flow.tick_params(axis='both', which='major', labelsize=12)  # 增大刻度标签字号
    
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold')  # 增大标题字号
    
    plt.tight_layout()
    return fig, ax_flow

def plot_normal_comparison_period_with_rainfall(obs_flow, pred_flow, period, basin_names, 
                                               time_array, save_dir, songliao_data_path=None, flow_unit="m³/s"):
    """
    绘制带降雨的正常模式对比图 - 使用松辽流域数据集
    """
    
    basin_idx = period['basin']
    start_idx = max(0, period['start'] - 48)  # 扩展时间窗口，显示事件前后
    end_idx = min(obs_flow.shape[1], period['end'] + 48)
    
    # 提取时间和流量数据
    if time_array is not None:
        time_subset = time_array[start_idx:end_idx]
        try:
            import pandas as pd
            time_subset_dt = pd.to_datetime(time_subset)
            use_datetime = True
        except:
            time_subset_dt = time_subset
            use_datetime = False
    else:
        time_subset_dt = range(start_idx, end_idx)
        use_datetime = False
    
    obs_subset = obs_flow[basin_idx, start_idx:end_idx]
    pred_subset = pred_flow[basin_idx, start_idx:end_idx]
    
    # 尝试加载降雨数据和观测径流数据
    rainfall_data = None
    obs_inflow_data = None
    flood_events_mask = None
    
    if basin_names is not None:
        try:
            basin_id = basin_names[basin_idx]
            if use_datetime and len(time_subset_dt) > 0:
                time_range = (time_subset_dt[0], time_subset_dt[-1])
            else:
                time_range = None
            
            # 加载松辽流域完整时间序列数据
            songliao_data = load_songliao_timeseries_data(basin_id, time_range)
            if songliao_data is not None:
                # 对齐时间序列到预测数据的时间范围
                if use_datetime:
                    songliao_data = songliao_data.set_index('time')
                    
                    # 创建时间索引，处理时区和格式差异
                    target_times = pd.to_datetime(time_subset_dt)
                    
                    # 尝试重新索引到目标时间
                    aligned_data = songliao_data.reindex(target_times, method='nearest', tolerance='1H')
                    
                    if not aligned_data.empty:
                        rainfall_data = aligned_data['rain'].values
                        if 'inflow' in aligned_data.columns:
                            obs_inflow_data = aligned_data['inflow'].values
                        if 'flood_event' in aligned_data.columns:
                            flood_events_mask = aligned_data['flood_event'].values == 1
                            
                        # 确保所有数组长度一致
                        min_len = min(len(rainfall_data), len(time_subset_dt), len(obs_subset), len(pred_subset))
                        if len(rainfall_data) != len(time_subset_dt):
                            print(f"Length mismatch - adjusting to {min_len} points")
                            rainfall_data = rainfall_data[:min_len]
                            if obs_inflow_data is not None:
                                obs_inflow_data = obs_inflow_data[:min_len]
                            if flood_events_mask is not None:
                                flood_events_mask = flood_events_mask[:min_len]
                            time_subset_dt = time_subset_dt[:min_len]
                            obs_subset = obs_subset[:min_len]
                            pred_subset = pred_subset[:min_len]
                            
                        print(f"Successfully aligned songliao data: {len(rainfall_data)} points")
                    else:
                        print("Failed to align songliao data to prediction time range")
                else:
                    # 简单截取相同长度
                    min_len = min(len(songliao_data), len(time_subset_dt))
                    rainfall_data = songliao_data['rain'].values[:min_len]
                    if 'inflow' in songliao_data.columns:
                        obs_inflow_data = songliao_data['inflow'].values[:min_len]
                    if 'flood_event' in songliao_data.columns:
                        flood_events_mask = songliao_data['flood_event'].values[:min_len] == 1
                        
                    # 调整其他数组长度
                    time_subset_dt = time_subset_dt[:min_len]
                    obs_subset = obs_subset[:min_len]
                    pred_subset = pred_subset[:min_len]
                    
                print(f"Loaded songliao data - Rain: {rainfall_data is not None}, "
                     f"Obs inflow: {obs_inflow_data is not None}, "
                     f"Flood events: {flood_events_mask is not None}")
        except Exception as e:
            print(f"Failed to load songliao data: {e}")
    
    # 创建图形
    if HYDROUTILS_AVAILABLE and rainfall_data is not None:
        try:
            # 使用hydroutils的专业绘图功能
            flow_list = [obs_subset, pred_subset]
            leg_list = ["Observed", "Predicted"]
            color_list = ["black", "blue"]
            linestyle_list = ["-", "--"]
            
            title_name = f'{basin_names[basin_idx] if basin_names is not None else f"Basin_{basin_idx}"} - Flood Event: {period.get("event_name", "Unknown")}'
            ylabel = f"Flow ({flow_unit})"
            prcp_ylabel = "Precipitation (mm/3h)"
            
            fig, ax = plot_rainfall_runoff(
                time_subset_dt,
                rainfall_data,
                flow_list,
                fig_size=(16, 10),
                title=title_name,
                ylabel=ylabel,
                prcp_ylabel=prcp_ylabel,
                leg_lst=leg_list,
                linewidth=2,
                color_list=color_list,
                linestyle_list=linestyle_list,
            )
        except Exception as e:
            print(f"hydroutils plotting failed, using custom plotting: {e}")
            # 回退到自定义绘图
            title_name = f'{basin_names[basin_idx] if basin_names is not None else f"Basin_{basin_idx}"} - {period.get("event_name", "Unknown")}'
            ylabel = f"Flow ({flow_unit})"
            prcp_ylabel = "Precipitation (mm/3h)"
            
            # 使用观测径流数据（如果可用）或者预测数据中的观测部分
            obs_flow_for_plot = obs_inflow_data if obs_inflow_data is not None else obs_subset
            
            fig, ax = plot_rainfall_runoff_with_flood_events(
                time_subset_dt, rainfall_data, obs_flow_for_plot, pred_subset,
                flood_events_mask=flood_events_mask,
                title=title_name, ylabel=ylabel, prcp_ylabel=prcp_ylabel,
                obs_label="Observed", pred_label="Predicted"
            )
    else:
        # 使用自定义绘图
        title_name = f'{basin_names[basin_idx] if basin_names is not None else f"Basin_{basin_idx}"} - {period.get("event_name", "Unknown")}'
        ylabel = f"Flow ({flow_unit})"
        prcp_ylabel = "Precipitation (mm/3h)"
        
        # 使用观测径流数据（如果可用）或者预测数据中的观测部分
        obs_flow_for_plot = obs_inflow_data if obs_inflow_data is not None else obs_subset
        
        fig, ax = plot_rainfall_runoff_with_flood_events(
            time_subset_dt, rainfall_data, obs_flow_for_plot, pred_subset,
            flood_events_mask=flood_events_mask,
            title=title_name, ylabel=ylabel, prcp_ylabel=prcp_ylabel,
            obs_label="Observed", pred_label="Predicted"
        )
    
    # 计算评价指标
    metrics = calculate_metrics(obs_subset, pred_subset)
    
    # 添加指标文本框
    if 'error' not in metrics:
        metrics_text = f"Evaluation Metrics:\n"
        metrics_text += f"NSE: {metrics['NSE']:.3f}\n"
        metrics_text += f"R²: {metrics['R2']:.3f}\n"  
        metrics_text += f"RMSE: {metrics['RMSE']:.2f}\n"
        metrics_text += f"MAE: {metrics['MAE']:.2f}\n"
        metrics_text += f"Peak Error: {metrics['peak_error_pct']:.1f}%\n"
        metrics_text += f"Duration: {period['duration']} timesteps"
        
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=12,  # 增大文本框字号
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 注意：洪水事件高亮已经在plot_rainfall_runoff_with_flood_events中绘制了
    # 这里不再重复绘制，避免重叠
    
    # 设置x轴格式
    if use_datetime:
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    # 保存图片
    basin_name = basin_names[basin_idx].replace(' ', '_') if basin_names is not None else f"Basin_{basin_idx}"
    event_name = period.get('event_name', f"period_{period['start']}_{period['end']}")
    filename = f"{basin_name}_{event_name}.png"
    filepath = save_dir / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return metrics

def plot_normal_comparison(data, save_dir, epoch="30", songliao_data_path="/mnt/data/ClassC/songliaorrevent"):
    """Plot normal mode comparison analysis with denormalized data and rainfall"""
    
    save_dir = Path(save_dir)
    # Sanitize epoch string for use in folder names (remove extensions like .pth)
    try:
        epoch_folder = Path(str(epoch)).stem
    except Exception:
        epoch_folder = str(epoch)

    # Create an epoch-specific subfolder under the existing songliao_3h_test_normal directory
    plots_dir = save_dir / f"epoch{epoch_folder}" / "normal_comparison_plots"
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    # Extract data
    obs_data = data['obs']
    pred_data = data['normal']
    
    if obs_data is None or pred_data is None:
        print("ERROR: Missing observation or prediction data")
        return
    
    # 尝试找到流量数据的变量名
    flow_var_names = ['inflow', 'flow', 'streamflow', 'discharge', 'qobs', 'qsim']
    
    # 提取观测数据
    obs_flow = None
    for var_name in flow_var_names:
        if var_name in obs_data:
            obs_flow = obs_data[var_name]
            print(f"Found observation flow data with variable name: {var_name}")
            break
    
    if obs_flow is None:
        print(f"Available variables in obs data: {list(obs_data.keys())}")
        # 使用第一个数据变量
        first_var = list(obs_data.keys())[0]
        obs_flow = obs_data[first_var]
        print(f"Using first available variable for observations: {first_var}")
    
    # 提取预测数据
    pred_flow = None
    for var_name in flow_var_names:
        if var_name in pred_data:
            pred_flow = pred_data[var_name]
            print(f"Found prediction flow data with variable name: {var_name}")
            break
    
    if pred_flow is None:
        print(f"Available variables in pred data: {list(pred_data.keys())}")
        # 使用第一个数据变量
        first_var = list(pred_data.keys())[0]
        pred_flow = pred_data[first_var]
        print(f"Using first available variable for predictions: {first_var}")
    
    # 检查数据形状
    print(f"Obs flow shape: {obs_flow.shape}")
    print(f"Pred flow shape: {pred_flow.shape}")
    
    if obs_flow.shape != pred_flow.shape:
        print(f"WARNING: Shape mismatch between observations {obs_flow.shape} and predictions {pred_flow.shape}")
        return
    
    # 获取流域名称和时间数组
    time_array = obs_data.get('time')
    basin_names = obs_data.get('basin')
    
    # Process basin names
    if basin_names is not None:
        try:
            if hasattr(basin_names[0], 'decode'):
                basin_names = [name.decode('utf-8') if hasattr(name, 'decode') else str(name) 
                              for name in basin_names]
        except:
            basin_names = [f"Basin_{i}" for i in range(obs_flow.shape[0])]
    else:
        basin_names = [f"Basin_{i}" for i in range(obs_flow.shape[0])]
    
    # 加载流域面积并进行单位转换（mm/3h → m³/s）
    if basin_names is not None:
        print("Loading basin areas for unit conversion...")
        basin_areas = {}
        for i, basin_id in enumerate(basin_names):
            attrs = load_basin_attributes(basin_id, songliao_data_path)
            if attrs is not None and 'area' in attrs:
                basin_areas[i] = attrs['area']
                print(f"  {basin_id}: {attrs['area']:.3f} km²")
            else:
                basin_areas[i] = None
                print(f"  {basin_id}: area not available")
        
        # 应用单位转换到所有流域（convert_flow_units_batch 返回 (data, unit)）
        obs_flow_converted, unit_obs = convert_flow_units_batch(obs_flow, basin_areas, time_unit="3h")
        pred_flow_converted, unit_pred = convert_flow_units_batch(pred_flow, basin_areas, time_unit="3h")
        
        obs_flow = obs_flow_converted
        pred_flow = pred_flow_converted
        # 使用转换后的单位（通常为 m³/s）
        flow_unit = unit_obs
    else:
        print("No basin names available, using original units")
        flow_unit = "mm/3h"
    
    print(f"Final data range:")
    print(f"  Obs data range: {np.nanmin(obs_flow):.3f} - {np.nanmax(obs_flow):.3f} {flow_unit}")
    print(f"  Pred data range: {np.nanmin(pred_flow):.3f} - {np.nanmax(pred_flow):.3f} {flow_unit}")
    
    print(f"Processing normal mode comparison for {len(basin_names)} basins:")
    for i, name in enumerate(basin_names):
        print(f"  {i}: {name}")
    
    # === 使用新的基于标签的洪水事件检测 ===
    print("\n=== 使用松辽数据集标签识别洪水事件 ===")
    periods = find_flood_events_from_labels(
        obs_flow, time_array, basin_names, songliao_data_path
    )
    
    print(f"\n基于松辽标签识别到 {len(periods)} 个洪水事件")
    
    if len(periods) == 0:
        print("未发现标记的洪水事件")
        return
    
    # 按流域分组显示结果
    basin_event_summary = {}
    for period in periods:
        basin_id = period['basin_id']
        if basin_id not in basin_event_summary:
            basin_event_summary[basin_id] = []
        basin_event_summary[basin_id].append(period['event_name'])
    
    print("\n各流域洪水事件汇总:")
    for basin_id, events in basin_event_summary.items():
        print(f"  {basin_id}: {', '.join(events)}")
    
    # Plot each period
    all_metrics = []
    
    for period in periods:
        basin_display_name = period['basin_id']
        print(f"\n正在绘制 {basin_display_name} - {period['event_name']}")
        
        # Create basin subfolder
        basin_name = period['basin_id'].replace(' ', '_')
        basin_dir = plots_dir / basin_name
        basin_dir.mkdir(exist_ok=True)
        
        metrics = plot_normal_comparison_period_with_rainfall(
            obs_flow, pred_flow, period, basin_names, time_array, basin_dir, 
            songliao_data_path=songliao_data_path, flow_unit=flow_unit
        )
        
        # Add period information to metrics
        if 'error' not in metrics:
            metrics.update({
                'basin': basin_display_name,
                'event_name': period['event_name'],
                'period_start': period['start'],
                'period_end': period['end'],
                'duration': period['duration'],
                'peak_value': period['peak_value'],
                'year': period['year'],
                'event_number': period['event_number'],
                'flood_source': period['flood_source']
            })
            all_metrics.append(metrics)
    
    # Save overall metrics statistics
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        csv_path = plots_dir / "normal_comparison_metrics.csv"
        df.to_csv(csv_path, index=False)
        
        # 生成总结报告
        print(f"\n=== 基于松辽标签的洪水事件分析结果 ===")
        print(f"总事件数: {len(all_metrics)}")
        print(f"涉及流域数: {len(set(df['basin']))}")
        
        # 按年份统计
        if 'year' in df.columns:
            year_summary = df.groupby('year').size()
            print(f"\n按年份统计:")
            for year, count in year_summary.items():
                print(f"  {year}年: {count} 个事件")
        
        # 整体评价指标
        avg_nse = df['NSE'].mean()
        avg_r2 = df['R2'].mean()
        avg_rmse = df['RMSE'].mean()
        avg_mae = df['MAE'].mean()
        avg_peak_error = df['peak_error_pct'].mean()
        
        print(f"\n整体平均指标:")
        print(f"NSE: {avg_nse:.3f} ± {df['NSE'].std():.3f}")
        print(f"R²: {avg_r2:.3f} ± {df['R2'].std():.3f}")
        print(f"RMSE: {avg_rmse:.3f} ± {df['RMSE'].std():.3f}")
        print(f"MAE: {avg_mae:.3f} ± {df['MAE'].std():.3f}")
        print(f"Peak Error: {avg_peak_error:.1f}% ± {df['peak_error_pct'].std():.1f}%")
        
        # Statistics by basin
        basin_stats = df.groupby('basin').agg({
            'NSE': ['count', 'mean', 'std'],
            'R2': ['mean', 'std'],
            'RMSE': ['mean', 'std'],
            'peak_error_pct': ['mean', 'std']
        }).round(3)
        
        print(f"\nPerformance by basin:")
        print(basin_stats)
        
        # Generate summary report
        summary_text = f"""基于松辽标签的洪水事件分析总结报告
        
分析事件总数: {len(all_metrics)}
涉及流域数: {len(set(df['basin']))}
数据来源: 松辽数据集 flood_event 标签

整体平均指标:
NSE: {avg_nse:.3f} ± {df['NSE'].std():.3f}
R²: {avg_r2:.3f} ± {df['R2'].std():.3f}
RMSE: {avg_rmse:.3f} ± {df['RMSE'].std():.3f}
MAE: {avg_mae:.3f} ± {df['MAE'].std():.3f}
Peak Error: {avg_peak_error:.1f}% ± {df['peak_error_pct'].std():.1f}%

按流域分析结果:
"""
        for basin in set(df['basin']):
            basin_data = df[df['basin'] == basin]
            basin_avg_nse = basin_data['NSE'].mean()
            basin_avg_r2 = basin_data['R2'].mean()
            basin_avg_rmse = basin_data['RMSE'].mean()
            basin_avg_peak_error = basin_data['peak_error_pct'].mean()
            basin_n_events = len(basin_data)
            
            summary_text += f"""
{basin} ({basin_n_events} 个事件):
  NSE: {basin_avg_nse:.3f}
  R²: {basin_avg_r2:.3f}
  RMSE: {basin_avg_rmse:.3f}
  Peak Error: {basin_avg_peak_error:.1f}%
"""
        
        # Save summary
        summary_path = plots_dir / "normal_comparison_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary_text)
    
    print(f"\n所有基于松辽标签的洪水事件分析结果已保存到: {plots_dir}")
    print("📊 分析内容包括:")
    print("   - 基于 flood_event==1 标签的洪水事件识别")
    print("   - 按年份和流域自动编号的事件命名")
    print("   - 降雨-径流对比图（带洪水事件高亮）")
    print("   - 详细的评价指标统计")


def main():
    """Main function to run the normal comparison analysis"""
    import argparse
    
    # 参数解析
    parser = argparse.ArgumentParser(description='Normal mode comparison analysis with flood events')
    parser.add_argument('--base_dir', '-d', default='/home/lizilin/code/torchhydro/results/3d_gnn_only/',
                        help='Base directory containing model results')
    parser.add_argument('--epoch', '-e', type=str, default='40',
                        help='Epoch identifier to analyze (string allowed, e.g. "40" or "best_model.pth")')
    parser.add_argument('--songliao_path', '-s', default='/mnt/data/ClassC/songliaorrevent',
                        help='Path to Songliao dataset')
    
    args = parser.parse_args()
    
    print(f"Loading normal mode comparison data from: {args.base_dir}")
    print(f"Using epoch: {args.epoch}")
    print(f"Songliao data path: {args.songliao_path}")
    
    # Load data
    data = load_normal_comparison_data(args.base_dir, epoch=args.epoch, songliao_data_path=args.songliao_path)
    
    if data['obs'] is None or data['normal'] is None:
        print("ERROR: Failed to load required data files")
        return
    
    # Generate plots
    # Use the standard songliao_3h_test_normal folder and let the plotting function
    # create an epoch subfolder inside it (epoch{epoch}/normal_comparison_plots)
    save_dir = Path(args.base_dir) / "songliao_3h_test_normal"
    plot_normal_comparison(data, save_dir, epoch=args.epoch, songliao_data_path=args.songliao_path)


if __name__ == "__main__":
    main()
