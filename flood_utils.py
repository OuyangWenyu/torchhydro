#!/usr/bin/env python3
"""
洪水事件分析工具函数模块

包含洪水事件检测、数据加载、单位转换等工具函数
分离自 plot_normal_comparison.py，提高代码模块化程度

Author: Assistant
Date: 2025-09-17
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import xarray as xr


def load_songliao_timeseries_data(basin_id, time_range=None, base_path="/mnt/data/ClassC/songliaorrevent"):
    """
    从松辽流域数据集加载时间序列数据（降雨、径流、洪水标签）
    
    Parameters:
    -----------
    basin_id : str
        流域ID（如 'songliao_20800900'）
    time_range : tuple, optional
        时间范围 (start_datetime, end_datetime)
    base_path : str
        数据集根目录路径
        
    Returns:
    --------
    pd.DataFrame : 包含time, rain, inflow, flood_event列的数据框
    """
    try:
        # 构建时间序列数据文件路径
        timeseries_path = os.path.join(base_path, "timeseries", "3h", f"{basin_id}.csv")
        
        if not os.path.exists(timeseries_path):
            print(f"Timeseries data file not found: {timeseries_path}")
            return None
            
        # 读取时间序列数据
        df = pd.read_csv(timeseries_path, parse_dates=["time"])
        print(f"Loaded timeseries data for {basin_id}: {len(df)} records")
        # print(f"Available columns: {list(df.columns)}")
        
        # 检查必需的列
        required_cols = ['time', 'rain', 'inflow', 'flood_event']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Warning: Missing columns {missing_cols}")
            # 尝试使用相似的列名
            col_mapping = {}
            for missing_col in missing_cols:
                if missing_col == 'rain':
                    rain_cols = [col for col in df.columns if 'rain' in col.lower() or 'prec' in col.lower()]
                    if rain_cols:
                        col_mapping['rain'] = rain_cols[0]
                elif missing_col == 'inflow':
                    flow_cols = [col for col in df.columns if 'flow' in col.lower() or 'discharge' in col.lower()]
                    if flow_cols:
                        col_mapping['inflow'] = flow_cols[0]
                elif missing_col == 'flood_event':
                    flood_cols = [col for col in df.columns if 'flood' in col.lower() or 'event' in col.lower()]
                    if flood_cols:
                        col_mapping['flood_event'] = flood_cols[0]
            
            # 重命名列
            if col_mapping:
                df = df.rename(columns=col_mapping)
                print(f"Renamed columns: {col_mapping}")
        
        # 时间范围筛选
        if time_range is not None:
            time_start = pd.to_datetime(time_range[0])
            time_end = pd.to_datetime(time_range[1])
            df = df[(df["time"] >= time_start) & (df["time"] <= time_end)]
            print(f"Filtered to time range {time_start} - {time_end}: {len(df)} records")
        
        # 确保包含基本列
        final_cols = ['time']
        if 'rain' in df.columns:
            final_cols.append('rain')
        else:
            df['rain'] = 0.0  # 默认无降雨
            final_cols.append('rain')
            
        if 'inflow' in df.columns:
            final_cols.append('inflow')
        else:
            print("Warning: No inflow data found")
            
        if 'flood_event' in df.columns:
            final_cols.append('flood_event')
        else:
            df['flood_event'] = 0  # 默认无洪水事件
            final_cols.append('flood_event')
        
        return df[final_cols]
        
    except Exception as e:
        print(f"Failed to load songliao timeseries data: {e}")
        return None


def load_basin_attributes(basin_id, base_path="/mnt/data/ClassC/songliaorrevent"):
    """
    加载流域属性信息
    
    Parameters:
    -----------
    basin_id : str
        流域ID
    base_path : str
        数据集根目录路径
        
    Returns:
    --------
    dict : 流域属性信息
    """
    try:
        # 构建属性文件路径
        attr_file = os.path.join(base_path, "attributes", "attributes.csv")
        
        if not os.path.exists(attr_file):
            print(f"Attributes file not found: {attr_file}")
            return None
            
        # 读取属性文件
        df = pd.read_csv(attr_file)
        print(f"Loaded attributes file with {len(df)} basins")
        
        # 查找对应的流域 - 使用basin_id列
        if 'basin_id' in df.columns:
            basin_data = df[df['basin_id'] == basin_id]
        else:
            print(f"Error: 'basin_id' column not found in attributes file")
            print(f"Available columns: {list(df.columns)}")
            return None
        
        if len(basin_data) > 0:
            attrs = basin_data.iloc[0].to_dict()
            
            # 特别检查area属性
            if 'area' in attrs:
                area_km2 = attrs['area']
                print(f"Loaded attributes for {basin_id}: area={area_km2:.3f} km²")
                return attrs
            else:
                print(f"Warning: 'area' not found in attributes for {basin_id}")
                print(f"Available attributes: {list(attrs.keys())}")
                return attrs
        else:
            print(f"Basin {basin_id} not found in attributes file")
            # 显示前几个basin_id作为参考
            if len(df) > 0:
                print(f"Sample basin_ids in file: {df['basin_id'].head().tolist()}")
            return None
        
    except Exception as e:
        print(f"Failed to load basin attributes: {e}")
        import traceback
        traceback.print_exc()
        return None


def convert_flow_units_batch(flow_data, basin_areas, time_unit="3h"):
    """
    批量转换流量单位：mm/时间单位 -> m³/s
    
    Parameters:
    -----------
    flow_data : np.ndarray
        流量数据，形状为 (n_basins, n_timesteps)
    basin_areas : dict
        流域面积字典，键为流域索引，值为面积(km²)
    time_unit : str
        时间单位（如'3h', '1D', '1h'）
    
    Returns:
    --------
    np.ndarray : 转换后的流量数据
    str : 单位字符串
    """
    
    if flow_data.ndim == 1:
        flow_data = flow_data.reshape(1, -1)
    
    n_basins, n_timesteps = flow_data.shape
    converted_data = np.copy(flow_data)
    
    # 确定时间转换因子
    if time_unit == "1D" or time_unit == "1d":
        time_factor = 24  # mm/day -> mm/h
        unit = "m³/s"
    elif time_unit == "3h" or time_unit == "3H":
        time_factor = 3   # mm/3h -> mm/h
        unit = "m³/s"
    elif time_unit == "1h" or time_unit == "1H":
        time_factor = 1   # mm/h -> mm/h
        unit = "m³/s"
    else:
        print(f"Warning: Unknown time unit {time_unit}, keeping original units")
        return flow_data, f"mm/{time_unit}"
    
    areas_converted = 0
    areas_missing = 0
    
    for basin_idx in range(n_basins):
        if basin_idx in basin_areas and basin_areas[basin_idx] is not None:
            area_km2 = basin_areas[basin_idx]
            # 转换公式: mm/时间单位 -> m³/s
            # mm/h * km² * (1000 m/km) * (1000 m/km) * (1 m/1000 mm) / (3600 s/h) 
            # = mm/h * km² / 3.6
            converted_data[basin_idx, :] = flow_data[basin_idx, :] / time_factor * area_km2 / 3.6
            areas_converted += 1
        else:
            # 保持原始单位，不转换
            areas_missing += 1
    
    print(f"Unit conversion summary:")
    print(f"  Converted {areas_converted} basins from mm/{time_unit} to {unit}")
    print(f"  {areas_missing} basins kept in original units (area not available)")
    
    if areas_converted > 0:
        return converted_data, unit
    else:
        return flow_data, f"mm/{time_unit}"


def find_flood_events_from_labels(obs_flow, time_array, basin_names, songliao_data_path="/mnt/data/ClassC/songliaorrevent"):
    """
    基于松辽数据集的flood_event标签提取洪水事件
    
    Parameters:
    -----------
    obs_flow : np.ndarray
        观测流量数据，形状为 (n_basins, n_time)
    time_array : np.ndarray
        时间数组，与obs_flow的时间维度对应
    basin_names : list
        流域名称列表
    songliao_data_path : str
        松辽数据集路径
    
    Returns:
    --------
    list : 洪水事件列表，每个事件包含详细信息
    """
    
    if obs_flow.ndim == 1:
        obs_flow = obs_flow.reshape(1, -1)
    
    n_basins, n_time = obs_flow.shape
    all_periods = []
    
    print("=== 基于松辽数据集标签提取洪水事件 ===")
    
    for basin_idx in range(n_basins):
        # Safely obtain basin_id: basin_names can be None, list, or numpy array; avoid truth-value on array
        if basin_names is not None:
            try:
                basin_id = basin_names[basin_idx]
                # decode bytes if necessary and ensure string type
                if isinstance(basin_id, (bytes, bytearray)):
                    basin_id = basin_id.decode('utf-8')
                basin_id = str(basin_id)
            except Exception:
                basin_id = f"Basin_{basin_idx}"
        else:
            basin_id = f"Basin_{basin_idx}"
        basin_flow = obs_flow[basin_idx, :]
        
        print(f"\n处理流域 {basin_idx}: {basin_id}")
        
        try:
            # 加载松辽时间序列数据
            if time_array is not None and len(time_array) > 0:
                time_start = pd.to_datetime(time_array[0])
                time_end = pd.to_datetime(time_array[-1])
                time_range = (time_start, time_end)
                print(f"  观测数据时间范围: {time_start} 到 {time_end}")
            else:
                time_range = None
                print("  未提供时间信息，加载全部松辽数据")
            
            # 加载对应流域的松辽数据
            songliao_data = load_songliao_timeseries_data(basin_id, time_range, songliao_data_path)
            
            if songliao_data is None:
                print(f"  无法加载松辽数据，跳过流域 {basin_id}")
                continue
            
            # 检查是否有洪水事件标签
            if 'flood_event' not in songliao_data.columns:
                print(f"  松辽数据中没有flood_event列，跳过流域 {basin_id}")
                continue
            
            # 时间对齐：将松辽数据的时间与观测数据的时间对齐
            aligned_flood_events = align_flood_events_to_observations(
                songliao_data, time_array, basin_flow
            )
            
            if aligned_flood_events is None:
                print(f"  时间对齐失败，跳过流域 {basin_id}")
                continue
            
            # 识别连续的洪水事件期间
            flood_periods = extract_continuous_flood_periods(
                aligned_flood_events, basin_flow, time_array, basin_idx, basin_id
            )
            
            print(f"  识别到 {len(flood_periods)} 个洪水事件")
            
            # 按年份分组并编号
            flood_periods_with_year = assign_year_and_event_number(flood_periods, time_array)
            
            # 显示识别结果
            for period in flood_periods_with_year:
                print(f"    事件: {period['event_name']}, "
                      f"持续时间: {period['duration']} 时间步, "
                      f"峰值: {period['peak_value']:.2f}")
            
            all_periods.extend(flood_periods_with_year)
            
        except Exception as e:
            print(f"  处理流域 {basin_id} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n总共识别到 {len(all_periods)} 个洪水事件")
    return all_periods


def align_flood_events_to_observations(songliao_data, obs_time_array, obs_flow):
    """
    将松辽数据的洪水事件标签与观测数据的时间对齐
    
    Parameters:
    -----------
    songliao_data : pd.DataFrame
        松辽时间序列数据，包含time和flood_event列
    obs_time_array : np.ndarray
        观测数据的时间数组
    obs_flow : np.ndarray
        观测流量数据（一维）
    
    Returns:
    --------
    np.ndarray : 对齐后的洪水事件标签数组，与obs_flow长度相同
    """
    
    try:
        # 将松辽数据的时间设为索引
        songliao_indexed = songliao_data.set_index('time')
        
        # 转换观测时间数组为pandas时间格式
        obs_times_pd = pd.to_datetime(obs_time_array)
        
        print(f"    松辽数据时间范围: {songliao_indexed.index.min()} 到 {songliao_indexed.index.max()}")
        print(f"    观测数据时间范围: {obs_times_pd.min()} 到 {obs_times_pd.max()}")
        
        # 使用最近邻方法对齐，容差为1小时
        aligned_data = songliao_indexed.reindex(obs_times_pd, method='nearest', tolerance='1H')
        
        # 提取洪水事件标签
        flood_events = aligned_data['flood_event'].values
        
        # 处理NaN值，默认为0（非洪水事件）
        flood_events = np.nan_to_num(flood_events, nan=0).astype(int)
        
        # 确保长度与观测数据一致
        if len(flood_events) != len(obs_flow):
            print(f"    长度不匹配，调整为观测数据长度: {len(obs_flow)}")
            if len(flood_events) > len(obs_flow):
                flood_events = flood_events[:len(obs_flow)]
            else:
                # 用0填充
                padded_events = np.zeros(len(obs_flow), dtype=int)
                padded_events[:len(flood_events)] = flood_events
                flood_events = padded_events
        
        flood_count = np.sum(flood_events == 1)
        print(f"    成功对齐，洪水时间步数: {flood_count}/{len(flood_events)}")
        
        return flood_events
        
    except Exception as e:
        print(f"    时间对齐失败: {e}")
        return None


def extract_continuous_flood_periods(flood_events, basin_flow, time_array, basin_idx, basin_id):
    """
    从洪水事件标签中提取连续的洪水期间
    
    Parameters:
    -----------
    flood_events : np.ndarray
        洪水事件标签数组（1=洪水，0=非洪水）
    basin_flow : np.ndarray
        流域流量数据
    time_array : np.ndarray
        时间数组
    basin_idx : int
        流域索引
    basin_id : str
        流域ID
        
    Returns:
    --------
    list : 连续洪水期间列表
    """
    
    periods = []
    
    # 寻找连续的洪水期间
    start_idx = None
    
    for i, is_flood in enumerate(flood_events):
        if is_flood == 1 and start_idx is None:
            # 洪水事件开始
            start_idx = i
        elif is_flood == 0 and start_idx is not None:
            # 洪水事件结束
            end_idx = i
            duration = end_idx - start_idx
            
            # 只保留持续时间大于等于1个时间步的事件
            if duration >= 1:
                period_info = extract_period_statistics(
                    start_idx, end_idx, basin_flow, basin_idx, basin_id
                )
                periods.append(period_info)
            
            start_idx = None
    
    # 处理延续到末尾的洪水事件
    if start_idx is not None:
        end_idx = len(flood_events)
        duration = end_idx - start_idx
        
        if duration >= 1:
            period_info = extract_period_statistics(
                start_idx, end_idx, basin_flow, basin_idx, basin_id
            )
            periods.append(period_info)
    
    return periods


def extract_period_statistics(start_idx, end_idx, basin_flow, basin_idx, basin_id):
    """
    提取洪水期间的统计信息
    
    Parameters:
    -----------
    start_idx : int
        开始索引
    end_idx : int
        结束索引
    basin_flow : np.ndarray
        流域流量数据
    basin_idx : int
        流域索引
    basin_id : str
        流域ID
        
    Returns:
    --------
    dict : 期间统计信息
    """
    
    duration = end_idx - start_idx
    period_flow = basin_flow[start_idx:end_idx]
    
    # 寻找峰值
    valid_flow = period_flow[~np.isnan(period_flow)]
    if len(valid_flow) > 0:
        peak_idx_relative = np.argmax(valid_flow)
        # 找到在原始period_flow中的位置
        valid_indices = np.where(~np.isnan(period_flow))[0]
        peak_idx = start_idx + valid_indices[peak_idx_relative]
        peak_value = valid_flow[peak_idx_relative]
        
        # 其他统计信息
        mean_flow = np.mean(valid_flow)
        total_volume = np.sum(valid_flow)
    else:
        peak_idx = start_idx
        peak_value = 0.0
        mean_flow = 0.0
        total_volume = 0.0
    
    return {
        'basin': basin_idx,
        'basin_id': basin_id,
        'start': start_idx,
        'end': end_idx,
        'duration': duration,
        'peak_idx': peak_idx,
        'peak_value': peak_value,
        'mean_flow': mean_flow,
        'total_volume': total_volume,
        'flood_source': 'songliao_labels'  # 标记来源
    }


def assign_year_and_event_number(flood_periods, time_array):
    """
    为洪水事件分配年份和事件编号
    
    Parameters:
    -----------
    flood_periods : list
        洪水期间列表
    time_array : np.ndarray
        时间数组
        
    Returns:
    --------
    list : 带有年份和事件编号的洪水期间列表
    """
    
    periods_with_year = []
    year_event_counts = {}
    
    for period in flood_periods:
        try:
            # 获取峰值时间的年份
            if time_array is not None and period['peak_idx'] < len(time_array):
                peak_time = time_array[period['peak_idx']]
                
                if hasattr(peak_time, 'year'):
                    year = peak_time.year
                else:
                    # 尝试pandas转换
                    try:
                        time_pd = pd.to_datetime(peak_time)
                        year = time_pd.year
                    except:
                        # 假设从2020年开始，3小时一个时间步
                        year = 2020 + (period['peak_idx'] // (365 * 8))
            else:
                # 粗略估算年份
                year = 2020 + (period['peak_idx'] // (365 * 8))
        
        except:
            # 默认年份
            year = 2020
        
        # 统计每年的事件数
        basin_year_key = f"{period['basin_id']}_{year}"
        if basin_year_key not in year_event_counts:
            year_event_counts[basin_year_key] = 0
        year_event_counts[basin_year_key] += 1
        
        # 添加年份和事件信息
        period['year'] = year
        period['event_number'] = year_event_counts[basin_year_key]
        period['event_name'] = f"{year}_Event{year_event_counts[basin_year_key]}"
        
        periods_with_year.append(period)
    
    return periods_with_year


def find_flood_events(flow_data, time_array=None, min_duration=8, percentile_threshold=50):
    """
    改进的洪水事件检测 - 基于动态阈值和连续性
    
    Parameters:
    -----------
    flow_data : np.ndarray
        Flow data with shape (n_basins, n_time) - 应该是反标准化后的数据
    time_array : np.ndarray, optional
        Time array for naming events by year
    min_duration : int, default=8
        Minimum duration (timesteps) for a flood event (8*3h = 24h)
    percentile_threshold : float, default=50
        Percentile threshold for identifying significant flow events
    
    Returns:
    --------
    list : List of flood event dictionaries
    """
    
    if flow_data.ndim == 1:
        flow_data = flow_data.reshape(1, -1)
    
    n_basins, n_time = flow_data.shape
    all_periods = []
    
    for basin_idx in range(n_basins):
        basin_flow = flow_data[basin_idx, :]
        
        # 改进的阈值计算：排除极值和NaN
        valid_flow = basin_flow[~np.isnan(basin_flow)]
        if len(valid_flow) < 20:  # 需要足够的数据点
            print(f"Basin {basin_idx}: Insufficient valid data ({len(valid_flow)} points)")
            continue
        
        # 使用更合理的阈值：均值 + 标准差
        flow_mean = np.mean(valid_flow)
        flow_std = np.std(valid_flow)
        flow_percentile = np.percentile(valid_flow, percentile_threshold)
        
        # 动态阈值：取百分位数和均值+标准差的较大值
        flow_threshold = max(flow_percentile, flow_mean + 0.5 * flow_std)
        
        print(f"Basin {basin_idx}: mean={flow_mean:.2f}, std={flow_std:.2f}, "
              f"threshold={flow_threshold:.2f} (p{percentile_threshold}={flow_percentile:.2f})")
        
        # 寻找连续超过阈值的期间
        above_threshold = basin_flow > flow_threshold
        periods = []
        
        start_idx = None
        for i, is_above in enumerate(above_threshold):
            if is_above and start_idx is None:
                start_idx = i
            elif not is_above and start_idx is not None:
                # 结束一个期间
                duration = i - start_idx
                if duration >= min_duration:
                    # 在期间内找峰值
                    period_flow = basin_flow[start_idx:i]
                    peak_idx = start_idx + np.argmax(period_flow)
                    peak_value = basin_flow[peak_idx]
                    
                    # 计算洪水事件的统计信息
                    volume = np.sum(period_flow[period_flow > flow_mean])  # 超过均值的体积
                    
                    periods.append({
                        'basin': basin_idx,
                        'start': start_idx,
                        'end': i,
                        'duration': duration,
                        'peak_idx': peak_idx,
                        'peak_value': peak_value,
                        'threshold': flow_threshold,
                        'volume': volume,
                        'mean_flow': np.mean(period_flow)
                    })
                start_idx = None
        
        # 处理延续到末尾的情况
        if start_idx is not None:
            duration = n_time - start_idx
            if duration >= min_duration:
                period_flow = basin_flow[start_idx:]
                peak_idx = start_idx + np.argmax(period_flow)
                peak_value = basin_flow[peak_idx]
                volume = np.sum(period_flow[period_flow > flow_mean])
                
                periods.append({
                    'basin': basin_idx,
                    'start': start_idx,
                    'end': n_time,
                    'duration': duration,
                    'peak_idx': peak_idx,
                    'peak_value': peak_value,
                    'threshold': flow_threshold,
                    'volume': volume,
                    'mean_flow': np.mean(period_flow)
                })
        
        # 按时间排序
        periods.sort(key=lambda x: x['start'])
        
        # 添加年份和事件命名
        if time_array is not None and len(time_array) == n_time:
            periods_with_year = []
            year_event_counts = {}
            
            for period in periods:
                try:
                    # 尝试从时间数组获取年份
                    peak_time = time_array[period['peak_idx']]
                    if hasattr(peak_time, 'year'):
                        year = peak_time.year
                    else:
                        # 尝试pandas转换
                        try:
                            time_pd = pd.to_datetime(peak_time)
                            year = time_pd.year
                        except:
                            # 假设从2020年开始，3小时一个时间步
                            year = 2020 + (period['peak_idx'] // (365*8))
                except:
                    year = 2020 + (period['peak_idx'] // 2920)  # 粗略估算
                
                # 统计每年事件数
                if year not in year_event_counts:
                    year_event_counts[year] = 0
                year_event_counts[year] += 1
                
                period['year'] = year
                period['event_number'] = year_event_counts[year]
                period['event_name'] = f"{year}_Event{year_event_counts[year]}"
                periods_with_year.append(period)
            
            periods = periods_with_year
        else:
            # 简单编号
            for i, period in enumerate(periods):
                period['event_name'] = f"FloodEvent_{i+1}"
        
        print(f"  Found {len(periods)} flood events for basin {basin_idx}")
        if len(periods) > 0:
            peak_values = [p['peak_value'] for p in periods]
            print(f"    Peak range: {min(peak_values):.2f} - {max(peak_values):.2f}")
        
        all_periods.extend(periods)
    
    return all_periods


def load_netcdf_data(file_path):
    """Load NetCDF file data"""
    try:
        ds = xr.open_dataset(file_path)
        
        # Extract data
        data_dict = {}
        for var in ds.data_vars:
            data_dict[var] = ds[var].values
        
        # Extract coordinate information
        if 'time' in ds.coords:
            data_dict['time'] = ds.coords['time'].values
        if 'basin' in ds.coords:
            data_dict['basin'] = ds.coords['basin'].values
            
        return data_dict
        
    except Exception as e:
        print(f"Failed to load file {file_path}: {e}")
        return None


def calculate_metrics(obs, pred):
    """Calculate evaluation metrics"""
    # Remove NaN values
    valid_mask = ~(np.isnan(obs) | np.isnan(pred))
    if valid_mask.sum() < 2:
        return {'error': 'insufficient_data'}
    
    obs_valid = obs[valid_mask]
    pred_valid = pred[valid_mask]
    
    # NSE
    numerator = np.sum((obs_valid - pred_valid) ** 2)
    denominator = np.sum((obs_valid - np.mean(obs_valid)) ** 2)
    nse = 1 - (numerator / denominator) if denominator > 0 else -np.inf
    
    # R²
    r2 = np.corrcoef(obs_valid, pred_valid)[0, 1] ** 2 if len(obs_valid) > 1 else 0
    
    # RMSE
    rmse = np.sqrt(np.mean((obs_valid - pred_valid) ** 2))
    
    # MAE
    mae = np.mean(np.abs(obs_valid - pred_valid))
    
    # Peak comparison
    obs_peak = np.max(obs_valid)
    pred_peak = np.max(pred_valid)
    peak_error = abs(pred_peak - obs_peak) / obs_peak * 100 if obs_peak > 0 else 0
    
    return {
        'NSE': nse,
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae,
        'obs_peak': obs_peak,
        'pred_peak': pred_peak,
        'peak_error_pct': peak_error
    }


def find_flood_events_threshold_based(flow_data, time_array=None, min_duration=8, percentile_threshold=50):
    """
    改进的洪水事件检测 - 基于动态阈值和连续性
    
    Parameters:
    -----------
    flow_data : np.ndarray
        Flow data with shape (n_basins, n_time) - 应该是反标准化后的数据
    time_array : np.ndarray, optional
        Time array for naming events by year
    min_duration : int, default=8
        Minimum duration (timesteps) for a flood event (8*3h = 24h)
    percentile_threshold : float, default=50
        Percentile threshold for identifying significant flow events
    
    Returns:
    --------
    list : List of flood event dictionaries
    """
    
    if flow_data.ndim == 1:
        flow_data = flow_data.reshape(1, -1)
    
    n_basins, n_time = flow_data.shape
    all_periods = []
    
    for basin_idx in range(n_basins):
        basin_flow = flow_data[basin_idx, :]
        
        # 改进的阈值计算：排除极值和NaN
        valid_flow = basin_flow[~np.isnan(basin_flow)]
        if len(valid_flow) < 20:  # 需要足够的数据点
            print(f"Basin {basin_idx}: Insufficient valid data ({len(valid_flow)} points)")
            continue
        
        # 使用更合理的阈值：均值 + 标准差
        flow_mean = np.mean(valid_flow)
        flow_std = np.std(valid_flow)
        flow_percentile = np.percentile(valid_flow, percentile_threshold)
        
        # 动态阈值：取百分位数和均值+标准差的较大值
        flow_threshold = max(flow_percentile, flow_mean + 0.5 * flow_std)
        
        print(f"Basin {basin_idx}: mean={flow_mean:.2f}, std={flow_std:.2f}, "
              f"threshold={flow_threshold:.2f} (p{percentile_threshold}={flow_percentile:.2f})")
        
        # 寻找连续超过阈值的期间
        above_threshold = basin_flow > flow_threshold
        periods = []
        
        start_idx = None
        for i, is_above in enumerate(above_threshold):
            if is_above and start_idx is None:
                start_idx = i
            elif not is_above and start_idx is not None:
                # 结束一个期间
                duration = i - start_idx
                if duration >= min_duration:
                    # 在期间内找峰值
                    period_flow = basin_flow[start_idx:i]
                    peak_idx = start_idx + np.argmax(period_flow)
                    peak_value = basin_flow[peak_idx]
                    
                    # 计算洪水事件的统计信息
                    volume = np.sum(period_flow[period_flow > flow_mean])  # 超过均值的体积
                    
                    periods.append({
                        'basin': basin_idx,
                        'start': start_idx,
                        'end': i,
                        'duration': duration,
                        'peak_idx': peak_idx,
                        'peak_value': peak_value,
                        'threshold': flow_threshold,
                        'volume': volume,
                        'mean_flow': np.mean(period_flow)
                    })
                start_idx = None
        
        # 处理延续到末尾的情况
        if start_idx is not None:
            duration = n_time - start_idx
            if duration >= min_duration:
                period_flow = basin_flow[start_idx:]
                peak_idx = start_idx + np.argmax(period_flow)
                peak_value = basin_flow[peak_idx]
                volume = np.sum(period_flow[period_flow > flow_mean])
                
                periods.append({
                    'basin': basin_idx,
                    'start': start_idx,
                    'end': n_time,
                    'duration': duration,
                    'peak_idx': peak_idx,
                    'peak_value': peak_value,
                    'threshold': flow_threshold,
                    'volume': volume,
                    'mean_flow': np.mean(period_flow)
                })
        
        # 按时间排序
        periods.sort(key=lambda x: x['start'])
        
        # 添加年份和事件命名
        if time_array is not None and len(time_array) == n_time:
            periods_with_year = []
            year_event_counts = {}
            
            for period in periods:
                try:
                    # 尝试从时间数组获取年份
                    peak_time = time_array[period['peak_idx']]
                    if hasattr(peak_time, 'year'):
                        year = peak_time.year
                    else:
                        # 尝试pandas转换
                        import pandas as pd
                        try:
                            time_pd = pd.to_datetime(peak_time)
                            year = time_pd.year
                        except:
                            # 假设从2020年开始，3小时一个时间步
                            year = 2020 + (period['peak_idx'] // (365*8))
                except:
                    year = 2020 + (period['peak_idx'] // 2920)  # 粗略估算
                
                # 统计每年事件数
                if year not in year_event_counts:
                    year_event_counts[year] = 0
                year_event_counts[year] += 1
                
                period['year'] = year
                period['event_number'] = year_event_counts[year]
                period['event_name'] = f"{year}_Event{year_event_counts[year]}"
                periods_with_year.append(period)
            
            periods = periods_with_year
        else:
            # 简单编号
            for i, period in enumerate(periods):
                period['event_name'] = f"FloodEvent_{i+1}"
        
        print(f"  Found {len(periods)} flood events for basin {basin_idx}")
        if len(periods) > 0:
            peak_values = [p['peak_value'] for p in periods]
            print(f"    Peak range: {min(peak_values):.2f} - {max(peak_values):.2f}")
        
        all_periods.extend(periods)
    
    return all_periods


try:
    from hydrodatasource.reader.data_source import SelfMadeHydroDataset
    HYDRODATASOURCE_AVAILABLE = True
except ImportError:
    HYDRODATASOURCE_AVAILABLE = False


def read_basin_info_from_hydro_dataset(data_path, time_unit):
    """
    从Hydro格式数据集自动读取流域信息,返回DataFrame(含basin_id、area等)。
    """
    if not HYDRODATASOURCE_AVAILABLE:
        print("Warning: hydrodatasource not available, cannot load basin info")
        return None
        
    try:
        dataset = SelfMadeHydroDataset(
            data_path=data_path,
            time_unit=[time_unit] if isinstance(time_unit, str) else time_unit,
            offset_to_utc=True,
        )
        attrs = dataset.read_site_info()
        return attrs
    except Exception as e:
        print(f"Failed to load basin info: {e}")
        return None
