#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多特征GC-GRU训练脚本
使用历史降雨、LSTM流量、实测流量数据进行单步预测
结合静态属性数据，采用逐时间步建模方式
"""

import os
import csv
import json
from datetime import datetime
import random

import lightning_fabric
import matplotlib.pyplot as plt
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.optim.lr_scheduler import ExponentialLR
from torchmetrics.functional import r2_score
import networkx as nx
from tqdm import tqdm
import pandas as pd
import numpy as np
import xarray as xr
import gconv_gru

# ================================
# 配置参数
# ================================

# 路径配置
PATH_CONFIG = {
    'data_dir': "/home/mahaoran/GNN/data_jinshajiang_32",
    'result_dir': "/home/mahaoran/GNN/result/test_jinshajiang_HF_k3_32_1112_eval",
    'log_dir': "/home/mahaoran/GNN/logs/test_jinshajiang_HF_k3_32_1112",
}

# 训练配置
TRAIN_CONFIG = {
    'num_epochs': 50,            # 训练轮数
    'batch_size': 16,           # 批次大小
    'learning_rate': 0.01,     # 学习率
    'lr_decay_factor': 0.96,    # 学习率衰减系数
    'max_grad_norm': 1.0,       # 梯度裁剪阈值
    'num_workers': 0,           # 数据加载器工作进程数
    'random_seed': 1234,        # 随机种子
    'use_confluence_nodes_only': True,  # 是否只对汇流节点进行训练和评估
}

# 模型配置
MODEL_CONFIG = {
    'hindcast_length': 30,      # 历史数据长度M
    'forecast_length': 1,       # 未来预报数据长度n (单步预测)
    'gcgru_out_channels': 32,   # GConvGRU输出通道数
    'gcgru_K': 3,              # Chebyshev滤波器大小
    'mlp_hidden_dim': 128,      # MLP隐藏层维度
    'mlp_dropout': 0.1,         # MLP Dropout率
    'node_attr_dim': 3,         # 节点静态属性维度
    'node_attr_hidden_dim': 32, # 节点属性处理隐藏层维度
    'use_node_attr': True,      # 是否使用节点属性
    'use_edge_attr': False,     # 是否使用边属性
}

# 数据配置
DATA_CONFIG = {
    'train_start': "2021-01-01",
    'train_end': "2024-12-31",
    'test_start': "2021-01-01",
    'test_end': "2024-12-31",
    'target_station': 'sanxia_13020000',
    'basin_id': 'sanxia_13020000',
    'normalization_scale': 2,  # 调整为2以更好适配Tanh和Sigmoid激活函数
    'selected_attrs': ['area', 'p_mean', 'sgr_dk_sav'],
}

# 评估配置
EVAL_CONFIG = {
    'eval_only': True,          # 是否仅进行评估（不训练）
    'eval_epoch': 50,          # 评估时加载的模型epoch，None表示加载best_model.pth
    'multistep_eval': True,     # 是否进行多步预测评估
    'forecast_steps': 3,         # 多步预测的步数
    'eval_samples': 10000,          # 多步预测评估的样本数量
}

# 图结构定义
graph_dict = {
    'sanxia_13020000': [
        ['sanxia_60100700', 'sanxia_60100800', 'sanxia_60101000', 'sanxia_60101100', 'sanxia_60101300', 'sanxia_60101398', 'sanxia_60101498', 'sanxia_60101698', 'sanxia_60101750', 'sanxia_60101850', 'sanxia_60101998', 'sanxia_60102100', 'sanxia_60102200', 'sanxia_16020000', 'sanxia_15020000', 'sanxia_14020000', 'sanxia_13020000'],
        ['sanxia_60201800', 'sanxia_60101300', 'sanxia_60101398', 'sanxia_60101498', 'sanxia_60101698', 'sanxia_60101750', 'sanxia_60101850', 'sanxia_60101998', 'sanxia_60102100', 'sanxia_60102200', 'sanxia_16020000', 'sanxia_15020000', 'sanxia_14020000', 'sanxia_13020000'],
        ['sanxia_60010080', 'sanxia_60101498', 'sanxia_60101698', 'sanxia_60101750', 'sanxia_60101850', 'sanxia_60101998', 'sanxia_60102100', 'sanxia_60102200', 'sanxia_16020000', 'sanxia_15020000', 'sanxia_14020000', 'sanxia_13020000'],
        ['sanxia_60301000', 'sanxia_60301011', 'sanxia_60301100', 'sanxia_60305011', 'sanxia_60300000', 'sanxia_60301400', 'sanxia_60300001', 'sanxia_60300002', 'sanxia_60300005', 'sanxia_60300011', 'sanxia_60102200', 'sanxia_16020000', 'sanxia_15020000', 'sanxia_14020000', 'sanxia_13020000'],
        ['sanxia_60301700', 'sanxia_60301011', 'sanxia_60301100', 'sanxia_60305011', 'sanxia_60300000', 'sanxia_60301400', 'sanxia_60300001', 'sanxia_60300002', 'sanxia_60300005', 'sanxia_60300011', 'sanxia_60102200', 'sanxia_16020000', 'sanxia_15020000', 'sanxia_14020000', 'sanxia_13020000'],
        ['sanxia_60303050', 'sanxia_60304050', 'sanxia_60300011', 'sanxia_60102200', 'sanxia_16020000', 'sanxia_15020000', 'sanxia_14020000', 'sanxia_13020000']
    ]
}

# ================================
# 工具函数
# ================================

def normalize_time_series(train_tensor, test_tensor, method='zscore', scale_factor=1.0, eps=1e-8, per_node=False):
    """
    通用时序归一化函数。
    
    Args:
        train_tensor: 训练期数据，形状 [time, basin]
        test_tensor: 测试期数据，形状 [time, basin]
        method: 'minmax' 或 'zscore'
        scale_factor: min-max 归一化输出缩放到 [0, scale_factor]，Z分数归一化不使用
        eps: 防止除零的小常数
        per_node: True 表示按站点逐列统计（对每个站点独立标准化）；False 表示全局统计
    Returns:
        (train_norm, test_norm, params)
        params 包含统计量和方法信息，便于反归一化或记录
    """
    if method == 'minmax':
        if per_node:
            min_vals = train_tensor.min(dim=0).values
            max_vals = train_tensor.max(dim=0).values
            train_norm = scale_factor * (train_tensor - min_vals) / (max_vals - min_vals + eps)
            test_norm = scale_factor * (test_tensor - min_vals) / (max_vals - min_vals + eps)
            params = {'method': 'minmax', 'min': min_vals, 'max': max_vals, 'scale_factor': scale_factor}
        else:
            min_val = train_tensor.min()
            max_val = train_tensor.max()
            train_norm = scale_factor * (train_tensor - min_val) / (max_val - min_val + eps)
            test_norm = scale_factor * (test_tensor - min_val) / (max_val - min_val + eps)
            params = {'method': 'minmax', 'min': float(min_val.item()), 'max': float(max_val.item()), 'scale_factor': scale_factor}
    elif method == 'zscore':
        if per_node:
            mean_vals = train_tensor.mean(dim=0)
            std_vals = train_tensor.std(dim=0)
            train_norm = (train_tensor - mean_vals) / (std_vals + eps)
            test_norm = (test_tensor - mean_vals) / (std_vals + eps)
            params = {'method': 'zscore', 'mean': mean_vals, 'std': std_vals}
        else:
            mean_val = train_tensor.mean()
            std_val = train_tensor.std()
            train_norm = (train_tensor - mean_val) / (std_val + eps)
            test_norm = (test_tensor - mean_val) / (std_val + eps)
            params = {'method': 'zscore', 'mean': float(mean_val.item()), 'std': float(std_val.item())}
    else:
        raise ValueError(f"不支持的归一化方法: {method}")
    return train_norm, test_norm, params

def denormalize_time_series(normalized_tensor, normalization_params, data_type='obs_flow'):
    """
    反归一化函数，将标准化后的数据恢复为原始尺度
    
    Args:
        normalized_tensor: 标准化后的张量
        normalization_params: 归一化参数字典
        data_type: 数据类型，可选 'precip', 'lstm_flow', 'obs_flow'
    
    Returns:
        反归一化后的张量
    """
    if normalization_params['method'] == 'zscore':
        # Z分数反归一化: original = normalized * std + mean
        if data_type == 'precip':
            mean = normalization_params['precip_mean']
            std = normalization_params['precip_std']
        elif data_type == 'lstm_flow':
            mean = normalization_params['lstm_flow_mean']
            std = normalization_params['lstm_flow_std']
        elif data_type == 'obs_flow':
            mean = normalization_params['obs_flow_mean']
            std = normalization_params['obs_flow_std']
        else:
            raise ValueError(f"不支持的数据类型: {data_type}")
        
        # 确保mean和std是张量且在正确的设备上
        if isinstance(normalized_tensor, torch.Tensor):
            if not isinstance(mean, torch.Tensor):
                mean = torch.tensor(mean, dtype=normalized_tensor.dtype, device=normalized_tensor.device)
            if not isinstance(std, torch.Tensor):
                std = torch.tensor(std, dtype=normalized_tensor.dtype, device=normalized_tensor.device)
            
            denormalized = normalized_tensor * std + mean
        else:
            # 如果是numpy数组
            if isinstance(mean, torch.Tensor):
                mean = mean.cpu().numpy()
            if isinstance(std, torch.Tensor):
                std = std.cpu().numpy()
            
            denormalized = normalized_tensor * std + mean
            
    elif normalization_params['method'] == 'minmax':
        # Min-Max反归一化: original = min + (max - min) * normalized / scale_factor
        if data_type == 'precip':
            min_val = normalization_params['precip_min']
            max_val = normalization_params['precip_max']
        elif data_type == 'lstm_flow':
            min_val = normalization_params['lstm_flow_min']
            max_val = normalization_params['lstm_flow_max']
        elif data_type == 'obs_flow':
            min_val = normalization_params['obs_flow_min']
            max_val = normalization_params['obs_flow_max']
        else:
            raise ValueError(f"不支持的数据类型: {data_type}")
        
        scale_factor = normalization_params.get('scale_factor', 1.0)
        
        if isinstance(normalized_tensor, torch.Tensor):
            if not isinstance(min_val, torch.Tensor):
                min_val = torch.tensor(min_val, dtype=normalized_tensor.dtype, device=normalized_tensor.device)
            if not isinstance(max_val, torch.Tensor):
                max_val = torch.tensor(max_val, dtype=normalized_tensor.dtype, device=normalized_tensor.device)
            
            denormalized = min_val + (max_val - min_val) * normalized_tensor / scale_factor
        else:
            # 如果是numpy数组
            if isinstance(min_val, torch.Tensor):
                min_val = min_val.cpu().numpy()
            if isinstance(max_val, torch.Tensor):
                max_val = max_val.cpu().numpy()
            
            denormalized = min_val + (max_val - min_val) * normalized_tensor / scale_factor
    else:
        raise ValueError(f"不支持的归一化方法: {normalization_params['method']}")
    
    return denormalized

def gen_nx_graph(basin_id, basin_names):
    """生成NetworkX图结构"""
    paths = graph_dict[basin_id]
    
    # 创建有向图
    dg = nx.DiGraph()
    
    # 添加节点
    for name in basin_names:
        dg.add_node(name)
    
    # 添加边
    for path in paths:
        for i in range(len(path) - 1):
            dg.add_edge(path[i], path[i + 1])
    
    # 转换为边列表
    edges = list(dg.edges())
    
    # 创建节点映射
    node_mapping = {name: idx for idx, name in enumerate(basin_names)}
    
    # 转换边索引
    edge_indices = [[node_mapping[edge[0]], node_mapping[edge[1]]] for edge in edges]
    
    return dg, edge_indices, node_mapping

def classify_nodes(dg, basin_names):
    """
    分类节点为源头节点和汇流节点
    
    Args:
        dg: NetworkX有向图
        basin_names: 节点名称列表
    
    Returns:
        source_nodes: 源头节点列表（没有入度的节点）
        confluence_nodes: 汇流节点列表（有入度的节点）
        node_mask: 布尔掩码，True表示汇流节点，False表示源头节点
    """
    source_nodes = []
    confluence_nodes = []
    
    for node in basin_names:
        # 检查节点的入度
        in_degree = dg.in_degree(node)
        if in_degree == 0:
            source_nodes.append(node)
        else:
            confluence_nodes.append(node)
    
    # 创建节点掩码：True表示汇流节点（需要训练），False表示源头节点（不训练）
    node_mask = []
    for node in basin_names:
        node_mask.append(node in confluence_nodes)
    
    print(f"源头节点 ({len(source_nodes)}个): {source_nodes}")
    print(f"汇流节点 ({len(confluence_nodes)}个): {confluence_nodes}")
    
    return source_nodes, confluence_nodes, node_mask

def interpolate_missing_data(data, method='linear'):
    """插值处理缺失数据"""
    # 创建原始有效数据的掩码
    mask = ~torch.isnan(data)
    
    # 转换为numpy进行插值
    data_np = data.numpy()
    
    for node_idx in range(data_np.shape[1]):
        node_data = data_np[:, node_idx]
        valid_mask = ~np.isnan(node_data)
        
        if valid_mask.sum() == 0:
            # 如果该节点完全没有数据，用0填充
            data_np[:, node_idx] = 0.0
        elif valid_mask.sum() == len(node_data):
            # 如果该节点没有缺失数据，保持不变
            continue
        else:
            # 进行插值
            if method == 'linear':
                # 线性插值
                valid_indices = np.where(valid_mask)[0]
                valid_values = node_data[valid_mask]
                
                # 对缺失值进行线性插值
                interpolated = np.interp(
                    np.arange(len(node_data)), 
                    valid_indices, 
                    valid_values
                )
                data_np[:, node_idx] = interpolated
            elif method == 'forward':
                # 前向填充
                pd_series = pd.Series(node_data)
                data_np[:, node_idx] = pd_series.fillna(method='ffill').fillna(0).values
            elif method == 'backward':
                # 后向填充
                pd_series = pd.Series(node_data)
                data_np[:, node_idx] = pd_series.fillna(method='bfill').fillna(0).values
    
    return torch.from_numpy(data_np).float(), mask

def load_multifeature_data(data_dir, train_start, train_end, test_start, test_end):
    """加载多特征数据：降水、LSTM流量、实测流量、以及水库出库流量(discharge)"""
    # 加载LSTM预测的区间流量数据
    lstm_pred_file = os.path.join(data_dir, "lstmpred.nc")
    lstm_data = xr.open_dataset(lstm_pred_file)
    
    # 加载时间序列数据（包含降水、实测流量，及可能的水库出库流量discharge）
    timeseries_file = os.path.join(data_dir, "timeseries_1D.nc")
    timeseries_data = xr.open_dataset(timeseries_file)
    
    # 提取basin坐标变量（站名信息）
    basin_names = timeseries_data.coords['basin'].values.tolist()
    print(f"加载的流域站点: {basin_names}")
    
    # 提取降水数据
    precipitation = timeseries_data['total_precipitation_hourly']
    
    # 提取实测流量数据
    observed_streamflow = timeseries_data['streamflow']

    # 可选：提取水库出库流量数据（若存在）
    discharge = None
    if 'discharge' in timeseries_data.data_vars:
        discharge = timeseries_data['discharge']
        print("检测到 discharge 变量：将进行提取与标准化")
    else:
        print("未检测到 discharge 变量：跳过水库出库流量的处理")
    
    # 提取LSTM模拟的区间流量数据
    lstm_streamflow = lstm_data['streamflow']
    
    # 转换为时间范围
    train_start_dt = pd.to_datetime(train_start)
    train_end_dt = pd.to_datetime(train_end)
    test_start_dt = pd.to_datetime(test_start)
    test_end_dt = pd.to_datetime(test_end)
    
    # 提取训练期数据
    train_precip = precipitation.sel(time=slice(train_start_dt, train_end_dt))
    train_lstm_flow = lstm_streamflow.sel(time=slice(train_start_dt, train_end_dt))
    train_obs_flow = observed_streamflow.sel(time=slice(train_start_dt, train_end_dt))
    train_discharge = discharge.sel(time=slice(train_start_dt, train_end_dt)) if discharge is not None else None
    
    # 提取测试期数据
    test_precip = precipitation.sel(time=slice(test_start_dt, test_end_dt))
    test_lstm_flow = lstm_streamflow.sel(time=slice(test_start_dt, test_end_dt))
    test_obs_flow = observed_streamflow.sel(time=slice(test_start_dt, test_end_dt))
    test_discharge = discharge.sel(time=slice(test_start_dt, test_end_dt)) if discharge is not None else None
    
    # 转换为torch张量并转置为[time, basin]格式
    train_precip_tensor = torch.from_numpy(train_precip.values).float().T  # [time, basin]
    train_lstm_flow_tensor = torch.from_numpy(train_lstm_flow.values).float().T  # [time, basin]
    train_obs_flow_tensor = torch.from_numpy(train_obs_flow.values).float().T  # [time, basin]
    # discharge 可能包含大量 NaN（非水库站点），此处暂不去插值，保留 NaN 以便后续在数据集内部根据是否存在值来判断节点类型
    if train_discharge is not None:
        train_discharge_np = train_discharge.values  # [time, basin]
        test_discharge_np = test_discharge.values   # [time, basin]
        # 使用忽略 NaN 的全局 Z 分数归一化（基于训练期统计量）
        train_mean = np.nanmean(train_discharge_np)
        train_std = np.nanstd(train_discharge_np)
        if np.isnan(train_mean) or np.isnan(train_std) or train_std == 0:
            # 若训练期完全为 NaN 或方差为0，则保持原值，不做归一化（后续数据集会按节点掩码处理）
            print("警告：训练期 discharge 无有效值或标准差为0，跳过标准化")
            train_discharge_tensor = torch.from_numpy(train_discharge_np).float().T
            test_discharge_tensor = torch.from_numpy(test_discharge_np).float().T
            discharge_params = {'mean': float('nan'), 'std': float('nan')}
        else:
            # 仅对非 NaN 位置进行标准化，NaN 保留
            train_norm_np = (train_discharge_np - train_mean) / (train_std + 1e-8)
            test_norm_np = (test_discharge_np - train_mean) / (train_std + 1e-8)
            train_discharge_tensor = torch.from_numpy(train_norm_np).float().T
            test_discharge_tensor = torch.from_numpy(test_norm_np).float().T
            discharge_params = {'mean': float(train_mean), 'std': float(train_std)}
    else:
        train_discharge_tensor = None
        test_discharge_tensor = None
        discharge_params = None
    
    test_precip_tensor = torch.from_numpy(test_precip.values).float().T  # [time, basin]
    test_lstm_flow_tensor = torch.from_numpy(test_lstm_flow.values).float().T  # [time, basin]
    test_obs_flow_tensor = torch.from_numpy(test_obs_flow.values).float().T  # [time, basin]
    
    # 数据标准化 - 使用Z分数归一化（基于训练期统计量，全局）
    train_precip_tensor, test_precip_tensor, precip_params = normalize_time_series(
        train_precip_tensor, test_precip_tensor, method='zscore', per_node=False
    )
    train_lstm_flow_tensor, test_lstm_flow_tensor, lstm_params = normalize_time_series(
        train_lstm_flow_tensor, test_lstm_flow_tensor, method='zscore', per_node=False
    )
    train_obs_flow_tensor, test_obs_flow_tensor, obs_params = normalize_time_series(
        train_obs_flow_tensor, test_obs_flow_tensor, method='zscore', per_node=False
    )
    
    print(f"数据标准化信息:")
    print(f"  - 方法: Z分数归一化（使用训练期全局均值与标准差）")
    print(f"  - 降水: mean={precip_params['mean']:.4f}, std={precip_params['std']:.4f} -> 约N(0,1)")
    print(f"  - LSTM流量: mean={lstm_params['mean']:.4f}, std={lstm_params['std']:.4f} -> 约N(0,1)")
    print(f"  - 实测流量: mean={obs_params['mean']:.4f}, std={obs_params['std']:.4f} -> 约N(0,1)")
    if train_discharge_tensor is not None and discharge_params is not None:
        if np.isnan(discharge_params['mean']) or np.isnan(discharge_params['std']):
            print(f"  - discharge: 训练期无有效值或std=0，未进行标准化")
        else:
            print(f"  - discharge: mean={discharge_params['mean']:.4f}, std={discharge_params['std']:.4f} -> 约N(0,1)")
    print(f"训练数据形状:")
    print(f"  - 降水: {train_precip_tensor.shape}")
    print(f"  - LSTM流量: {train_lstm_flow_tensor.shape}")
    print(f"  - 实测流量: {train_obs_flow_tensor.shape}")
    if train_discharge_tensor is not None:
        print(f"  - discharge: {train_discharge_tensor.shape}")
    
    # 返回标准化参数以便后续反标准化使用
    normalization_params = {
        'method': 'zscore',
        'precip_mean': precip_params['mean'], 'precip_std': precip_params['std'],
        'lstm_flow_mean': lstm_params['mean'], 'lstm_flow_std': lstm_params['std'],
        'obs_flow_mean': obs_params['mean'], 'obs_flow_std': obs_params['std']
    }
    if discharge_params is not None:
        normalization_params['discharge_mean'] = discharge_params['mean']
        normalization_params['discharge_std'] = discharge_params['std']
    
    return (train_precip_tensor, train_lstm_flow_tensor, train_obs_flow_tensor,
            test_precip_tensor, test_lstm_flow_tensor, test_obs_flow_tensor,
            train_discharge_tensor, test_discharge_tensor,
            basin_names, normalization_params)

def load_node_attributes(data_dir, basin_names, selected_attrs=None):
    """加载节点静态属性"""
    if selected_attrs is None:
        selected_attrs = ['area', 'p_mean', 'sgr_dk_sav']
    
    attr_file = os.path.join(data_dir, "attributes.nc")
    
    if not os.path.exists(attr_file):
        print(f"警告：找不到节点属性文件 {attr_file}")
        return None
    
    # 读取NetCDF属性数据
    attr_data = xr.open_dataset(attr_file)
    
    # 获取可用属性
    available_attrs = []
    for attr in selected_attrs:
        if attr in attr_data.data_vars:
            available_attrs.append(attr)
        else:
            print(f"警告：属性 {attr} 在文件中不存在")
    
    if not available_attrs:
        print(f"警告：指定的属性 {selected_attrs} 在文件中都不存在")
        return None
    
    # 确保站点顺序与basin_names一致
    attr_matrix = []
    for basin in basin_names:
        if basin in attr_data.coords['basin'].values:
            basin_idx = list(attr_data.coords['basin'].values).index(basin)
            basin_attrs = []
            for attr in available_attrs:
                value = float(attr_data[attr].values[basin_idx])
                if np.isnan(value):
                    value = 0.0
                basin_attrs.append(value)
            attr_matrix.append(basin_attrs)
        else:
            # 如果站点不存在，用零填充
            attr_matrix.append([0.0] * len(available_attrs))
            print(f"警告：站点 {basin} 的属性数据缺失，用零填充")
    
    node_attributes = np.array(attr_matrix, dtype=np.float32)
    
    # 标准化
    for i in range(node_attributes.shape[1]):
        attr_col = node_attributes[:, i]
        mean_val = attr_col.mean()
        std_val = attr_col.std()
        if std_val > 0:
            node_attributes[:, i] = (attr_col - mean_val) / std_val
    
    print(f"加载了 {len(available_attrs)} 个节点属性: {available_attrs}")
    print(f"节点属性形状: {node_attributes.shape}")
    
    return torch.tensor(node_attributes, dtype=torch.float32)

def set_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ================================
# 数据集类
# ================================

class MultiFeatureStreamDataset(TensorDataset):
    """多特征数据集类 - 使用降水、LSTM流量、实测流量和静态属性"""
    
    def __init__(self, hindcast_length, forecast_length, precipitation_data, 
                 lstm_streamflow_data, observed_streamflow_data, basin_names, 
                 basin_id, node_attributes=None, use_confluence_nodes_only=True,
                 discharge_data=None):
        """
        初始化多特征数据集
        
        Args:
            hindcast_length: 历史数据长度
            forecast_length: 预测长度
            precipitation_data: 降水数据 [time_steps, num_stations]
            lstm_streamflow_data: LSTM流量数据 [time_steps, num_stations]
            observed_streamflow_data: 实测流量数据 [time_steps, num_stations]
            basin_names: 站点名称列表
            basin_id: 流域ID
            node_attributes: 节点属性 [num_stations, num_attributes]
            use_confluence_nodes_only: 是否只对汇流节点进行训练和评估
        """
        super().__init__(precipitation_data, lstm_streamflow_data, observed_streamflow_data)
        self.tensors = (precipitation_data, lstm_streamflow_data, observed_streamflow_data)
        self.hindcast_length = hindcast_length
        self.forecast_length = forecast_length
        self.basin_names = basin_names
        self.basin_id = basin_id
        self.node_attributes = node_attributes
        self.use_confluence_nodes_only = use_confluence_nodes_only
        self.discharge_data = discharge_data  # [time, basin] 或 None；可能包含 NaN 表示非水库节点
        
        # 处理缺失数据
        self.precipitation_data, self.precip_mask = interpolate_missing_data(precipitation_data)
        self.lstm_streamflow_data, self.lstm_mask = interpolate_missing_data(lstm_streamflow_data)
        self.observed_streamflow_data, self.obs_mask = interpolate_missing_data(observed_streamflow_data)
        # discharge 保持原始 NaN（用于节点类型判断），不进行插值；但类型转换为 torch.float
        if self.discharge_data is not None:
            if not torch.is_tensor(self.discharge_data):
                self.discharge_data = torch.from_numpy(self.discharge_data).float()
            else:
                self.discharge_data = self.discharge_data.float()
        
        # 根据参数决定是否计算节点掩码（区分源头节点和汇流节点）
        if self.use_confluence_nodes_only:
            dg, _, _ = gen_nx_graph(basin_id, basin_names)
            source_nodes, confluence_nodes, self.node_mask = classify_nodes(dg, basin_names)
            self.node_mask = torch.tensor(self.node_mask, dtype=torch.bool)  # 转换为PyTorch张量
        else:
            self.node_mask = None
            # 仍然计算节点分类用于信息显示，但不使用掩码
            dg, _, _ = gen_nx_graph(basin_id, basin_names)
            source_nodes, confluence_nodes, _ = classify_nodes(dg, basin_names)
        
        print(f"多特征数据集创建完成:")
        print(f"  - 降水数据形状: {self.precipitation_data.shape}")
        print(f"  - LSTM流量形状: {self.lstm_streamflow_data.shape}")
        print(f"  - 实测流量形状: {self.observed_streamflow_data.shape}")
        if self.discharge_data is not None:
            print(f"  - discharge形状: {self.discharge_data.shape}")
        if self.node_attributes is not None:
            print(f"  - 节点属性形状: {self.node_attributes.shape}")
        print(f"  - 源头节点数量: {len(source_nodes)}")
        print(f"  - 汇流节点数量: {len(confluence_nodes)}")
        print(f"  - 源头节点: {source_nodes}")
        print(f"  - 汇流节点: {confluence_nodes}")
        
        if self.use_confluence_nodes_only:
            print("  - 使用节点掩码：只对汇流节点进行训练和评估")
        else:
            print("  - 不使用节点掩码：对所有节点进行训练和评估")
    
    def __len__(self):
        # 由于降雨和LSTM数据需要往后错一个时段获取未来信息，需要额外减去1
        return self.precipitation_data.shape[0] - self.hindcast_length - self.forecast_length + 1 - 1
    
    def __getitem__(self, idx):
        """获取单个样本 - 引入未来降雨和LSTM预测信息"""
        start_idx = idx + self.hindcast_length

        # 历史实测流量数据 [M, num_nodes] - 保持原有时间窗口
        past_obs_flow = self.observed_streamflow_data[idx : start_idx]
        # 若存在 discharge，则对每个时间步的非 NaN 位置直接替换为 discharge；NaN 位置保留原观测流量
        if self.discharge_data is not None:
            window_discharge = self.discharge_data[idx : start_idx]  # [M, num_nodes]
            valid_discharge = ~torch.isnan(window_discharge)
            past_obs_flow = torch.where(valid_discharge, window_discharge, past_obs_flow)
        
        # 降雨和LSTM数据往后错一个时段，引入未来信息 [M, num_nodes]
        # 这样可以获得未来1天的降雨预报和LSTM预测
        past_rain = self.precipitation_data[idx + 1 : start_idx + 1]
        past_lstm_flow = self.lstm_streamflow_data[idx + 1 : start_idx + 1]
        
        # 未来目标数据 [forecast_length, num_nodes] —— 保持为观测流量，不修改
        future_obs_flow = self.observed_streamflow_data[start_idx : start_idx + self.forecast_length]
        
        # 数据预处理：确保非负值
        past_rain = torch.clamp(past_rain, min=0)
        past_lstm_flow = torch.clamp(past_lstm_flow, min=0)
        past_obs_flow = torch.clamp(past_obs_flow, min=0)
        
        # 构建时间序列特征：[num_nodes, M, 3] -> 每个时间步有3个特征（降水、LSTM流量、实测流量/水库出库）
        # 注意：现在降雨和LSTM包含未来1天的信息，而实测流量是历史信息
        time_series_features = torch.stack([past_rain.T, past_lstm_flow.T, past_obs_flow.T], dim=2)  # [num_nodes, M, 3]
        
        if self.node_attributes is not None:
            # 将静态属性扩展到每个时间步，在特征维度进行拼接
            # node_attributes: [num_nodes, attr_dim]
            # 扩展为: [num_nodes, M, attr_dim]
            M = time_series_features.shape[1]
            expanded_static_attrs = self.node_attributes.unsqueeze(1).expand(-1, M, -1)  # [num_nodes, M, attr_dim]
            
            # 在特征维度拼接：[num_nodes, M, 3 + attr_dim]
            input_features = torch.cat([time_series_features, expanded_static_attrs], dim=2)
        else:
            input_features = time_series_features
        
        # 转换为期望的格式：[M, num_nodes, features]
        # 从 [num_nodes, M, features] 转换为 [M, num_nodes, features]
        input_features = input_features.transpose(0, 1)  # [M, num_nodes, features]
        
        # 根据参数决定是否返回节点掩码
        if self.use_confluence_nodes_only:
            # 返回输入特征、目标数据和节点掩码
            return input_features.float(), future_obs_flow.squeeze(0).float(), self.node_mask  # [M, num_nodes, features], [num_nodes], [num_nodes]
        else:
            # 不返回节点掩码
            return input_features.float(), future_obs_flow.squeeze(0).float()  # [M, num_nodes, features], [num_nodes]
    
    def get_future_data(self, idx, forecast_steps):
        """
        获取未来的降水和LSTM数据，用于自回归预测
        
        Args:
            idx: 样本索引
            forecast_steps: 预测步数
            
        Returns:
            future_precipitation: [forecast_steps, num_nodes] 未来降水数据
            future_lstm: [forecast_steps, num_nodes] 未来LSTM数据
        """
        start_idx = idx + self.hindcast_length
        
        # 获取未来的降水和LSTM数据
        # 注意：这里的未来数据是相对于历史窗口的"未来"，即从start_idx+1开始
        end_idx = start_idx + 1 + forecast_steps
        
        # 确保不超出数据范围
        max_idx = min(end_idx, self.precipitation_data.shape[0])
        actual_steps = max_idx - (start_idx + 1)
        
        if actual_steps > 0:
            future_precipitation = self.precipitation_data[start_idx + 1:max_idx]
            future_lstm = self.lstm_streamflow_data[start_idx + 1:max_idx]
            
            # 如果实际步数少于请求步数，用最后一个值填充
            if actual_steps < forecast_steps:
                last_precip = future_precipitation[-1:].expand(forecast_steps - actual_steps, -1)
                last_lstm = future_lstm[-1:].expand(forecast_steps - actual_steps, -1)
                future_precipitation = torch.cat([future_precipitation, last_precip], dim=0)
                future_lstm = torch.cat([future_lstm, last_lstm], dim=0)
        else:
            # 如果没有未来数据，使用最后一个历史值
            num_nodes = self.precipitation_data.shape[1]
            last_precip = self.precipitation_data[start_idx:start_idx+1].expand(forecast_steps, -1)
            last_lstm = self.lstm_streamflow_data[start_idx:start_idx+1].expand(forecast_steps, -1)
            future_precipitation = last_precip
            future_lstm = last_lstm
        
        # 数据预处理：确保非负值
        future_precipitation = torch.clamp(future_precipitation, min=0)
        future_lstm = torch.clamp(future_lstm, min=0)
        
        return future_precipitation, future_lstm
    
    def get_multistep_sample(self, idx, forecast_steps):
        """
        获取多步预测的完整样本（包括输入、目标和未来数据）
        
        Args:
            idx: 样本索引
            forecast_steps: 预测步数
            
        Returns:
            input_features: [hindcast_length, num_nodes, features] 输入特征
            future_targets: [forecast_steps, num_nodes] 未来目标
            future_precipitation: [forecast_steps, num_nodes] 未来降水
            future_lstm: [forecast_steps, num_nodes] 未来LSTM
            node_mask: [num_nodes] 节点掩码（如果使用）
        """
        start_idx = idx + self.hindcast_length
        
        # 获取输入特征（与单步预测相同）
        input_features, _, node_mask = self.__getitem__(idx) if self.use_confluence_nodes_only else (*self.__getitem__(idx), None)
        
        # 获取多步目标数据 —— 保持为观测流量，不修改
        end_target_idx = start_idx + forecast_steps
        if end_target_idx <= self.observed_streamflow_data.shape[0]:
            future_targets = self.observed_streamflow_data[start_idx:end_target_idx]
        else:
            available_targets = self.observed_streamflow_data[start_idx:]
            if len(available_targets) > 0:
                last_target = available_targets[-1:].expand(forecast_steps - len(available_targets), -1)
                future_targets = torch.cat([available_targets, last_target], dim=0)
            else:
                # 如果没有可用目标，用零填充
                future_targets = torch.zeros(forecast_steps, self.observed_streamflow_data.shape[1])
        
        # 获取未来的降水和LSTM数据
        future_precipitation, future_lstm = self.get_future_data(idx, forecast_steps)
        
        if self.use_confluence_nodes_only:
            return input_features, future_targets, future_precipitation, future_lstm, node_mask
        else:
            return input_features, future_targets, future_precipitation, future_lstm

# ================================
# 模型定义
# ================================

class GConvGRULayer(nn.Module):
    """优化的GConvGRU包装层 - 支持完全批处理

    直接利用GConvGRU的批处理能力，移除内层batch循环，实现完全并行化。
    输入形状为 [batch_size, seq_len, num_nodes, in_channels]。
    """
    def __init__(self, in_channels, out_channels, K):
        super().__init__()
        self.cell = gconv_gru.GConvGRU(in_channels, out_channels, K)
        self.out_channels = out_channels

    def forward(self, X_seq, edge_index, edge_weight=None, H=None):
        # X_seq: [batch_size, seq_len, num_nodes, in_channels]
        batch_size, seq_len, num_nodes, in_channels = X_seq.shape
        
        # 初始化隐藏状态
        if H is None:
            H = torch.zeros(batch_size, num_nodes, self.out_channels, 
                          device=X_seq.device, dtype=X_seq.dtype)
        
        outputs = []
        for t in range(seq_len):
            # 取出当前时间步的所有batch数据: [batch_size, num_nodes, in_channels]
            X_t = X_seq[:, t, :, :]
            # GConvGRU支持批处理，直接传入整个batch
            H = self.cell(X_t, edge_index, edge_weight, H)
            outputs.append(H)
        
        # 返回所有时间步的输出和最后的隐藏状态
        # outputs: [seq_len, batch_size, num_nodes, out_channels]
        return torch.stack(outputs, dim=0), H

class MultiFeatureStreamGCGRU(nn.Module):
    """多特征Stream GC-GRU模型 - 使用降水、LSTM流量、实测流量和静态属性"""
    
    def __init__(self, edge_index, config, node_attributes=None):
        super(MultiFeatureStreamGCGRU, self).__init__()
        
        # 将edge_index注册为buffer，这样它会自动跟随模型移动到正确的设备
        self.register_buffer('edge_index', edge_index)
        self.config = config
        self.node_attributes = node_attributes
        
        # 输入特征维度：3个时间序列特征 + 静态属性维度
        self.time_series_dim = 3  # 降水、LSTM流量、实测流量
        self.static_attr_dim = config['node_attr_dim'] if config['use_node_attr'] and node_attributes is not None else 0
        self.input_feature_dim = self.time_series_dim + self.static_attr_dim
        
        # GConvGRU层（外部时间循环）：直接使用多通道输入特征，时间维度由外层循环处理
        self.gcgru_layer = GConvGRULayer(
            in_channels=self.input_feature_dim,
            out_channels=config['gcgru_out_channels'],
            K=config['gcgru_K']
        )
        
        # 输出MLP
        self.output_mlp = nn.Sequential(
            nn.Linear(config['gcgru_out_channels'], config['mlp_hidden_dim']),
            nn.ReLU(),
            nn.Dropout(config['mlp_dropout']),
            nn.Linear(config['mlp_hidden_dim'], config['forecast_length'])
        )
        
        print(f"多特征模型创建完成:")
        print(f"  - 时间序列特征维度: {self.time_series_dim}")
        print(f"  - 静态属性维度: {self.static_attr_dim}")
        print(f"  - 原始输入特征维度: {self.input_feature_dim}")
        print(f"  - GConvGRU输出维度: {config['gcgru_out_channels']}")
        print(f"  - 预测长度: {config['forecast_length']}")
        
    def forward(self, x):
        """前向传播 - 优化的批处理版本，移除外层batch循环"""
        # x: [batch_size, hindcast_length, num_nodes, features]
        batch_size, seq_len, num_nodes, features = x.shape

        # 直接使用批处理的GConvGRULayer，无需逐batch循环
        # 输入: [batch_size, seq_len, num_nodes, features]
        # 输出: outputs: [seq_len, batch_size, num_nodes, gcgru_out_channels]
        #       h_last: [batch_size, num_nodes, gcgru_out_channels]
        outputs, h_last = self.gcgru_layer(x, self.edge_index, edge_weight=None, H=None)

        # 步骤2: 输出预测（基于最后一个隐藏状态）
        predictions = self.output_mlp(h_last)  # [batch_size, num_nodes, forecast_length]

        # 转换维度顺序以匹配targets的格式：[batch_size, forecast_length, num_nodes]
        if self.config['forecast_length'] > 1:
            predictions = predictions.transpose(1, 2)  # [batch_size, forecast_length, num_nodes]
        else:
            # 单步预测时去掉最后一个维度
            predictions = predictions.squeeze(-1)  # [batch_size, num_nodes]

        return predictions
    
    def autoregressive_predict(self, x, forecast_steps, precipitation_future=None, lstm_future=None, targets_future=None, teacher_forcing_ratio=0.0):
        """
        自回归多步预测
        
        Args:
            x: 初始输入序列 [batch_size, hindcast_length, num_nodes, features]
            forecast_steps: 预测步数
            precipitation_future: 未来降水数据 [batch_size, forecast_steps, num_nodes] (可选)
            lstm_future: 未来LSTM预测数据 [batch_size, forecast_steps, num_nodes] (可选)
            
        Returns:
            predictions: [batch_size, forecast_steps, num_nodes] 多步预测结果
        """
        batch_size, seq_len, num_nodes, features = x.shape
        device = x.device
        
        # 存储所有预测结果
        all_predictions = []
        
        # 当前输入序列（滑动窗口）
        current_input = x.clone()  # [batch_size, hindcast_length, num_nodes, features]
        
        for step in range(forecast_steps):
            # 单步预测（保留梯度用于训练）
            logits = self.forward(current_input)  # [batch_size, forecast_length, num_nodes] 或 [batch_size, num_nodes]
            # 取第1个预测步作为单步预测
            if logits.dim() == 3:
                step_prediction = logits[:, 0, :]  # [batch_size, num_nodes]
            else:
                step_prediction = logits  # 已是单步
            
            # 保存当前步的预测用于损失计算
            step_pred_unsq = step_prediction.unsqueeze(1)  # [batch_size, 1, num_nodes]
            all_predictions.append(step_pred_unsq)
            
            # 准备下一步的输入（除最后一步外）
            if step < forecast_steps - 1:
                # 未来外生量（若未提供则回退为最后时间步）
                if precipitation_future is not None:
                    future_precip = precipitation_future[:, step:step+1, :]
                else:
                    future_precip = current_input[:, -1:, :, 0]
                if lstm_future is not None:
                    future_lstm = lstm_future[:, step:step+1, :]
                else:
                    future_lstm = current_input[:, -1:, :, 1]
                
                # teacher forcing: 用真实流量替代下一步输入的预测流量（按比率）
                input_flow = step_pred_unsq  # 默认用预测
                if (targets_future is not None) and (teacher_forcing_ratio > 0.0):
                    gt_flow = targets_future[:, step:step+1, :]  # [batch_size, 1, num_nodes]
                    tf_mask = (torch.rand(batch_size, 1, 1, device=device) < teacher_forcing_ratio)
                    input_flow = torch.where(tf_mask, gt_flow, input_flow)
                
                # 构建新的时间步特征 [batch_size, 1, num_nodes, 3]
                new_time_step = torch.stack([
                    future_precip,              # 降水 [batch_size, 1, num_nodes]
                    future_lstm,                # LSTM流量 [batch_size, 1, num_nodes]
                    input_flow                  # 实测流量（预测或真值） [batch_size, 1, num_nodes]
                ], dim=3)
                
                # 拼接静态属性
                if self.static_attr_dim > 0:
                    static_attrs = current_input[:, -1:, :, 3:]  # [batch_size, 1, num_nodes, static_attr_dim]
                    new_time_step = torch.cat([new_time_step, static_attrs], dim=3)
                
                # 滑动窗口前移并追加新时间步
                current_input = torch.cat([current_input[:, 1:, :, :], new_time_step], dim=1)
        
        predictions = torch.cat(all_predictions, dim=1)
        return predictions

# ================================
# 损失函数和评估指标
# ================================

def masked_mse_loss(predictions, targets, mask=None, node_mask=None):
    """带掩码的RMSE损失（逐节点均值），适配forecast_length=1及节点选择性训练"""
    # 统一预测与目标为三维 [batch, time, nodes]
    if predictions.dim() == 2 and targets.dim() == 2:
        predictions = predictions.unsqueeze(1)
        targets = targets.unsqueeze(1)
    elif predictions.dim() == 3 and targets.dim() == 2:
        targets = targets.unsqueeze(1)
    elif predictions.dim() == 2 and targets.dim() == 3:
        predictions = predictions.unsqueeze(1)

    # 数据掩码：默认忽略targets中的NaN
    if mask is None:
        mask = ~torch.isnan(targets)
    else:
        mask = mask.bool()
        # 将mask扩展到与targets一致的形状
        if mask.dim() == 1:  # [num_nodes]
            mask = mask.unsqueeze(0).unsqueeze(0).expand_as(targets)
        elif mask.dim() == 2 and targets.dim() == 3:  # [batch_size, num_nodes]
            mask = mask.unsqueeze(1).expand_as(targets)
        elif mask.shape != targets.shape:
            try:
                mask = mask.expand_as(targets)
            except Exception:
                if mask.dim() == 2 and mask.shape[-1] == targets.shape[-1]:
                    mask = mask.unsqueeze(1).expand_as(targets)

    # 节点掩码：仅对指定节点计算损失
    if node_mask is not None:
        nm = node_mask.bool()
        if nm.dim() == 1:  # [num_nodes]
            nm = nm.unsqueeze(0).unsqueeze(0).expand_as(targets)
        elif nm.dim() == 2 and targets.dim() == 3:  # [batch_size, num_nodes]
            nm = nm.unsqueeze(1).expand_as(targets)
        elif nm.shape != targets.shape:
            try:
                nm = nm.expand_as(targets)
            except Exception:
                pass
        mask = mask & nm

    # 逐节点MSE：对每个节点在所有 batch/time 的有效元素求 MSE，再对节点均值
    # 将无效位置的误差置零以便做节点聚合
    se = (predictions - targets) ** 2
    se = torch.where(mask, se, torch.zeros_like(se))

    # 每节点的有效计数与误差累积
    valid_counts = mask.sum(dim=(0, 1))      # [num_nodes]
    per_node_sum = se.sum(dim=(0, 1))        # [num_nodes]

    # 有效节点：至少有一个有效元素
    valid_nodes = valid_counts > 0
    if valid_nodes.sum() == 0:
        return torch.tensor(0.0, device=predictions.device, requires_grad=True)

    per_node_mse = per_node_sum[valid_nodes] / valid_counts[valid_nodes].clamp_min(1)
    mse = per_node_mse.mean()
    rmse = torch.sqrt(mse + 1e-12)
    return rmse

def calculate_nse(predictions, targets, mask=None, node_mask=None):
    """计算NSE指标，支持节点选择性评估"""
    if mask is None:
        mask = ~torch.isnan(targets)
    
    # 如果提供了节点掩码，只对指定节点计算NSE
    if node_mask is not None:
        # 处理node_mask的维度，确保它与targets兼容
        # node_mask可能是 [num_nodes] 或 [batch_size, num_nodes] 或 [batch_size, num_nodes, forecast_length]
        
        # 如果node_mask是1维的，扩展到与targets相同的形状
        if len(node_mask.shape) == 1:  # [num_nodes]
            if len(targets.shape) == 3:  # [batch_size, forecast_length, num_nodes]
                # node_mask: [num_nodes] -> [1, 1, num_nodes] -> [batch_size, forecast_length, num_nodes]
                expanded_node_mask = node_mask.unsqueeze(0).unsqueeze(0).expand_as(targets)
            elif len(targets.shape) == 2:  # [batch_size, num_nodes]
                expanded_node_mask = node_mask.unsqueeze(0).expand_as(targets)
            else:
                expanded_node_mask = node_mask
        else:
            # 如果node_mask已经是多维的，直接使用或调整到正确的形状
            if node_mask.shape == targets.shape:
                expanded_node_mask = node_mask
            elif len(node_mask.shape) == 2 and len(targets.shape) == 3:
                # node_mask: [batch_size, num_nodes], targets: [batch_size, forecast_length, num_nodes]
                expanded_node_mask = node_mask.unsqueeze(1).expand_as(targets)
            elif len(node_mask.shape) == 3 and len(targets.shape) == 2:
                # node_mask: [batch_size, forecast_length, num_nodes], targets: [batch_size, num_nodes]
                expanded_node_mask = node_mask.squeeze(1)
            else:
                expanded_node_mask = node_mask
        
        # 结合数据掩码和节点掩码
        mask = mask & expanded_node_mask
    
    valid_predictions = predictions[mask]
    valid_targets = targets[mask]
    
    if len(valid_targets) == 0:
        return 0.0
    
    # NSE = 1 - SS_res / SS_tot
    ss_res = torch.sum((valid_targets - valid_predictions) ** 2)
    ss_tot = torch.sum((valid_targets - torch.mean(valid_targets)) ** 2)
    
    nse = 1 - ss_res / (ss_tot + 1e-8)
    
    return nse.item()

def calculate_nse_per_node(predictions, targets, node_mask=None):
    """计算每个节点的NSE，支持节点选择性评估"""
    # try:
        # 确保输入是2D张量 [samples, nodes]
    if predictions.dim() == 1:
        predictions = predictions.unsqueeze(0)
    elif predictions.dim() == 3:
        # [batch, time, nodes] -> [batch*time, nodes]
        predictions = predictions.reshape(-1, predictions.shape[-1])
    
    if targets.dim() == 1:
        targets = targets.unsqueeze(0)
    elif targets.dim() == 3:
        # [batch, time, nodes] -> [batch*time, nodes]
        targets = targets.reshape(-1, targets.shape[-1])
    
    # 确保predictions和targets形状一致
    if predictions.shape != targets.shape:
        # 如果形状不一致，尝试广播到相同形状
        if predictions.shape[0] != targets.shape[0]:
            if targets.shape[0] == 1:
                targets = targets.expand_as(predictions)
            elif predictions.shape[0] == 1:
                predictions = predictions.expand_as(targets)
            else:
                raise ValueError(f"无法匹配predictions形状{predictions.shape}和targets形状{targets.shape}")
        
    num_nodes = predictions.shape[1]
    nse_per_node = []
    
    # 处理节点掩码
    if node_mask is not None:
        # 如果node_mask是3维的，取第一个样本的掩码
        if node_mask.dim() == 3:
            node_mask = node_mask[0, :, 0]  # [batch_size, num_nodes, 1] -> [num_nodes]
        elif node_mask.dim() == 2:
            node_mask = node_mask[0, :]  # [batch_size, num_nodes] -> [num_nodes]
        # 如果是1维的，直接使用
    
    # 计算每个节点的NSE
    for node_idx in range(num_nodes):
        # 如果提供了节点掩码，检查当前节点是否为汇流节点
        if node_mask is not None and not node_mask[node_idx]:
            nse_per_node.append(-999.0)  # 源头节点不计算NSE
            continue
            
        pred_node = predictions[:, node_idx]
        target_node = targets[:, node_idx]
        
        # 移除NaN值
        mask = ~(torch.isnan(pred_node) | torch.isnan(target_node))
        if mask.sum() == 0:
            nse_per_node.append(-999.0)
            continue
            
        pred_clean = pred_node[mask]
        target_clean = target_node[mask]
        
        # 计算该节点的NSE
        numerator = torch.sum((target_clean - pred_clean) ** 2)
        denominator = torch.sum((target_clean - torch.mean(target_clean)) ** 2)
        
        if denominator == 0:
            nse_per_node.append(-999.0)
        else:
            nse = 1 - (numerator / denominator)
            nse_per_node.append(nse.item())
    
    # 计算整体NSE（只对汇流节点）
    # 改为：整体NSE = 逐节点NSE的均值（忽略 -999）
    valid_nses = [n for n in nse_per_node if n > -999]
    overall_nse = float(np.mean(valid_nses)) if len(valid_nses) > 0 else 0.0
    
    return nse_per_node, overall_nse
    # except Exception as e:
        # 如果出错，返回默认值
        # print(f"Error in calculate_nse_per_node: {e}")
        # num_nodes = predictions.shape[1] if predictions.dim() > 1 else 1
        # return [-999.0] * num_nodes, -999.0

# ================================
# 训练和评估函数
# ================================

def train_model(model, train_loader, val_loader, config, device, log_dir=None, basin_names=None):
    """训练模型"""
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = ExponentialLR(optimizer, gamma=config['lr_decay_factor'])
    
    train_losses = []
    val_losses = []
    train_nses = []
    val_nses = []
    train_nses_per_node = []
    val_nses_per_node = []
    
    # 如果num_epochs为0，跳过训练直接返回
    if config['num_epochs'] == 0:
        print("num_epochs=0，跳过训练阶段")
        return train_losses, val_losses, train_nses, val_nses, train_nses_per_node, val_nses_per_node
    
    # 创建模型保存目录
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        model_save_dir = os.path.join(log_dir, "model_checkpoints")
        os.makedirs(model_save_dir, exist_ok=True)
        # 为本次训练创建/确定一个按epoch实时写入的训练历史JSON文件路径
        training_start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        epoch_history_path = os.path.join(log_dir, f"training_history_{training_start_time}.json")
        epoch_history = []
        # 尝试获取站点名称，便于记录每节点NSE的名称映射
        epoch_basin_names = None
        if basin_names is not None:
            epoch_basin_names = list(basin_names)
        else:
            try:
                ds = train_loader.dataset
                # 若为Subset，向内层dataset取属性
                if hasattr(ds, 'basin_names'):
                    epoch_basin_names = list(ds.basin_names)
                elif hasattr(ds, 'dataset') and hasattr(ds.dataset, 'basin_names'):
                    epoch_basin_names = list(ds.dataset.basin_names)
            except Exception:
                epoch_basin_names = None
    
    best_val_nse = -float('inf')
    best_epoch = 0
    
    for epoch in range(config['num_epochs']):
        # 训练阶段
        model.train()
        # 线性退火的 teacher forcing 比例（方案B）
        num_epochs = config['num_epochs']
        tf_ratio = 1.0 - (epoch / max(num_epochs - 1, 1))
        tf_ratio = max(0.0, min(1.0, tf_ratio))
        
        train_loss = 0.0
        train_batches = 0
        train_predictions = []
        train_targets = []
        
        for batch_idx, batch_data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")):
            # 处理数据集返回的三个元素：输入、目标、节点掩码
            if len(batch_data) == 3:
                inputs, targets, node_mask = batch_data
                inputs, targets = inputs.to(device), targets.to(device)
                node_mask = node_mask[0].to(device)  # 取第一个样本的节点掩码（所有样本的节点掩码相同）
            else:
                # 兼容旧版本数据集
                inputs, targets = batch_data
                inputs, targets = inputs.to(device), targets.to(device)
                node_mask = None
            
            optimizer.zero_grad()
            outputs = model.autoregressive_predict(
                 inputs,
                 model.config['forecast_length'],
                 targets_future=targets.unsqueeze(1),  # [batch, nodes] -> [batch, 1, nodes]
                 teacher_forcing_ratio=tf_ratio
             )
            
            # 使用节点掩码计算损失（只对汇流节点计算损失）
            loss = masked_mse_loss(outputs, targets, node_mask=node_mask)
            
            # 检查损失是否为NaN或无穷大
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"警告：在epoch {epoch+1}, batch {batch_idx}中检测到异常损失值: {loss.item()}")
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            # 收集预测值和真实值用于计算NSE
            train_predictions.append(outputs.detach().cpu())
            train_targets.append(targets.detach().cpu())
        
        avg_train_loss = train_loss / max(train_batches, 1)
        train_losses.append(avg_train_loss)
        
        # 计算训练NSE（使用节点掩码只评估汇流节点）
        if train_predictions:
            train_pred_tensor = torch.cat(train_predictions, dim=0)
            train_target_tensor = torch.cat(train_targets, dim=0)
            
            # 获取节点掩码（假设所有批次的节点掩码相同）
            if 'node_mask' in locals() and node_mask is not None:
                train_nse_per_node, train_nse_overall = calculate_nse_per_node(train_pred_tensor, train_target_tensor, node_mask=node_mask.cpu())
            else:
                train_nse_per_node, train_nse_overall = calculate_nse_per_node(train_pred_tensor, train_target_tensor)
            train_nses.append(train_nse_overall)
            train_nses_per_node.append(train_nse_per_node)
        else:
            train_nses.append(-999.0)
            train_nses_per_node.append([])
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_batches = 0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(val_loader):
                # 处理数据集返回的三个元素：输入、目标、节点掩码
                if len(batch_data) == 3:
                    inputs, targets, node_mask = batch_data
                    inputs, targets = inputs.to(device), targets.to(device)
                    node_mask = node_mask[0].to(device)  # 取第一个样本的节点掩码
                else:
                    # 兼容旧版本数据集
                    inputs, targets = batch_data
                    inputs, targets = inputs.to(device), targets.to(device)
                    node_mask = None
                    
                outputs = model.autoregressive_predict(
                    inputs,
                    model.config['forecast_length'],
                    targets_future=targets.unsqueeze(1),  # [batch, nodes] -> [batch, 1, nodes]
                    teacher_forcing_ratio=0.0
                )
                
                # 使用节点掩码计算损失（只对汇流节点计算损失）
                loss = masked_mse_loss(outputs, targets, node_mask=node_mask)
                
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    val_loss += loss.item()
                    val_batches += 1
                    
                    val_predictions.append(outputs.cpu())
                    val_targets.append(targets.cpu())
        
        avg_val_loss = val_loss / max(val_batches, 1)
        val_losses.append(avg_val_loss)
        
        # 计算验证NSE（使用节点掩码只评估汇流节点）
        if val_predictions:
            val_pred_tensor = torch.cat(val_predictions, dim=0)
            val_target_tensor = torch.cat(val_targets, dim=0)
            
            # 获取节点掩码（假设所有批次的节点掩码相同）
            if 'node_mask' in locals() and node_mask is not None:
                val_nse_per_node, val_nse_overall = calculate_nse_per_node(val_pred_tensor, val_target_tensor, node_mask=node_mask.cpu())
            else:
                val_nse_per_node, val_nse_overall = calculate_nse_per_node(val_pred_tensor, val_target_tensor)
            val_nses.append(val_nse_overall)
            val_nses_per_node.append(val_nse_per_node)
        else:
            val_nses.append(-999.0)
            val_nses_per_node.append([])
        
        # 更新学习率
        scheduler.step()
        
        # 保存当前epoch的模型参数
        if log_dir is not None:
            epoch_model_path = os.path.join(model_save_dir, f"model_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_nse': train_nses[-1],
                'val_nse': val_nses[-1],
            }, epoch_model_path)
            
            # 如果当前验证NSE是最好的，保存为best_model
            if val_nses[-1] > best_val_nse:
                best_val_nse = val_nses[-1]
                best_epoch = epoch + 1
                best_model_path = os.path.join(log_dir, "best_model.pth")
                torch.save(model.state_dict(), best_model_path)
                print(f"保存最佳模型 (Epoch {best_epoch}, Val NSE: {best_val_nse:.4f})")
        
        print(f"Epoch {epoch+1}/{config['num_epochs']}: "
              f"Train Loss: {avg_train_loss:.6f}, Train NSE: {train_nses[-1]:.4f}, "
              f"Val Loss: {avg_val_loss:.6f}, Val NSE: {val_nses[-1]:.4f}, "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")

        # 按epoch实时写入训练历史JSON，便于边训边看
        if log_dir is not None:
            try:
                epoch_entry = {
                    'epoch': epoch + 1,
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'train_nse': train_nses[-1],
                    'val_nse': val_nses[-1],
                    'train_nse_per_node': train_nses_per_node[-1] if len(train_nses_per_node) > 0 else [],
                    'val_nse_per_node': val_nses_per_node[-1] if len(val_nses_per_node) > 0 else [],
                    'learning_rate': scheduler.get_last_lr()[0],
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'basin_names': epoch_basin_names
                }
                epoch_history.append(epoch_entry)
                with open(epoch_history_path, 'w') as f:
                    json.dump(epoch_history, f, indent=2)
            except Exception as e:
                print(f"警告：写入训练历史JSON失败：{e}")
    
    if log_dir is not None:
        print(f"所有模型参数已保存到: {model_save_dir}")
        print(f"最佳模型 (Epoch {best_epoch}) 已保存到: {os.path.join(log_dir, 'best_model.pth')}")
    
    return train_losses, val_losses, train_nses, val_nses, train_nses_per_node, val_nses_per_node

def evaluate_multistep_model(model, test_dataset, device, config, forecast_steps, num_samples=100, log_dir=None, epoch=None):
    """
    评估多步自回归预测模型
    
    Args:
        model: 模型
        test_dataset: 测试数据集
        device: 设备
        config: 配置
        forecast_steps: 预测步数
        num_samples: 评估样本数量
        log_dir: 日志目录
        epoch: 指定加载的epoch
        
    Returns:
        avg_loss: 平均损失
        nse_results: NSE结果
        all_predictions: 所有预测结果 [num_samples, forecast_steps, num_nodes]
        all_targets: 所有目标结果 [num_samples, forecast_steps, num_nodes]
    """
    model.to(device)
    
    # 如果指定了epoch，加载对应的模型参数
    if epoch is not None and log_dir is not None:
        model_path = os.path.join(log_dir, "model_checkpoints", f"model_epoch_{epoch}.pth")
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"已加载第{epoch}代模型参数: {model_path}")
        else:
            print(f"警告: 未找到第{epoch}代模型文件: {model_path}")
    elif log_dir is not None:
        # 尝试加载最佳模型
        best_model_path = os.path.join(log_dir, "best_model.pth")
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            print(f"已加载最佳模型参数: {best_model_path}")
        else:
            print("警告: 未找到最佳模型文件，使用当前模型参数进行评估")
    
    model.eval()
    
    # 限制评估样本数量
    max_samples = min(num_samples, len(test_dataset))
    
    all_predictions = []
    all_targets = []
    total_loss = 0.0
    
    print(f"开始多步预测评估 (预测步数: {forecast_steps}, 样本数: {max_samples})...")
    
    with torch.no_grad():
        for i in tqdm(range(max_samples), desc="多步预测评估"):
            # 获取多步样本
            if config.get('use_confluence_nodes_only', False):
                input_features, future_targets, future_precipitation, future_lstm, node_mask = test_dataset.get_multistep_sample(i, forecast_steps)
                node_mask = node_mask.to(device)
            else:
                input_features, future_targets, future_precipitation, future_lstm = test_dataset.get_multistep_sample(i, forecast_steps)
                node_mask = None
            
            # 转移到设备
            input_features = input_features.unsqueeze(0).to(device)  # 添加batch维度
            future_targets = future_targets.to(device)
            future_precipitation = future_precipitation.unsqueeze(0).to(device)  # 添加batch维度
            future_lstm = future_lstm.unsqueeze(0).to(device)  # 添加batch维度
            
            # 进行自回归预测
            predictions = model.autoregressive_predict(
                input_features, 
                forecast_steps, 
                precipitation_future=future_precipitation,
                lstm_future=future_lstm
            )  # [1, forecast_steps, num_nodes]
            
            predictions = predictions.squeeze(0)  # 移除batch维度 [forecast_steps, num_nodes]
            
            # 计算损失
            # 统一使用逐节点均值RMSE损失
            if node_mask is not None:
                loss = masked_mse_loss(predictions, future_targets, node_mask=node_mask)
            else:
                loss = masked_mse_loss(predictions, future_targets)
            total_loss += loss.item()
            
            # 保存结果
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(future_targets.cpu().numpy())
    
    # 计算平均损失
    avg_loss = total_loss / max_samples
    
    # 转换为numpy数组
    all_predictions = np.array(all_predictions)  # [num_samples, forecast_steps, num_nodes]
    all_targets = np.array(all_targets)  # [num_samples, forecast_steps, num_nodes]
    
    # 计算NSE（对每个预测步数分别计算）
    nse_results = {}
    for step in range(forecast_steps):
        step_predictions = all_predictions[:, step, :]  # [num_samples, num_nodes]
        step_targets = all_targets[:, step, :]  # [num_samples, num_nodes]
        
        # 计算每个节点的NSE
        step_nse_per_node = []
        for node_idx in range(step_predictions.shape[1]):
            node_pred = step_predictions[:, node_idx]
            node_target = step_targets[:, node_idx]
            
            # 过滤有效值
            valid_mask = ~np.isnan(node_target) & ~np.isnan(node_pred)
            if np.sum(valid_mask) > 0:
                valid_pred = node_pred[valid_mask]
                valid_target = node_target[valid_mask]
                
                # 计算NSE
                ss_res = np.sum((valid_target - valid_pred) ** 2)
                ss_tot = np.sum((valid_target - np.mean(valid_target)) ** 2)
                nse = 1 - ss_res / (ss_tot + 1e-8)
                step_nse_per_node.append(nse)
            else:
                step_nse_per_node.append(-999.0)  # 无效值
        
        nse_results[f'step_{step+1}'] = {
            'nse_per_node': step_nse_per_node,
            'overall_nse': np.mean([nse for nse in step_nse_per_node if nse > -999])
        }
    
    print(f"多步预测评估完成:")
    print(f"  - 平均损失: {avg_loss:.6f}")
    for step in range(forecast_steps):
        overall_nse = nse_results[f'step_{step+1}']['overall_nse']
        print(f"  - 第{step+1}步NSE: {overall_nse:.4f}")
    
    return avg_loss, nse_results, all_predictions, all_targets

def evaluate_model(model, test_loader, device, config, log_dir=None, epoch=None):
    """评估模型
    
    Args:
        model: 模型
        test_loader: 测试数据加载器
        device: 设备
        config: 配置
        log_dir: 日志目录
        epoch: 指定加载的epoch，如果为None则加载best_model.pth
    """
    model.to(device)
    
    # 如果指定了epoch，加载对应的模型参数
    if epoch is not None and log_dir is not None:
        model_path = os.path.join(log_dir, "model_checkpoints", f"model_epoch_{epoch}.pth")
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"已加载第 {epoch} 代的模型参数: {model_path}")
        else:
            print(f"警告: 未找到第 {epoch} 代的模型文件: {model_path}")
            print("将使用当前模型参数进行评估")
    elif log_dir is not None:
        # 默认加载best_model.pth
        best_model_path = os.path.join(log_dir, "best_model.pth")
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            print(f"已加载最佳模型参数: {best_model_path}")
        else:
            print(f"警告: 未找到最佳模型文件: {best_model_path}")
            print("将使用当前模型参数进行评估")
    
    model.eval()
    criterion = nn.MSELoss()
    test_loss = 0.0
    test_batches = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc="Evaluating"):
            # 处理数据集返回的三个元素：输入、目标、节点掩码
            if len(batch_data) == 3:
                inputs, targets, node_mask = batch_data
                inputs, targets = inputs.to(device), targets.to(device)
                node_mask = node_mask[0].to(device)  # 取第一个样本的节点掩码（所有样本的节点掩码相同）
            else:
                # 兼容旧版本数据集
                inputs, targets = batch_data
                inputs, targets = inputs.to(device), targets.to(device)
                node_mask = None
            
            outputs = model(inputs)
            
            # 使用统一的掩码损失函数，自动处理维度和节点掩码
            loss = masked_mse_loss(outputs, targets, node_mask=node_mask)
            
            test_loss += loss.item()
            test_batches += 1
            
            all_predictions.append(outputs.cpu())
            all_targets.append(targets.cpu())
    
    avg_test_loss = test_loss / test_batches
    
    # 计算NSE
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # 使用节点掩码计算NSE
    if hasattr(test_loader.dataset, 'use_confluence_nodes_only') and test_loader.dataset.use_confluence_nodes_only:
        # 获取节点掩码
        node_mask = test_loader.dataset.node_mask
        nse_per_node, overall_nse = calculate_nse_per_node(all_predictions, all_targets, node_mask=node_mask)
    else:
        nse_per_node, overall_nse = calculate_nse_per_node(all_predictions, all_targets)
    
    print(f"Test Loss: {avg_test_loss:.6f}, Test NSE: {overall_nse:.4f}")
    
    return avg_test_loss, overall_nse, all_predictions, all_targets

def save_results(model, train_losses, val_losses, train_nses, val_nses, train_nses_per_node, val_nses_per_node, 
                test_loss, test_nse, test_predictions, test_targets, basin_names, config, result_dir, multistep_nse_results=None, normalization_params=None):
    """保存训练结果"""
    os.makedirs(result_dir, exist_ok=True)
    
    # 保存模型
    model_path = os.path.join(result_dir, "best_model.pth")
    torch.save(model.state_dict(), model_path)
    
    # 保存训练历史
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_nses': train_nses,
        'val_nses': val_nses,
        'train_nses_per_node': train_nses_per_node,
        'val_nses_per_node': val_nses_per_node,
        'test_loss': test_loss,
        'test_nse': test_nse,
        'basin_names': basin_names,
        'config': config,
        'multistep_nse_results': multistep_nse_results
    }
    
    # 在文件名中加入当前时间
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_path = os.path.join(result_dir, f"training_history_{current_time}.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # 保存evaluate_and_plot.py需要的npz文件
    if test_predictions is not None and test_targets is not None:
        # 转换为numpy数组
        if isinstance(test_predictions, torch.Tensor):
            predicted = test_predictions.numpy()
        else:
            predicted = test_predictions
            
        if isinstance(test_targets, torch.Tensor):
            observed = test_targets.numpy()
        else:
            observed = test_targets
        
        # 确保数据是3维的 (batch_size, time_steps, num_nodes)
        # 如果是2维的 (time_steps, num_nodes)，添加batch维度
        if predicted.ndim == 2:
            predicted = predicted[np.newaxis, :, :]  # 添加batch维度
        if observed.ndim == 2:
            observed = observed[np.newaxis, :, :]    # 添加batch维度
        
        # 找到目标站点的索引
        target_station = config.get('target_station', DATA_CONFIG['target_station'])
        try:
            output_idx = basin_names.index(target_station)
        except ValueError:
            # 如果找不到目标站点，使用最后一个站点
            output_idx = len(basin_names) - 1
            print(f"警告: 未找到目标站点 {target_station}，使用索引 {output_idx}")
        
        # 如果提供了归一化参数，进行反归一化
        if normalization_params is not None:
            print("正在进行反归一化处理...")
            predicted_denorm = denormalize_time_series(predicted, normalization_params, data_type='obs_flow')
            observed_denorm = denormalize_time_series(observed, normalization_params, data_type='obs_flow')
            
            # 保存反归一化后的结果
            evaluation_results_path = os.path.join(result_dir, "evaluation_results.npz")
            np.savez(evaluation_results_path,
                    predicted=predicted_denorm,
                    observed=observed_denorm,
                    basin_names=np.array(basin_names),
                    output_idx=output_idx,
                    normalization_params=normalization_params)
            
            print(f"反归一化后的评估结果已保存到: {evaluation_results_path}")
            print(f"数据形状: predicted={predicted_denorm.shape}, observed={observed_denorm.shape}")
            print(f"目标站点: {target_station} (索引: {output_idx})")
        else:
            # 保存原始归一化后的结果
            evaluation_results_path = os.path.join(result_dir, "evaluation_results.npz")
            np.savez(evaluation_results_path,
                    predicted=predicted,
                    observed=observed,
                    basin_names=np.array(basin_names),
                    output_idx=output_idx)
            
            print(f"评估结果已保存到: {evaluation_results_path} (未进行反归一化)")
            print(f"数据形状: predicted={predicted.shape}, observed={observed.shape}")
            print(f"目标站点: {target_station} (索引: {output_idx})")
    
    # 保存每个节点的NSE结果（支持单步与多步）
    nse_path = os.path.join(result_dir, "nse_per_node.csv")
    if multistep_nse_results is not None:
        # 多步：将每一步的每节点NSE写成多列 step_1_nse, step_2_nse, ...
        # 构造行：每个basin一行，包含所有步的NSE
        # 确定最大步数
        steps = sorted([k for k in multistep_nse_results.keys() if k.startswith('step_')], key=lambda s: int(s.split('_')[1]))
        rows = []
        for i, basin in enumerate(basin_names):
            row = {'basin': basin}
            for s in steps:
                nse_list = multistep_nse_results[s].get('nse_per_node', [])
                row[f"{s}_nse"] = nse_list[i] if i < len(nse_list) else -999.0
            rows.append(row)
        # 写入CSV，字段名包含所有step列
        fieldnames = ['basin'] + [f"{s}_nse" for s in steps]
        with open(nse_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    elif hasattr(test_nse, '__len__') and len(test_nse) > 1:
        # 单步：test_nse为每节点NSE列表
        rows = []
        for i, basin in enumerate(basin_names):
            rows.append({'basin': basin, 'test_nse': test_nse[i] if i < len(test_nse) else -999.0})
        with open(nse_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['basin', 'test_nse'])
            writer.writeheader()
            writer.writerows(rows)
    
    print(f"结果已保存到: {result_dir}")

def evaluate_only(epoch=None, forecast_steps=1, multistep_eval=False, eval_samples=100):
    """仅评估模式，加载指定epoch的模型进行评估"""
    # 设置随机种子
    set_seed(TRAIN_CONFIG['random_seed'])
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载多特征数据
    print("加载多特征数据...")
    (train_precip, train_lstm_flow, train_obs_flow,
     test_precip, test_lstm_flow, test_obs_flow,
     train_discharge, test_discharge,
     basin_names, normalization_params) = load_multifeature_data(
        PATH_CONFIG['data_dir'], 
        DATA_CONFIG['train_start'], DATA_CONFIG['train_end'],
        DATA_CONFIG['test_start'], DATA_CONFIG['test_end']
    )
    
    # 加载节点属性
    print("加载节点属性...")
    node_attributes = load_node_attributes(
        PATH_CONFIG['data_dir'], 
        basin_names, 
        DATA_CONFIG['selected_attrs']
    )
    
    # 生成图结构
    print("生成图结构...")
    dg, edge_indices, node_mapping = gen_nx_graph(DATA_CONFIG['basin_id'], basin_names)
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    
    # 创建测试数据集
    print("创建测试数据集...")
    test_dataset = MultiFeatureStreamDataset(
        MODEL_CONFIG['hindcast_length'], MODEL_CONFIG['forecast_length'],
        test_precip, test_lstm_flow, test_obs_flow,
        basin_names, DATA_CONFIG['basin_id'], node_attributes,
        use_confluence_nodes_only=TRAIN_CONFIG['use_confluence_nodes_only'],
        discharge_data=test_discharge
    )
    
    test_loader = DataLoader(test_dataset, batch_size=TRAIN_CONFIG['batch_size'], 
                            shuffle=False, num_workers=TRAIN_CONFIG['num_workers'])
    
    print(f"测试集大小: {len(test_dataset)}")
    
    # 创建模型
    print("创建模型...")
    model = MultiFeatureStreamGCGRU(edge_index, MODEL_CONFIG, node_attributes)
    
    # 评估模型
    if multistep_eval and forecast_steps > 1:
        print(f"进行多步预测评估 (epoch={epoch}, 预测步数={forecast_steps})...")
        test_loss, nse_results, test_predictions, test_targets = evaluate_multistep_model(
            model, test_dataset, device, TRAIN_CONFIG, forecast_steps, eval_samples,
            log_dir=PATH_CONFIG['log_dir'], epoch=epoch
        )
        
        print(f"多步预测评估结果:")
        print(f"  - 平均测试损失: {test_loss:.6f}")
        for step in range(forecast_steps):
            overall_nse = nse_results[f'step_{step+1}']['overall_nse']
            print(f"  - 第{step+1}步NSE: {overall_nse:.4f}")
        
        # 保存多步预测结果
        # 将多步预测结果重塑为与单步预测兼容的格式
        # test_predictions: [num_samples, forecast_steps, num_nodes]
        # test_targets: [num_samples, forecast_steps, num_nodes]
        # 为了兼容现有的save_results函数，我们将其重塑为 [num_samples*forecast_steps, num_nodes]
        reshaped_predictions = test_predictions.reshape(-1, test_predictions.shape[-1])
        reshaped_targets = test_targets.reshape(-1, test_targets.shape[-1])
        
        # 使用第一步的每节点NSE用于CSV保存
        test_nse_per_node = nse_results['step_1']['nse_per_node']
        
        save_results(model, [], [], [], [], [], [],
                    test_loss, test_nse_per_node, reshaped_predictions, reshaped_targets, basin_names, 
                    {**TRAIN_CONFIG, **MODEL_CONFIG, **DATA_CONFIG, 'forecast_steps': forecast_steps}, 
                    PATH_CONFIG['result_dir'], multistep_nse_results=nse_results, normalization_params=normalization_params)
        
        return test_loss, nse_results
    else:
        print(f"进行单步预测评估 (epoch={epoch})...")
        test_loss, test_nse, test_predictions, test_targets = evaluate_model(
            model, test_loader, device, TRAIN_CONFIG, log_dir=PATH_CONFIG['log_dir'], epoch=epoch
        )
        
        print(f"单步预测评估结果:")
        print(f"  - 测试损失: {test_loss:.6f}")
        print(f"  - 测试NSE: {test_nse:.4f}")
        
        # 保存结果（写每节点NSE以输出 CSV）
        # 重新计算每节点NSE列表，以触发 nse_per_node.csv 保存
        if hasattr(test_loader.dataset, 'use_confluence_nodes_only') and test_loader.dataset.use_confluence_nodes_only:
            node_mask = test_loader.dataset.node_mask
            per_node_nse, _ = calculate_nse_per_node(test_predictions, test_targets, node_mask=node_mask)
        else:
            per_node_nse, _ = calculate_nse_per_node(test_predictions, test_targets)
        
        save_results(model, [], [], [], [], [], [],
                    test_loss, per_node_nse, test_predictions, test_targets, basin_names, 
                    {**TRAIN_CONFIG, **MODEL_CONFIG, **DATA_CONFIG}, PATH_CONFIG['result_dir'], normalization_params=normalization_params)
        
        return test_loss, test_nse

# ================================
# 主函数
# ================================

def main():
    """主函数"""
    
    # 如果设置为仅评估模式，进入评估流程
    if EVAL_CONFIG['eval_only']:
        print("进入仅评估模式...")
        evaluate_only(
            epoch=EVAL_CONFIG['eval_epoch'],
            forecast_steps=EVAL_CONFIG['forecast_steps'],
            multistep_eval=EVAL_CONFIG['multistep_eval'],
            eval_samples=EVAL_CONFIG['eval_samples']
        )
        return
    
    # 设置随机种子
    set_seed(TRAIN_CONFIG['random_seed'])
    
    # 设置设备
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(PATH_CONFIG['result_dir'], exist_ok=True)
    os.makedirs(PATH_CONFIG['log_dir'], exist_ok=True)
    
    # 加载多特征数据
    print("加载多特征数据...")
    (train_precip, train_lstm_flow, train_obs_flow,
     test_precip, test_lstm_flow, test_obs_flow,
     train_discharge, test_discharge,
     basin_names, normalization_params) = load_multifeature_data(
        PATH_CONFIG['data_dir'], 
        DATA_CONFIG['train_start'], DATA_CONFIG['train_end'],
        DATA_CONFIG['test_start'], DATA_CONFIG['test_end']
    )
    
    # 加载节点属性
    print("加载节点属性...")
    node_attributes = load_node_attributes(
        PATH_CONFIG['data_dir'], 
        basin_names, 
        DATA_CONFIG['selected_attrs']
    )
    
    # 生成图结构
    print("生成图结构...")
    dg, edge_indices, node_mapping = gen_nx_graph(DATA_CONFIG['basin_id'], basin_names)
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    
    print(f"图结构信息:")
    print(f"  - 节点数: {len(basin_names)}")
    print(f"  - 边数: {len(edge_indices)}")
    
    # 创建数据集
    print("创建数据集...")
    train_dataset = MultiFeatureStreamDataset(
        MODEL_CONFIG['hindcast_length'], MODEL_CONFIG['forecast_length'],
        train_precip, train_lstm_flow, train_obs_flow,
        basin_names, DATA_CONFIG['basin_id'], node_attributes,
        use_confluence_nodes_only=TRAIN_CONFIG['use_confluence_nodes_only'],
        discharge_data=train_discharge
    )
    
    test_dataset = MultiFeatureStreamDataset(
        MODEL_CONFIG['hindcast_length'], MODEL_CONFIG['forecast_length'],
        test_precip, test_lstm_flow, test_obs_flow,
        basin_names, DATA_CONFIG['basin_id'], node_attributes,
        use_confluence_nodes_only=TRAIN_CONFIG['use_confluence_nodes_only'],
        discharge_data=test_discharge
    )
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    # 创建数据加载器
    train_loader = DataLoader(train_subset, batch_size=TRAIN_CONFIG['batch_size'], 
                             shuffle=True, num_workers=TRAIN_CONFIG['num_workers'])
    val_loader = DataLoader(val_subset, batch_size=TRAIN_CONFIG['batch_size'], 
                           shuffle=False, num_workers=TRAIN_CONFIG['num_workers'])
    test_loader = DataLoader(test_dataset, batch_size=TRAIN_CONFIG['batch_size'], 
                            shuffle=False, num_workers=TRAIN_CONFIG['num_workers'])
    
    print(f"数据集大小:")
    print(f"  - 训练集: {len(train_subset)}")
    print(f"  - 验证集: {len(val_subset)}")
    print(f"  - 测试集: {len(test_dataset)}")
    
    # 创建模型
    print("创建模型...")
    model = MultiFeatureStreamGCGRU(edge_index, MODEL_CONFIG, node_attributes)
    
    # 训练模型
    print("开始训练...")
    train_losses, val_losses, train_nses, val_nses, train_nses_per_node, val_nses_per_node = train_model(
        model, train_loader, val_loader, TRAIN_CONFIG, device, log_dir=PATH_CONFIG['log_dir'], basin_names=basin_names
    )
    
    # 评估模型
    print("评估模型...")
    test_loss, test_nse, test_predictions, test_targets = evaluate_model(
        model, test_loader, device, TRAIN_CONFIG, log_dir=PATH_CONFIG['log_dir']
    )
    
    print(f"最终结果:")
    print(f"  - 测试损失: {test_loss:.6f}")
    print(f"  - 测试NSE: {test_nse:.4f}")
    
    # 保存结果
    save_results(model, train_losses, val_losses, train_nses, val_nses, train_nses_per_node, val_nses_per_node,
                test_loss, test_nse, test_predictions, test_targets, basin_names, 
                {**TRAIN_CONFIG, **MODEL_CONFIG, **DATA_CONFIG}, PATH_CONFIG['result_dir'], 
                normalization_params=normalization_params)
    
    # 打印每个节点的NSE（如果test_nse是列表的话）
    if hasattr(test_nse, '__len__') and len(test_nse) > 1:
        print("\n各节点NSE结果:")
        for i, basin in enumerate(basin_names):
            nse_val = test_nse[i] if i < len(test_nse) else -999.0
            print(f"  {basin}: {nse_val:.4f}")
    else:
        print(f"\n整体NSE结果: {test_nse:.4f}")

if __name__ == "__main__":
    main()