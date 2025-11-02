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
from improved_gconv_gru import SimpleImprovedStreamGCGRU
from improved_gconv_gru_with_receptive_field import ConfigurableStreamGCGRU, RECEPTIVE_FIELD_PRESETS

# ================================
# 配置参数区域
# ================================

# 路径配置
PATH_CONFIG = {
    'data_dir': "/home/mahaoran/GNN/data_jialingjiang_29",
    'result_dir': "/home/mahaoran/GNN/result/test_jialingjiang_HF_autoregressive_29",
    'log_dir': "/home/mahaoran/GNN/logs/test_jialingjiang_HF_autoregressive_29",
}

# 训练参数配置
TRAIN_CONFIG = {
    'num_epochs': 30,            # 训练轮数
    'batch_size': 16,           # 批次大小（自回归训练通常需要较小的batch size）
    'learning_rate': 0.005,     # 学习率（自回归模型通常需要较小的学习率）
    'lr_decay_factor': 0.95,    # 学习率衰减系数
    'max_grad_norm': 1.0,       # 梯度裁剪阈值
    'num_workers': 0,           # 数据加载器工作进程数
    'random_seed': 1234,        # 随机种子
}

# 模型参数配置
MODEL_CONFIG = {
    'hindcast_length': 30,      # 历史数据长度M（建议7-30天范围）
    'forecast_length': 3,       # 未来预报数据长度n
    'gcgru_out_channels': 32,   # GConvGRU输出通道数
    'gcgru_K': 15,              # Chebyshev滤波器大小
    'mlp_hidden_dim': 128,      # MLP隐藏层维度
    'mlp_dropout': 0.1,         # MLP Dropout率
    'node_attr_dim': 3,         # 节点静态属性维度
    'node_attr_hidden_dim': 32, # 节点属性处理隐藏层维度
    # 感受野配置
    # 'receptive_field': {
    #     'type': 'multihop',
    #     'config': {'num_hops': 2}
    # }
    'receptive_field': {
        'type': 'attention',
        'config': {'heads': 4, 'dropout': 0.1}
    }
}

# 数据配置
DATA_CONFIG = {
    'train_start': "2021-01-01",
    'train_end': "2024-12-31",
    'test_start': "2021-01-01",
    'test_end': "2024-12-31",
    'target_station': 'sanxia_60703800',
    'basin_id': 'sanxia_60703800',
    'normalization_scale': 4,
    'selected_attrs': ['area', 'p_mean', 'sgr_dk_sav'],
}

# 图拓扑配置
graph_dict = {
    'sanxia_60703800': [
        ['sanxia_60718300', 'sanxia_60703780', 'sanxia_60703800'],
        ['sanxia_60716700', 'sanxia_60715400', 'sanxia_60715955', 'sanxia_60703780', 'sanxia_60703800'],
        ['sanxia_60716250', 'sanxia_60715000', 'sanxia_60715400', 'sanxia_60715955', 'sanxia_60703780', 'sanxia_60703800'],
        ['sanxia_60701001', 'sanxia_60715000', 'sanxia_60715400', 'sanxia_60715955', 'sanxia_60703780', 'sanxia_60703800'],
        ['sanxia_60717050', 'sanxia_60715400', 'sanxia_60715955', 'sanxia_60703780', 'sanxia_60703800'],
        ['sanxia_60710650', 'sanxia_60703700', 'sanxia_60703780', 'sanxia_60703800'],
        ['sanxia_60706825', 'sanxia_60700001', 'sanxia_60700002', 'sanxia_60703700', 'sanxia_60703780', 'sanxia_60703800'],
        ['sanxia_60701750', 'sanxia_60700002', 'sanxia_60703700', 'sanxia_60703780', 'sanxia_60703800'],
        ['sanxia_60709859', 'sanxia_60703700', 'sanxia_60703780', 'sanxia_60703800'],
        ['sanxia_60713800', 'sanxia_60712000', 'sanxia_60703780', 'sanxia_60703800'],
        ['sanxia_60713730', 'sanxia_60711800', 'sanxia_60712000', 'sanxia_60703780', 'sanxia_60703800'],
        ['sanxia_60713630', 'sanxia_60711800', 'sanxia_60712000', 'sanxia_60703780', 'sanxia_60703800'],
        ['sanxia_60713400', 'sanxia_60706011', 'sanxia_60713170', 'sanxia_60711800', 'sanxia_60712000', 'sanxia_60703780', 'sanxia_60703800'],
        ['sanxia_60713270', 'sanxia_60713170', 'sanxia_60711800', 'sanxia_60712000', 'sanxia_60703780', 'sanxia_60703800'],
        ['sanxia_60712900', 'sanxia_60711800', 'sanxia_60712000', 'sanxia_60703780', 'sanxia_60703800'],
        ['sanxia_60701101', 'sanxia_60711800', 'sanxia_60712000', 'sanxia_60703780', 'sanxia_60703800'],
        ['sanxia_60711600', 'sanxia_60711800', 'sanxia_60712000', 'sanxia_60703780', 'sanxia_60703800']
    ]
}

# ================================
# 工具函数
# ================================

def gen_nx_graph(basin_id, basin_names):
    """生成网络图，确保节点索引与数据中basin顺序一致"""
    basin_path_lists = graph_dict.get(basin_id)
    dg = nx.DiGraph()
    for path in basin_path_lists:
        nx.add_path(dg, path)
    
    basin_names_list = list(basin_names)
    node_mapping = {}
    
    for node in dg.nodes:
        if node in basin_names_list:
            node_mapping[node] = basin_names_list.index(node)
        else:
            print(f"警告：图中的节点 {node} 在数据中不存在")
            continue
    
    edges = [(node_mapping[u], node_mapping[v]) for u, v in dg.edges 
             if u in node_mapping and v in node_mapping]
    
    return dg, edges, node_mapping

def get_upstream_nodes(basin_id, target_node, basin_names):
    """
    获取指定节点的直接上游节点
    
    Args:
        basin_id: 流域ID
        target_node: 目标节点名称
        basin_names: 所有节点名称列表
        
    Returns:
        upstream_indices: 直接上游节点的索引列表
    """
    dg, edges, node_mapping = gen_nx_graph(basin_id, basin_names)
    
    if target_node not in node_mapping:
        return []
    
    target_idx = node_mapping[target_node]
    upstream_indices = []
    
    # 查找直接上游节点
    for u, v in edges:
        if v == target_idx:
            upstream_indices.append(u)
    
    return upstream_indices

def interpolate_missing_data(data, method='linear'):
    """
    处理缺失数据
    
    Args:
        data: 输入数据张量 [time_steps, num_nodes]
        method: 插值方法 ('linear', 'forward', 'backward')
        
    Returns:
        interpolated_data: 插值后的数据
        mask: 原始有效数据的掩码
    """
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

def load_ts_data(data_dir, train_start, train_end, test_start, test_end):
    """加载恒江流域的NetCDF数据"""
    # 加载LSTM预测的区间流量数据
    lstm_pred_file = os.path.join(data_dir, "lstmpred.nc")
    lstm_data = xr.open_dataset(lstm_pred_file)
    
    # 加载时间序列数据（包含降水和实测流量）
    timeseries_file = os.path.join(data_dir, "timeseries_1D.nc")
    timeseries_data = xr.open_dataset(timeseries_file)
    
    # 提取basin坐标变量（站名信息）
    basin_names = timeseries_data.coords['basin'].values

    # 提取降水数据
    precipitation = timeseries_data['total_precipitation_hourly']
    
    # 提取实测流量数据
    observed_streamflow = timeseries_data['streamflow']
    
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
    
    # 提取测试期数据
    test_precip = precipitation.sel(time=slice(test_start_dt, test_end_dt))
    test_lstm_flow = lstm_streamflow.sel(time=slice(test_start_dt, test_end_dt))
    test_obs_flow = observed_streamflow.sel(time=slice(test_start_dt, test_end_dt))
    
    # 转换为torch张量并转置为[time, basin]格式
    train_precip_tensor = torch.from_numpy(train_precip.values).float().T  # [time, basin]
    train_lstm_flow_tensor = torch.from_numpy(train_lstm_flow.values).float().T  # [time, basin]
    train_obs_flow_tensor = torch.from_numpy(train_obs_flow.values).float().T  # [time, basin]
    
    test_precip_tensor = torch.from_numpy(test_precip.values).float().T  # [time, basin]
    test_lstm_flow_tensor = torch.from_numpy(test_lstm_flow.values).float().T  # [time, basin]
    test_obs_flow_tensor = torch.from_numpy(test_obs_flow.values).float().T  # [time, basin]
    
    return (train_precip_tensor, train_lstm_flow_tensor, train_obs_flow_tensor,
            test_precip_tensor, test_lstm_flow_tensor, test_obs_flow_tensor, basin_names)

def load_node_attributes(data_dir, basin_names, selected_attrs=None):
    """加载节点静态属性"""
    if selected_attrs is None:
        selected_attrs = ['ele_mt_sav', 'area', 'p_mean']
    
    attr_file = os.path.join(data_dir, "attributes.nc")
    attr_data = xr.open_dataset(attr_file)
    
    node_attrs = []
    for basin in basin_names:
        basin_attrs = []
        for attr_name in selected_attrs:
            if attr_name in attr_data.variables:
                basin_idx = list(attr_data.coords['basin'].values).index(basin)
                attr_value = float(attr_data[attr_name].values[basin_idx])
                if np.isnan(attr_value):
                    attr_value = 0.0
                basin_attrs.append(attr_value)
            else:
                print(f"警告：属性 {attr_name} 不存在，使用默认值0")
                basin_attrs.append(0.0)
        node_attrs.append(basin_attrs)
    
    node_attrs = torch.tensor(node_attrs, dtype=torch.float32)
    
    # 标准化
    for i in range(node_attrs.shape[1]):
        attr_col = node_attrs[:, i]
        mean_val = attr_col.mean()
        std_val = attr_col.std()
        if std_val > 0:
            node_attrs[:, i] = (attr_col - mean_val) / std_val
    
    print(f"加载了 {len(selected_attrs)} 个节点属性: {selected_attrs}")
    print(f"节点属性形状: {node_attrs.shape}")
    
    return node_attrs

def set_seed(seed=42):
    """设置所有随机种子以确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# ================================
# 自回归数据集类
# ================================

class AutoregressiveHengjiangDataset(TensorDataset):
    def __init__(self, hindcast_length, forecast_length, precipitation_data, 
                 lstm_streamflow_data, observed_streamflow_data, basin_names, 
                 basin_id, node_attributes=None):
        """
        自回归恒江流域数据集
        
        Args:
            hindcast_length: 历史数据时间窗口长度M
            forecast_length: 预报数据时间窗口长度n
            precipitation_data: 降水数据 [time_steps, num_stations]
            lstm_streamflow_data: LSTM模拟流量数据 [time_steps, num_stations]
            observed_streamflow_data: 实测流量数据 [time_steps, num_stations]
            basin_names: 流域站点名称列表
            basin_id: 流域ID
            node_attributes: 节点静态属性 [num_stations, num_attributes]
        """
        super().__init__(precipitation_data, lstm_streamflow_data, observed_streamflow_data)
        self.tensors = (precipitation_data, lstm_streamflow_data, observed_streamflow_data)
        self.hindcast_length = hindcast_length
        self.forecast_length = forecast_length
        self.node_attributes = node_attributes
        self.basin_names = basin_names
        self.basin_id = basin_id
        
        # 处理缺失数据
        self.precipitation_data, self.precip_mask = interpolate_missing_data(precipitation_data)
        self.lstm_streamflow_data, self.lstm_mask = interpolate_missing_data(lstm_streamflow_data)
        self.observed_streamflow_data, self.obs_mask = interpolate_missing_data(observed_streamflow_data)
        
        # 为每个节点构建上游节点映射
        self.upstream_mapping = {}
        for i, basin_name in enumerate(basin_names):
            upstream_indices = get_upstream_nodes(basin_id, basin_name, basin_names)
            self.upstream_mapping[i] = upstream_indices
        
        print(f"上游节点映射: {self.upstream_mapping}")

    def __len__(self):
        return len(self.tensors[0]) - self.hindcast_length - self.forecast_length + 1

    def __getitem__(self, idx):
        start_idx = idx + self.hindcast_length
        
        # 历史数据
        past_rain = self.precipitation_data[idx : start_idx]  # [M, num_nodes]
        past_lstm_flow = self.lstm_streamflow_data[idx : start_idx]  # [M, num_nodes]
        past_obs_flow = self.observed_streamflow_data[idx : start_idx]  # [M, num_nodes]
        
        # 未来数据
        future_rain = self.precipitation_data[start_idx : start_idx + self.forecast_length]  # [n, num_nodes]
        future_lstm_flow = self.lstm_streamflow_data[start_idx : start_idx + self.forecast_length]  # [n, num_nodes]
        future_obs_flow = self.observed_streamflow_data[start_idx : start_idx + self.forecast_length]  # [n, num_nodes] (目标)
        
        # 构建上游实测流量数据
        past_upstream_flow = self._extract_upstream_flows(past_obs_flow)  # [M, num_nodes]
        
        # 数据预处理：确保非负值
        past_rain = torch.clamp(past_rain, min=0)
        future_rain = torch.clamp(future_rain, min=0)
        past_lstm_flow = torch.clamp(past_lstm_flow, min=0)
        future_lstm_flow = torch.clamp(future_lstm_flow, min=0)
        past_obs_flow = torch.clamp(past_obs_flow, min=0)
        past_upstream_flow = torch.clamp(past_upstream_flow, min=0)
        
        # 构建输入样本
        if self.node_attributes is not None:
            sample = (
                past_rain.float(),           # 历史降水
                past_lstm_flow.float(),      # 历史LSTM流量
                past_upstream_flow.float(),  # 历史上游实测流量
                future_rain.float(),         # 未来降水
                future_lstm_flow.float(),    # 未来LSTM流量
                self.node_attributes.float() # 节点属性
            )
        else:
            sample = (
                past_rain.float(),
                past_lstm_flow.float(),
                past_upstream_flow.float(),
                future_rain.float(),
                future_lstm_flow.float()
            )
        
        return sample, future_obs_flow.float()
    
    def _extract_upstream_flows(self, obs_flow_data):
        """
        提取上游节点的实测流量数据，使用面积加权聚合
        
        Args:
            obs_flow_data: 实测流量数据 [time_steps, num_nodes]
            
        Returns:
            upstream_flows: 上游流量数据 [time_steps, num_nodes]
        """
        time_steps, num_nodes = obs_flow_data.shape
        upstream_flows = torch.zeros_like(obs_flow_data)
        
        for node_idx in range(num_nodes):
            upstream_indices = self.upstream_mapping.get(node_idx, [])
            
            if upstream_indices:
                # 使用面积加权聚合上游节点流量
                upstream_data = obs_flow_data[:, upstream_indices]  # [time_steps, num_upstream]
                
                if hasattr(self, 'node_areas') and self.node_areas is not None:
                    # 获取上游节点的面积权重
                    upstream_areas = self.node_areas[upstream_indices]  # [num_upstream]
                    
                    # 归一化权重，避免除零错误
                    area_weights = upstream_areas / (upstream_areas.sum() + 1e-8)  # [num_upstream]
                    
                    # 扩展权重维度以匹配时间步
                    area_weights = area_weights.unsqueeze(0).expand(time_steps, -1)  # [time_steps, num_upstream]
                    
                    # 面积加权求和
                    upstream_flows[:, node_idx] = (upstream_data * area_weights).sum(dim=1)  # [time_steps]
                else:
                    # 如果没有面积信息，回退到均值
                    upstream_flows[:, node_idx] = upstream_data.mean(dim=1)  # [time_steps]
            else:
                # 如果没有上游节点，使用0
                upstream_flows[:, node_idx] = 0.0
        
        return upstream_flows

# ================================
# 自回归模型定义
# ================================

class AutoregressiveStreamGCGRU(nn.Module):
    def __init__(self, edge_index, config, upstream_mapping=None, node_areas=None):
        super().__init__()
        self.hindcast_length = config['hindcast_length']
        self.forecast_length = config['forecast_length']
        self.node_attr_dim = config.get('node_attr_dim', 0)
        self.gcgru_out_channels = config['gcgru_out_channels']
        self.upstream_mapping = upstream_mapping or {}
        self.register_buffer('edge_index', edge_index)
        
        # 存储节点面积信息用于面积加权聚合
        if node_areas is not None:
            self.register_buffer('node_areas', torch.tensor(node_areas, dtype=torch.float32))
        else:
            self.node_areas = None
        
        # 历史数据处理的GConvGRU层
        self.gcgru_past_rain = gconv_gru.GConvGRU(
            in_channels=config['hindcast_length'], 
            out_channels=config['gcgru_out_channels'], 
            K=config['gcgru_K']
        )
        self.gcgru_past_lstm_flow = gconv_gru.GConvGRU(
            in_channels=config['hindcast_length'], 
            out_channels=config['gcgru_out_channels'], 
            K=config['gcgru_K']
        )
        self.gcgru_past_upstream_flow = gconv_gru.GConvGRU(
            in_channels=config['hindcast_length'], 
            out_channels=config['gcgru_out_channels'], 
            K=config['gcgru_K']
        )
        
        # 当前时刻数据处理的GConvGRU层（用于自回归）
        self.gcgru_current_rain = gconv_gru.GConvGRU(
            in_channels=1,  # 单天数据
            out_channels=config['gcgru_out_channels'], 
            K=config['gcgru_K']
        )
        self.gcgru_current_lstm_flow = gconv_gru.GConvGRU(
            in_channels=1,  # 单天数据
            out_channels=config['gcgru_out_channels'], 
            K=config['gcgru_K']
        )
        
        # 节点属性处理层
        if self.node_attr_dim > 0:
            self.node_attr_mlp = nn.Sequential(
                nn.Linear(self.node_attr_dim, config['node_attr_hidden_dim']),
                nn.ReLU(),
                nn.Linear(config['node_attr_hidden_dim'], config['gcgru_out_channels'])
            )
            fusion_input_dim = config['gcgru_out_channels'] * 6  # 包含节点属性
        else:
            self.node_attr_mlp = None
            fusion_input_dim = config['gcgru_out_channels'] * 5  # 不包含节点属性
        
        # 添加节点类型感知的特征权重
        self.feature_weights = nn.ModuleDict({
            'source_nodes': nn.Linear(fusion_input_dim, fusion_input_dim),
            'confluence_nodes': nn.Linear(fusion_input_dim, fusion_input_dim)
        })
        
        # 为源头节点和汇流节点设置不同的权重初始化
        self._init_feature_weights()
        
        # 融合层
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, config['mlp_hidden_dim']),
            nn.ReLU(),
            nn.Dropout(config['mlp_dropout']),
            nn.Linear(config['mlp_hidden_dim'], config['gcgru_out_channels'])
        )
        
        # 输出层（单天预测）
        self.output_linear = nn.Linear(config['gcgru_out_channels'], 1)
        
    def forward(self, x, autoregressive=True):
        """
        前向传播
        
        Args:
            x: 输入数据元组
                x[0]: past_rain [batch, M, num_nodes]
                x[1]: past_lstm_flow [batch, M, num_nodes]
                x[2]: past_upstream_flow [batch, M, num_nodes]
                x[3]: future_rain [batch, n, num_nodes]
                x[4]: future_lstm_flow [batch, n, num_nodes]
                x[5]: node_attributes [num_nodes, node_attr_dim] (可选)
            autoregressive: 是否使用自回归模式
            
        Returns:
            output: 预测结果 [batch, n, num_nodes]
        """
        batch_size = x[0].shape[0]
        num_nodes = x[0].shape[2]
        
        if autoregressive:
            return self._autoregressive_forward(x, batch_size, num_nodes)
        else:
            return self._single_step_forward(x, batch_size, num_nodes, day_idx=0)
    
    def _init_feature_weights(self):
        """初始化特征权重，体现不同节点类型的特征重要性"""
        # 获取特征权重矩阵的维度
        feature_dim = self.feature_weights['source_nodes'].weight.shape[0]
        gcgru_channels = self.gcgru_out_channels
        
        # 判断是否包含节点属性
        has_node_attr = (feature_dim == gcgru_channels * 6)  # 5个动态特征 + 1个节点属性特征
        
        if has_node_attr:
            # 包含节点属性的情况：6个特征组
            source_weights = torch.zeros(6)  # [rain_past, lstm_past, upstream_past, rain_current, lstm_current, node_attr]
            source_weights[0] = 1.0  # 历史降雨权重大
            source_weights[1] = 1.5  # LSTM流量权重中等
            source_weights[2] = 0.0  # 上游流量权重为0
            source_weights[3] = 1.0  # 当前降雨权重最大
            source_weights[4] = 3.0  # 当前LSTM流量权重小
            source_weights[5] = 1.0  # 节点属性权重中等偏大
            
            confluence_weights = torch.zeros(6)
            confluence_weights[0] = 1.0  # 历史降雨权重小
            confluence_weights[1] = 1.0  # LSTM流量权重大
            confluence_weights[2] = 4.0  # 上游流量权重最大
            confluence_weights[3] = 0.5  # 当前降雨权重小
            confluence_weights[4] = 1.5  # 当前LSTM流量权重大
            confluence_weights[5] = 1.0  # 节点属性权重中等
        else:
            # 不包含节点属性的情况：5个特征组
            source_weights = torch.zeros(5)  # [rain_past, lstm_past, upstream_past, rain_current, lstm_current]
            source_weights[0] = 1.0  # 历史降雨权重大
            source_weights[1] = 1.5  # LSTM流量权重中等
            source_weights[2] = 0.0  # 上游流量权重为0
            source_weights[3] = 1.0  # 当前降雨权重最大
            source_weights[4] = 3.0  # 当前LSTM流量权重小
            
            confluence_weights = torch.zeros(5)
            confluence_weights[0] = 1.0  # 历史降雨权重小
            confluence_weights[1] = 1.0  # LSTM流量权重大
            confluence_weights[2] = 4.0  # 上游流量权重最大
            confluence_weights[3] = 0.5  # 当前降雨权重小
            confluence_weights[4] = 1.5  # 当前LSTM流量权重大
        
        # 将权重扩展到所有通道
        source_weights_expanded = source_weights.repeat_interleave(gcgru_channels)
        confluence_weights_expanded = confluence_weights.repeat_interleave(gcgru_channels)
        
        # 初始化权重矩阵为对角矩阵乘以权重
        with torch.no_grad():
            self.feature_weights['source_nodes'].weight.copy_(
                torch.diag(source_weights_expanded)
            )
            self.feature_weights['confluence_nodes'].weight.copy_(
                torch.diag(confluence_weights_expanded)
            )
    
    def _apply_node_type_weighting(self, combined_features, batch_size, num_nodes):
        """根据节点类型应用不同的特征权重"""
        weighted_features = torch.zeros_like(combined_features)
        
        for node_idx in range(num_nodes):
            # 判断节点类型
            has_upstream = (node_idx in self.upstream_mapping and 
                          len(self.upstream_mapping[node_idx]) > 0)
            
            if has_upstream:
                # 汇流节点：使用汇流节点权重
                weighted_features[:, node_idx, :] = self.feature_weights['confluence_nodes'](
                    combined_features[:, node_idx, :]
                )
            else:
                # 源头节点：使用源头节点权重
                weighted_features[:, node_idx, :] = self.feature_weights['source_nodes'](
                    combined_features[:, node_idx, :]
                )
        
        return weighted_features



    def _single_step_forward(self, x, batch_size, num_nodes, day_idx=0):
        """单步前向传播（添加节点类型感知的特征加权）"""
        # 处理历史数据 - 输入形状: [batch, time_steps, num_nodes]
        # GConvGRU期望输入形状: [batch, num_nodes, time_steps]
        
        # 处理历史数据，使用与原始模型相同的permute操作
        past_rain_features = self.gcgru_past_rain(
            x[0].permute(0, 2, 1), self.edge_index, num_nodes=num_nodes
        )  # 输出: [batch, num_nodes, out_channels]
        
        past_lstm_flow_features = self.gcgru_past_lstm_flow(
            x[1].permute(0, 2, 1), self.edge_index, num_nodes=num_nodes
        )
        
        past_upstream_flow_features = self.gcgru_past_upstream_flow(
            x[2].permute(0, 2, 1), self.edge_index, num_nodes=num_nodes
        )
        
        # 处理当前时刻数据 - 需要添加时间维度
        # x[3][:, day_idx, :] 形状: [batch, num_nodes]
        # 需要转换为 [batch, num_nodes, 1] 以匹配GConvGRU期望的输入
        current_rain = x[3][:, day_idx:day_idx+1, :].permute(0, 2, 1)  # [batch, num_nodes, 1]
        current_lstm_flow = x[4][:, day_idx:day_idx+1, :].permute(0, 2, 1)  # [batch, num_nodes, 1]
        
        current_rain_features = self.gcgru_current_rain(
            current_rain, self.edge_index, num_nodes=num_nodes
        )  # 输出: [batch, num_nodes, out_channels]
        
        current_lstm_flow_features = self.gcgru_current_lstm_flow(
            current_lstm_flow, self.edge_index, num_nodes=num_nodes
        )
        
        # 特征融合
        if len(x) > 5 and self.node_attr_mlp is not None:
            # 处理节点属性
            node_attrs = x[5]
            node_attr_features = self.node_attr_mlp(node_attrs)
            # 如果节点属性没有batch维度，添加batch维度
            if node_attr_features.dim() == 2:
                node_attr_features = node_attr_features.unsqueeze(0).expand(batch_size, -1, -1)
            
            combined_features = torch.cat([
                past_rain_features, past_lstm_flow_features, past_upstream_flow_features,
                current_rain_features, current_lstm_flow_features, node_attr_features
            ], dim=-1)
        else:
            combined_features = torch.cat([
                past_rain_features, past_lstm_flow_features, past_upstream_flow_features,
                current_rain_features, current_lstm_flow_features
            ], dim=-1)
        
        # 应用节点类型感知的特征加权
        weighted_features = self._apply_node_type_weighting(combined_features, batch_size, num_nodes)
        
        # 通过融合层和输出层
        fused = self.fusion_mlp(weighted_features)
        output = self.output_linear(fused)  # [batch, num_nodes, 1]
        
        return output.permute(0,2,1)  # [batch, 1, num_nodes]
    
    def _autoregressive_forward(self, x, batch_size, num_nodes):
        """自回归前向传播"""
        outputs = []
        
        # 初始化历史数据
        current_past_rain = x[0].clone()  # [batch, M, num_nodes]
        current_past_lstm_flow = x[1].clone()  # [batch, M, num_nodes]
        current_past_upstream_flow = x[2].clone()  # [batch, M, num_nodes]
        
        for day in range(self.forecast_length):
            # 构建当前输入
            current_input = (
                current_past_rain,
                current_past_lstm_flow,
                current_past_upstream_flow,
                x[3],  # future_rain
                x[4],  # future_lstm_flow
            )
            
            # 添加节点属性（如果存在）
            if len(x) > 5:
                current_input = current_input + (x[5],)
            
            # 单步预测
            day_output = self._single_step_forward(
                current_input, batch_size, num_nodes, day_idx=day
            )  # [batch, 1, num_nodes]
            
            outputs.append(day_output)
            
            # 更新历史数据（滑动窗口）
            if day < self.forecast_length - 1:
                # 更新历史降水和LSTM流量
                new_rain = x[3][:, day:day+1, :]  # [batch, 1, num_nodes]
                new_lstm_flow = x[4][:, day:day+1, :]  # [batch, 1, num_nodes]
                
                current_past_rain = torch.cat([
                    current_past_rain[:, 1:, :], new_rain
                ], dim=1)
                current_past_lstm_flow = torch.cat([
                    current_past_lstm_flow[:, 1:, :], new_lstm_flow
                ], dim=1)
                
                # 更新历史上游流量（使用预测结果）
                # 这里需要根据拓扑关系更新上游流量
                new_upstream_flow = self._update_upstream_flow(
                    day_output, current_past_upstream_flow
                )  # [batch, 1, num_nodes]
                
                current_past_upstream_flow = torch.cat([
                    current_past_upstream_flow[:, 1:, :], new_upstream_flow
                ], dim=1)
        
        # 拼接所有输出
        output = torch.cat(outputs, dim=1)  # [batch, n, num_nodes]
        return output
    
    def _update_upstream_flow(self, predicted_flow, past_upstream_flow):
        """
        根据预测结果更新上游流量，使用面积加权聚合
        
        Args:
            predicted_flow: 当前预测结果 [batch, 1, num_nodes]
            past_upstream_flow: 历史上游流量 [batch, M, num_nodes]
            
        Returns:
            new_upstream_flow: 更新后的上游流量 [batch, 1, num_nodes]
        """
        batch_size, _, num_nodes = predicted_flow.shape
        new_upstream_flow = torch.zeros_like(predicted_flow)
        
        # 使用与训练阶段一致的拓扑关系计算上游流量
        for node_idx in range(num_nodes):
            # 获取当前节点的上游节点索引
            upstream_indices = self.upstream_mapping.get(node_idx, [])
            
            if upstream_indices:
                # 获取上游节点的预测流量
                upstream_flows = predicted_flow[:, :, upstream_indices]  # [batch, 1, num_upstream]
                
                if self.node_areas is not None:
                    # 使用面积加权聚合
                    upstream_areas = self.node_areas[upstream_indices]  # [num_upstream]
                    # 计算面积权重（归一化）
                    area_weights = upstream_areas / upstream_areas.sum()  # [num_upstream]
                    # 扩展权重维度以匹配流量数据
                    area_weights = area_weights.unsqueeze(0).unsqueeze(0)  # [1, 1, num_upstream]
                    # 面积加权聚合
                    new_upstream_flow[:, :, node_idx] = (upstream_flows * area_weights).sum(dim=2)
                else:
                    # 如果没有面积信息，回退到简单均值
                    new_upstream_flow[:, :, node_idx] = upstream_flows.mean(dim=2)  # [batch, 1]
            else:
                # 如果没有上游节点，使用0
                new_upstream_flow[:, :, node_idx] = 0.0
        
        return new_upstream_flow

# ================================
# 损失函数和评估函数
# ================================

def masked_mse_loss(predictions, targets, mask=None):
    """计算忽略NaN值的MSE损失"""
    if mask is None:
        mask = ~torch.isnan(targets)
    
    valid_predictions = predictions[mask]
    valid_targets = targets[mask]
    
    if valid_predictions.numel() == 0:
        return torch.tensor(0.0, device=predictions.device, requires_grad=True)
    
    loss = torch.mean((valid_predictions - valid_targets) ** 2)
    return loss

def masked_nse_loss(predictions, targets, mask=None):
    """计算忽略NaN值的NSE损失（1 - NSE，用于最小化）"""
    if mask is None:
        mask = ~torch.isnan(targets)
    
    valid_predictions = predictions[mask]
    valid_targets = targets[mask]
    
    if valid_predictions.numel() == 0:
        return torch.tensor(1.0, device=predictions.device, requires_grad=True)
    
    # 计算NSE
    numerator = torch.sum((valid_targets - valid_predictions) ** 2)
    denominator = torch.sum((valid_targets - torch.mean(valid_targets)) ** 2)
    
    if denominator == 0:
        return torch.tensor(1.0, device=predictions.device, requires_grad=True)
    
    nse = 1 - numerator / denominator
    nse_loss = 1 - nse  # 转换为损失（最小化）
    
    return nse_loss

def calculate_nse(predictions, targets, mask=None):
    """计算NSE系数"""
    if mask is None:
        mask = ~torch.isnan(targets)
    
    valid_predictions = predictions[mask]
    valid_targets = targets[mask]
    
    if valid_predictions.numel() == 0:
        return float('nan')
    
    numerator = torch.sum((valid_targets - valid_predictions) ** 2)
    denominator = torch.sum((valid_targets - torch.mean(valid_targets)) ** 2)
    
    if denominator == 0:
        return float('nan')
    
    nse = 1 - numerator / denominator
    return nse.item()

def calculate_nse_all_stations_autoregressive(model, dataset, device, basin_names, forecast_day=None, 
                                           cached_predictions=None, cached_targets=None):
    """计算自回归数据集上所有站点的NSE指标
    
    Args:
        model: 训练好的模型
        dataset: 自回归数据集
        device: 设备
        basin_names: 站点名称列表
        forecast_day: 指定预报天数(1-based)，如果为None则计算整个预报期的NSE
        cached_predictions: 缓存的预测结果，避免重复推理
        cached_targets: 缓存的目标结果，避免重复推理
    """
    # 如果没有缓存结果，进行模型推理
    if cached_predictions is None or cached_targets is None:
        model.eval()
        # 增大批处理大小以提高效率，同时确保内存不会溢出
        batch_size = 32  # 从16增加到32
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        original_samples = []
        predicted_samples = []
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = tuple(inp.to(device) for inp in inputs)
                targets = targets.to(device)
                outputs = model(inputs, autoregressive=True)
                
                # 立即移动到CPU以节省GPU内存
                original_samples.append(targets.cpu())
                predicted_samples.append(outputs.cpu())
                
                # 清理GPU缓存
                del inputs, targets, outputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # 合并收集的样本（在CPU上进行）
        original_samples = torch.cat(original_samples, dim=0)
        predicted_samples = torch.cat(predicted_samples, dim=0)
        
        # 反归一化（与原版本保持一致）
        scale_factor = DATA_CONFIG['normalization_scale']
        # 获取归一化参数 - 这里需要从数据集中获取或传入
        # 为了保持一致性，我们使用与原版本相同的反归一化方式
        # 注意：这里假设数据集有归一化参数，实际使用时可能需要调整
        min_val = 0  # 这里需要根据实际情况设置
        max_val = 1  # 这里需要根据实际情况设置
        
        # 暂时跳过反归一化，保持与当前数据一致
        # predicted_samples_denorm = min_val + (max_val - min_val) * predicted_samples / scale_factor
        # original_samples_denorm = min_val + (max_val - min_val) * original_samples / scale_factor
        predicted_samples_denorm = predicted_samples
        original_samples_denorm = original_samples
    else:
        predicted_samples_denorm = cached_predictions
        original_samples_denorm = cached_targets
    
    # 确保数据在CPU上进行NSE计算
    predicted_samples_denorm = predicted_samples_denorm.cpu()
    original_samples_denorm = original_samples_denorm.cpu()
    
    # 计算每个站点的NSE
    nse_dict = {}
    for i, station_name in enumerate(basin_names):
        if forecast_day is not None:
            # 计算指定预报天数的NSE
            day_idx = forecast_day - 1  # 转换为0-based索引
            if day_idx < predicted_samples_denorm.shape[1]:
                pred_data = predicted_samples_denorm[:, day_idx, i].flatten()
                true_data = original_samples_denorm[:, day_idx, i].flatten()
            else:
                print(f"警告：预报天数 {forecast_day} 超出范围，跳过站点 {station_name}")
                continue
        else:
            # 计算整个预报期的NSE
            pred_data = predicted_samples_denorm[:, :, i].flatten()
            true_data = original_samples_denorm[:, :, i].flatten()
        
        # 过滤掉NaN值
        valid_mask = ~(torch.isnan(pred_data) | torch.isnan(true_data))
        if valid_mask.sum() > 0:  # 确保有有效数据
            pred_valid = pred_data[valid_mask]
            true_valid = true_data[valid_mask]
            # 确保r2_score在CPU上计算
            nse = r2_score(pred_valid.cpu(), true_valid.cpu())
            nse_dict[station_name] = nse.item()
        else:
            nse_dict[station_name] = 0.0  # 没有有效数据时返回0
    
    return nse_dict, predicted_samples_denorm, original_samples_denorm

def calculate_nse_multiple_days_autoregressive(model, dataset, device, basin_names, forecast_length):
    """计算自回归模型多个预报天数的NSE指标 - 优化版本，避免重复推理"""
    nse_by_day = {}
    
    # 只进行一次模型推理，获取所有预测结果
    nse_all, cached_predictions, cached_targets = calculate_nse_all_stations_autoregressive(
        model, dataset, device, basin_names
    )
    nse_by_day['all_days'] = nse_all
    
    # 使用缓存结果计算每一天的NSE，避免重复推理
    for day in range(1, forecast_length + 1):
        nse_day, _, _ = calculate_nse_all_stations_autoregressive(
            model, dataset, device, basin_names, forecast_day=day,
            cached_predictions=cached_predictions, cached_targets=cached_targets
        )
        nse_by_day[f'day_{day}'] = nse_day
    
    return nse_by_day

def calculate_loss_on_dataset_autoregressive(model, dataset, device):
    """计算自回归数据集上的平均损失 - 优化版本"""
    model.eval()
    # 增加批处理大小以提高效率
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=True)
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = tuple(inp.to(device) for inp in inputs)
            targets = targets.to(device)
            outputs = model(inputs, autoregressive=True)
            loss = masked_mse_loss(outputs, targets)
            
            if not (loss.isnan() or loss.isinf()):
                total_loss += loss.item()
                num_batches += 1
            
            # 清理GPU内存
            del inputs, targets, outputs, loss
            torch.cuda.empty_cache()
    
    return total_loss / num_batches if num_batches > 0 else float('inf')

def save_training_metrics_json(epoch, train_loss, train_nse_dict, test_loss, test_nse_dict, current_lr, log_file):
    """保存训练指标到JSON文件"""
    # 检查文件是否存在，如果不存在则创建空列表
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = []
    
    # 添加当前epoch的数据
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    epoch_data = {
        'epoch': epoch,
        'train_loss': round(train_loss, 6),
        'train_nse': train_nse_dict,
        'test_loss': round(test_loss, 6),
        'test_nse': test_nse_dict,
        'learning_rate': current_lr,
        'timestamp': timestamp
    }
    
    data.append(epoch_data)
    
    # 保存到文件
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ================================
# 训练和评估函数
# ================================

def train_autoregressive_model(model, train_dataset, test_dataset, config, device, basin_names, log_file):
    """训练自回归模型"""
    model.to(device)
    
    # 优化器和调度器
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = ExponentialLR(optimizer, gamma=config['lr_decay_factor'])
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=config['num_workers'], 
        drop_last=True
    )
    
    print(f"开始训练，共 {config['num_epochs']} 个epoch")
    print(f"训练指标将保存到: {log_file}")
    
    # 训练循环
    for epoch in range(config['num_epochs']):
        # ===== 训练阶段 =====
        model.train()
        train_total_loss = 0.0
        train_num_batches = 0
        
        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        
        # 使用tqdm显示训练进度
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} [Train] LR:{current_lr:.6f}", leave=True)
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            # 将数据移动到设备
            inputs = tuple(inp.to(device) for inp in inputs)
            targets = targets.to(device)
            
            # 检查输入数据是否包含NaN
            if any(torch.isnan(inp).any() for inp in inputs) or torch.isnan(targets).any():
                print(f"跳过包含NaN的批次 {batch_idx}")
                continue
            
            # 前向传播（非自回归模式用于训练）
            outputs = model(inputs, autoregressive=False)
            
            # 计算损失（只使用第一天的预测进行训练）
            loss = masked_mse_loss(outputs[:, 0, :], targets[:, 0, :])
            
            # 更严格的loss检查
            if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 1000:
                print(f"跳过异常loss的批次 {batch_idx}: {loss.item()}")
                continue
            
            optimizer.zero_grad()
            loss.backward()
            
            # 添加梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['max_grad_norm'])
            
            optimizer.step()
            train_total_loss += loss.item()
            train_num_batches += 1
            
            # 更新进度条显示当前loss
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{current_lr:.6f}"})
        
        # 计算训练期平均loss
        train_avg_loss = train_total_loss / train_num_batches if train_num_batches > 0 else float('inf')
        
        # ===== 评估阶段 =====
        print(f"Epoch {epoch+1}: 开始评估...")
        
        # 计算训练期多个预报天数的NSE
        train_nse_dict = calculate_nse_multiple_days_autoregressive(
            model, train_dataset, device, basin_names, MODEL_CONFIG['forecast_length']
        )
        
        # 计算测试期loss和多个预报天数的NSE
        test_avg_loss = calculate_loss_on_dataset_autoregressive(model, test_dataset, device)
        test_nse_dict = calculate_nse_multiple_days_autoregressive(
            model, test_dataset, device, basin_names, MODEL_CONFIG['forecast_length']
        )
        
        # 打印统计信息（显示目标站点的NSE）
        target_station = DATA_CONFIG['target_station']
        target_train_nse_all = train_nse_dict['all_days'].get(target_station, 0.0)
        target_test_nse_all = test_nse_dict['all_days'].get(target_station, 0.0)
        
        print(f"Epoch [{epoch+1}/{config['num_epochs']}]:")
        print(f"  学习率: {current_lr:.6f}")
        print(f"  训练 - Loss: {train_avg_loss:.6f}")
        print(f"    {target_station} NSE (全期): {target_train_nse_all:.6f}")
        print(f"  测试 - Loss: {test_avg_loss:.6f}")
        print(f"    {target_station} NSE (全期): {target_test_nse_all:.6f}")
        
        # 保存指标到JSON文件（包含多个预报天数的NSE）
        save_training_metrics_json(
            epoch+1, train_avg_loss, train_nse_dict, test_avg_loss, 
            test_nse_dict, current_lr, log_file
        )
        
        # 保存模型
        torch.save(
            model.state_dict(), 
            os.path.join(PATH_CONFIG['result_dir'], "model_pth", f"basin_hengjiang_autoregressive_model_{epoch+1}.pth")
        )
        
        # 更新学习率（在epoch结束后）
        scheduler.step()
        next_lr = optimizer.param_groups[0]['lr']
        print(f"  下一个epoch学习率: {next_lr:.6f}")
        
        print("-" * 60)
    
    print("训练完成！")
    print(f"训练指标已保存到: {log_file}")
    
    return {
        'train_losses': [],  # 保持兼容性
        'test_losses': [],
        'train_nses': [],
        'test_nses': []
    }

def evaluate_autoregressive_model(model, test_dataset, device):
    """评估自回归模型"""
    model.eval()
    
    # 创建数据加载器
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating"):
            inputs = tuple(inp.to(device) for inp in inputs)
            targets = targets.to(device)
            
            # 自回归预测
            outputs = model(inputs, autoregressive=True)
            
            all_predictions.append(outputs.cpu())
            all_targets.append(targets.cpu())
    
    # 拼接所有结果
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    # 计算整体NSE
    overall_nse = calculate_nse(predictions, targets)
    
    # 计算每个预测天数的NSE
    day_nses = []
    for day in range(predictions.shape[1]):
        day_nse = calculate_nse(predictions[:, day, :], targets[:, day, :])
        day_nses.append(day_nse)
    
    # 计算每个节点的NSE
    node_nses = []
    for node in range(predictions.shape[2]):
        node_nse = calculate_nse(predictions[:, :, node], targets[:, :, node])
        node_nses.append(node_nse)
    
    return {
        'overall_nse': overall_nse,
        'day_nses': day_nses,
        'node_nses': node_nses,
        'predictions': predictions,
        'targets': targets
    }

# ================================
# 主函数
# ================================

def main():
    """主函数"""
    # 设置随机种子
    set_seed(TRAIN_CONFIG['random_seed'])
    
    # 创建输出目录
    os.makedirs(PATH_CONFIG['result_dir'], exist_ok=True)
    os.makedirs(PATH_CONFIG['log_dir'], exist_ok=True)
    os.makedirs(os.path.join(PATH_CONFIG['result_dir'], "model_pth"), exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 生成日志文件路径
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(PATH_CONFIG['log_dir'], f"training_metrics_{timestamp}.json")
    
    # 加载数据
    print("加载数据...")
    (train_precip, train_lstm_flow, train_obs_flow,
     test_precip, test_lstm_flow, test_obs_flow, basin_names) = load_ts_data(
        PATH_CONFIG['data_dir'], 
        DATA_CONFIG['train_start'], DATA_CONFIG['train_end'],
        DATA_CONFIG['test_start'], DATA_CONFIG['test_end']
    )
    
    print(f"训练数据形状: 降水{train_precip.shape}, LSTM流量{train_lstm_flow.shape}, 实测流量{train_obs_flow.shape}")
    print(f"测试数据形状: 降水{test_precip.shape}, LSTM流量{test_lstm_flow.shape}, 实测流量{test_obs_flow.shape}")
    print(f"流域站点数量: {len(basin_names)}")
    
    # 加载节点属性
    node_attributes = load_node_attributes(
        PATH_CONFIG['data_dir'], basin_names, DATA_CONFIG['selected_attrs']
    )
    
    # 生成图结构
    dg, edges, node_mapping = gen_nx_graph(DATA_CONFIG['basin_id'], basin_names)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    print(f"图边数: {len(edges)}")
    
    # 创建数据集
    print("创建自回归数据集...")
    train_dataset = AutoregressiveHengjiangDataset(
        MODEL_CONFIG['hindcast_length'], MODEL_CONFIG['forecast_length'],
        train_precip, train_lstm_flow, train_obs_flow,
        basin_names, DATA_CONFIG['basin_id'], node_attributes
    )
    
    test_dataset = AutoregressiveHengjiangDataset(
        MODEL_CONFIG['hindcast_length'], MODEL_CONFIG['forecast_length'],
        test_precip, test_lstm_flow, test_obs_flow,
        basin_names, DATA_CONFIG['basin_id'], node_attributes
    )
    
    print(f"训练样本数: {len(train_dataset)}")
    print(f"测试样本数: {len(test_dataset)}")
    
    # 提取面积信息用于面积加权聚合
    node_areas = None
    if node_attributes is not None and 'area' in DATA_CONFIG['selected_attrs']:
        area_idx = DATA_CONFIG['selected_attrs'].index('area')
        node_areas = node_attributes[:, area_idx].numpy()  # 提取面积列
        print(f"已加载节点面积信息，形状: {node_areas.shape}")
    
    # 创建模型
    print("创建自回归模型...")
    model = AutoregressiveStreamGCGRU(edge_index, MODEL_CONFIG, train_dataset.upstream_mapping, node_areas)
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    print(f"上游节点映射: {train_dataset.upstream_mapping}")
    print(f"面积加权聚合: {'启用' if node_areas is not None else '禁用（回退到简单均值）'}")
    
    # 如果epochs为0，加载预训练模型并仅进行评估
    if TRAIN_CONFIG['num_epochs'] == 0:
        model_path = PATH_CONFIG.get('model_path')
        if model_path and os.path.exists(model_path):
            print(f"加载预训练模型: {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=device))
            
            # 仅进行评估并保存结果
            print("开始模型评估...")
            evaluation_results = evaluate_autoregressive_model(model, test_dataset, device)
            print("评估完成！")
        else:
            print(f"错误：找不到预训练模型文件 {model_path}")
            print("请先训练模型或检查模型路径")
    else:
        # 训练模型
        print("开始训练...")
        training_history = train_autoregressive_model(
            model, train_dataset, test_dataset, TRAIN_CONFIG, device, basin_names, log_file
        )
        
        print("训练完成！")
        print(f"训练指标已保存到: {log_file}")

        # 训练完成后也保存最终的模拟结果
        print("保存最终模拟结果...")
        evaluation_results = evaluate_autoregressive_model(model, test_dataset, device)
        
        print(f"整体NSE: {evaluation_results['overall_nse']:.4f}")
        print("各天NSE:", [f"{nse:.4f}" for nse in evaluation_results['day_nses']])
        print("各节点NSE:", [f"{nse:.4f}" for nse in evaluation_results['node_nses']])
        
        # 保存评估结果
        results_path = os.path.join(PATH_CONFIG['result_dir'], 'evaluation_results.npz')
        np.savez(
            results_path,
            predictions=evaluation_results['predictions'].numpy(),
            targets=evaluation_results['targets'].numpy(),
            overall_nse=evaluation_results['overall_nse'],
            day_nses=evaluation_results['day_nses'],
            node_nses=evaluation_results['node_nses'],
            basin_names=basin_names
        )
        print(f"评估结果已保存到: {results_path}")
    
    print("自回归模型训练和评估完成！")

if __name__ == "__main__":
    main()