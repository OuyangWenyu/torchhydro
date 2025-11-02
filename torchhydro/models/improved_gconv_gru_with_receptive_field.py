#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的图卷积GRU模型 - 支持可配置感受野

基于 improved_gconv_gru.py，添加了多种感受野配置选项
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class DirectedGConv(MessagePassing):
    """
    有向图卷积层 - 原始版本（1跳感受野）
    """
    def __init__(self, in_channels, out_channels, aggr='add'):
        super().__init__(aggr=aggr)
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.lin_upstream = nn.Linear(in_channels, out_channels)
        self.lin_self = nn.Linear(in_channels, out_channels)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_upstream.weight)
        nn.init.xavier_uniform_(self.lin_self.weight)
        nn.init.zeros_(self.lin_upstream.bias)
        nn.init.zeros_(self.lin_self.bias)
        nn.init.constant_(self.alpha, 0.5)
    
    def forward(self, x, edge_index):
        upstream_info = self.propagate(edge_index, x=x)
        upstream_transformed = self.lin_upstream(upstream_info)
        self_transformed = self.lin_self(x)
        
        alpha_norm = torch.sigmoid(self.alpha)
        out = alpha_norm * upstream_transformed + (1 - alpha_norm) * self_transformed
        return out
    
    def message(self, x_j):
        return x_j

class MultiHopDirectedGConv(MessagePassing):
    """
    多跳有向图卷积层 - 扩展感受野
    """
    def __init__(self, in_channels, out_channels, num_hops=2, aggr='add'):
        super().__init__(aggr=aggr)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_hops = num_hops
        
        # 为每一跳创建变换矩阵
        self.hop_transforms = nn.ModuleList([
            nn.Linear(in_channels, out_channels) for _ in range(num_hops + 1)
        ])
        
        # 跳数权重
        self.hop_weights = nn.Parameter(torch.ones(num_hops + 1))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        for transform in self.hop_transforms:
            nn.init.xavier_uniform_(transform.weight)
            nn.init.zeros_(transform.bias)
        nn.init.ones_(self.hop_weights)
    
    def forward(self, x, edge_index):
        # 收集不同跳数的信息
        hop_outputs = [self.hop_transforms[0](x)]  # 0跳：自身
        
        current_x = x
        for hop in range(1, self.num_hops + 1):
            current_x = self.propagate(edge_index, x=current_x)
            hop_outputs.append(self.hop_transforms[hop](current_x))
        
        # 加权融合
        weights = F.softmax(self.hop_weights, dim=0)
        output = sum(w * h for w, h in zip(weights, hop_outputs))
        
        return output
    
    def message(self, x_j):
        return x_j

class AttentionDirectedGConv(MessagePassing):
    """
    基于注意力机制的有向图卷积 - 简化版本避免CUDA错误
    """
    def __init__(self, in_channels, out_channels, heads=4, dropout=0.1, aggr='add'):
        super().__init__(aggr=aggr)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        
        # 简化为单头注意力避免复杂的张量重塑
        self.lin_query = nn.Linear(in_channels, out_channels)
        self.lin_key = nn.Linear(in_channels, out_channels)
        self.lin_value = nn.Linear(in_channels, out_channels)
        self.lin_self = nn.Linear(in_channels, out_channels)
        
        # 融合参数
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_query.weight)
        nn.init.xavier_uniform_(self.lin_key.weight)
        nn.init.xavier_uniform_(self.lin_value.weight)
        nn.init.xavier_uniform_(self.lin_self.weight)
        nn.init.constant_(self.alpha, 0.5)
    
    def forward(self, x, edge_index):
        # 简化的注意力计算
        query = self.lin_query(x)
        key = self.lin_key(x)
        value = self.lin_value(x)
        
        # 注意力传播
        attended_info = self.propagate(edge_index, query=query, key=key, value=value)
        
        # 自身信息
        self_info = self.lin_self(x)
        
        # 融合
        alpha_norm = torch.sigmoid(self.alpha)
        out = alpha_norm * attended_info + (1 - alpha_norm) * self_info
        
        return out
    
    def message(self, query_i, key_j, value_j):
        # 简化的注意力分数计算
        attention_scores = torch.sum(query_i * key_j, dim=-1, keepdim=True)
        attention_weights = torch.sigmoid(attention_scores)
        
        # 应用dropout
        attention_weights = F.dropout(attention_weights, p=self.dropout, training=self.training)
        
        # 应用注意力权重
        attended_values = attention_weights * value_j
        
        return attended_values

class StackedDirectedGConv(nn.Module):
    """
    堆叠有向图卷积层 - 通过深度扩展感受野
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList()
        
        # 输入层
        self.layers.append(DirectedGConv(in_channels, hidden_channels))
        
        # 隐藏层
        for _ in range(num_layers - 2):
            self.layers.append(DirectedGConv(hidden_channels, hidden_channels))
        
        # 输出层
        self.layers.append(DirectedGConv(hidden_channels, out_channels))
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        return x

class ConfigurableFlowAwareGRU(nn.Module):
    """
    可配置感受野的流感知GRU
    """
    def __init__(self, input_dim, hidden_dim, edge_index, node_attr_dim=0, 
                 receptive_field_type='original', receptive_field_config=None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.edge_index = edge_index
        self.node_attr_dim = node_attr_dim
        
        # 节点属性处理
        if node_attr_dim > 0:
            self.node_attr_transform = nn.Linear(node_attr_dim, hidden_dim)
            total_input_dim = input_dim + hidden_dim
        else:
            self.node_attr_transform = None
            total_input_dim = input_dim
        
        # 根据配置选择图卷积类型
        if receptive_field_config is None:
            receptive_field_config = {}
        
        if receptive_field_type == 'original':
            self.gconv = DirectedGConv(total_input_dim, hidden_dim)
            
        elif receptive_field_type == 'multihop':
            num_hops = receptive_field_config.get('num_hops', 2)
            self.gconv = MultiHopDirectedGConv(total_input_dim, hidden_dim, num_hops=num_hops)
            
        elif receptive_field_type == 'attention':
            heads = receptive_field_config.get('heads', 4)
            dropout = receptive_field_config.get('dropout', 0.1)
            self.gconv = AttentionDirectedGConv(total_input_dim, hidden_dim, heads=heads, dropout=dropout)
            
        elif receptive_field_type == 'stacked':
            num_layers = receptive_field_config.get('num_layers', 3)
            hidden_channels = receptive_field_config.get('hidden_channels', hidden_dim)
            dropout = receptive_field_config.get('dropout', 0.1)
            self.gconv = StackedDirectedGConv(total_input_dim, hidden_channels, hidden_dim, 
                                            num_layers=num_layers, dropout=dropout)
        else:
            raise ValueError(f"Unknown receptive field type: {receptive_field_type}")
        
        # GRU单元
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
    
    def forward(self, x, h=None, node_attrs=None):
        """
        前向传播
        
        Args:
            x: 输入特征 [num_nodes, input_dim]
            h: 隐藏状态 [num_nodes, hidden_dim]
            node_attrs: 节点属性 [num_nodes, node_attr_dim]
        
        Returns:
            new_h: 新的隐藏状态 [num_nodes, hidden_dim]
        """
        # 处理节点属性
        if self.node_attr_transform is not None and node_attrs is not None:
            attr_features = self.node_attr_transform(node_attrs)
            x = torch.cat([x, attr_features], dim=-1)
        
        # 图卷积
        gconv_out = self.gconv(x, self.edge_index)
        
        # GRU更新
        if h is None:
            h = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
        
        new_h = self.gru(gconv_out, h)
        return new_h

class ConfigurableStreamGCGRU(nn.Module):
    """
    可配置感受野的流域图卷积GRU模型
    """
    def __init__(self, edge_index, config):
        super().__init__()
        self.edge_index = edge_index
        self.config = config
        
        # 提取配置
        hindcast_length = config['hindcast_length']
        forecast_length = config['forecast_length']
        gcgru_out_channels = config['gcgru_out_channels']
        mlp_hidden_dim = config['mlp_hidden_dim']
        mlp_dropout = config['mlp_dropout']
        node_attr_dim = config.get('node_attr_dim', 0)
        
        # 感受野配置
        rf_config = config.get('receptive_field', {
            'type': 'original',
            'config': {}
        })
        rf_type = rf_config['type']
        rf_params = rf_config['config']
        
        # 三个GRU层，每个都可以配置不同的感受野
        self.past_rain_gru = ConfigurableFlowAwareGRU(
            input_dim=1, hidden_dim=gcgru_out_channels, edge_index=edge_index,
            node_attr_dim=node_attr_dim, receptive_field_type=rf_type, 
            receptive_field_config=rf_params
        )
        
        self.past_flow_gru = ConfigurableFlowAwareGRU(
            input_dim=1, hidden_dim=gcgru_out_channels, edge_index=edge_index,
            node_attr_dim=node_attr_dim, receptive_field_type=rf_type,
            receptive_field_config=rf_params
        )
        
        self.future_rain_gru = ConfigurableFlowAwareGRU(
            input_dim=1, hidden_dim=gcgru_out_channels, edge_index=edge_index,
            node_attr_dim=node_attr_dim, receptive_field_type=rf_type,
            receptive_field_config=rf_params
        )
        
        # 输出MLP
        mlp_input_dim = gcgru_out_channels * 3
        self.output_mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden_dim // 2, forecast_length)
        )
    
    def forward(self, inputs):
        """
        前向传播
        
        Args:
            inputs: [past_rain, future_rain, past_flow, node_attrs(可选)] - 兼容原始训练脚本格式
        
        Returns:
            output: 预测结果 [batch_size, forecast_length, num_nodes]
        """
        if len(inputs) == 4:
            past_rain, future_rain, past_flow, node_attrs = inputs
        else:
            past_rain, future_rain, past_flow = inputs
            node_attrs = None
        
        batch_size, hindcast_length, num_nodes = past_rain.shape
        forecast_length = future_rain.shape[1]
        
        # 处理节点属性
        if node_attrs is not None:
            if node_attrs.dim() == 2:
                node_attrs = node_attrs.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 初始化隐藏状态
        h_rain = None
        h_flow = None
        h_future = None
        
        # 处理历史降雨
        for t in range(hindcast_length):
            rain_input = past_rain[:, t, :].unsqueeze(-1)  # [batch_size, num_nodes, 1]
            
            batch_h_rain = []
            for b in range(batch_size):
                attrs_b = node_attrs[b] if node_attrs is not None else None
                h_b = h_rain[b] if h_rain is not None else None
                new_h = self.past_rain_gru(rain_input[b], h_b, attrs_b)
                batch_h_rain.append(new_h)
            h_rain = torch.stack(batch_h_rain)
        
        # 处理历史流量
        for t in range(hindcast_length):
            flow_input = past_flow[:, t, :].unsqueeze(-1)
            
            batch_h_flow = []
            for b in range(batch_size):
                attrs_b = node_attrs[b] if node_attrs is not None else None
                h_b = h_flow[b] if h_flow is not None else None
                new_h = self.past_flow_gru(flow_input[b], h_b, attrs_b)
                batch_h_flow.append(new_h)
            h_flow = torch.stack(batch_h_flow)
        
        # 处理未来降雨
        for t in range(forecast_length):
            future_rain_input = future_rain[:, t, :].unsqueeze(-1)
            
            batch_h_future = []
            for b in range(batch_size):
                attrs_b = node_attrs[b] if node_attrs is not None else None
                h_b = h_future[b] if h_future is not None else None
                new_h = self.future_rain_gru(future_rain_input[b], h_b, attrs_b)
                batch_h_future.append(new_h)
            h_future = torch.stack(batch_h_future)
        
        # 融合特征
        combined_features = torch.cat([h_rain, h_flow, h_future], dim=-1)
        
        # 输出预测
        output = self.output_mlp(combined_features)
        
        # 调整输出维度以匹配训练脚本期望的格式: [batch_size, forecast_length, num_nodes]
        output = output.permute(0, 2, 1)
        
        return output

def create_model_with_receptive_field(edge_index, base_config, rf_type='original', rf_config=None):
    """
    创建具有指定感受野配置的模型
    
    Args:
        edge_index: 图的边索引
        base_config: 基础模型配置
        rf_type: 感受野类型 ('original', 'multihop', 'attention', 'stacked')
        rf_config: 感受野具体配置
    
    Returns:
        model: 配置好的模型
    """
    config = base_config.copy()
    config['receptive_field'] = {
        'type': rf_type,
        'config': rf_config or {}
    }
    
    return ConfigurableStreamGCGRU(edge_index, config)

# 预定义的感受野配置
RECEPTIVE_FIELD_PRESETS = {
    'small': {
        'type': 'original',
        'config': {},
        'description': '小感受野 - 仅直接邻居'
    },
    'medium': {
        'type': 'multihop',
        'config': {'num_hops': 2},
        'description': '中等感受野 - 2跳邻居'
    },
    'large': {
        'type': 'multihop',
        'config': {'num_hops': 3},
        'description': '大感受野 - 3跳邻居'
    },
    'adaptive': {
        'type': 'attention',
        'config': {'heads': 4, 'dropout': 0.1},
        'description': '自适应感受野 - 注意力机制'
    },
    'deep': {
        'type': 'stacked',
        'config': {'num_layers': 3, 'hidden_channels': 64, 'dropout': 0.1},
        'description': '深层感受野 - 3层堆叠'
    }
}

if __name__ == "__main__":
    print("可配置感受野的图卷积GRU模型")
    print("=" * 40)
    
    # 示例配置
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    
    base_config = {
        'hindcast_length': 30,
        'forecast_length': 1,
        'gcgru_out_channels': 64,
        'mlp_hidden_dim': 128,
        'mlp_dropout': 0.1,
        'node_attr_dim': 3
    }
    
    print("\n可用的感受野预设:")
    for name, preset in RECEPTIVE_FIELD_PRESETS.items():
        print(f"  {name}: {preset['description']}")
    
    print("\n创建不同感受野配置的模型:")
    for name, preset in RECEPTIVE_FIELD_PRESETS.items():
        try:
            model = create_model_with_receptive_field(
                edge_index, base_config, preset['type'], preset['config']
            )
            num_params = sum(p.numel() for p in model.parameters())
            print(f"  {name:>8}: 参数数量 {num_params:,}")
        except Exception as e:
            print(f"  {name:>8}: 创建失败 - {e}")