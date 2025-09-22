import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Callable, Union, Any, Literal
from torch.nn import Module, ModuleList
from torch_geometric.nn import GATConv, GCNConv, GCN2Conv, Linear
from torch_geometric.utils import add_self_loops


class SpatioTemporalFloodNet3D(Module):
    """
    基于3维batch处理的时空洪水预测网络
    核心特点：每个batch变成大子图，使用batch_vector进行节点分组
    借鉴gnn_st_flood_simple.py的多种聚合方式和成功的模型架构
    
    主要改进：
    1. 移除输出层Softplus激活，直接线性输出
    2. 支持时间维度MLP映射（use_time_mlp=True）或简单截断（use_time_mlp=False）
    3. 支持三种模式对比：
       - LSTM-only模式（use_lstm=True, use_gnn=False）：仅时序建模
       - GNN-only模式（use_lstm=False, use_gnn=True）：仅空间建模
       - LSTM+GNN模式（use_lstm=True, use_gnn=True）：时序+空间联合建模
    
    Args:
        use_lstm (bool): 是否使用LSTM时序建模，默认True
        use_gnn (bool): 是否使用GNN空间建模，默认True
        use_time_mlp (bool): 是否使用时间维度MLP映射而非简单截断，默认True
        其他参数保持不变...
    """
    
    def __init__(
        self,
        in_channels: int,  # 改为in_channels以兼容现有配置
        hidden_channels: int = 64,
        num_hidden: int = 2,
        output_size: int = 1,
        seq_len: int = 24,
        output_time: int = 72,
        gnn_type: str = "gcn",  # "gcn", "gat", "gcn2", "mlp"
        aggregation: Literal["mean", "attention", "multiscale"] = "mean",
        edge_orientation: Optional[str] = None,
        edge_weights: Optional[torch.Tensor] = None,
        dropout: float = 0.1,
        param_sharing: bool = False,
        use_gnn: bool = True,  # 🔧 新增：是否使用GNN，False时仅用LSTM
        use_lstm: bool = True, # 🔧 新增：是否使用LSTM，False时仅用GNN
        **kwargs,  # 接受额外的参数以提高兼容性
    ) -> None:
        super().__init__()
        
        # 基础参数
        self.num_features = in_channels  # 兼容性：使用in_channels作为num_features
        self.seq_len = seq_len
        self.output_time = output_time
        self.output_size = output_size
        self.edge_weights = edge_weights
        self.edge_orientation = edge_orientation
        self.gnn_type = gnn_type
        self.aggregation_type = aggregation
        self.use_gnn = use_gnn  # 存储use_gnn参数
        self.use_lstm = use_lstm  # 存储use_lstm参数
        
        # 验证参数组合的合理性
        if not use_gnn and not use_lstm:
            raise ValueError("At least one of use_gnn or use_lstm must be True")
        
        # === 借鉴gnn copy 4.py的统一隐藏层维度设计 ===
        self.unified_hidden_size = hidden_channels
        self.num_gnn_layers = num_hidden 
        
        # === 强化时序建模（借鉴SimpleLSTM架构）===
        if self.use_lstm:
            self.temporal_input_size = in_channels  # 使用in_channels
            self.temporal_hidden_size = self.unified_hidden_size
            
            # 简化架构：输入投影 + LSTM + Dropout（移除冗余的输出投影）
            self.temporal_input_projection = nn.Linear(self.temporal_input_size, self.temporal_hidden_size)
            self.temporal_lstm = nn.LSTM(
                input_size=self.temporal_hidden_size,
                hidden_size=self.temporal_hidden_size,
                #dropout=dropout if dropout > 0 else 0.0,
                batch_first=False,  # [T, N, F]
            )
            self.temporal_dropout = nn.Dropout(p=dropout)
        
        # === 轻量空间建模 ===
        if self.use_gnn:
            # 如果使用LSTM，空间建模接收LSTM输出；否则直接处理原始输入
            spatial_input_size = self.temporal_hidden_size if self.use_lstm else in_channels
            self.spatial_hidden = self.unified_hidden_size * 4
            
            # 空间特征投影 - 确保维度匹配
            self.spatial_input_projection = nn.Linear(spatial_input_size, self.spatial_hidden)
            
            # GNN层
            self.layers = self._create_gnn_layers(gnn_type, param_sharing)
            
            # 可选的dropout
            self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        else:
            # GNN-only模式：直接处理原始输入
            self.spatial_hidden = self.unified_hidden_size * 4
            self.spatial_input_projection = nn.Linear(in_channels, self.spatial_hidden)
            self.layers = self._create_gnn_layers(gnn_type, param_sharing)
            self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        # === 图级聚合策略（借鉴gnn_st_flood_simple.py）===
        if aggregation == "attention":
            self.attention_aggregation = AttentionAggregation(self.spatial_hidden)
            self.graph_feature_dim = self.spatial_hidden
        elif aggregation == "multiscale":
            self.multiscale_aggregation = MultiScaleAggregation(self.spatial_hidden)
            self.graph_feature_dim = self.spatial_hidden
        else:
            # mean聚合
            self.graph_feature_dim = self.spatial_hidden
        
        # 图级特征处理
        self.graph_aggregation = nn.Linear(self.graph_feature_dim, self.spatial_hidden)
        
        # === 最终输出投影：移除激活函数，直接线性降维 ===
        if self.use_gnn:
            # LSTM+GNN或GNN-only模式：从空间特征预测
            self.output_projection = nn.Sequential(
                nn.Linear(self.spatial_hidden, self.spatial_hidden // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.spatial_hidden // 2, output_size)  # 移除Softplus激活
            )
        
        # LSTM-only模式的输出投影层（直接从LSTM隐藏特征）
        if self.use_lstm and not self.use_gnn:
            self.output_projection_lstm_only = nn.Sequential(
                nn.Linear(self.temporal_hidden_size, self.temporal_hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.temporal_hidden_size // 2, output_size)  # 移除Softplus激活
            )
        
        # GNN-only模式的输出投影层（直接从GNN隐藏特征）
        if self.use_gnn and not self.use_lstm:
            self.output_projection_gnn_only = nn.Sequential(
                nn.Linear(self.spatial_hidden, self.spatial_hidden // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.spatial_hidden // 2, output_size)  # 移除Softplus激活
            )            
        
        # === 时间维度处理方案（替代简单截断）===
        # 支持多种时间映射方案
        # 兼容性：将use_time_mlp转换为time_mapping_strategy
        use_time_mlp = kwargs.get('use_time_mlp', None)
        if use_time_mlp is not None:
            # 如果传入use_time_mlp参数，根据其值设置time_mapping_strategy
            self.time_mapping_strategy = 'mlp_simple' if use_time_mlp else 'truncate'
        else:
            # 默认使用传入的time_mapping_strategy参数
            self.time_mapping_strategy = kwargs.get('time_mapping_strategy', 'truncate')
        
        if output_time != seq_len and self.time_mapping_strategy != 'truncate':
            if self.time_mapping_strategy == 'linear':
                # 线性插值：保持时序连续性
                self.time_mapper = nn.Linear(seq_len, output_time, bias=False)
                # 初始化为插值权重
                with torch.no_grad():
                    self.time_mapper.weight.data = self._create_interpolation_weights(seq_len, output_time)
                    
            elif self.time_mapping_strategy == 'attention':
                # 注意力选择：学习重要时间步
                self.time_attention = nn.Sequential(
                    nn.Linear(self.spatial_hidden, self.spatial_hidden // 4),
                    nn.Tanh(),
                    nn.Linear(self.spatial_hidden // 4, output_time)
                )
                
            elif self.time_mapping_strategy == 'conv1d':
                # 1D卷积：保持局部时序模式
                kernel_size = min(3, seq_len)
                self.time_conv = nn.Conv1d(
                    in_channels=self.spatial_hidden,
                    out_channels=self.spatial_hidden,
                    kernel_size=kernel_size,
                    padding=kernel_size//2
                )
                # 自适应池化到目标长度
                self.time_pool = nn.AdaptiveAvgPool1d(output_time)
                
            elif self.time_mapping_strategy == 'mlp_simple':
                # 简化MLP：减少参数，避免过拟合
                self.time_mlp = nn.Linear(seq_len, output_time)
            else:
                self.time_mapper = None
        else:
            self.time_mapper = None
        
        # 权重初始化
        self.apply(self._init_weights)

    def _create_gnn_layers(self, gnn_type: str, param_sharing: bool) -> ModuleList:
        """创建GNN层"""
        def layer_gen():
            if gnn_type == "gcn":
                return GCNConv(self.spatial_hidden, self.spatial_hidden, add_self_loops=False)
            elif gnn_type == "gat":
                return GATConv(self.spatial_hidden, self.spatial_hidden, add_self_loops=False)
            elif gnn_type == "gcn2":
                return GCN2Conv(self.spatial_hidden, alpha=0.5, add_self_loops=False)
            elif gnn_type == "mlp":
                return Linear(self.spatial_hidden, self.spatial_hidden, weight_initializer="kaiming_uniform")
            else:
                raise ValueError(f"Unknown gnn_type: {gnn_type}")
        
        if param_sharing:
            shared_layer = layer_gen()
            return ModuleList([shared_layer for _ in range(self.num_gnn_layers)])
        else:
            return ModuleList([layer_gen() for _ in range(self.num_gnn_layers)])

    def _init_weights(self, module):
        """权重初始化 - 借鉴成功模型，确保数值稳定性"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LSTM, nn.GRU)):
            for param in module.parameters():
                if len(param.shape) >= 2:
                    torch.nn.init.xavier_uniform_(param.data)
                else:
                    torch.nn.init.zeros_(param.data)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        batch_vector: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        前向传播: 强化时序建模 -> 空间交互(GNN逐时步) -> 图级聚合 -> 输出预测
        
        Args:
            x: 节点特征 [total_nodes, seq_len, features] (3维batch处理)
            edge_index: 边索引 [2, total_edges]
            edge_weight: 边权重 [total_edges]
            batch_vector: batch向量，指示每个节点属于batch中的哪个样本 [total_nodes]
        Returns:
            output: [batch_size, output_time, output_size]
        """
        if batch_vector is None:
            raise ValueError("batch_vector is required for 3D batch processing.")
        
        # 首先获取模型的设备
        device = next(self.parameters()).device
        
        # 确保所有输入都在同一设备上
        x = x.to(device)
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device)
        batch_vector = batch_vector.to(device)
        
        # 处理边权重和边方向
        edge_index, edge_weight = self._process_edges(edge_index, edge_weight, device)
        
        # === 步骤1：时序特征提取（如果启用LSTM）===
        if self.use_lstm:
            # 强化时序特征提取（使用LSTM，保留所有时间步）
            x_temporal = self._extract_temporal_features_with_lstm(x)
            N, T, Ht = x_temporal.shape
        else:
            # GNN-only模式：跳过LSTM，直接使用原始输入
            x_temporal = x  # [N, T, num_features]
            N, T, num_features = x_temporal.shape
        
        # === 处理不同模式 ===
        if not self.use_gnn and self.use_lstm:
            # LSTM-only模式：跳过GNN，直接进行平均池化聚合
            batch_size = int(batch_vector.max().item()) + 1
            
            # 节点到图聚合（平均池化）
            x_graph = self._aggregate_nodes_to_graph_mean_only(x_temporal, batch_vector, batch_size)  # [B, T, Ht]
            
            # 时间维度处理（MLP映射 vs 简单截断）
            B, T = x_graph.shape[:2]
            if hasattr(self, 'time_mlp') and self.time_mlp is not None and T != self.output_time:
                # 使用时间MLP进行映射：[B, T, Ht] -> [B, output_time, Ht]
                x_graph = x_graph.transpose(1, 2)  # [B, Ht, T]
                x_graph = self.time_mlp(x_graph)   # [B, Ht, output_time]
                x_graph = x_graph.transpose(1, 2)  # [B, output_time, Ht]
            else:
                # 简单截断方式
                if self.output_time <= T:
                    x_graph = x_graph[:, -self.output_time:, :]
                else:
                    pad_length = self.output_time - T
                    padding = torch.zeros(B, pad_length, x_graph.size(-1), device=x_graph.device, dtype=x_graph.dtype)
                    x_graph = torch.cat([padding, x_graph], dim=1)
            
            # 最终输出预测（直接从LSTM隐藏特征预测）
            output = self.output_projection_lstm_only(x_graph)
            return output
        
        elif self.use_gnn and not self.use_lstm:
            # GNN-only模式：跳过LSTM，直接进行GNN空间建模
            # === 步骤2：空间特征投影（对每个时间步）===
            x_proj = self.spatial_input_projection(x_temporal)  # [N, T, Hs]
            x_proj = F.relu(x_proj)
            if self.dropout is not None:
                x_proj = self.dropout(x_proj)
            
            # === 步骤3：轻量GNN空间建模（逐时间步应用相同图结构）===
            E = edge_index.size(1)
            offsets = (torch.arange(T, device=device) * N).view(1, T, 1)  # [1, T, 1]
            ei_tiled = edge_index.unsqueeze(1).expand(2, T, E) + offsets  # [2, T, E]
            edge_index_tiled = ei_tiled.reshape(2, T * E)                  # [2, T*E]
            edge_weight_tiled = edge_weight.repeat(T)                      # [T*E]

            # 将节点特征堆叠为 [T*N, Hs]，时间块连续，以匹配edge_index_tiled的偏移策略
            x_big = x_proj.permute(1, 0, 2).contiguous().view(T * N, -1)   # [T*N, Hs]
            x0_big = x_big.clone()
            for layer in self.layers:
                x_big = self._apply_layer(layer, x_big, edge_index_tiled, edge_weight_tiled, x0_big)
                x_big = F.relu(x_big)
                if self.dropout is not None:
                    x_big = self.dropout(x_big)
            # 还原为 [N, T, Hs]
            x_spatial = x_big.view(T, N, -1).permute(1, 0, 2).contiguous()
            
            # === 步骤4：图级聚合（对每个时间步进行节点到图的聚合）===
            batch_size = int(batch_vector.max().item()) + 1
            x_graph = self._aggregate_nodes_to_graph(x_spatial, batch_vector, batch_size)  # [B, T, H]
            
            # === 步骤5：时间维度处理（简单截断，因为没有LSTM的时序建模）===
            B, T = x_graph.shape[:2]
            if self.output_time <= T:
                x_graph = x_graph[:, -self.output_time:, :]
            else:
                pad_length = self.output_time - T
                padding = torch.zeros(B, pad_length, x_graph.size(-1), device=x_graph.device, dtype=x_graph.dtype)
                x_graph = torch.cat([padding, x_graph], dim=1)
            
            # 最终输出预测（直接从GNN隐藏特征预测）
            output = self.output_projection_gnn_only(x_graph)
            return output
        
        # LSTM+GNN模式：继续原有的完整处理流程
        
        # === 步骤2：空间特征投影（对每个时间步）===
        # 线性层可以直接处理3D张量，对最后一维进行变换
        x_proj = self.spatial_input_projection(x_temporal)  # [N, T, Hs]
        x_proj = F.relu(x_proj)
        if self.dropout is not None:
            x_proj = self.dropout(x_proj)
        # x_proj 已经是 [N, T, Hs] 形状

        # === 保持时序一一对应：不进行时间维度压缩 ===
        # 直接使用原始时间长度T进行GNN处理
        
        # === 步骤3：轻量GNN空间建模（逐时间步应用相同图结构）===
        # 将时间维度转为"批次"的图副本，一次完成GNN传播，避免Python层面的for循环
        # 构造时间展开的边索引和权重
        E = edge_index.size(1)
        offsets = (torch.arange(T, device=device) * N).view(1, T, 1)  # [1, T, 1]
        ei_tiled = edge_index.unsqueeze(1).expand(2, T, E) + offsets  # [2, T, E]
        edge_index_tiled = ei_tiled.reshape(2, T * E)                  # [2, T*E]
        edge_weight_tiled = edge_weight.repeat(T)                      # [T*E]

        # 将节点特征堆叠为 [T*N, Hs]，时间块连续，以匹配edge_index_tiled的偏移策略
        x_big = x_proj.permute(1, 0, 2).contiguous().view(T * N, -1)   # [T*N, Hs]
        x0_big = x_big.clone()
        for layer in self.layers:
            x_big = self._apply_layer(layer, x_big, edge_index_tiled, edge_weight_tiled, x0_big)
            x_big = F.relu(x_big)
            if self.dropout is not None:
                x_big = self.dropout(x_big)
        # 还原为 [N, T, Hs]
        x_spatial = x_big.view(T, N, -1).permute(1, 0, 2).contiguous()
        
        # === 步骤4：图级聚合（对每个时间步进行节点到图的聚合）===
        batch_size = int(batch_vector.max().item()) + 1
        x_graph = self._aggregate_nodes_to_graph(x_spatial, batch_vector, batch_size)  # [B, T, H]
        
        # === 步骤5：图聚合后的特征处理 ===
        # 线性层可以直接处理3D张量，对最后一维进行变换
        #x_graph = F.relu(self.graph_aggregation(x_graph))  # [B, T, Hs]暂且少用一层网络
        
        # === 步骤6：时间维度处理（MLP映射 vs 简单截断）===
        B, T = x_graph.shape[:2]
        
        if hasattr(self, 'time_mlp') and self.time_mlp is not None and T != self.output_time:
            # 使用时间MLP进行映射：[B, T, Hs] -> [B, output_time, Hs]
            # 转置以便MLP处理时间维度：[B, Hs, T] -> [B, Hs, output_time] -> [B, output_time, Hs]
            x_graph = x_graph.transpose(1, 2)  # [B, Hs, T]
            x_graph = self.time_mlp(x_graph)   # [B, Hs, output_time]
            x_graph = x_graph.transpose(1, 2)  # [B, output_time, Hs]
        else:
            # 简单截断方式（保留作为对比）
            if self.output_time <= T:
                # 取最后output_time个时间步
                x_graph = x_graph[:, -self.output_time:, :]  # [B, output_time, Hs]
            else:
                # 如果output_time > seq_len，需要进行padding
                pad_length = self.output_time - T
                padding = torch.zeros(B, pad_length, x_graph.size(-1), device=x_graph.device, dtype=x_graph.dtype)
                x_graph = torch.cat([padding, x_graph], dim=1)  # [B, output_time, Hs]
        
        # === 步骤7：最终输出预测 ===
        # 线性层可以直接处理3D张量，对最后一维进行变换
        output = self.output_projection(x_graph)  # [B, To, output_size]
        
        return output

    def _process_edges(self, edge_index: torch.Tensor, edge_weight: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """处理边权重和边方向"""
        # if self.edge_weights is not None:
        #     # 根据batch数量复制边权重
        #     num_graphs = edge_index.size(1) // len(self.edge_weights) if len(self.edge_weights) > 0 else 1
        #     edge_weight = torch.cat(num_graphs * [self.edge_weights], dim=0).to(device)
        #     edge_weight = edge_weight.abs()
        # else:
        #     if edge_weight is None:
        #         edge_weight = torch.ones(edge_index.size(1)).to(device)
        
        # # 处理边方向
        # if self.edge_orientation is not None:
        #     if self.edge_orientation == "upstream":
        #         edge_index = edge_index[[1, 0]]
        #     elif self.edge_orientation == "bidirectional":
        #         edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
        #         edge_weight = torch.cat(2 * [edge_weight], dim=0)
        
        # 添加自环
        if self.edge_weights is not None and len(self.edge_weights) > 0:
            loop_fill_value = 1.0 if (self.edge_weights == 0).all() else "mean"
            edge_index, edge_weight = add_self_loops(edge_index, edge_weight, fill_value=loop_fill_value)
        
        return edge_index, edge_weight

    def _extract_temporal_features_with_lstm(self, x: torch.Tensor) -> torch.Tensor:
        """
        使用LSTM提取时序特征 - 简化架构，避免冗余投影层
        输入: [total_nodes, seq_len, features]
        输出: [total_nodes, seq_len, temporal_hidden]
        """
        total_nodes, seq_len, num_features = x.shape
        
        # 重塑为 [seq_len, total_nodes, features] 以匹配LSTM输入
        x = x.permute(1, 0, 2)  # [seq_len, total_nodes, num_features]
        
        # === 简化架构：线性层 → LSTM → Dropout ===
        # 1. 输入投影
        x = F.relu(self.temporal_input_projection(x))  # [seq_len, total_nodes, temporal_hidden]
        
        # 2. LSTM层（核心时序建模）
        lstm_out, _ = self.temporal_lstm(x)  # [seq_len, total_nodes, temporal_hidden]
        
        # 3. Dropout层
        lstm_out = self.temporal_dropout(lstm_out)  # [seq_len, total_nodes, temporal_hidden]
        
        # 4. 返回所有时间步的输出，恢复到 [N, T, H]，保持时序一一对应
        x_all = lstm_out.permute(1, 0, 2)
        
        return x_all

    def _apply_layer(
        self,
        layer: Module,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        x_0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """应用单个GNN层"""
        if self.gnn_type == "gcn" and isinstance(layer, GCNConv):
            return layer(x, edge_index, edge_weight)
        elif self.gnn_type == "gat" and isinstance(layer, GATConv):
            # GAT处理边权重
            if edge_weight is not None and edge_weight.numel() > 0:
                valid_edges = edge_weight != 0
                if valid_edges.sum() > 0:
                    filtered_edge_index = edge_index[:, valid_edges]
                    filtered_edge_weight = edge_weight[valid_edges]
                    edge_attr = filtered_edge_weight.unsqueeze(-1) if filtered_edge_weight.dim() == 1 else filtered_edge_weight
                    return layer(x, filtered_edge_index, edge_attr=edge_attr)
            return layer(x, edge_index)
        elif self.gnn_type == "gcn2" and isinstance(layer, GCN2Conv):
            if x_0 is None:
                x_0 = x
            return layer(x, x_0, edge_index, edge_weight)
        elif self.gnn_type == "mlp" and isinstance(layer, Linear):
            # MLP忽略图结构
            return layer(x)
        else:
            # 通用处理
            try:
                return layer(x, edge_index, edge_weight)
            except:
                return layer(x, edge_index)

    def _expand_to_temporal(self, x: torch.Tensor, batch_vector: torch.Tensor) -> torch.Tensor:
        """
        将空间特征扩展为时序特征，以便进行图级聚合
        输入: [total_nodes, spatial_hidden]
        输出: [total_nodes, seq_len, spatial_hidden]
        """
        # 简单策略：将空间特征复制seq_len次
        # 更复杂的策略可以使用learnable parameters或attention机制
        total_nodes, spatial_hidden = x.shape
        x_expanded = x.unsqueeze(1).expand(total_nodes, self.seq_len, spatial_hidden)
        return x_expanded

    def _aggregate_nodes_to_graph(
        self, 
        x: torch.Tensor, 
        batch_vector: torch.Tensor,
        batch_size: int
    ) -> torch.Tensor:
        """
        节点到图聚合 - 借鉴gnn_st_flood_simple.py的多种聚合方式
        输入: [total_nodes, seq_len, spatial_hidden]
        输出: [batch_size, seq_len, graph_feature_dim]
        """
        device = x.device
        
        if self.aggregation_type == "attention":
            # 注意力聚合
            return self.attention_aggregation(x, batch_vector, batch_size)
        
        elif self.aggregation_type == "multiscale":
            # 多重聚合
            return self.multiscale_aggregation(x, batch_vector, batch_size)
        
        else:
            # 简单平均聚合（默认）
            basin_outputs = []
            for b in range(batch_size):
                mask = (batch_vector == b)
                if mask.sum() == 0:
                    basin_outputs.append(torch.zeros(x.size(1), x.size(2), device=device))
                    continue
                basin_nodes = x[mask]  # [n_nodes_in_basin, seq_len, spatial_hidden]
                basin_repr = basin_nodes.mean(dim=0)  # [seq_len, spatial_hidden]
                basin_outputs.append(basin_repr)
            
            return torch.stack(basin_outputs, dim=0)  # [batch_size, seq_len, spatial_hidden]

    def _aggregate_nodes_to_graph_mean_only(
        self, 
        x: torch.Tensor, 
        batch_vector: torch.Tensor,
        batch_size: int
    ) -> torch.Tensor:
        """
        LSTM-only模式的简单平均聚合（无GNN处理）
        输入: [total_nodes, seq_len, temporal_hidden]
        输出: [batch_size, seq_len, temporal_hidden]
        """
        device = x.device
        basin_outputs = []
        for b in range(batch_size):
            mask = (batch_vector == b)
            if mask.sum() == 0:
                basin_outputs.append(torch.zeros(x.size(1), x.size(2), device=device))
                continue
            basin_nodes = x[mask]  # [n_nodes_in_basin, seq_len, temporal_hidden]
            basin_repr = basin_nodes.mean(dim=0)  # [seq_len, temporal_hidden]
            basin_outputs.append(basin_repr)
        
        return torch.stack(basin_outputs, dim=0)  # [batch_size, seq_len, temporal_hidden]


# === 借鉴gnn_st_flood_simple.py的聚合组件 ===
class AttentionAggregation(nn.Module):
    """节点级注意力聚合 - 借鉴gnn_st_flood_simple.py"""
    def __init__(self, hidden_channels: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 4),
            nn.Tanh(),
            nn.Linear(hidden_channels // 4, 1),
        )
        
    def forward(self, x: torch.Tensor, batch_vector: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        输入: [total_nodes, seq_len, hidden_channels]
        输出: [batch_size, seq_len, hidden_channels]
        """
        device = x.device
        N, T, H = x.shape
        
        # 计算注意力权重
        att_weights = self.attention(x)  # [N, T, 1]
        
        outputs = []
        for b in range(batch_size):
            mask = (batch_vector == b)
            if mask.sum() == 0:
                outputs.append(torch.zeros(T, H, device=device))
                continue
            
            basin_x = x[mask]  # [n_nodes_in_basin, T, H]
            basin_att = att_weights[mask]  # [n_nodes_in_basin, T, 1]
            
            # 沿节点维度进行softmax归一化
            basin_att = F.softmax(basin_att, dim=0)  # [n_nodes_in_basin, T, 1]
            
            # 加权聚合
            basin_repr = (basin_x * basin_att).sum(dim=0)  # [T, H]
            outputs.append(basin_repr)
        
        return torch.stack(outputs, dim=0)  # [batch_size, T, H]


class MultiScaleAggregation(nn.Module):
    """多尺度聚合 - 借鉴gnn_st_flood_simple.py"""
    def __init__(self, hidden_channels: int):
        super().__init__()
        self.mean_proj = nn.Linear(hidden_channels, hidden_channels // 4)
        self.max_proj = nn.Linear(hidden_channels, hidden_channels // 4)
        self.std_proj = nn.Linear(hidden_channels, hidden_channels // 4)
        self.sum_proj = nn.Linear(hidden_channels, hidden_channels // 4)
        self.final_proj = nn.Linear(hidden_channels, hidden_channels)
        
    def forward(self, x: torch.Tensor, batch_vector: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        输入: [total_nodes, seq_len, hidden_channels]
        输出: [batch_size, seq_len, hidden_channels]
        """
        device = x.device
        N, T, H = x.shape
        
        outputs = []
        for b in range(batch_size):
            mask = (batch_vector == b)
            if mask.sum() == 0:
                outputs.append(torch.zeros(T, H, device=device))
                continue
            
            basin_x = x[mask]  # [n_nodes_in_basin, T, H]
            
            # 多种统计聚合
            mean_feat = self.mean_proj(basin_x.mean(dim=0))  # [T, H//4]
            max_feat = self.max_proj(basin_x.max(dim=0)[0])   # [T, H//4]
            std_feat = self.std_proj(basin_x.std(dim=0))     # [T, H//4]
            sum_feat = self.sum_proj(basin_x.sum(dim=0))     # [T, H//4]
            
            # 拼接多尺度特征
            multi_feat = torch.cat([mean_feat, max_feat, std_feat, sum_feat], dim=-1)  # [T, H]
            
            # 最终投影
            basin_repr = self.final_proj(multi_feat)  # [T, H]
            outputs.append(basin_repr)
        
        return torch.stack(outputs, dim=0)  # [batch_size, T, H]


class SpatialFirstTemporalFloodNet3D(nn.Module):
    """
    先空间后时间的洪水预测网络 + Transformer时序建模
    
    架构流程：
    1. 逐时间步进行空间特征提取（GNN）
    2. 使用Transformer进行时序建模和预测
    3. 直接输出最终预测结果
    
    优势：
    - 空间信息融合在前，每个时间步都有空间交互
    - Transformer处理长距离时序依赖
    - 更符合水文物理过程（空间传播 -> 时序演化）
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_hidden: int = 2,
        output_size: int = 1,
        seq_len: int = 24,
        output_time: int = 72,
        gnn_type: str = "gcn",
        aggregation: str = "mean",
        n_heads: int = 8,
        transformer_layers: int = 4,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        
        self.seq_len = seq_len
        self.output_time = output_time
        self.output_size = output_size
        self.hidden_channels = hidden_channels
        self.aggregation_type = aggregation
        
        # 输入特征投影
        self.input_projection = nn.Linear(in_channels, hidden_channels)
        
        # GNN层（用于空间特征提取）
        self.num_gnn_layers = num_hidden
        self.gnn_layers = self._create_gnn_layers(gnn_type, hidden_channels)
        
        # 聚合策略
        if aggregation == "attention":
            self.attention_aggregation = AttentionAggregation(hidden_channels)
        elif aggregation == "multiscale":
            self.multiscale_aggregation = MultiScaleAggregation(hidden_channels)
        
        # Transformer时序建模
        self.pos_encoding = PositionalEncoding(hidden_channels, dropout, seq_len)
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_channels,
            nhead=n_heads,
            dim_feedforward=hidden_channels * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True  # 使用batch_first=True，输入形状为[B, T, H]
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, transformer_layers)
        
        # 时序投影层（如果输出时间长度不同）
        if output_time != seq_len:
            self.time_projection = nn.Linear(seq_len, output_time)
        else:
            self.time_projection = nn.Identity()
        
        # 输出投影
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, output_size)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.apply(self._init_weights)
    
    def _create_gnn_layers(self, gnn_type: str, hidden_channels: int):
        """创建GNN层"""
        layers = nn.ModuleList()
        for _ in range(self.num_gnn_layers):
            if gnn_type == "gcn":
                layers.append(GCNConv(hidden_channels, hidden_channels, add_self_loops=False))
            elif gnn_type == "gat":
                layers.append(GATConv(hidden_channels, hidden_channels, add_self_loops=False))
            elif gnn_type == "gcn2":
                layers.append(GCN2Conv(hidden_channels, alpha=0.5, add_self_loops=False))
            else:
                raise ValueError(f"Unsupported gnn_type: {gnn_type}")
        return layers
    
    def _init_weights(self, module):
        """权重初始化 - 数值稳定版本"""
        if isinstance(module, nn.Linear):
            # 使用更保守的初始化
            torch.nn.init.xavier_normal_(module.weight, gain=0.5)  # 减小gain值，防止梯度爆炸
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.TransformerEncoderLayer):
            # 特别处理Transformer层的初始化
            for p in module.parameters():
                if p.dim() > 1:
                    torch.nn.init.xavier_normal_(p, gain=0.1)  # 非常保守的初始化
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        batch_vector: torch.Tensor,
        **kwargs
    ):
        """
        前向传播：先空间后时间
        
        Args:
            x: [N, T, F] 节点特征
            edge_index: [2, E] 边索引
            edge_weight: [E] 边权重
            batch_vector: [N] batch向量
        
        Returns:
            output: [B, output_time, output_size]
        """
        # 首先获取模型的设备
        device = next(self.parameters()).device
        
        # 确保所有输入都在同一设备上
        x = x.to(device)
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device)
        batch_vector = batch_vector.to(device)
        
        N, T, num_features = x.shape  # 修复：将F重命名为num_features以避免与torch.nn.functional.F冲突
        batch_size = int(batch_vector.max().item()) + 1
        
        # === 步骤1：输入特征投影 ===
        x = self.input_projection(x)  # [N, T, H]
        
        # === 步骤2：批量空间特征提取（避免for循环）===
        # 将时间维度展开为大图的批量处理，类似原架构的做法
        # 构造时间展开的边索引和权重
        E = edge_index.size(1)
        offsets = (torch.arange(T, device=device) * N).view(1, T, 1)  # [1, T, 1]
        ei_tiled = edge_index.unsqueeze(1).expand(2, T, E) + offsets  # [2, T, E]
        edge_index_tiled = ei_tiled.reshape(2, T * E)                  # [2, T*E]
        edge_weight_tiled = edge_weight.repeat(T)                      # [T*E]

        # 将节点特征堆叠为 [T*N, H]，时间块连续
        x_big = x.permute(1, 0, 2).contiguous().view(T * N, -1)   # [T*N, H]
        x0_big = x_big.clone()
        
        # 批量GNN处理所有时间步
        for gnn_layer in self.gnn_layers:
            x_big = self._apply_gnn_layer(gnn_layer, x_big, edge_index_tiled, edge_weight_tiled, x0_big)
            x_big = F.relu(x_big)
            x_big = self.dropout(x_big)
        
        # 还原为 [N, T, H]
        x_spatial = x_big.view(T, N, -1).permute(1, 0, 2).contiguous()  # [N, T, H]
        
        # 批量节点到图聚合（对所有时间步一次性处理）
        spatial_features = self._batch_aggregate_nodes_to_graph(x_spatial, batch_vector, batch_size)  # [B, T, H]
        
        # === 步骤3：Transformer时序建模 ===
        # 位置编码（现在自动处理设备匹配）
        spatial_features = self.pos_encoding(spatial_features)
        
        # Transformer处理
        temporal_features = self.transformer(spatial_features)  # [B, T, H]
        
        # === 步骤4：时间维度调整 ===
        if self.output_time != T:
            # 转置进行时间投影：[B, T, H] -> [B, H, T] -> [B, H, output_time] -> [B, output_time, H]
            temporal_features = temporal_features.transpose(1, 2)  # [B, H, T]
            temporal_features = self.time_projection(temporal_features)  # [B, H, output_time]
            temporal_features = temporal_features.transpose(1, 2)  # [B, output_time, H]
        
        # === 步骤5：输出预测 ===
        output = self.output_projection(temporal_features)  # [B, output_time, output_size]
        
        return output
    
    def _apply_gnn_layer(self, layer, x, edge_index, edge_weight, x0=None):
        """应用GNN层"""
        if isinstance(layer, GCNConv):
            return layer(x, edge_index, edge_weight)
        elif isinstance(layer, GATConv):
            return layer(x, edge_index)
        elif isinstance(layer, GCN2Conv):
            return layer(x, x0, edge_index, edge_weight)
        else:
            return layer(x, edge_index, edge_weight)
    
    def _aggregate_nodes_to_graph(self, x, batch_vector, batch_size):
        """节点到图聚合"""
        device = x.device
        
        if self.aggregation_type == "attention":
            # 注意力聚合（需要适配2D输入）
            x_3d = x.unsqueeze(1)  # [N, 1, H] 模拟时间维度
            result = self.attention_aggregation(x_3d, batch_vector, batch_size)  # [B, 1, H]
            return result.squeeze(1)  # [B, H]
        
        elif self.aggregation_type == "multiscale":
            # 多尺度聚合（需要适配2D输入）
            x_3d = x.unsqueeze(1)  # [N, 1, H] 模拟时间维度
            result = self.multiscale_aggregation(x_3d, batch_vector, batch_size)  # [B, 1, H]
            return result.squeeze(1)  # [B, H]
        
        else:
            # 平均聚合
            outputs = []
            for b in range(batch_size):
                mask = (batch_vector == b)
                if mask.sum() == 0:
                    outputs.append(torch.zeros(x.size(-1), device=device))
                    continue
                outputs.append(x[mask].mean(dim=0))  # [H]
            return torch.stack(outputs, dim=0)  # [B, H]
    
    def _batch_aggregate_nodes_to_graph(self, x, batch_vector, batch_size):
        """
        批量节点到图聚合，避免for循环
        输入: [N, T, H] 节点特征
        输出: [B, T, H] 图特征
        """
        device = x.device
        N, T, H = x.shape
        
        # 确保batch_vector在正确的设备上
        batch_vector = batch_vector.to(device)
        
        if self.aggregation_type == "attention":
            # 注意力聚合 - 使用现有的AttentionAggregation（已经优化过）
            return self.attention_aggregation(x, batch_vector, batch_size)
        
        elif self.aggregation_type == "multiscale":
            # 多尺度聚合 - 使用现有的MultiScaleAggregation（已经优化过）
            return self.multiscale_aggregation(x, batch_vector, batch_size)
        
        else:
            # 批量平均聚合 - 使用矩阵操作避免for循环
            # 创建聚合掩码矩阵 [B, N]
            batch_mask = torch.zeros(batch_size, N, device=device, dtype=x.dtype)
            batch_mask[batch_vector, torch.arange(N, device=device)] = 1.0
            
            # 计算每个batch的节点数量
            batch_counts = batch_mask.sum(dim=1, keepdim=True)  # [B, 1]
            batch_counts = batch_counts.clamp(min=1.0)  # 避免除零
            
            # 归一化掩码（平均池化权重）
            batch_mask = batch_mask / batch_counts  # [B, N]
            
            # 批量矩阵乘法进行聚合：[B, N] × [N, T, H] -> [B, T, H]
            output = torch.einsum('bn,nth->bth', batch_mask, x)
            
            return output


class PositionalEncoding(nn.Module):
    """Transformer位置编码 - 设备兼容版本"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]
    
    def forward(self, x):
        # x: [B, T, H]
        # 确保位置编码与输入张量在同一设备上
        device = x.device
        pe_slice = self.pe[:, :x.size(1), :].to(device)
        x = x + pe_slice
        return self.dropout(x)


def create_spatial_first_flood_model(
    in_channels: int,
    hidden_channels: int = 64,
    num_hidden: int = 2,
    output_size: int = 1,
    seq_len: int = 24,
    output_time: int = 72,
    gnn_type: str = "gcn",
    aggregation: str = "mean",
    n_heads: int = 8,
    transformer_layers: int = 4,
    dropout: float = 0.1,
    **kwargs
) -> SpatialFirstTemporalFloodNet3D:
    """
    创建先空间后时间的洪水预测模型
    """
    return SpatialFirstTemporalFloodNet3D(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        num_hidden=num_hidden,
        output_size=output_size,
        seq_len=seq_len,
        output_time=output_time,
        gnn_type=gnn_type,
        aggregation=aggregation,
        n_heads=n_heads,
        transformer_layers=transformer_layers,
        dropout=dropout,
        **kwargs
    )


# === 聚合策略的模型包装器 ===
def MeanAggregationFloodNet(
    in_channels: int,
    hidden_channels: int = 64,
    num_hidden: int = 2,
    output_size: int = 1,
    seq_len: int = 24,
    output_time: int = 72,
    **kwargs
) -> SpatioTemporalFloodNet3D:
    """
    平均聚合版本的3D洪水预测GCN
    """
    return SpatioTemporalFloodNet3D(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        num_hidden=num_hidden,
        output_size=output_size,
        seq_len=seq_len,
        output_time=output_time,
        aggregation="mean",
        **kwargs
    )


def AttentionAggregationFloodNet(
    in_channels: int,
    hidden_channels: int = 64,
    num_hidden: int = 2,
    output_size: int = 1,
    seq_len: int = 24,
    output_time: int = 72,
    **kwargs
) -> SpatioTemporalFloodNet3D:
    """
    注意力聚合版本的3D洪水预测GCN
    """
    return SpatioTemporalFloodNet3D(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        num_hidden=num_hidden,
        output_size=output_size,
        seq_len=seq_len,
        output_time=output_time,
        aggregation="attention",
        **kwargs
    )


def MultiScaleAggregationFloodNet(
    in_channels: int,
    hidden_channels: int = 64,
    num_hidden: int = 2,
    output_size: int = 1,
    seq_len: int = 24,
    output_time: int = 72,
    **kwargs
) -> SpatioTemporalFloodNet3D:
    """
    多尺度聚合版本的3D洪水预测GCN
    """
    return SpatioTemporalFloodNet3D(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        num_hidden=num_hidden,
        output_size=output_size,
        seq_len=seq_len,
        output_time=output_time,
        aggregation="multiscale",
        **kwargs
    )
