import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Callable, Union, Any
from torch.nn import Module, ModuleList
from torch_geometric.nn import GATConv, GCNConv, GCN2Conv, Linear, BatchNorm
from torch_geometric.utils import add_self_loops


class GNNBaseModel(Module, ABC):
    """
    改进的GNN基础模型 - 借鉴SimpleLSTM的成功经验
    核心改进：使用真正的LSTM进行时序建模 + 轻量GNN进行空间建模
    架构：时序LSTM -> 空间GNN -> 输出预测
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_hidden: int,
        param_sharing: bool,
        layerfun: Callable[[], Module],
        edge_orientation: Optional[str],
        edge_weights: Optional[torch.Tensor],
        output_size: int = 1,
        aggregate_to_graph: bool = False,
        # 时序参数
        seq_len: Optional[int] = None,
        num_features: Optional[int] = None,
        output_time: Optional[int] = None,
        dropout: float = 0.1,
        use_temporal_modeling: bool = True,
    ) -> None:
        super().__init__()
        
        # 基础参数
        self.output_size = output_size
        self.aggregate_to_graph = aggregate_to_graph
        self.edge_weights = edge_weights
        self.edge_orientation = edge_orientation
        self.seq_len = seq_len
        self.num_features = num_features
        self.output_time = output_time or 1
        self.use_temporal_modeling = use_temporal_modeling
        
        # === 强化时序建模（借鉴SimpleLSTM架构）===
        if use_temporal_modeling and seq_len is not None and num_features is not None:
            # 统一隐藏层维度，确保时序和空间建模的维度匹配
            unified_hidden_size = min(hidden_channels, 64)
            
            # 时序特征维度设置
            self.temporal_input_size = num_features
            self.temporal_hidden_size = unified_hidden_size
            
            # 借鉴SimpleLSTM的三层架构：输入投影 + LSTM + 输出投影
            self.temporal_input_projection = nn.Linear(self.temporal_input_size, self.temporal_hidden_size)
            self.temporal_lstm = nn.LSTM(
                input_size=self.temporal_hidden_size,
                hidden_size=self.temporal_hidden_size,
                batch_first=False,  # 保持与SimpleLSTM一致的维度顺序
                dropout=dropout if dropout > 0 else 0.0
            )
            # 时序特征输出维度
            spatial_input_dim = self.temporal_hidden_size
        else:
            # 如果不使用时序建模，直接使用展平特征
            self.temporal_input_projection = None
            self.temporal_lstm = None
            spatial_input_dim = in_channels
            unified_hidden_size = min(hidden_channels, 32)

        # === 轻量空间建模 ===
        self.spatial_hidden = unified_hidden_size  # 使用统一的隐藏层维度
        self.num_gnn_layers = max(1, min(num_hidden, 2))  # 最多2层GNN
        
        # 空间特征投影 - 确保维度匹配
        self.spatial_input_projection = nn.Linear(spatial_input_dim, self.spatial_hidden)
        
        # GNN层
        if param_sharing:
            self.shared_layer = layerfun()
            self.layers = ModuleList([self.shared_layer for _ in range(self.num_gnn_layers)])
        else:
            self.layers = ModuleList([layerfun() for _ in range(self.num_gnn_layers)])
        
        # 可选的dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        # === 输出预测层 ===
        if aggregate_to_graph:
            # 图级输出：节点聚合 + 时序预测
            # 聚合后输出 mean+max+std 三部分拼接，故输入维度为 3 * spatial_hidden
            self.graph_aggregation = nn.Linear(self.spatial_hidden * 3, self.spatial_hidden)
            # 正确的输出投影：直接输出时间序列×特征数
            self.output_projection = nn.Linear(self.spatial_hidden, output_size * self.output_time)
        else:
            # 节点级输出
            self.output_projection = nn.Linear(self.spatial_hidden, output_size * self.output_time)

        # 自环设置
        if self.edge_weights is not None:
            self.register_buffer("static_edge_weights", edge_weights)
        else:
            self.static_edge_weights = None

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        batch_vector: Optional[torch.Tensor] = None,
        evo_tracking: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        前向传播: 强化时序建模 -> 轻量空间建模 -> 输出预测
        
        Args:
            x: 节点特征 [batch, nodes, seq_len, features] 或 [total_nodes, flattened_features]
            edge_index: 边索引
            edge_weight: 边权重
            batch_vector: batch向量，指示每个节点属于batch中的哪个样本 [total_nodes]
            evo_tracking: 是否跟踪演化
        """
        device = next(self.parameters()).device
        x = x.to(device)
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device)
        if batch_vector is not None:
            batch_vector = batch_vector.to(device)
        
        original_shape = x.shape
        evolution = [] if evo_tracking else None
        
        # === 步骤1：强化时序特征提取（使用LSTM）===
        x = self._extract_temporal_features_with_lstm(x)
        if evo_tracking:
            evolution.append(x.detach())
        
        # === 步骤2：空间特征投影 ===
        x = self.spatial_input_projection(x)
        x = F.relu(x)
        if self.dropout is not None:
            x = self.dropout(x)
        if evo_tracking:
            evolution.append(x.detach())
        
        # === 步骤3：轻量GNN空间建模 ===
        x_0 = x.clone()  # 保存初始特征，用于需要x_0的模型（如GCNII）
        for layer in self.layers:
            x = self.apply_layer(layer, x, edge_index, edge_weight, x_0)
            x = F.relu(x)
            if self.dropout is not None:
                x = self.dropout(x)
            if evo_tracking:
                evolution.append(x.detach())
        
        # === 步骤4：输出预测 ===
        if self.aggregate_to_graph:
            # 图级输出：节点聚合 + 预测
            x = self._aggregate_nodes_to_graph(x, batch_vector)
            x = F.relu(self.graph_aggregation(x))
            x = self.output_projection(x)
        else:
            # 节点级输出
            x = self.output_projection(x)
        
        # === 步骤5：格式化输出 ===
        x = self._format_output(x, batch_vector, original_shape)
        
        return (x, evolution) if evo_tracking else x
    
    def _extract_temporal_features_with_lstm(self, x: torch.Tensor) -> torch.Tensor:
        """
        使用LSTM提取时序特征 - 借鉴SimpleLSTM的架构
        """
        
        if not self.use_temporal_modeling or self.temporal_lstm is None:
            # 传统模式：展平特征
            if x.dim() == 4:
                batch_size, num_nodes, window_size, num_features = x.shape
                x = x.view(batch_size * num_nodes, window_size * num_features)
            return x
        
        # LSTM时序建模模式
        if x.dim() == 4:
            batch_size, num_nodes, seq_len, num_features = x.shape
            # 重塑为 [seq_len, batch*nodes, features] 以匹配LSTM输入
            x = x.permute(2, 0, 1, 3)  # [seq_len, batch, nodes, features]
            x = x.contiguous().view(seq_len, batch_size * num_nodes, num_features)
        elif x.dim() == 2:
            total_nodes = x.shape[0]
            flattened_features = x.shape[1]
            
            # 推断时序维度
            if self.num_features and flattened_features % self.num_features == 0:
                seq_len = flattened_features // self.num_features
                num_features = self.num_features
            elif self.seq_len and flattened_features % self.seq_len == 0:
                seq_len = self.seq_len
                num_features = flattened_features // self.seq_len
            else:
                # 无法推断，回退传统模式
                return x
            
            x = x.view(total_nodes, seq_len, num_features)
            x = x.permute(1, 0, 2)  # [seq_len, total_nodes, num_features]
        else:
            raise ValueError(f"Unsupported x shape: {x.shape}")
        
        # === 借鉴SimpleLSTM的三层架构 ===
        # 1. 输入投影（相当于SimpleLSTM的linearIn）
        x = F.relu(self.temporal_input_projection(x))  # [seq_len, total_nodes, temporal_hidden]
        
        # 2. LSTM层（核心时序建模）
        lstm_out, _ = self.temporal_lstm(x)  # [seq_len, total_nodes, temporal_hidden]
        
        # 3. 取最后时间步的输出（或者可以尝试其他聚合方式）
        x = lstm_out[-1]  # [total_nodes, temporal_hidden]
        
        return x
    
    def _aggregate_nodes_to_graph(
        self, 
        x: torch.Tensor, 
        batch_vector: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        节点到图聚合 - 使用多种聚合策略
        """
        if batch_vector is not None:
            # 多样本批处理
            num_samples = batch_vector.max().item() + 1
            graph_features = []
            
            for sample_id in range(num_samples):
                mask = (batch_vector == sample_id)
                if mask.sum() > 0:
                    node_features = x[mask]  # [num_nodes_in_sample, features]
                    
                    # 多重聚合策略，结合不同的统计信息
                    mean_feat = node_features.mean(dim=0)      # 平均值
                    max_feat = node_features.max(dim=0)[0]     # 最大值
                    std_feat = node_features.std(dim=0)       # 标准差
                    
                    # 组合多种统计特征
                    graph_feature = torch.cat([mean_feat, max_feat, std_feat], dim=0)
                    graph_features.append(graph_feature)
                else:
                    # 对应的零特征
                    zero_feat = torch.zeros(self.spatial_hidden * 3, device=x.device)
                    graph_features.append(zero_feat)
            
            return torch.stack(graph_features, dim=0)
        else:
            # 单样本：多重聚合
            mean_feat = x.mean(dim=0, keepdim=True)
            max_feat = x.max(dim=0, keepdim=True)[0]
            std_feat = x.std(dim=0, keepdim=True)
            return torch.cat([mean_feat, max_feat, std_feat], dim=1)
    
    def _extract_temporal_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取时序特征"""
        
        if not self.use_temporal_modeling or self.temporal_encoder is None:
            # 传统模式：展平特征
            if x.dim() == 4:
                batch_size, num_nodes, window_size, num_features = x.shape
                x = x.view(batch_size * num_nodes, window_size * num_features)
            return x
        
        # 时序建模模式
        if x.dim() == 4:
            batch_size, num_nodes, seq_len, num_features = x.shape
            x = x.view(batch_size * num_nodes, seq_len, num_features)
        elif x.dim() == 2:
            total_nodes = x.shape[0]
            flattened_features = x.shape[1]
            
            if self.num_features and flattened_features % self.num_features == 0:
                seq_len = flattened_features // self.num_features
                num_features = self.num_features
            elif self.seq_len and flattened_features % self.seq_len == 0:
                seq_len = self.seq_len
                num_features = flattened_features // self.seq_len
            else:
                # 无法推断，回退传统模式
                return x
            
            x = x.view(total_nodes, seq_len, num_features)
        else:
            raise ValueError(f"Unsupported x shape: {x.shape}")
        
        # 简单的全连接时序编码
        x = x.squeeze(-1)  # [total_nodes, seq_len]
        x = self.temporal_encoder(x)  # [total_nodes, temporal_hidden]
        
        return x
    
    def _aggregate_nodes_to_graph_simple(
        self, 
        x: torch.Tensor, 
        batch_vector: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        简单的节点到图聚合 - 用平均池化替代复杂的注意力机制
        """
        if batch_vector is not None:
            # 多样本批处理
            num_samples = batch_vector.max().item() + 1
            graph_features = []
            
            for sample_id in range(num_samples):
                mask = (batch_vector == sample_id)
                if mask.sum() > 0:
                    # 简单平均聚合
                    graph_feature = x[mask].mean(dim=0)
                    graph_features.append(graph_feature)
                else:
                    graph_features.append(torch.zeros(self.spatial_hidden, device=x.device))
            
            return torch.stack(graph_features, dim=0)
        else:
            # 单样本：简单平均
            return x.mean(dim=0, keepdim=True)
    
    def _format_output(
        self,
        x: torch.Tensor,
        batch_vector: Optional[torch.Tensor],
        original_shape: torch.Size
    ) -> torch.Tensor:
        """格式化输出为正确的维度"""
        
        if self.aggregate_to_graph:
            # 图级输出: [batch_size, output_size * output_time] -> [batch_size, output_time, output_size]
            batch_size = x.shape[0]
            # 重塑为正确的时间序列格式：[batch_size, output_time, output_size]
            x = x.view(batch_size, self.output_time, self.output_size)
            return x
        else:
            # 节点级输出: [total_nodes, output_size * output_time] -> [batch, nodes, output_time, output_size]
            if x.dim() == 2 and len(original_shape) == 4:
                batch_size, num_nodes, _, _ = original_shape
                if x.shape[0] == batch_size * num_nodes:
                    # 重塑为正确的维度：[batch, nodes, output_time, output_size]
                    x = x.view(batch_size, num_nodes, self.output_time, self.output_size)
            elif x.dim() == 2:
                # 如果无法推断原始形状，添加时间维度
                x = x.view(x.shape[0], self.output_time, self.output_size)
            return x

    @abstractmethod
    def apply_layer(
        self,
        layer: Module,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        x_0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """应用单个GNN层"""
        pass


class GCNModel(GNNBaseModel):
    """简化的GCN模型"""
    
    def apply_layer(
        self,
        layer: Module,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        x_0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if isinstance(layer, GCNConv):
            return layer(x, edge_index, edge_weight)
        else:
            return layer(x, edge_index)


class GATModel(GNNBaseModel):
    """简化的GAT模型"""
    
    def apply_layer(
        self,
        layer: Module,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        x_0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if isinstance(layer, GATConv):
            # GAT使用edge_attr参数
            if edge_weight is not None and edge_weight.numel() > 0:
                # 确保edge_weight的维度正确
                if edge_weight.dim() == 1:
                    edge_attr = edge_weight.unsqueeze(-1)
                else:
                    edge_attr = edge_weight
                return layer(x, edge_index, edge_attr=edge_attr)
            else:
                return layer(x, edge_index)
        else:
            return layer(x, edge_index)


class GCN2Model(GNNBaseModel):
    """简化的GCNII模型"""
    
    def apply_layer(
        self,
        layer: Module,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        x_0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if isinstance(layer, GCN2Conv):
            if x_0 is None:
                raise ValueError("GCNII requires x_0 (initial features)")
            return layer(x, x_0, edge_index, edge_weight)
        else:
            return layer(x, edge_index)


class MLPModel(GNNBaseModel):
    """简化的MLP模型（作为baseline）"""
    
    def apply_layer(
        self,
        layer: Module,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        x_0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # MLP忽略图结构
        if isinstance(layer, nn.Linear):
            return layer(x)
        else:
            return x


class GNNMLP(GNNBaseModel):
    """简化的GNN MLP模型，专为小数据集设计"""
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_hidden: int = 1,
        param_sharing: bool = False,
        output_size: int = 1,
        aggregate_to_graph: bool = False,
        seq_len: Optional[int] = None,
        num_features: Optional[int] = None,
        output_time: Optional[int] = None,
        dropout: float = 0.1,
        use_temporal_modeling: bool = True,
    ) -> None:
        
        def layer_gen() -> Linear:
            # 使用与基类一致的隐藏维度
            unified_hidden_size = min(hidden_channels, 64)
            return Linear(unified_hidden_size, unified_hidden_size, weight_initializer="kaiming_uniform")

        super().__init__(
            in_channels,
            hidden_channels,
            num_hidden,
            param_sharing,
            layer_gen,
            None,  # edge_orientation
            None,  # edge_weights
            output_size,
            aggregate_to_graph,
            seq_len,
            num_features,
            output_time,
            dropout,
            use_temporal_modeling,
        )

    def apply_layer(
        self,
        layer: Module,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        x_0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # MLP忽略图结构，只使用线性层
        return layer(x)


class GCN(GNNBaseModel):
    """简化的GCN模型，专为小数据集设计"""
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_hidden: int = 2,
        param_sharing: bool = False,
        edge_orientation: Optional[str] = None,
        edge_weights: Optional[torch.Tensor] = None,
        output_size: int = 1,
        aggregate_to_graph: bool = False,
        seq_len: Optional[int] = None,
        num_features: Optional[int] = None,
        output_time: Optional[int] = None,
        dropout: float = 0.1,
        use_temporal_modeling: bool = True,
    ) -> None:
        
        # 统一隐藏层维度，确保GCN层使用正确的维度
        unified_hidden_size = min(hidden_channels, 64)
        
        def layer_gen() -> GCNConv:
            return GCNConv(unified_hidden_size, unified_hidden_size, add_self_loops=False)

        super().__init__(
            in_channels,
            hidden_channels,
            num_hidden,
            param_sharing,
            layer_gen,
            edge_orientation,
            edge_weights,
            output_size,
            aggregate_to_graph,
            seq_len,
            num_features,
            output_time,
            dropout,
            use_temporal_modeling,
        )

    def apply_layer(
        self,
        layer: Module,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        x_0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return layer(x, edge_index, edge_weight)


class ResGCN(GCN):
    """带残差连接的简化GCN"""
    
    def apply_layer(
        self,
        layer: Module,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        x_0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 残差连接：x + GCN(x)
        return x + layer(x, edge_index, edge_weight)


class GCNII(GNNBaseModel):
    """简化的GCNII模型"""
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_hidden: int = 2,
        param_sharing: bool = False,
        edge_orientation: Optional[str] = None,
        edge_weights: Optional[torch.Tensor] = None,
        output_size: int = 1,
        aggregate_to_graph: bool = False,
        seq_len: Optional[int] = None,
        num_features: Optional[int] = None,
        output_time: Optional[int] = None,
        dropout: float = 0.1,
        use_temporal_modeling: bool = True,
    ) -> None:
        
        # 统一隐藏层维度，确保与基类一致
        unified_hidden_size = min(hidden_channels, 64)
        
        def layer_gen() -> GCN2Conv:
            return GCN2Conv(unified_hidden_size, alpha=0.5, add_self_loops=False)

        super().__init__(
            in_channels,
            hidden_channels,
            num_hidden,
            param_sharing,
            layer_gen,
            edge_orientation,
            edge_weights,
            output_size,
            aggregate_to_graph,
            seq_len,
            num_features,
            output_time,
            dropout,
            use_temporal_modeling,
        )

    def apply_layer(
        self,
        layer: Module,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        x_0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if isinstance(layer, GCN2Conv):
            if x_0 is None:
                # 如果没有提供初始特征，使用当前特征作为初始特征
                x_0 = x
            return layer(x, x_0, edge_index, edge_weight)
        else:
            return layer(x, edge_index)


class ResGAT(GNNBaseModel):
    """简化的GAT模型"""
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_hidden: int = 2,
        param_sharing: bool = False,
        edge_orientation: Optional[str] = None,
        edge_weights: Optional[torch.Tensor] = None,
        output_size: int = 1,
        aggregate_to_graph: bool = False,
        seq_len: Optional[int] = None,
        num_features: Optional[int] = None,
        output_time: Optional[int] = None,
        dropout: float = 0.1,
        use_temporal_modeling: bool = True,
    ) -> None:
        
        # 统一隐藏层维度，确保与基类一致
        unified_hidden_size = min(hidden_channels, 64)
        
        def layer_gen() -> GATConv:
            return GATConv(unified_hidden_size, unified_hidden_size, add_self_loops=False)

        super().__init__(
            in_channels,
            hidden_channels,
            num_hidden,
            param_sharing,
            layer_gen,
            edge_orientation,
            edge_weights,
            output_size,
            aggregate_to_graph,
            seq_len,
            num_features,
            output_time,
            dropout,
            use_temporal_modeling,
        )

    def apply_layer(
        self,
        layer: Module,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        x_0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if isinstance(layer, GATConv):
            # GAT不直接使用edge_weight，而是通过edge_attr
            # 如果edge_weight是一维的，需要重新构建edge_index以移除零权重边
            if edge_weight is not None and edge_weight.dim() == 1:
                valid_edges = edge_weight != 0
                if valid_edges.sum() > 0:
                    filtered_edge_index = edge_index[:, valid_edges]
                    filtered_edge_weight = edge_weight[valid_edges]
                    # GAT使用edge_attr参数而不是edge_weight
                    return x + layer(x, filtered_edge_index, edge_attr=filtered_edge_weight.unsqueeze(-1) if filtered_edge_weight.dim() == 1 else filtered_edge_weight)
                else:
                    # 如果没有有效边，只处理自环
                    return x + layer(x, edge_index)
            else:
                return x + layer(x, edge_index)
        else:
            # 其他类型的层
            return x + layer(x, edge_index)


# 保持向后兼容性的别名
SimpleGCNModel = GCNModel
SimpleGATModel = GATModel
SimpleGCN2Model = GCN2Model
SimpleMLP = MLPModel