import torch
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Callable, Union, Any
from torch.nn import Module, ModuleList
from torch.nn.functional import relu
from torch_geometric.nn import GATConv, GCNConv, GCN2Conv, Linear
from torch_geometric.utils import add_self_loops


class GNNBaseModel(Module, ABC):
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
        root_gauge_idx: Optional[int] = None,
    ) -> None:
        super().__init__()
        # 修改：支持多时段输出
        self.output_size = output_size
        self.root_gauge_idx = root_gauge_idx

        self.encoder = Linear(
            in_channels, hidden_channels, weight_initializer="kaiming_uniform"
        )
        if param_sharing:
            self.layers = ModuleList(num_hidden * [layerfun()])
        else:
            self.layers = ModuleList([layerfun() for _ in range(num_hidden)])
        # 传统的decoder（用于所有节点输出）
        self.decoder = Linear(
            hidden_channels, output_size, weight_initializer="kaiming_uniform"
        )
        # 聚合层：将所有节点的信息聚合到根节点
        if root_gauge_idx is not None:
            # 这个层将在forward中动态创建，需要知道确切的节点数
            self.aggregation_layer: Optional[Linear] = None

        self.edge_weights = edge_weights
        self.edge_orientation = edge_orientation
        # 设置自环填充值，优先使用预设的edge_weights，如果没有则在forward中动态设置
        if self.edge_weights is not None:
            self.loop_fill_value: Union[float, str] = (
                1.0 if (self.edge_weights == 0).all() else "mean"
            )
        else:
            # 如果没有预设权重，使用默认值，在forward中可能会根据输入的edge_weight调整
            self.loop_fill_value: Union[float, str] = "mean"

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor = None, evo_tracking: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        # x的形状: [batch_size, num_nodes, window_size, num_features]
        batch_size, num_nodes, window_size, num_features = x.shape
        
        # 重塑x为GNN期望的格式: [batch_size * num_nodes, window_size * num_features]
        x = x.view(batch_size * num_nodes, window_size * num_features)

        # 处理edge_index和edge_weight
        # edge_index形状: [2, num_edges] 或者 [batch_size, 2, num_edges]
        # edge_weight形状: [batch_size, num_edges] 或者 [num_edges]
        
        if edge_index.dim() == 3:
            # 如果是batch格式: [batch_size, 2, num_edges]
            # 需要为每个batch创建偏移的节点索引
            
            # 创建节点偏移量
            node_offsets = torch.arange(batch_size, device=edge_index.device) * num_nodes
            node_offsets = node_offsets.view(-1, 1, 1)  # [batch_size, 1, 1]
            
            # 添加偏移量到edge_index
            edge_index_offset = edge_index + node_offsets  # [batch_size, 2, num_edges]
            
            # 重塑为 [2, batch_size * num_edges]
            edge_index = edge_index_offset.transpose(0, 1).contiguous().view(2, -1)
        else:
            # 如果已经是标准格式: [2, num_edges]，需要复制到所有batch
            if batch_size > 1:
                # 为每个batch创建偏移的edge_index
                edge_indices = []
                for b in range(batch_size):
                    offset = b * num_nodes
                    edge_indices.append(edge_index + offset)
                edge_index = torch.cat(edge_indices, dim=1)

        edge_index = edge_index.to(x.device)

        # 处理edge_weight
        if edge_weight is not None:
            if edge_weight.dim() == 2:
                # edge_weight形状: [batch_size, num_edges]
                edge_weights = edge_weight.view(-1).abs().to(x.device)
            else:
                # edge_weight形状: [num_edges]，需要复制到所有batch
                edge_weights = edge_weight.repeat(batch_size).abs().to(x.device)
            # 动态设置自环填充值
            loop_fill_value = 1.0 if (edge_weights == 0).all() else "mean"
        elif self.edge_weights is not None:
            # 如果没有输入edge_weight但模型有预设的edge_weights，使用预设值
            edge_weights = self.edge_weights.repeat(batch_size).abs().to(x.device)
            loop_fill_value = self.loop_fill_value
        else:
            # 如果都没有，创建单位权重
            num_edges_total = edge_index.size(-1)
            edge_weights = torch.ones(num_edges_total, device=x.device)
            loop_fill_value = 1.0

        # 处理边的方向
        if self.edge_orientation is not None:
            if self.edge_orientation == "upstream":
                edge_index = edge_index[[1, 0]]  # 交换源和目标节点
            elif self.edge_orientation == "bidirectional":
                edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
                edge_weights = torch.cat([edge_weights, edge_weights], dim=0)
            elif self.edge_orientation != "downstream":
                raise ValueError("unknown edge direction", self.edge_orientation)
        
        # 添加自环
        edge_index, edge_weights = add_self_loops(
            edge_index, edge_weights, fill_value=loop_fill_value, num_nodes=batch_size * num_nodes
        )

        x_0 = self.encoder(x)
        evolution: Optional[List[torch.Tensor]] = (
            [x_0.detach()] if evo_tracking else None
        )

        x = x_0
        for layer in self.layers:
            x = self.apply_layer(layer, x, x_0, edge_index, edge_weights)
            if evo_tracking:
                evolution.append(x.detach())
        x = self.decoder(x)

        # 如果指定了根节点，进行聚合操作
        if self.root_gauge_idx is not None:
            if evo_tracking:
                evolution.append(x.detach())

            # 重新整形为: [batch_size, num_nodes, output_size]
            x = x.view(batch_size, num_nodes, self.output_size)

            # 动态创建聚合层（如果还没创建）
            if self.aggregation_layer is None:
                input_dim = num_nodes * self.output_size  # num_nodes * output_size
                self.aggregation_layer = Linear(
                    input_dim, self.output_size, weight_initializer="kaiming_uniform"
                ).to(x.device)

            # 展平所有节点的特征: [batch_size, num_nodes * output_size]
            x_flat = x.view(batch_size, -1)

            # 聚合到根节点输出: [batch_size, output_size]
            x = self.aggregation_layer(x_flat)
            
            if evo_tracking:
                return x, evolution
        else:
            # 如果没有根节点聚合，重新整形为: [batch_size, num_nodes, output_size]
            x = x.view(batch_size, num_nodes, self.output_size)

        return (x, evolution) if evo_tracking else x

    @abstractmethod
    def apply_layer(
        self,
        layer: Module,
        x: torch.Tensor,
        x_0: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: torch.Tensor,
    ) -> torch.Tensor:
        pass


class GNNMLP(GNNBaseModel):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_hidden: int,
        param_sharing: bool,
        output_size: int = 1,
        root_gauge_idx: Optional[int] = None,
    ) -> None:
        def layer_gen() -> Linear:
            return Linear(
                hidden_channels, hidden_channels, weight_initializer="kaiming_uniform"
            )

        super().__init__(
            in_channels,
            hidden_channels,
            num_hidden,
            param_sharing,
            layer_gen,
            None,
            None,
            output_size,
            root_gauge_idx,
        )

    def apply_layer(
        self,
        layer: Module,
        x: torch.Tensor,
        x_0: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: torch.Tensor,
    ) -> torch.Tensor:
        return relu(layer(x))


class GCN(GNNBaseModel):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_hidden: int,
        param_sharing: bool,
        edge_orientation: Optional[str] = None,
        edge_weights: Optional[torch.Tensor] = None,  # 保持向后兼容
        output_size: int = 1,
        root_gauge_idx: Optional[int] = None,
    ) -> None:
        def layer_gen() -> GCNConv:
            return GCNConv(hidden_channels, hidden_channels, add_self_loops=False)

        super().__init__(
            in_channels,
            hidden_channels,
            num_hidden,
            param_sharing,
            layer_gen,
            edge_orientation,
            edge_weights,  # 可以为None，将通过forward参数传入
            output_size,
            root_gauge_idx,
        )

    def apply_layer(
        self,
        layer: Module,
        x: torch.Tensor,
        x_0: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: torch.Tensor,
    ) -> torch.Tensor:
        return relu(layer(x, edge_index, edge_weights))


class ResGCN(GCN):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_hidden: int,
        param_sharing: bool,
        edge_orientation: Optional[str] = None,
        edge_weights: Optional[torch.Tensor] = None,  # 保持向后兼容
        output_size: int = 1,
        root_gauge_idx: Optional[int] = None,
    ) -> None:
        super().__init__(
            in_channels,
            hidden_channels,
            num_hidden,
            param_sharing,
            edge_orientation,
            edge_weights,
            output_size,
            root_gauge_idx,
        )

    def apply_layer(
        self,
        layer: Module,
        x: torch.Tensor,
        x_0: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: torch.Tensor,
    ) -> torch.Tensor:
        return x + super().apply_layer(layer, x, x_0, edge_index, edge_weights)


class GCNII(GNNBaseModel):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_hidden: int,
        param_sharing: bool,
        edge_orientation: Optional[str] = None,
        edge_weights: Optional[torch.Tensor] = None,  # 保持向后兼容
        output_size: int = 1,
        root_gauge_idx: Optional[int] = None,
    ) -> None:
        def layer_gen() -> GCN2Conv:
            return GCN2Conv(hidden_channels, alpha=0.5, add_self_loops=False)

        super().__init__(
            in_channels,
            hidden_channels,
            num_hidden,
            param_sharing,
            layer_gen,
            edge_orientation,
            edge_weights,
            output_size,
            root_gauge_idx,
        )

    def apply_layer(
        self,
        layer: Module,
        x: torch.Tensor,
        x_0: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: torch.Tensor,
    ) -> torch.Tensor:
        return relu(layer(x, x_0, edge_index, edge_weights))


class ResGAT(GNNBaseModel):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_hidden: int,
        param_sharing: bool,
        edge_orientation: Optional[str] = None,
        edge_weights: Optional[torch.Tensor] = None,  # 保持向后兼容
        output_size: int = 1,
        root_gauge_idx: Optional[int] = None,
    ) -> None:
        def layer_gen() -> GATConv:
            return GATConv(hidden_channels, hidden_channels, add_self_loops=False)

        super().__init__(
            in_channels,
            hidden_channels,
            num_hidden,
            param_sharing,
            layer_gen,
            edge_orientation,
            edge_weights,
            output_size,
            root_gauge_idx,
        )

    def apply_layer(
        self,
        layer: Module,
        x: torch.Tensor,
        x_0: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: torch.Tensor,
    ) -> torch.Tensor:
        if edge_weights.dim() == 1:
            edge_index = edge_index[:, edge_weights != 0]
        return x + relu(layer(x, edge_index, edge_weights))
