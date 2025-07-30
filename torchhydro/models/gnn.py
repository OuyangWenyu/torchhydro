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
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        batch_vector: Optional[torch.Tensor] = None,
        evo_tracking: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        通用GNN前向传播，兼容两种输入模式：
        1. batch_vector模式（变节点数batch，PyG风格，适合大规模异构图/多流域拼接）
        2. 传统batch维度模式（[batch, num_nodes, ...]，适合定长节点数）

        参数:
            x: 节点特征张量，shape见上
            edge_index: 边索引，PyG格式
            edge_weight: 边权重，若为None则自动补1
            batch_vector: 节点到batch的映射（如有）
            evo_tracking: 是否记录每层输出
        返回:
            预测结果，或(预测, 演化序列)
        """
        # 保险：所有输入 tensor 强制同步到 encoder.device，防止 device 不一致
        device = self.encoder.weight.device
        x = x.to(device)
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device)
        if batch_vector is not None:
            batch_vector = batch_vector.to(device)
        if batch_vector is not None:
            # 支持 x 为 [batch, num_nodes, window_size, num_features] 或 [total_nodes, window_size, num_features]
            if x.dim() == 4:
                x = x.view(-1, x.size(2), x.size(3))
            elif x.dim() != 3:
                raise ValueError(f"Unsupported x shape for GNN batch_vector mode: {x.shape}")
            x = x.view(x.size(0), -1)
            x_0 = self.encoder(x)
            evolution = [x_0.detach()] if evo_tracking else None
            x = x_0
            for layer in self.layers:
                x = self.apply_layer(layer, x, x_0, edge_index, edge_weight)
                if evo_tracking:
                    evolution.append(x.detach())
            x = self.decoder(x)
            if self.root_gauge_idx is not None:
                if evo_tracking:
                    evolution.append(x.detach())
                batch_size = batch_vector.max().item() + 1
                # PyTorch原生实现batch mean聚合
                out_sum = torch.zeros(batch_size, x.size(-1), device=x.device)
                out_sum = out_sum.index_add(0, batch_vector, x)
                count = torch.zeros(batch_size, device=x.device)
                count = count.index_add(0, batch_vector, torch.ones_like(batch_vector, dtype=x.dtype))
                count = count.clamp_min(1).unsqueeze(-1)
                x = out_sum / count
                # 保证输出 shape 为 [batch, time, feature]，即 [batch, output_size, 1]（如果 output_size=时间步，特征数=1）
                if x.dim() == 2:
                    x = x.unsqueeze(-1)
                if evo_tracking:
                    return x, evolution
            else:
                # 保证输出 shape 为 [node, time, feature]，即 [N, output_size, 1]
                if x.dim() == 2:
                    x = x.unsqueeze(-1)
            return (x, evolution) if evo_tracking else x
        else:
            # 标准 batch 模式，要求 edge_weight 必须输入，和 edge_index 一致
            batch_size, num_nodes, window_size, num_features = x.shape
            x = x.view(batch_size * num_nodes, window_size * num_features)
            # edge_index: [2, num_edges] 或 [batch, 2, num_edges]
            if edge_index.dim() == 3:
                node_offsets = torch.arange(batch_size, device=edge_index.device) * num_nodes
                node_offsets = node_offsets.view(-1, 1, 1)
                edge_index_offset = edge_index + node_offsets
                edge_index = edge_index_offset.transpose(0, 1).contiguous().view(2, -1)
            else:
                if batch_size > 1:
                    edge_indices = []
                    for b in range(batch_size):
                        offset = b * num_nodes
                        edge_indices.append(edge_index + offset)
                    edge_index = torch.cat(edge_indices, dim=1)
            # 添加自环（如有需要，可在数据集预处理）
            # edge_index, edge_weight = add_self_loops(edge_index, edge_weight, num_nodes=batch_size * num_nodes)
            x_0 = self.encoder(x)
            evolution: Optional[List[torch.Tensor]] = [x_0.detach()] if evo_tracking else None
            x = x_0
            for layer in self.layers:
                x = self.apply_layer(layer, x, x_0, edge_index, edge_weight)
                if evo_tracking:
                    evolution.append(x.detach())
            x = self.decoder(x)
            if self.root_gauge_idx is not None:
                if evo_tracking:
                    evolution.append(x.detach())
                x = x.view(batch_size, num_nodes, self.output_size)
                if self.aggregation_layer is None:
                    input_dim = num_nodes * self.output_size
                    self.aggregation_layer = Linear(
                        input_dim, self.output_size, weight_initializer="kaiming_uniform"
                    ).to(x.device)
                x_flat = x.view(batch_size, -1)
                x = self.aggregation_layer(x_flat)
                if evo_tracking:
                    return x, evolution
            else:
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
