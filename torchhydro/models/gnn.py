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
        if self.edge_weights is not None:
            self.loop_fill_value: Union[float, str] = (
                1.0 if (self.edge_weights == 0).all() else "mean"
            )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, evo_tracking: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        # x的形状: [batch_size * num_nodes, window_size, num_features]
        # 从edge_index推断batch_size和num_nodes
        if self.edge_weights is not None:
            # num_graphs就是batch_size
            num_graphs = edge_index.size(1) // len(self.edge_weights)
            batch_size = num_graphs
        else:
            # 如果没有edge_weights，从x的第一维推断
            # 假设所有图都有相同的节点数
            batch_size = 1  # 需要根据实际情况调整

        # 计算每个图的节点数
        num_nodes = x.size(0) // batch_size

        x = x.flatten(1)  # 展平时间和特征维度: [batch_size * num_nodes, features]

        if self.edge_weights is not None:
            edge_weights = torch.cat(batch_size * [self.edge_weights], dim=0).to(
                x.device
            )
            edge_weights = edge_weights.abs()
        else:
            edge_weights = torch.zeros(edge_index.size(1)).to(x.device)

        if self.edge_orientation is not None:
            if self.edge_orientation == "upstream":
                edge_index = edge_index[[1, 0]].to(x.device)
            elif self.edge_orientation == "bidirectional":
                edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1).to(
                    x.device
                )
                edge_weights = torch.cat(2 * [edge_weights], dim=0).to(x.device)
            elif self.edge_orientation != "downstream":
                raise ValueError("unknown edge direction", self.edge_orientation)
        if self.edge_weights is not None:
            edge_index, edge_weights = add_self_loops(
                edge_index, edge_weights, fill_value=self.loop_fill_value
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

            # 重新整形: [batch_size, num_nodes, hidden_channels]
            x = x.view(batch_size, num_nodes, -1)

            # 动态创建聚合层（如果还没创建）
            if self.aggregation_layer is None:
                input_dim = num_nodes * x.size(-1)  # num_nodes * hidden_channels
                self.aggregation_layer = Linear(
                    input_dim, self.output_size, weight_initializer="kaiming_uniform"
                ).to(x.device)

            # 展平所有节点的特征: [batch_size, num_nodes * hidden_channels]
            x_flat = x.view(batch_size, -1)

            # 聚合到根节点输出: [batch_size, output_size]
            x = self.aggregation_layer(x_flat)
            if evo_tracking:
                return x, evolution

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
        edge_orientation: Optional[str],
        edge_weights: Optional[torch.Tensor],
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
        return relu(layer(x, edge_index, edge_weights))


class ResGCN(GCN):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_hidden: int,
        param_sharing: bool,
        edge_orientation: Optional[str],
        edge_weights: Optional[torch.Tensor],
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
        edge_orientation: Optional[str],
        edge_weights: Optional[torch.Tensor],
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
        edge_orientation: Optional[str],
        edge_weights: Optional[torch.Tensor],
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
