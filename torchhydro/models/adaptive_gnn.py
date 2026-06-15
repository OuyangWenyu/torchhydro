"""
自适应图网络模型适配器
======================

将 LamaH-CE 中的 WDNEInspiredFloodNet / WDNEAdaptiveRoutingNet 适配到
torchhydro 现有的训练框架中。

核心思路：
1. 接受现有 GNN 模型的输入格式 (x, edge_index, edge_weight, batch_vector)
2. 内部构造 torch_geometric Data 对象
3. 自动从 x 中分离静态属性（末尾的 constant_cols 在每个时间步重复拼接的）
4. 自动推断根节点（出口）位置
5. 提取根节点预测作为 batch 级输出，对齐现有训练 target 格式
6. 调用底层模型并返回匹配现有接口的输出

现有接口完全不变：GNNDataset / gnn_collate_fn / model_infer 都不需要修改。
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
from torch_geometric.data import Data

from torchhydro.models.wdne_inspired_flood_net import WDNEInspiredFloodNet
from torchhydro.models.wdne_adaptive_routing_net import WDNEAdaptiveRoutingNet


class WDNEAdapter(nn.Module):
    """
    将 WDNE 系列模型适配到 torchhydro 现有接口。

    现有接口: forward(x, edge_index, edge_weight, batch_vector, **kwargs)
    新模型接口: forward(batch: Data)

    适配器负责：
    - 格式转换：tensor kwargs → Data 对象
    - 静态特征分离：x 末尾的 constant_cols → static_attr
    - 根节点推断：出度为 0 的节点 → root_local_idx
    - 输出提取：per-node 输出 → 根节点（出口）per-graph 输出

    Parameters
    ----------
    model_type : str
        "wdne"  → WDNEInspiredFloodNet (静态 GCN 图传播)
        "wdne_adaptive" → WDNEAdaptiveRoutingNet (自适应边权重路由)

    num_static_features : int
        x 末尾的静态特征维度数。当前 dataset 将 constant_cols 复制拼接到
        每个时间步，所以 x 的最后一维 = dynamic_features + static_features。
        设 0 则模型使用可学习 embedding 替代静态属性。

    root_output_mode : str
        如何从 per-node 输出得到 per-graph 预测：
        - "root" (默认): 取出口节点（出度为0）的预测
        - "mean": 所有节点取平均（适合没有明确出口的场景）
    """

    def __init__(
        self,
        model_type: str = "wdne_adaptive",
        num_static_features: int = 0,
        root_output_mode: str = "root",
        **model_hyperparams: Any,
    ) -> None:
        super().__init__()

        if model_type not in ("wdne", "wdne_adaptive"):
            raise ValueError(
                f"model_type must be 'wdne' or 'wdne_adaptive', got '{model_type}'"
            )
        if root_output_mode not in ("root", "mean"):
            raise ValueError(
                f"root_output_mode must be 'root' or 'mean', got '{root_output_mode}'"
            )

        self.model_type = model_type
        self.num_static_features = num_static_features
        self.root_output_mode = root_output_mode

        # 构建底层模型
        if model_type == "wdne":
            self.inner: WDNEInspiredFloodNet = WDNEInspiredFloodNet(
                **model_hyperparams,
            )
        else:
            self.inner = WDNEAdaptiveRoutingNet(
                **model_hyperparams,
            )

        # 强制初始化 LazyLinear 层，避免保存 checkpoint 时含 UninitializedParameter
        # (WDNE 模型的 attribute_encoder 使用了 nn.LazyLinear，需要一次前向才能确定维度)
        self._init_lazy_modules()

        # 暴露底层模型的关键属性，方便外部追踪
        self._last_edge_weights: Optional[torch.Tensor] = None

    def _init_lazy_modules(self) -> None:
        """用 dummy 输入强制所有 LazyLinear 完成初始化"""
        dummy_static_attr = torch.randn(1, max(self.num_static_features, 1))
        try:
            _ = self.inner.attribute_encoder(dummy_static_attr)
        except Exception:
            pass  # 某些配置下 attribute_encoder 可能不存在或已初始化

    # ---- 辅助函数 ----

    @staticmethod
    def _build_ptr(
        batch_vector: torch.Tensor, num_graphs: int
    ) -> torch.Tensor:
        """构建每个子图的起止指针，类似 torch_geometric 的 ptr"""
        counts = torch.bincount(batch_vector, minlength=num_graphs)
        ptr = torch.zeros(num_graphs + 1, device=batch_vector.device, dtype=torch.long)
        ptr[1:] = torch.cumsum(counts, dim=0)
        return ptr

    @staticmethod
    def _infer_root_local_idx(
        edge_index: torch.Tensor,
        batch_vector: torch.Tensor,
    ) -> torch.Tensor:
        """
        为 batch 中的每个子图推断出口节点（局部索引）。

        在有向河流网络中，出口节点的出度为 0（没有下游节点）。
        对每个子图，找到第一个出度为 0 的节点作为出口。

        Returns
        -------
        torch.Tensor [batch_size] : 每个子图的出口局部索引
        """
        num_nodes = int(batch_vector.size(0))
        num_graphs = int(batch_vector.max().item()) + 1

        # 计算每个节点的出度
        out_degree = torch.zeros(num_nodes, device=edge_index.device, dtype=torch.long)
        ones = torch.ones(edge_index.size(1), device=edge_index.device, dtype=torch.long)
        out_degree.scatter_add_(0, edge_index[0], ones)

        roots = []
        for g in range(num_graphs):
            mask = batch_vector == g
            graph_out_degree = out_degree[mask]
            zero_out = (graph_out_degree == 0).nonzero(as_tuple=False).view(-1)

            if zero_out.numel() > 0:
                roots.append(int(zero_out[0].item()))
            else:
                # 环形图或无向图：取最后一个节点
                roots.append(int(mask.sum().item() - 1))

        return torch.tensor(roots, device=edge_index.device, dtype=torch.long)

    def _extract_static_attr(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        从 x 中分离动态特征和静态属性。

        当前 dataset 将 constant_cols 复制拼接到每个时间步的末尾，
        所以 x[:, 0, -num_static:] 就是每个节点的静态属性。

        Returns
        -------
        (x_dynamic, static_attr)
            x_dynamic: [N, T, dynamic_dim]
            static_attr: [N, static_dim] 或 None
        """
        if self.num_static_features <= 0:
            return x, None

        static_attr = x[:, 0, -self.num_static_features:]
        x_dynamic = x[:, :, :-self.num_static_features]
        return x_dynamic, static_attr

    def _extract_root_output(
        self,
        output: torch.Tensor,
        batch_vector: torch.Tensor,
        root_local_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        从 per-node 输出中提取每个子图出口节点的预测。

        Parameters
        ----------
        output : [total_nodes, output_time, output_size]
        batch_vector : [total_nodes]
        root_local_idx : [batch_size]

        Returns
        -------
        torch.Tensor : [batch_size, output_time, output_size]
        """
        num_graphs = int(batch_vector.max().item()) + 1
        ptr = self._build_ptr(batch_vector, num_graphs)

        root_outputs = []
        for g in range(num_graphs):
            start = int(ptr[g].item())
            root_global = start + int(root_local_idx[g].item())
            root_outputs.append(output[root_global])

        return torch.stack(root_outputs, dim=0)

    @staticmethod
    def _mean_pool_output(
        output: torch.Tensor,
        batch_vector: torch.Tensor,
    ) -> torch.Tensor:
        """
        对每个子图的所有节点取平均，得到 per-graph 预测。

        Parameters
        ----------
        output : [total_nodes, output_time, output_size]
        batch_vector : [total_nodes]

        Returns
        -------
        torch.Tensor : [batch_size, output_time, output_size]
        """
        num_graphs = int(batch_vector.max().item()) + 1
        pooled = []
        for g in range(num_graphs):
            mask = batch_vector == g
            pooled.append(output[mask].mean(dim=0))
        return torch.stack(pooled, dim=0)

    # ---- 主前向传播 ----

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        batch_vector: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        适配器前向传播。

        Parameters
        ----------
        x : [total_nodes, seq_len, features]
            节点特征，最后一维 = dynamic_features + num_static_features
        edge_index : [2, total_edges]
        edge_weight : [total_edges]
        batch_vector : [total_nodes]
            每个节点属于 batch 中第几个样本

        Returns
        -------
        torch.Tensor : [batch_size, output_time, output_size]  或
                       [batch_size, output_size] (当 output_time=1)
            预测结果，已从 per-node 提取为 per-graph，对齐训练 target 维度
        """
        device = next(self.parameters()).device

        x = x.to(device)
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device)
        if batch_vector is not None:
            batch_vector = batch_vector.to(device)

        # 1. 分离静态属性
        x_dynamic, static_attr = self._extract_static_attr(x)

        # 2. 推断根节点（出口）
        if batch_vector is not None and batch_vector.numel() > 0:
            root_local_idx = self._infer_root_local_idx(edge_index, batch_vector)
        else:
            root_local_idx = None

        # 3. 构造 Data 对象
        data = Data(
            x=x_dynamic,
            edge_index=edge_index,
            edge_weight=edge_weight,
            batch=batch_vector,
            static_attr=static_attr,
            root_local_idx=root_local_idx,
        )

        # 4. 调用底层模型 → per-node 输出
        output = self.inner(data)  # [total_nodes, output_time, output_size]

        # 5. 提取 per-graph 预测以对齐训练 target
        if batch_vector is not None and batch_vector.numel() > 0:
            if self.root_output_mode == "root" and root_local_idx is not None:
                output = self._extract_root_output(output, batch_vector, root_local_idx)
            elif self.root_output_mode == "mean":
                output = self._mean_pool_output(output, batch_vector)

        # 6. 收集调试信息
        if isinstance(self.inner, WDNEAdaptiveRoutingNet):
            self._last_edge_weights = getattr(
                self.inner, "last_adaptive_edge_weights", None
            )

        return output

    # ---- 辅助访问 ----

    @property
    def last_edge_weights(self) -> Optional[torch.Tensor]:
        """最近一次前向传播的自适应边权重（仅 WDNEAdaptiveRoutingNet）"""
        return self._last_edge_weights

    def get_inner_model(self) -> nn.Module:
        """获取底层模型，用于调试或单独访问"""
        return self.inner
