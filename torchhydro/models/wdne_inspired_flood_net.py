"""WDNE-inspired structure-aware flood forecasting baseline.

This module keeps the original SandwichFloodNet path untouched and provides a
comparison model that explicitly separates:

1. local temporal encoding via LSTM;
2. structural-role embedding from topology + static attributes; and
3. graph propagation over the river network.
"""

from __future__ import annotations

from collections import deque
import hashlib
from typing import Any, Optional

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


def _build_ptr(batch_vector: Optional[torch.Tensor], num_nodes: int, device: torch.device) -> torch.Tensor:
    if batch_vector is None or batch_vector.numel() == 0:
        return torch.tensor([0, num_nodes], device=device, dtype=torch.long)
    batch_vector = batch_vector.to(device=device, dtype=torch.long)
    num_graphs = int(batch_vector.max().item()) + 1 if batch_vector.numel() else 1
    counts = torch.bincount(batch_vector, minlength=num_graphs)
    ptr = torch.zeros(num_graphs + 1, device=device, dtype=torch.long)
    ptr[1:] = torch.cumsum(counts, dim=0)
    return ptr


def _infer_root_indices(
    edge_index: torch.Tensor,
    ptr: torch.Tensor,
    root_local_idx: Optional[torch.Tensor],
) -> list[int]:
    roots: list[int] = []
    total_nodes = int(ptr[-1].item())
    out_degree = torch.bincount(edge_index[0], minlength=total_nodes) if edge_index.numel() else torch.zeros(total_nodes, dtype=torch.long, device=ptr.device)
    for graph_idx in range(ptr.numel() - 1):
        start = int(ptr[graph_idx].item())
        end = int(ptr[graph_idx + 1].item())
        if root_local_idx is not None and graph_idx < root_local_idx.numel():
            roots.append(start + int(root_local_idx[graph_idx].item()))
            continue
        graph_out = out_degree[start:end]
        sink_nodes = torch.nonzero(graph_out == 0, as_tuple=False).view(-1)
        if sink_nodes.numel() > 0:
            roots.append(start + int(sink_nodes[0].item()))
        else:
            roots.append(end - 1)
    return roots


def _graph_distances_to_outlet(
    edge_index: torch.Tensor,
    ptr: torch.Tensor,
    roots: list[int],
    device: torch.device,
) -> torch.Tensor:
    total_nodes = int(ptr[-1].item())
    dist = torch.full((total_nodes,), -1.0, dtype=torch.float32, device=device)
    if total_nodes == 0:
        return dist
    reverse_adj: list[list[int]] = [[] for _ in range(total_nodes)]
    for src, dst in edge_index.t().tolist():
        reverse_adj[int(dst)].append(int(src))
    for root in roots:
        queue: deque[int] = deque([int(root)])
        dist[root] = 0.0
        while queue:
            node = queue.popleft()
            node_dist = float(dist[node].item())
            for parent in reverse_adj[node]:
                if dist[parent] >= 0:
                    continue
                dist[parent] = node_dist + 1.0
                queue.append(parent)
    return dist


def _graph_upstream_depth(edge_index: torch.Tensor, ptr: torch.Tensor, device: torch.device) -> torch.Tensor:
    total_nodes = int(ptr[-1].item())
    depth = torch.zeros((total_nodes,), dtype=torch.float32, device=device)
    if total_nodes == 0:
        return depth
    children: list[list[int]] = [[] for _ in range(total_nodes)]
    indegree = torch.zeros((total_nodes,), dtype=torch.long, device=device)
    for src, dst in edge_index.t().tolist():
        src_i = int(src)
        dst_i = int(dst)
        children[src_i].append(dst_i)
        indegree[dst_i] += 1
    for graph_idx in range(ptr.numel() - 1):
        start = int(ptr[graph_idx].item())
        end = int(ptr[graph_idx + 1].item())
        local_indegree = indegree[start:end].clone()
        queue: deque[int] = deque((torch.nonzero(local_indegree == 0, as_tuple=False).view(-1) + start).tolist())
        while queue:
            node = queue.popleft()
            for child in children[node]:
                depth[child] = torch.maximum(depth[child], depth[node] + 1.0)
                local_indegree[child - start] -= 1
                if local_indegree[child - start] == 0:
                    queue.append(child)
    return depth


class WDNEInspiredFloodNet(nn.Module):
    """Comparison model with structure embedding, local LSTM, and graph propagation."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        output_size: int = 1,
        seq_len: int = 24,
        output_time: int = 1,
        forecast_mode: str = "single",
        output_mode: str = "node",
        structure_embedding_dim: int = 32,
        structure_hidden_dim: Optional[int] = None,
        topology_encoder_type: str = "mlp",
        topology_graph_layers: int = 2,
        topology_graph_dropout: Optional[float] = None,
        local_lstm_layers: int = 1,
        route_steps: int = 3,
        route_decay: float = 0.7,
        route_mode: str = "reachability",
        route_residual_scale: float = 1.0,
        route_gate_bias: float = -0.5,
        graph_output_mode: str = "residual",
        graph_blend_bias: float = -1.0,
        local_aux_loss_weight: float = 0.0,
        local_aux_loss_name: str = "mse",
        structure_fusion: str = "gated",
        structure_ablation_mode: str = "fusion",
        use_structure_embedding: bool = True,
        local_branch_use_structure_input: bool = True,
        temporal_use_static_input: bool = False,
        dropout: float = 0.1,
        root_gauge_idx: Optional[int] = None,
        **_: Any,
    ) -> None:
        super().__init__()
        if output_mode.lower() != "node":
            raise ValueError("WDNEInspiredFloodNet currently supports only output_mode='node'")
        if route_mode.lower() not in {"direct", "reachability"}:
            raise ValueError("route_mode must be 'direct' or 'reachability'")
        normalized_graph_output_mode = graph_output_mode.lower()
        if normalized_graph_output_mode not in {"residual", "blend_residual", "main"}:
            raise ValueError("graph_output_mode must be 'residual', 'blend_residual', or 'main'")
        if structure_fusion.lower() not in {"gated", "mean"}:
            raise ValueError("structure_fusion must be 'gated' or 'mean'")
        normalized_ablation_mode = structure_ablation_mode.lower()
        if normalized_ablation_mode == "static_only":
            normalized_ablation_mode = "attribute_only"
        normalized_topology_encoder_type = topology_encoder_type.lower()
        if normalized_topology_encoder_type == "graph":
            normalized_topology_encoder_type = "gcn"
        if normalized_topology_encoder_type not in {"mlp", "gcn"}:
            raise ValueError("topology_encoder_type must be 'mlp', 'gcn', or 'graph'")
        if normalized_ablation_mode not in {"fusion", "topology_only", "attribute_only"}:
            raise ValueError(
                "structure_ablation_mode must be one of 'fusion', 'topology_only', 'attribute_only', or 'static_only'"
            )

        self.input_dim = int(in_channels)
        self.hidden_channels = int(hidden_channels)
        self.output_size = int(output_size)
        self.output_time = int(output_time)
        self.output_horizon = self.output_time
        self.forecast_mode = str(forecast_mode).lower()
        self.output_mode = "node"
        self.prediction_level = "node"
        self.target_aggregation = "none"
        self.supports_sequence_output = self.forecast_mode == "multi" and self.output_time > 1
        self.is_sequence_output = self.supports_sequence_output
        self.output_structure = "sequence" if self.supports_sequence_output else "scalar"
        self.accepts_data_batch = True
        self.requires_batch_vector = False
        self.requires_full_batch = False
        self.requires_edge_weight_input = False
        self.root_gauge_idx = root_gauge_idx

        self.structure_embedding_dim = int(structure_embedding_dim)
        self.structure_hidden_dim = int(structure_hidden_dim or hidden_channels)
        self.topology_encoder_type = normalized_topology_encoder_type
        self.topology_graph_layer_count = max(1, int(topology_graph_layers))
        self.topology_graph_dropout_rate = max(
            0.0,
            float(dropout if topology_graph_dropout is None else topology_graph_dropout),
        )
        self.local_lstm_layers = int(local_lstm_layers)
        self.graph_layers_count = max(1, int(route_steps))
        self.route_steps = max(1, int(route_steps))
        self.route_decay = float(route_decay)
        self.route_mode = route_mode.lower()
        self.route_residual_scale = float(route_residual_scale)
        self.graph_output_mode = normalized_graph_output_mode
        self.local_aux_loss_weight = max(0.0, float(local_aux_loss_weight))
        self.local_aux_loss_name = str(local_aux_loss_name)
        self.structure_fusion = structure_fusion.lower()
        self.structure_ablation_mode = normalized_ablation_mode
        self.use_structure_embedding = bool(use_structure_embedding)
        self.local_branch_use_structure_input = bool(local_branch_use_structure_input)
        self.temporal_use_static_input = bool(temporal_use_static_input)

        self.topology_encoder = nn.Sequential(
            nn.Linear(6, self.structure_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.structure_hidden_dim, self.structure_embedding_dim),
        )
        if self.topology_encoder_type == "gcn":
            self.topology_graph_input_projection = nn.Linear(6, self.structure_hidden_dim)
            self.topology_graph_layers = nn.ModuleList(
                GCNConv(self.structure_hidden_dim, self.structure_hidden_dim, add_self_loops=True, normalize=True)
                for _ in range(self.topology_graph_layer_count)
            )
            self.topology_graph_norms = nn.ModuleList(
                nn.LayerNorm(self.structure_hidden_dim) for _ in range(self.topology_graph_layer_count)
            )
            self.topology_graph_output_projection = nn.Sequential(
                nn.Linear(self.structure_hidden_dim, self.structure_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.structure_hidden_dim, self.structure_embedding_dim),
            )
            self.topology_graph_dropout = nn.Dropout(self.topology_graph_dropout_rate)
        self.attribute_encoder = nn.Sequential(
            nn.LazyLinear(self.structure_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.structure_hidden_dim, self.structure_embedding_dim),
        )
        self.no_static_embedding = nn.Parameter(torch.zeros(self.structure_embedding_dim))
        self.structure_gate = nn.Linear(self.structure_embedding_dim * 2, self.structure_embedding_dim)

        local_input_dim = self.input_dim + self.structure_embedding_dim if self.local_branch_use_structure_input else self.input_dim
        self.local_input_projection = nn.Linear(local_input_dim, self.hidden_channels)
        self.local_encoder = nn.LSTM(
            input_size=self.hidden_channels,
            hidden_size=self.hidden_channels,
            num_layers=self.local_lstm_layers,
            batch_first=True,
            dropout=dropout if self.local_lstm_layers > 1 else 0.0,
        )
        if self.local_branch_use_structure_input:
            self.init_h = nn.Linear(self.structure_embedding_dim, self.local_lstm_layers * self.hidden_channels)
            self.init_c = nn.Linear(self.structure_embedding_dim, self.local_lstm_layers * self.hidden_channels)
        else:
            self.init_h = None
            self.init_c = None

        merged_dim = self.hidden_channels + self.structure_embedding_dim if self.local_branch_use_structure_input else self.hidden_channels
        # local_head 改为逐时间步预测: 对最后 output_time 个LSTM隐藏状态各自做预测
        # 而非从一个向量直接映射到所有 output_time 步
        self.local_head = nn.Sequential(
            nn.Linear(merged_dim, self.hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_channels, self.output_size),  # 每步输出 output_size
        )

        driver_context_dim = self.input_dim * 2
        self.driver_context_proj = nn.Sequential(
            nn.Linear(driver_context_dim, self.hidden_channels),
            nn.ReLU(),
        )
        self.local_prior_proj = nn.Sequential(
            nn.Linear(self.output_time * self.output_size, self.hidden_channels),
            nn.ReLU(),
        )

        self.graph_seed_proj = nn.Linear(merged_dim + self.hidden_channels * 2, self.hidden_channels)
        self.graph_structure_gate = nn.Sequential(
            nn.Linear(self.structure_embedding_dim + self.hidden_channels * 2, self.hidden_channels),
            nn.Sigmoid(),
        )
        self.graph_layers = nn.ModuleList(
            GCNConv(self.hidden_channels, self.hidden_channels, add_self_loops=False, normalize=True)
            for _ in range(self.graph_layers_count)
        )
        self.graph_norms = nn.ModuleList(nn.LayerNorm(self.hidden_channels) for _ in range(self.graph_layers_count))
        self.graph_dropout = nn.Dropout(dropout)
        self.graph_context_proj = nn.Sequential(
            nn.Linear(self.hidden_channels + self.structure_embedding_dim, self.hidden_channels),
            nn.ReLU(),
        )
        propagation_input_dim = self.hidden_channels * 4 + self.structure_embedding_dim
        self.propagation_head = nn.Sequential(
            nn.Linear(propagation_input_dim, self.hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_channels, self.output_time * self.output_size),
        )
        self.propagation_gate = nn.Linear(
            propagation_input_dim,
            self.output_time * self.output_size,
        )
        nn.init.constant_(self.propagation_gate.bias, float(route_gate_bias))
        self.graph_prediction_head = nn.Sequential(
            nn.Linear(propagation_input_dim, self.hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_channels, self.output_time * self.output_size),
        )
        self.graph_blend_gate = nn.Linear(
            propagation_input_dim,
            self.output_time * self.output_size,
        )
        nn.init.constant_(self.graph_blend_gate.bias, float(graph_blend_bias))

        self.last_structure_embedding: Optional[torch.Tensor] = None
        self.last_structure_topology_embedding: Optional[torch.Tensor] = None
        self.last_structure_attribute_embedding: Optional[torch.Tensor] = None
        self.last_structure_consistency: Optional[torch.Tensor] = None
        self.last_base_prediction: Optional[torch.Tensor] = None
        self.last_local_prediction: Optional[torch.Tensor] = None
        self.last_graph_prediction: Optional[torch.Tensor] = None
        self.last_graph_blend: Optional[torch.Tensor] = None
        self.last_route_correction: Optional[torch.Tensor] = None
        self.last_graph_state: Optional[torch.Tensor] = None
        self._latest_local_prediction: Optional[torch.Tensor] = None
        self._topology_feature_cache: dict[str, torch.Tensor] = {}
        self._topology_feature_cache_order: deque[str] = deque()
        self._topology_feature_cache_max_entries = 8

    def _tensor_digest(self, tensor: Optional[torch.Tensor]) -> bytes:
        if tensor is None or tensor.numel() == 0:
            return b""
        tensor_cpu = tensor.detach().to(device="cpu").contiguous()
        return tensor_cpu.numpy().tobytes()

    def _make_topology_cache_key(
        self,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor],
        batch_vector: Optional[torch.Tensor],
        root_local_idx: Optional[torch.Tensor],
        num_nodes: int,
        dtype: torch.dtype,
        device: torch.device,
        normalize: bool,
    ) -> str:
        hasher = hashlib.sha1()
        hasher.update(str(num_nodes).encode("utf-8"))
        hasher.update(str(dtype).encode("utf-8"))
        hasher.update(str(device).encode("utf-8"))
        hasher.update(str(normalize).encode("utf-8"))
        hasher.update(self._tensor_digest(edge_index))
        hasher.update(self._tensor_digest(edge_weight))
        hasher.update(self._tensor_digest(batch_vector))
        hasher.update(self._tensor_digest(root_local_idx))
        return hasher.hexdigest()

    def _get_cached_topology_features(self, cache_key: str) -> Optional[torch.Tensor]:
        cached = self._topology_feature_cache.get(cache_key)
        if cached is None:
            return None
        try:
            self._topology_feature_cache_order.remove(cache_key)
        except ValueError:
            pass
        self._topology_feature_cache_order.append(cache_key)
        return cached

    def _store_cached_topology_features(self, cache_key: str, features: torch.Tensor) -> None:
        self._topology_feature_cache[cache_key] = features.detach()
        try:
            self._topology_feature_cache_order.remove(cache_key)
        except ValueError:
            pass
        self._topology_feature_cache_order.append(cache_key)
        while len(self._topology_feature_cache_order) > self._topology_feature_cache_max_entries:
            oldest_key = self._topology_feature_cache_order.popleft()
            self._topology_feature_cache.pop(oldest_key, None)

    def _build_transition(self, num_nodes: int, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        transition = torch.zeros((num_nodes, num_nodes), dtype=dtype, device=device)
        if edge_index.numel() == 0:
            return transition
        src = edge_index[0].long()
        dst = edge_index[1].long()
        if edge_weight is None or edge_weight.numel() != src.numel():
            weights = torch.ones((src.numel(),), dtype=dtype, device=device)
        else:
            weights = edge_weight.to(device=device, dtype=dtype).clamp_min(0.0)
        transition[dst, src] = weights
        row_sum = transition.sum(dim=1, keepdim=True)
        transition = torch.where(row_sum > 0, transition / row_sum.clamp_min(1e-6), transition)
        return transition

    def _build_route_kernel(self, transition: torch.Tensor) -> torch.Tensor:
        kernel = transition.clone()
        if self.route_mode == "direct" or self.route_steps <= 1:
            return kernel
        running = transition.clone()
        for step in range(2, self.route_steps + 1):
            running = running @ transition
            kernel = kernel + (self.route_decay ** (step - 1)) * running
        row_sum = kernel.sum(dim=1, keepdim=True)
        kernel = torch.where(row_sum > 0, kernel / row_sum.clamp_min(1e-6), kernel)
        return kernel

    def _compute_topology_features(
        self,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor],
        batch_vector: Optional[torch.Tensor],
        root_local_idx: Optional[torch.Tensor],
        num_nodes: int,
        dtype: torch.dtype,
        device: torch.device,
        normalize: bool = True,
    ) -> torch.Tensor:
        cache_key = self._make_topology_cache_key(
            edge_index=edge_index,
            edge_weight=edge_weight,
            batch_vector=batch_vector,
            root_local_idx=root_local_idx,
            num_nodes=num_nodes,
            dtype=dtype,
            device=device,
            normalize=normalize,
        )
        cached_features = self._get_cached_topology_features(cache_key)
        if cached_features is not None:
            return cached_features

        ptr = _build_ptr(batch_vector, num_nodes, device)
        transition = self._build_transition(num_nodes, edge_index, edge_weight, dtype, device)
        kernel = self._build_route_kernel(transition)

        in_degree = (transition > 0).sum(dim=1).to(dtype=dtype)
        out_degree = (transition > 0).sum(dim=0).to(dtype=dtype)
        upstream_span = (kernel > 0).sum(dim=1).to(dtype=dtype)
        downstream_span = (kernel.transpose(0, 1) > 0).sum(dim=1).to(dtype=dtype)

        roots = _infer_root_indices(edge_index, ptr, root_local_idx)
        distance_to_outlet = _graph_distances_to_outlet(edge_index, ptr, roots, device=device).to(dtype=dtype)
        upstream_depth = _graph_upstream_depth(edge_index, ptr, device=device).to(dtype=dtype)

        features = torch.stack(
            [
                in_degree,
                out_degree,
                upstream_span,
                downstream_span,
                torch.clamp_min(distance_to_outlet, 0.0),
                upstream_depth,
            ],
            dim=-1,
        )

        if normalize:
            for graph_idx in range(ptr.numel() - 1):
                start = int(ptr[graph_idx].item())
                end = int(ptr[graph_idx + 1].item())
                if end <= start:
                    continue
                graph_slice = features[start:end]
                denom = graph_slice.abs().amax(dim=0, keepdim=True).clamp_min(1.0)
                features[start:end] = graph_slice / denom

        features = torch.nan_to_num(features)
        self._store_cached_topology_features(cache_key, features)
        return features

    def _fuse_structure_embeddings(self, topo_embedding: torch.Tensor, attr_embedding: torch.Tensor) -> torch.Tensor:
        if self.structure_fusion == "mean":
            return 0.5 * (topo_embedding + attr_embedding)
        gate = torch.sigmoid(self.structure_gate(torch.cat([topo_embedding, attr_embedding], dim=-1)))
        return gate * topo_embedding + (1.0 - gate) * attr_embedding

    def _encode_topology(
        self,
        topo_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.topology_encoder_type == "mlp":
            return self.topology_encoder(topo_features)

        topology_state = torch.relu(self.topology_graph_input_projection(topo_features))
        for layer, norm in zip(self.topology_graph_layers, self.topology_graph_norms):
            propagated = layer(topology_state, edge_index, edge_weight=edge_weight)
            propagated = torch.relu(propagated)
            propagated = self.topology_graph_dropout(propagated)
            topology_state = norm(topology_state + propagated)
        return self.topology_graph_output_projection(topology_state)

    def _select_structure_embedding(self, topo_embedding: torch.Tensor, attr_embedding: torch.Tensor) -> torch.Tensor:
        if self.structure_ablation_mode == "topology_only":
            return topo_embedding
        if self.structure_ablation_mode == "attribute_only":
            return attr_embedding
        return self._fuse_structure_embeddings(topo_embedding, attr_embedding)

    def _reshape_output(self, prediction: torch.Tensor) -> torch.Tensor:
        if self.output_time == 1:
            return prediction[:, 0, :]
        return prediction

    def _build_local_input(self, x: torch.Tensor, structure_embedding: torch.Tensor) -> torch.Tensor:
        if self.local_branch_use_structure_input:
            structure_seq = structure_embedding.unsqueeze(1).expand(-1, x.size(1), -1)
            local_input = torch.cat([x, structure_seq], dim=-1)
        else:
            local_input = x
        return torch.relu(self.local_input_projection(local_input))

    def _build_local_initial_state(
        self,
        structure_embedding: torch.Tensor,
        *,
        num_nodes: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.local_branch_use_structure_input:
            init_h = self.init_h(structure_embedding).view(
                self.local_lstm_layers, num_nodes, self.hidden_channels
            ).contiguous()
            init_c = self.init_c(structure_embedding).view(
                self.local_lstm_layers, num_nodes, self.hidden_channels
            ).contiguous()
            return init_h, init_c

        init_h = torch.zeros(
            self.local_lstm_layers,
            num_nodes,
            self.hidden_channels,
            dtype=dtype,
            device=device,
        )
        init_c = torch.zeros_like(init_h)
        return init_h, init_c

    def _build_local_head_input(self, local_hidden: torch.Tensor, structure_embedding: torch.Tensor) -> torch.Tensor:
        if self.local_branch_use_structure_input:
            return torch.cat([local_hidden, structure_embedding], dim=-1)
        return local_hidden

    def auxiliary_loss_components(self) -> list[dict[str, Any]]:
        if self.local_aux_loss_weight <= 0.0 or self._latest_local_prediction is None:
            return []
        return [
            {
                "name": "local_prediction",
                "pred": self._latest_local_prediction,
                "weight": self.local_aux_loss_weight,
                "loss_name": self.local_aux_loss_name,
            }
        ]

    def forward(self, batch: Data | Any) -> torch.Tensor:
        if not hasattr(batch, "x") or not hasattr(batch, "edge_index"):
            raise ValueError("WDNEInspiredFloodNet expects a torch_geometric Data batch")

        x = batch.x
        if x.dim() != 3:
            raise ValueError(f"Expected batch.x to be [N, T, F], got {tuple(x.shape)}")

        if not self.temporal_use_static_input and x.size(-1) > self.input_dim:
            x = x[:, :, : self.input_dim]
        if x.size(-1) != self.input_dim:
            raise ValueError(
                f"Temporal input feature size mismatch: expected {self.input_dim}, got {x.size(-1)}"
            )

        device = x.device
        dtype = x.dtype
        num_nodes, seq_len, _ = x.shape
        edge_index = batch.edge_index.to(device=device)
        edge_weight = getattr(batch, "edge_weight", None)
        if edge_weight is not None:
            edge_weight = edge_weight.to(device=device, dtype=dtype)
        batch_vector = getattr(batch, "batch", None)
        root_local_idx = getattr(batch, "root_local_idx", None)
        if root_local_idx is not None:
            root_local_idx = root_local_idx.view(-1).to(device=device, dtype=torch.long)

        topo_features = self._compute_topology_features(
            edge_index=edge_index,
            edge_weight=edge_weight,
            batch_vector=batch_vector,
            root_local_idx=root_local_idx,
            num_nodes=num_nodes,
            dtype=dtype,
            device=device,
        )
        topo_embedding = self._encode_topology(topo_features, edge_index, edge_weight)

        static_attr = getattr(batch, "static_attr", None)
        if static_attr is not None and static_attr.numel() > 0 and static_attr.size(-1) > 0:
            static_attr = static_attr.to(device=device, dtype=dtype)
            attr_embedding = self.attribute_encoder(static_attr)
        else:
            attr_embedding = self.no_static_embedding.unsqueeze(0).expand(num_nodes, -1)

        structure_embedding = self._select_structure_embedding(topo_embedding, attr_embedding)
        if not self.use_structure_embedding:
            structure_embedding = torch.zeros_like(structure_embedding)
        structure_consistency = torch.mean((topo_embedding - attr_embedding) ** 2)

        local_input = self._build_local_input(x, structure_embedding)
        init_h, init_c = self._build_local_initial_state(
            structure_embedding,
            num_nodes=num_nodes,
            dtype=dtype,
            device=device,
        )
        local_seq, _ = self.local_encoder(local_input, (init_h, init_c))
        # 取最后 output_time 步做逐时间步预测（而非只取最后1步）
        local_hidden_steps = local_seq[:, -self.output_time:, :]  # [N, output_time, hidden]
        local_hidden = local_seq[:, -1, :]  # 最后一步用于图传播的种子

        # 逐时间步预测: 对每个时刻的 hidden state 独立预测
        N_steps = local_hidden_steps.size(0)
        T_out = local_hidden_steps.size(1)
        # 展开为 [N*T, hidden]，每步独立过 local_head
        merged_per_step = self._build_local_head_input(
            local_hidden_steps.reshape(N_steps * T_out, -1),
            structure_embedding.unsqueeze(1).expand(-1, T_out, -1).reshape(N_steps * T_out, -1)
        )
        base_prediction = self.local_head(merged_per_step).view(
            num_nodes, self.output_time, self.output_size
        )
        self._latest_local_prediction = base_prediction

        # 用最后一步的 hidden 作为图传播种子（结构信息 + 时序摘要）
        merged_hidden = self._build_local_head_input(local_hidden, structure_embedding)

        driver_summary = torch.cat([x.mean(dim=1), x[:, -1, :]], dim=-1)
        driver_context = self.driver_context_proj(driver_summary)
        local_prior_context = self.local_prior_proj(base_prediction.reshape(num_nodes, -1))

        graph_seed = self.graph_seed_proj(torch.cat([merged_hidden, driver_context, local_prior_context], dim=-1))
        graph_gate = self.graph_structure_gate(torch.cat([structure_embedding, driver_context, local_prior_context], dim=-1))
        graph_state = graph_seed * graph_gate

        for layer_idx, (layer, norm) in enumerate(zip(self.graph_layers, self.graph_norms)):
            propagated = layer(graph_state, edge_index, edge_weight=edge_weight)
            propagated = torch.relu(propagated)
            propagated = self.graph_dropout(propagated)
            graph_state = norm(graph_state + (self.route_decay ** layer_idx) * propagated)

        graph_context = self.graph_context_proj(torch.cat([graph_state, structure_embedding], dim=-1))
        propagation_hidden = torch.cat(
            [local_hidden, graph_context, driver_context, local_prior_context, structure_embedding],
            dim=-1,
        )
        propagation_delta = self.propagation_head(propagation_hidden).view(num_nodes, self.output_time, self.output_size)
        propagation_gate = torch.sigmoid(
            self.propagation_gate(propagation_hidden)
        ).view(num_nodes, self.output_time, self.output_size)
        graph_correction = self.route_residual_scale * propagation_gate * propagation_delta
        graph_prediction = self.graph_prediction_head(propagation_hidden).view(
            num_nodes, self.output_time, self.output_size
        )
        graph_blend = None
        if self.graph_output_mode == "main":
            output = graph_prediction
        elif self.graph_output_mode == "blend_residual":
            graph_blend = torch.sigmoid(self.graph_blend_gate(propagation_hidden)).view(
                num_nodes, self.output_time, self.output_size
            )
            output = base_prediction + graph_blend * (graph_prediction - base_prediction) + graph_correction
        else:
            output = base_prediction + graph_correction

        self.last_structure_embedding = structure_embedding.detach()
        self.last_structure_topology_embedding = topo_embedding.detach()
        self.last_structure_attribute_embedding = attr_embedding.detach()
        self.last_structure_consistency = structure_consistency.detach()
        self.last_base_prediction = base_prediction.detach()
        self.last_local_prediction = base_prediction.detach()
        self.last_graph_prediction = graph_prediction.detach()
        self.last_graph_blend = graph_blend.detach() if graph_blend is not None else None
        self.last_route_correction = graph_correction.detach()
        self.last_graph_state = graph_state.detach()

        return self._reshape_output(output)