"""WDNE variant with adaptive downstream propagation."""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import softmax as pyg_softmax

from torchhydro.models.wdne_inspired_flood_net import WDNEInspiredFloodNet


class WDNEAdaptiveRoutingNet(WDNEInspiredFloodNet):
	"""WDNE model that learns event-conditioned downstream routing weights."""

	def __init__(
		self,
		*args: Any,
		adaptive_route_hidden_dim: Optional[int] = None,
		adaptive_route_temperature: float = 1.0,
		adaptive_route_use_base_weight: bool = True,
		adaptive_route_use_edge_gate: bool = True,
		adaptive_route_self_residual: float = 1.0,
		**kwargs: Any,
	) -> None:
		super().__init__(*args, **kwargs)
		self.adaptive_route_hidden_dim = int(adaptive_route_hidden_dim or self.hidden_channels)
		self.adaptive_route_temperature = max(float(adaptive_route_temperature), 1e-3)
		self.adaptive_route_use_base_weight = bool(adaptive_route_use_base_weight)
		self.adaptive_route_use_edge_gate = bool(adaptive_route_use_edge_gate)
		self.adaptive_route_self_residual = float(adaptive_route_self_residual)

		self.graph_layers = nn.ModuleList()

		edge_input_dim = self.hidden_channels * 4 + self.structure_embedding_dim * 2 + 1
		node_input_dim = self.hidden_channels * 4 + self.structure_embedding_dim

		self.adaptive_edge_gate = nn.Sequential(
			nn.Linear(edge_input_dim, self.adaptive_route_hidden_dim),
			nn.ReLU(),
			nn.Linear(self.adaptive_route_hidden_dim, 1),
		)
		self.adaptive_message_proj = nn.Sequential(
			nn.Linear(self.hidden_channels, self.hidden_channels),
			nn.ReLU(),
		)
		self.adaptive_update_proj = nn.Sequential(
			nn.Linear(self.hidden_channels, self.hidden_channels),
			nn.Tanh(),
		)
		self.adaptive_update_gate = nn.Linear(node_input_dim, self.hidden_channels)

		self.last_adaptive_edge_weights: Optional[torch.Tensor] = None
		self.last_edge_logits: Optional[torch.Tensor] = None

	def _destination_softmax(self, logits: torch.Tensor, dst: torch.Tensor, num_nodes: int) -> torch.Tensor:
		softmax_dtype = torch.float32 if logits.dtype in {torch.float16, torch.bfloat16} else logits.dtype
		logits_for_softmax = logits.to(dtype=softmax_dtype)
		if logits.numel() == 0:
			return torch.zeros_like(logits)
		weights = pyg_softmax(logits_for_softmax, dst, num_nodes=num_nodes)
		return weights.to(dtype=logits.dtype)

	def _prepare_edge_context(
		self,
		driver_context: torch.Tensor,
		local_prior_context: torch.Tensor,
		structure_embedding: torch.Tensor,
		edge_index: torch.Tensor,
		edge_weight: Optional[torch.Tensor],
		dtype: torch.dtype,
		device: torch.device,
	) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
		src = edge_index[0].long()
		dst = edge_index[1].long()
		if edge_weight is None or edge_weight.numel() != src.numel():
			base_weight = torch.ones((src.numel(), 1), device=device, dtype=dtype)
		else:
			base_weight = edge_weight.to(device=device, dtype=dtype).view(-1, 1)

		edge_context = torch.cat(
			[
				driver_context.index_select(0, src),
				local_prior_context.index_select(0, dst),
				structure_embedding.index_select(0, src),
				structure_embedding.index_select(0, dst),
				base_weight,
			],
			dim=-1,
		)
		base_log_weight = None
		if self.adaptive_route_use_base_weight:
			base_log_weight = torch.log(base_weight.squeeze(-1).clamp_min(1e-6))
		return src, dst, base_weight, edge_context, base_log_weight

	def _adaptive_route(
		self,
		graph_state: torch.Tensor,
		src: torch.Tensor,
		dst: torch.Tensor,
		base_weight: torch.Tensor,
		edge_context: torch.Tensor,
		base_log_weight: Optional[torch.Tensor],
	) -> tuple[torch.Tensor, torch.Tensor]:
		num_nodes = graph_state.size(0)
		dtype = graph_state.dtype
		device = graph_state.device
		if src.numel() == 0:
			return torch.zeros_like(graph_state), torch.zeros((0,), device=device, dtype=dtype)

		edge_inputs = torch.cat(
			[
				graph_state.index_select(0, src),
				graph_state.index_select(0, dst),
				edge_context,
			],
			dim=-1,
		)
		if self.adaptive_route_use_edge_gate:
			edge_logits = self.adaptive_edge_gate(edge_inputs).squeeze(-1) / self.adaptive_route_temperature
		else:
			edge_logits = torch.zeros(src.size(0), device=device, dtype=dtype)
		if base_log_weight is not None:
			edge_logits = edge_logits + base_log_weight
		self.last_edge_logits = edge_logits.detach()
		edge_weights = self._destination_softmax(edge_logits, dst, num_nodes).to(dtype=dtype)

		messages = self.adaptive_message_proj(graph_state.index_select(0, src)).to(dtype=dtype)
		messages = messages * edge_weights.unsqueeze(-1)
		aggregated = torch.zeros_like(graph_state)
		aggregated.index_add_(0, dst, messages)
		return aggregated, edge_weights

	def forward(self, batch: Data | Any) -> torch.Tensor:
		if not hasattr(batch, "x") or not hasattr(batch, "edge_index"):
			raise ValueError("WDNEAdaptiveRoutingNet expects a torch_geometric Data batch")

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
		structure_consistency = torch.mean((topo_embedding - attr_embedding) ** 2)

		local_input = self._build_local_input(x, structure_embedding)
		init_h, init_c = self._build_local_initial_state(
			structure_embedding,
			num_nodes=num_nodes,
			dtype=dtype,
			device=device,
		)
		local_seq, _ = self.local_encoder(local_input, (init_h, init_c))
		# 取最后 output_time 步做逐时间步预测
		local_hidden_steps = local_seq[:, -self.output_time:, :]
		local_hidden = local_seq[:, -1, :]  # 最后一步用于图传播种子

		N_steps = local_hidden_steps.size(0)
		T_out = local_hidden_steps.size(1)
		merged_per_step = self._build_local_head_input(
			local_hidden_steps.reshape(N_steps * T_out, -1),
			structure_embedding.unsqueeze(1).expand(-1, T_out, -1).reshape(N_steps * T_out, -1)
		)
		base_prediction = self.local_head(merged_per_step).view(
			num_nodes, self.output_time, self.output_size
		)
		self._latest_local_prediction = base_prediction

		merged_hidden = self._build_local_head_input(local_hidden, structure_embedding)

		driver_summary = torch.cat([x.mean(dim=1), x[:, -1, :]], dim=-1)
		driver_context = self.driver_context_proj(driver_summary)
		local_prior_context = self.local_prior_proj(base_prediction.reshape(num_nodes, -1))
		edge_src, edge_dst, base_weight, edge_context, base_log_weight = self._prepare_edge_context(
			driver_context=driver_context,
			local_prior_context=local_prior_context,
			structure_embedding=structure_embedding,
			edge_index=edge_index,
			edge_weight=edge_weight,
			dtype=dtype,
			device=device,
		)

		graph_seed = self.graph_seed_proj(torch.cat([merged_hidden, driver_context, local_prior_context], dim=-1))
		graph_gate = self.graph_structure_gate(torch.cat([structure_embedding, driver_context, local_prior_context], dim=-1))
		graph_state = graph_seed * graph_gate

		latest_edge_weights = None
		for layer_idx, norm in enumerate(self.graph_norms):
			aggregated, latest_edge_weights = self._adaptive_route(
				graph_state=graph_state,
				src=edge_src,
				dst=edge_dst,
				base_weight=base_weight,
				edge_context=edge_context,
				base_log_weight=base_log_weight,
			)
			update_input = torch.cat(
				[graph_state, aggregated, driver_context, local_prior_context, structure_embedding],
				dim=-1,
			)
			update_gate = torch.sigmoid(self.adaptive_update_gate(update_input))
			proposal = self.adaptive_update_proj(aggregated)
			graph_state = norm(
				self.adaptive_route_self_residual * graph_state
				+ (self.route_decay ** layer_idx) * update_gate * proposal
			)

		graph_context = self.graph_context_proj(torch.cat([graph_state, structure_embedding], dim=-1))
		propagation_hidden = torch.cat(
			[local_hidden, graph_context, driver_context, local_prior_context, structure_embedding],
			dim=-1,
		)
		propagation_delta = self.propagation_head(propagation_hidden).view(num_nodes, self.output_time, self.output_size)
		propagation_gate = torch.sigmoid(self.propagation_gate(propagation_hidden)).view(
			num_nodes, self.output_time, self.output_size
		)
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
		self.last_adaptive_edge_weights = None if latest_edge_weights is None else latest_edge_weights.detach()
		return output
