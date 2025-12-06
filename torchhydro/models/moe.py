"""
Author: Wenyu Ouyang
Date: 2025-11-25
LastEditTime: 2025-11-25 19:18:45
LastEditors: Wenyu Ouyang
Description: Mixture of Experts (MoE) model
FilePath: \torchhydro\torchhydro\models\moe.py
Copyright (c) 2021-2025 Wenyu Ouyang. All rights reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Mapping, Type


class MoE(nn.Module):
    """
    A simple Mixture of Experts (MoE) model.
    """

    def __init__(
        self,
        n_experts: int,
        experts_cfgs: list,
        gating_cfgs: dict,
        gating_mode: str = "full",
        **kwargs: dict,
    ) -> None:
        """
        __init__ method for the MoE model.

        Parameters
        ----------
        n_experts
            Number of expert models.
        experts_cfgs
            A list of configurations for each expert model.
            Each configuration is a dictionary with 'model_name' and
            'model_hyperparam'.
        gating_cfgs
            Configuration for the gating network. It is a dictionary with
            'model_name' and 'model_hyperparam'.
        gating_mode
            'full' or 'last'. If 'full', the full input is passed to the gating
            network. If 'last', only the last time step is used (for sequence
            data).
        """
        super(MoE, self).__init__()
        # to avoid circular import, import model_dict_function here
        from torchhydro.models.model_dict_function import pytorch_model_dict

        assert len(experts_cfgs) == n_experts
        self.gating_mode = gating_mode
        self.experts = nn.ModuleList(
            [
                self._create_expert(expert_cfg, pytorch_model_dict)
                for expert_cfg in experts_cfgs
            ]
        )

        gating_model_name = gating_cfgs["model_name"]
        gating_hyperparam = gating_cfgs["model_hyperparam"]
        # The output of gating network is number of experts
        gating_hyperparam["ny"] = n_experts
        gating_cls = pytorch_model_dict[gating_model_name]
        self.gating_network = gating_cls(**gating_hyperparam)

    def _create_expert(
        self,
        expert_cfg: Dict[str, Any],
        pytorch_model_dict: Mapping[str, Type[nn.Module]],
    ) -> nn.Module:
        """
        Helper function to create an expert model from its configuration.
        """
        model_name = expert_cfg["model_name"]
        model_hyperparam = expert_cfg["model_hyperparam"]
        cls = pytorch_model_dict[model_name]
        return cls(**model_hyperparam)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MoE model.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        torch.Tensor
            The output of the MoE model.
        """
        if self.gating_mode == "last":
            # Use the last time step for gating, assuming input is a sequence
            gating_input = x[:, -1, :]
        else:  # 'full'
            gating_input = x

        # Get gating scores
        gating_scores = self.gating_network(gating_input)

        # Get weights for each expert
        gating_weights = F.softmax(gating_scores, dim=-1)

        # Get outputs from all experts
        expert_outputs = []
        for expert in self.experts:
            output = expert(x)
            # Some models like SimpleLSTM return a tuple (output, state)
            if isinstance(output, tuple):
                output = output[0]
            expert_outputs.append(output)

        # Stack expert outputs along a new dimension
        # Assuming all experts have the same output shape
        stacked_expert_outputs = torch.stack(expert_outputs, dim=-1)

        # Weight the expert outputs
        # gating_weights needs to be reshaped to be broadcastable with stacked_expert_outputs
        # From [batch_size, n_experts] to [batch_size, 1, ..., 1, n_experts]
        while len(gating_weights.shape) < len(stacked_expert_outputs.shape):
            gating_weights = gating_weights.unsqueeze(1)

        weighted_outputs = stacked_expert_outputs * gating_weights

        # Sum the weighted outputs
        return torch.sum(weighted_outputs, dim=-1)


class DenseMoE(nn.Module):
    def __init__(
        self,
        n_experts: int,
        experts_cfgs: list,
        gating_cfgs: dict,
        static_features: int,
        **kwargs: dict,
    ) -> None:
        super().__init__()

        assert len(experts_cfgs) == n_experts

        from torchhydro.models.model_dict_function import pytorch_model_dict

        self.n_experts = n_experts
        self.static_features = static_features
        self.experts = nn.ModuleList(
            [
                self._create_expert(expert_cfg, pytorch_model_dict)
                for expert_cfg in experts_cfgs
            ]
        )

        gating_model_name = gating_cfgs["model_name"]
        gating_hyperparam = gating_cfgs["model_hyperparam"]
        # The output of gating network is number of experts
        gating_hyperparam["ny"] = n_experts
        gating_cls = pytorch_model_dict[gating_model_name]
        self.gating_network = gating_cls(**gating_hyperparam)

    def _create_expert(
        self,
        expert_cfg: Dict[str, Any],
        pytorch_model_dict: Mapping[str, Type[nn.Module]],
    ) -> nn.Module:
        """
        Helper function to create an expert model from its configuration.
        """
        model_name = expert_cfg["model_name"]
        model_hyperparam = expert_cfg["model_hyperparam"]
        cls = pytorch_model_dict[model_name]
        return cls(**model_hyperparam)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(1)
        basin_attr = x[0, :, -self.static_features :]  # [batch_size, static_features]
        gating_logits = self.gating_network(basin_attr)
        gating_weights = F.softmax(gating_logits, dim=-1)

        expert_outputs = []
        for expert in self.experts:
            out = expert(x)
            if isinstance(out, tuple):
                out = out[0]
            expert_outputs.append(out)

        concat_expert_outputs = torch.stack(expert_outputs, dim=-1)

        gating_weights = gating_weights.view(1, B, 1, self.n_experts)

        weighted_outputs = concat_expert_outputs * gating_weights

        # Sum the weighted outputs
        return torch.sum(weighted_outputs, dim=-1)


class SparseMoE(nn.Module):
    """
    Sparse Mixture of Experts with top-k routing.

    Each sample is routed to top-k experts based on static basin attributes.
    Includes auxiliary load balancing loss.
    """

    def __init__(
        self,
        n_experts: int,
        experts_cfgs: list,  # Fixed: was expert_cfg (singular)
        gating_cfgs: dict,
        static_features: int,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        load_balance_loss_coef: float = 1e-2,
        jitter_noise: float = 0.0,
        **kwargs: dict,
    ) -> None:
        super().__init__()
        assert len(experts_cfgs) == n_experts

        from torchhydro.models.model_dict_function import pytorch_model_dict

        self.n_experts = n_experts
        self.static_features = static_features
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.load_balance_loss_coef = load_balance_loss_coef
        self.jitter_noise = jitter_noise
        self.aux_loss = torch.tensor(0.0)  # Added: initialize aux_loss

        self.experts = nn.ModuleList(
            [
                self._create_expert(expert_cfg, pytorch_model_dict)
                for expert_cfg in experts_cfgs
            ]
        )

        gating_model_name = gating_cfgs["model_name"]
        gating_hyperparam = gating_cfgs[
            "model_hyperparam"
        ].copy()  # Added: copy to avoid mutation
        gating_hyperparam["ny"] = n_experts
        gating_cls = pytorch_model_dict[gating_model_name]
        self.gating_network = gating_cls(**gating_hyperparam)

    def _create_expert(
        self,
        expert_cfg: Dict[str, Any],
        pytorch_model_dict: Mapping[str, Type[nn.Module]],
    ) -> nn.Module:
        """Helper function to create an expert model from its configuration."""
        model_name = expert_cfg["model_name"]
        model_hyperparam = expert_cfg["model_hyperparam"]
        cls = pytorch_model_dict[model_name]
        return cls(**model_hyperparam)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for SparseMoE.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [sequence, batch, features].

        Returns
        -------
        torch.Tensor
            Output tensor of shape [sequence, batch, output_features].
        """
        S, B, _ = x.shape
        device = x.device
        dtype = x.dtype

        # Extract static basin attributes
        basin_attr = x[0, :, -self.static_features :]  # [B, static_features]
        gating_logits = self.gating_network(basin_attr)  # [B, n_experts]

        # Add jitter noise during training for exploration
        if self.training and self.jitter_noise > 0:
            noise = torch.randn_like(gating_logits) * self.jitter_noise
            gating_logits = gating_logits + noise

        # Top-k routing
        top_k_logits, top_k_idx = torch.topk(
            gating_logits, self.top_k, dim=-1
        )  # [B, k]
        top_k_gates = F.softmax(top_k_logits, dim=-1)  # [B, k]  [8,2]

        # Calculate capacity
        capacity = int(B * self.top_k / self.n_experts * self.capacity_factor)
        capacity = max(1, capacity)

        # Flatten for assignment
        expert_indices = top_k_idx.reshape(-1)  # [B*k]  [16]
        gates = top_k_gates.reshape(-1)  # [B*k]

        # Sort by gate values (priority dispatch)
        token_indices = torch.arange(B, device=device).repeat_interleave(self.top_k)
        _, perm = gates.sort(descending=True)
        token_indices = token_indices[perm]
        expert_indices = expert_indices[perm]
        gates = gates[perm]

        # Assign tokens to experts with capacity constraint
        expert_load = torch.zeros(self.n_experts, device=device, dtype=torch.int)
        assigned_tokens = [[] for _ in range(self.n_experts)]
        assigned_gates = [[] for _ in range(self.n_experts)]

        for t, e, g in zip(token_indices, expert_indices, gates):
            if expert_load[e] < capacity:
                assigned_tokens[e].append(t)
                assigned_gates[e].append(g)
                expert_load[e] += 1

        # Compute auxiliary load balancing loss
        router_probs = F.softmax(gating_logits, dim=-1)  # [B, n_experts]
        prob_mean = router_probs.mean(dim=0)  # [n_experts]

        total_dispatched = sum(len(assigned_tokens[i]) for i in range(self.n_experts))
        if total_dispatched > 0:
            # Fixed: renamed from F to frac to avoid shadowing torch.nn.functional
            frac = torch.tensor(
                [
                    len(assigned_tokens[i]) / total_dispatched
                    for i in range(self.n_experts)
                ],
                device=device,
                dtype=dtype,
            )
        else:
            frac = torch.zeros(self.n_experts, device=device, dtype=dtype)

        self.aux_loss = (
            self.load_balance_loss_coef * self.n_experts * torch.sum(prob_mean * frac)
        )

        # Get output shape
        with torch.no_grad():
            example_out = self.experts[0](x[:, :1, :])
            if isinstance(example_out, tuple):
                example_out = example_out[0]
        S_out, _, D_out = example_out.shape

        # Initialize output
        output = torch.zeros((S_out, B, D_out), device=device, dtype=dtype)

        # Route and compute expert outputs
        for i in range(self.n_experts):
            if len(assigned_tokens[i]) == 0:
                continue

            token_ids = torch.tensor(
                assigned_tokens[i], device=device, dtype=torch.long
            )
            gates_e = torch.stack(assigned_gates[i])  # [n_tokens]
            expert_input = x[:, token_ids, :]

            out = self.experts[i](expert_input)
            if isinstance(out, tuple):
                out = out[0]

            # Vectorized accumulation (optimized from per-element loop)
            # gates_e: [n_tokens] -> [1, n_tokens, 1] for broadcasting
            weighted_out = out * gates_e.view(1, -1, 1)
            output[:, token_ids, :] += weighted_out

        return output

    def get_aux_loss(self) -> torch.Tensor:
        """Return the auxiliary load balancing loss."""
        return self.aux_loss


class SwitchMoE(nn.Module):
    """
    Switch Transformer style MoE with top-1 routing.

    Each sample is routed to exactly one expert based on static basin attributes.
    Includes auxiliary load balancing loss to encourage balanced expert utilization.

    Reference: https://arxiv.org/abs/2101.03961
    """

    def __init__(
        self,
        n_experts: int,
        experts_cfgs: list,
        gating_cfgs: dict,
        static_features: int,
        capacity_factor: float = 1.25,
        load_balance_loss_coef: float = 1e-2,
        jitter_noise: float = 0.0,
        **kwargs: dict,
    ) -> None:
        """
        Parameters
        ----------
        n_experts : int
            Number of expert models.
        experts_cfgs : list
            List of configurations for each expert model.
        gating_cfgs : dict
            Configuration for the gating network.
        static_features : int
            Number of static features at the end of input for gating.
        capacity_factor : float
            Multiplier for expert capacity (how many samples an expert can handle).
        load_balance_loss_coef : float
            Coefficient for the auxiliary load balancing loss.
        jitter_noise : float
            Amount of noise to add to gating logits during training for exploration.
        """
        super().__init__()
        assert len(experts_cfgs) == n_experts

        from torchhydro.models.model_dict_function import pytorch_model_dict

        self.n_experts = n_experts
        self.static_features = static_features
        self.capacity_factor = capacity_factor
        self.load_balance_loss_coef = load_balance_loss_coef
        self.jitter_noise = jitter_noise
        self.aux_loss = torch.tensor(0.0)

        # Create experts
        self.experts = nn.ModuleList(
            [
                self._create_expert(expert_cfg, pytorch_model_dict)
                for expert_cfg in experts_cfgs
            ]
        )

        # Create gating network
        gating_model_name = gating_cfgs["model_name"]
        gating_hyperparam = gating_cfgs["model_hyperparam"].copy()
        gating_hyperparam["ny"] = n_experts
        gating_cls = pytorch_model_dict[gating_model_name]
        self.gating_network = gating_cls(**gating_hyperparam)

    def _create_expert(
        self,
        expert_cfg: Dict[str, Any],
        pytorch_model_dict: Mapping[str, Type[nn.Module]],
    ) -> nn.Module:
        """Helper function to create an expert model from its configuration."""
        model_name = expert_cfg["model_name"]
        model_hyperparam = expert_cfg["model_hyperparam"]
        cls = pytorch_model_dict[model_name]
        return cls(**model_hyperparam)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for SwitchMoE.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [sequence, batch, features].

        Returns
        -------
        torch.Tensor
            Output tensor of shape [sequence, batch, output_features].
        """
        S, B, _ = x.shape
        device = x.device
        dtype = x.dtype

        # Extract static basin attributes from the first time step
        # Shape: [batch_size, static_features]
        basin_attr = x[0, :, -self.static_features :]

        # Get gating logits
        gating_logits = self.gating_network(basin_attr)  # [B, n_experts]

        # Add jitter noise during training for exploration
        if self.training and self.jitter_noise > 0:
            noise = torch.randn_like(gating_logits) * self.jitter_noise
            gating_logits = gating_logits + noise

        # Compute router probabilities (for auxiliary loss)
        router_probs = F.softmax(gating_logits, dim=-1)  # [B, n_experts]

        # Top-1 routing: each sample goes to exactly one expert
        top1_logits, top1_indices = torch.topk(gating_logits, k=1, dim=-1)
        top1_gates = F.softmax(top1_logits, dim=-1).squeeze(-1)  # [B]
        expert_indices = top1_indices.squeeze(-1)  # [B]

        # Calculate capacity per expert
        capacity = int(B / self.n_experts * self.capacity_factor)
        capacity = max(1, capacity)

        # Compute auxiliary load balancing loss
        # P_i = mean probability of routing to expert i
        P = router_probs.mean(dim=0)  # [n_experts]

        # f_i = fraction of samples routed to expert i
        expert_counts = torch.zeros(self.n_experts, device=device, dtype=dtype)
        for i in range(self.n_experts):
            expert_counts[i] = (expert_indices == i).sum().float()
        f = expert_counts / B  # [n_experts]

        # Auxiliary loss: encourages uniform routing
        self.aux_loss = self.load_balance_loss_coef * self.n_experts * torch.sum(P * f)

        # Get output shape by running one sample through the first expert
        with torch.no_grad():
            example_out = self.experts[0](x[:, :1, :])
            if isinstance(example_out, tuple):
                example_out = example_out[0]
        S_out, _, D_out = example_out.shape

        # Initialize output tensor
        output = torch.zeros((S_out, B, D_out), device=device, dtype=dtype)

        # Route samples to experts
        for i in range(self.n_experts):
            # Find samples assigned to this expert
            mask = expert_indices == i
            if not mask.any():
                continue

            sample_indices = mask.nonzero(as_tuple=True)[0]

            # Apply capacity constraint
            if len(sample_indices) > capacity:
                # Keep only top-capacity samples by their gate values
                sample_gates = top1_gates[sample_indices]
                _, keep_indices = torch.topk(sample_gates, k=capacity)
                sample_indices = sample_indices[keep_indices]

            # Get input for this expert
            expert_input = x[:, sample_indices, :]  # [S, n_samples, features]

            # Run expert
            expert_out = self.experts[i](expert_input)
            if isinstance(expert_out, tuple):
                expert_out = expert_out[0]

            # Get gate values for these samples
            gates = top1_gates[sample_indices]  # [n_samples]

            # Weight output by gate values and accumulate
            # gates: [n_samples] -> [1, n_samples, 1] for broadcasting
            weighted_out = expert_out * gates.view(1, -1, 1)

            # Scatter back to output
            output[:, sample_indices, :] = weighted_out

        return output

    def get_aux_loss(self) -> torch.Tensor:
        """Return the auxiliary load balancing loss."""
        return self.aux_loss


class HydrologyTopKMoE(nn.Module):
    """
    Hydrology-friendly Top-k Sparse MoE for basin-level forecasting.

    设计要点：
    - x: [S, B, Din] （sequence-first）
    - basin_attr: [B, D_attr] 作为 gating 输入
    - 每个 basin 至少分配 1 个 expert，最多 top_k 个 expert
    - 若因 capacity 导致某些额外 expert 分配失败，会重新对“实际成功的 experts”
      的 logits 做 softmax 归一化（per-token）
    - 不 drop 流域，只可能少于 top_k 个 expert
    """

    def __init__(
        self,
        n_experts: int,
        experts_cfgs: list,
        gating_cfgs: dict,
        static_features: int = 17,
        top_k: int = 1,
        capacity_factor: float = 1.25,
        load_balance_loss_coef: float = 1e-2,
    ):
        super().__init__()

        from torchhydro.models.model_dict_function import pytorch_model_dict

        assert top_k >= 1, "top_k must be >= 1"

        self.n_experts = n_experts
        self.static_features = static_features
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.load_balance_loss_coef = load_balance_loss_coef

        # ----------- Build experts -----------
        self.experts = nn.ModuleList(
            self._create_expert(cfg, pytorch_model_dict) for cfg in experts_cfgs
        )

        # ----------- Build gating network -----------
        gating_hyper = dict(gating_cfgs["model_hyperparam"])
        gating_hyper["ny"] = n_experts
        gating_cls = pytorch_model_dict[gating_cfgs["model_name"]]
        self.gating_network = gating_cls(**gating_hyper)

        # aux loss 缓存
        self.aux_loss = torch.tensor(0.0)

    def _create_expert(self, cfg, model_dict):
        cls = model_dict[cfg["model_name"]]
        return cls(**cfg["model_hyperparam"])

    # ===============================================================
    # Forward
    # ===============================================================
    def forward(self, x: torch.Tensor):
        """
        x: [S, B, Din]
        basin_attr: [B, D_attr]
        return: [S, B, Dout]
        """
        S, B, _ = x.shape
        device = x.device
        basin_attr = x[0, :, -self.static_features :]
        # -------- 1. Compute logits --------
        # 对每个 basin 计算 expert logits
        logits = self.gating_network(basin_attr)  # [B, n_experts]

        # 对每个 basin 得到 experts 的排序（从大到小）
        sorted_logits, sorted_idx = torch.sort(
            logits, dim=-1, descending=True
        )  # [B, E]

        # -------- 2. 计算 per-expert capacity（全局上限） --------
        # 理论期望：总分配约 = B * top_k
        # 平均每个 expert 分配约 = B * top_k / n_experts
        # 乘一个 capacity_factor，给一点冗余空间
        base = (B * self.top_k) / float(self.n_experts)
        capacity = int(base * self.capacity_factor)
        capacity = max(1, capacity)

        # -------- 3. 为每个 basin 分配 experts（token-first） --------
        # assigned_experts[b] = [e1, e2, ...] 每个是 expert index
        assigned_experts: List[List[int]] = [[] for _ in range(B)]
        # 每个 expert 当前已经分配了多少个 basins（可重复计数：同一个 basin 多个 expert）
        expert_load = torch.zeros(self.n_experts, device=device, dtype=torch.int)

        # 3.1 第一次 pass：保证每个 basin 至少 1 个 expert
        for b in range(B):
            # 遍历该 basin 的所有 experts 排序，从高到低
            assigned = False
            for rank in range(self.n_experts):
                e = int(sorted_idx[b, rank].item())
                if expert_load[e] < capacity:
                    assigned_experts[b].append(e)
                    expert_load[e] += 1
                    assigned = True
                    break

            # 理论上只要 capacity_factor 不极端小，都会成功 assigned
            # 如果实在所有 expert 都满了，就强行忽略 capacity，避免 basin 被 drop
            if not assigned:
                e = int(sorted_idx[b, 0].item())
                assigned_experts[b].append(e)
                # 不增加 expert_load，避免把 load 统计搞乱

        # 3.2 后续 pass：在不违反 capacity 的前提下再给每个 basin 分配额外 expert
        if self.top_k > 1:
            # 对每一层“额外 expert”（第 2 个、第 3 个...）
            for extra_idx in range(1, self.top_k):
                for b in range(B):
                    # 已经分配的 expert 数 >= extra_idx+1，说明这个 basin 已经有足够 expert 了
                    if len(assigned_experts[b]) >= extra_idx + 1:
                        continue

                    # 找该 basin 下一个最优的、尚未分配过的 expert
                    for rank in range(self.n_experts):
                        e = int(sorted_idx[b, rank].item())
                        if e in assigned_experts[b]:
                            continue
                        if expert_load[e] < capacity:
                            assigned_experts[b].append(e)
                            expert_load[e] += 1
                            break
                    # 如果没找到合适的 expert，就保持现状（少于 top_k 也可以）

        # -------- 4. 基于实际“成功分配”的 experts 重新计算 per-token gate --------
        # gates_per_token[b] = [g_b,e1, g_b,e2, ...] 与 assigned_experts[b] 对齐
        gates_per_token: List[List[torch.Tensor]] = [[] for _ in range(B)]

        for b in range(B):
            experts_b = assigned_experts[b]
            # 取对应 logits
            e_ids = torch.tensor(experts_b, device=device, dtype=torch.long)
            logits_b = logits[b, e_ids]  # [m_b]
            # 在“实际成功分配的 experts 上”做 softmax 归一化
            gates_b = F.softmax(logits_b, dim=-1)  # [m_b]
            # 拆成 list 存
            gates_per_token[b] = [gates_b[i] for i in range(len(experts_b))]

        # -------- 5. 负载均衡损失：P_i + F_i（修改为 hydrology-friendly） --------
        # P_i: full softmax(logits) 的平均值（每个 expert 分到多少概率质量）
        router_probs = F.softmax(logits, dim=-1)  # [B, E]
        P = router_probs.mean(dim=0)  # [E]

        # F_i: 实际分配到 expert i 的 token（basin）数量占比
        expert_assign_counts = torch.zeros(
            self.n_experts, device=device, dtype=torch.float
        )
        total_assign = 0.0
        for b in range(B):
            for e in assigned_experts[b]:
                expert_assign_counts[e] += 1.0
                total_assign += 1.0

        if total_assign > 0:
            F_vec = expert_assign_counts / total_assign
        else:
            F_vec = torch.zeros_like(P)

        aux_loss = self.n_experts * torch.sum(P * F_vec)
        self.aux_loss = self.load_balance_loss_coef * aux_loss

        # -------- 6. Expert forward（真正 sparse 执行） --------
        # 用一个小 batch 探查输出维度：输入 [S, 1, Din] → 输出 [S, 1, Dout]
        example_out = self.experts[0](x[:, :1, :])
        if isinstance(example_out, tuple):
            example_out = example_out[0]
        _, _, D_out = example_out.shape

        output = torch.zeros((S, B, D_out), device=device, dtype=example_out.dtype)

        # 为每个 expert 收集它要处理的 basin 序号及对应 gate
        tokens_for_expert: List[List[int]] = [[] for _ in range(self.n_experts)]
        gates_for_expert: List[List[torch.Tensor]] = [[] for _ in range(self.n_experts)]

        for b in range(B):
            for e_idx, e in enumerate(assigned_experts[b]):
                tokens_for_expert[e].append(b)
                gates_for_expert[e].append(gates_per_token[b][e_idx])

        # 逐 expert 执行
        for e in range(self.n_experts):
            if len(tokens_for_expert[e]) == 0:
                continue

            token_ids = torch.tensor(
                tokens_for_expert[e], device=device, dtype=torch.long
            )  # [N_e]
            gates_e = torch.stack(gates_for_expert[e])  # [N_e]

            # 聚合输入：从 batch 维（B）gather
            expert_input = x[:, token_ids, :]  # [S, N_e, Din]

            out_e = self.experts[e](expert_input)
            if isinstance(out_e, tuple):
                out_e = out_e[0]  # 兼容 LSTM 返回 (output, state)
            # out_e: [S, N_e, Dout]

            # 向 output scatter：按 gate 权重加权
            # 尽量向量化，避免 Python 循环
            # gates_e: [N_e] → [1, N_e, 1] 方便 broadcast
            gate_scale = gates_e.view(1, -1, 1)  # [1, N_e, 1]
            # out_e * gate_scale: [S, N_e, Dout]
            output[:, token_ids, :] += out_e * gate_scale

        return output


class GlobalTopKMoE(nn.Module):
    """
    Global-priority, capacity-constrained Top-k MoE with soft fallback (no token drop).

    - 输入 x: [S, B, Din]  (sequence-first)
    - gating 输入: 静态流域属性（输入最后 static_features 维）
    - 路由策略：
        1. 对所有 token×expert 的概率全局排序（大到小）
        2. 按顺序尝试分配 (token, expert)，受 expert capacity 和 token top_k 约束
        3. 分配完成后，若某个 token 仍然没有 expert：
           → 在该 token 的 experts 中，按概率从大到小找第一个“尚未满载”的 expert 作为 fallback
        4. 如果极端情况下所有 expert 都满载：
           → fallback 到该 token 概率最大的 expert，但不再增加其 load（避免破坏 capacity 统计）

    - 每个 token 至少 1 个 expert，最多 top_k 个 expert
    - 对“实际成功分配的 experts”重新 softmax logits，保证 Σ_e g_{b,e} = 1
    - self.aux_loss 存储 Switch 风格的负载均衡损失（P×F）
    """

    def __init__(
        self,
        n_experts: int,
        experts_cfgs: List[Dict[str, Any]],
        gating_cfgs: Dict[str, Any],
        static_features: int,
        top_k: int = 1,
        capacity_factor: float = 1.25,
        load_balance_loss_coef: float = 1e-2,
        jitter_noise: float = 0.0,
        **kwargs: dict,
    ) -> None:
        super().__init__()
        assert len(experts_cfgs) == n_experts
        assert top_k >= 1

        from torchhydro.models.model_dict_function import pytorch_model_dict

        self.n_experts = n_experts
        self.static_features = static_features
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.load_balance_loss_coef = load_balance_loss_coef
        self.jitter_noise = jitter_noise

        # ------------- build experts -------------
        self.experts = nn.ModuleList(
            [
                self._create_expert(expert_cfg, pytorch_model_dict)
                for expert_cfg in experts_cfgs
            ]
        )

        # ------------- build gating network -------------
        gating_model_name = gating_cfgs["model_name"]
        gating_hyperparam = gating_cfgs["model_hyperparam"].copy()
        gating_hyperparam["ny"] = n_experts
        gating_cls = pytorch_model_dict[gating_model_name]
        self.gating_network = gating_cls(**gating_hyperparam)

        self.aux_loss = torch.tensor(0.0)

    def _create_expert(
        self,
        expert_cfg: Dict[str, Any],
        pytorch_model_dict: Mapping[str, Type[nn.Module]],
    ) -> nn.Module:
        model_name = expert_cfg["model_name"]
        model_hyperparam = expert_cfg["model_hyperparam"]
        cls = pytorch_model_dict[model_name]
        return cls(**model_hyperparam)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [S, B, Din]
        return: [S, B, Dout]
        """
        S, B, _ = x.shape
        device = x.device
        dtype = x.dtype
        E = self.n_experts
        top_k = self.top_k

        # -------- 1. 计算 gating logits（静态流域属性作为路由输入） --------
        basin_attr = x[0, :, -self.static_features :]  # [B, static_features]
        logits = self.gating_network(basin_attr)  # [B, E]

        # jitter noise 做探索（可选）
        if self.training and self.jitter_noise > 0.0:
            logits = logits + self.jitter_noise * torch.randn_like(logits)

        # Router 概率（用于负载均衡损失）
        router_probs = F.softmax(logits, dim=-1)  # [B, E]

        # -------- 2. 全局排序的概率（用 probs 排序，更直观） --------
        probs = router_probs  # [B, E]

        # 计算 per-expert capacity
        # 期望总 pair 数 ≈ B * top_k，平均每个 expert ≈ (B * top_k / E)
        expected_pairs = B * top_k
        capacity = int(expected_pairs / float(E) * self.capacity_factor)
        capacity = max(1, capacity)

        # 展平 B×E 为 BE
        flat_probs = probs.reshape(-1)  # [B*E]
        flat_exp_idx = torch.arange(E, device=device).repeat(B)  # [B*E]
        flat_tok_idx = torch.arange(B, device=device).repeat_interleave(E)  # [B*E]

        # 全局按概率降序排序
        sorted_probs, perm = flat_probs.sort(descending=True)
        sorted_tok = flat_tok_idx[perm]  # [B*E]
        sorted_exp = flat_exp_idx[perm]  # [B*E]

        # -------- 3. Global priority 分配 + capacity + per-token top_k --------
        expert_load = torch.zeros(E, device=device, dtype=torch.int32)
        assigned_experts: List[List[int]] = [[] for _ in range(B)]

        # 3.1 全局优先分配：遵守 capacity 和 per-token top_k
        for t, e in zip(sorted_tok.tolist(), sorted_exp.tolist()):
            if len(assigned_experts[t]) >= top_k:
                continue
            if expert_load[e] < capacity:
                assigned_experts[t].append(e)
                expert_load[e] += 1

        # 3.2 Soft fallback：保证每个 token 至少 1 个 expert（不破 capacity）
        for t in range(B):
            if len(assigned_experts[t]) == 0:
                probs_t = probs[t]  # [E]
                sorted_e = torch.argsort(probs_t, descending=True)  # [E]

                fallback = None
                for e in sorted_e.tolist():
                    if expert_load[e] < capacity:
                        fallback = e
                        break

                if fallback is not None:
                    assigned_experts[t].append(fallback)
                    expert_load[fallback] += 1
                else:
                    # 极端情况：所有 expert 都满载，强制 fallback 到概率最大的 expert
                    # 不增加 expert_load，避免破坏 capacity 统计
                    best_e = int(sorted_e[0].item())
                    assigned_experts[t].append(best_e)

        # -------- 4. 基于“实际成功分配的 experts”重新 softmax gate --------
        gates_per_token: List[List[torch.Tensor]] = [[] for _ in range(B)]

        for b in range(B):
            experts_b = assigned_experts[b]  # List[int]
            e_ids = torch.tensor(experts_b, device=device, dtype=torch.long)  # [m_b]
            logits_b = logits[b, e_ids]  # [m_b]
            gates_b = F.softmax(logits_b, dim=-1)  # [m_b]，保证 Σ g = 1

            gates_per_token[b] = [gates_b[i] for i in range(len(experts_b))]

        # -------- 5. 负载均衡损失（Switch 风格：P_i, F_i） --------
        # P_i: 平均 router 概率
        P = router_probs.mean(dim=0)  # [E]

        # F_i: 实际分配到 expert i 的 (token, expert) pair 占比
        expert_assign_counts = torch.zeros(E, device=device, dtype=torch.float32)
        total_assign = 0.0
        for b in range(B):
            for e in assigned_experts[b]:
                expert_assign_counts[e] += 1.0
                total_assign += 1.0

        if total_assign > 0:
            F_vec = expert_assign_counts / total_assign
        else:
            F_vec = torch.zeros_like(P)

        aux_loss = E * torch.sum(P * F_vec)
        self.aux_loss = self.load_balance_loss_coef * aux_loss

        # -------- 6. sparse expert forward（只跑被分配过的 experts） --------
        # 用一个样本探测输出维度
        example_out = self.experts[0](x[:, :1, :])
        if isinstance(example_out, tuple):
            example_out = example_out[0]
        S_out, _, D_out = example_out.shape

        output = torch.zeros((S_out, B, D_out), device=device, dtype=example_out.dtype)

        # 为每个 expert 收集 tokens + gates
        tokens_for_expert: List[List[int]] = [[] for _ in range(E)]
        gates_for_expert: List[List[torch.Tensor]] = [[] for _ in range(E)]

        for b in range(B):
            for j, e in enumerate(assigned_experts[b]):
                tokens_for_expert[e].append(b)
                gates_for_expert[e].append(gates_per_token[b][j])

        # 逐 expert 执行
        for e in range(E):
            if len(tokens_for_expert[e]) == 0:
                continue

            token_ids = torch.tensor(
                tokens_for_expert[e], device=device, dtype=torch.long
            )  # [N_e]
            gates_e = torch.stack(gates_for_expert[e])  # [N_e]

            expert_input = x[:, token_ids, :]  # [S, N_e, Din]

            out_e = self.experts[e](expert_input)
            if isinstance(out_e, tuple):
                out_e = out_e[0]
            # out_e: [S, N_e, Dout]

            gate_scale = gates_e.view(1, -1, 1)  # [1, N_e, 1]
            output[:, token_ids, :] += out_e * gate_scale

        return output

    def get_aux_loss(self) -> torch.Tensor:
        return self.aux_loss
