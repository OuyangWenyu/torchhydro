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
        basin_attr = x[0, :, -self.static_features :]
        gating_logits = self.gating_network(basin_attr)  # [B, n_experts]

        # Top-k routing
        top_k_logits, top_k_idx = torch.topk(gating_logits, self.top_k, dim=-1)
        top_k_gates = F.softmax(top_k_logits, dim=-1)  # [B, k]

        # Calculate capacity
        capacity = int(B * self.top_k / self.n_experts * self.capacity_factor)
        capacity = max(1, capacity)

        # Flatten for assignment
        expert_indices = top_k_idx.reshape(-1)  # [B*k]
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
