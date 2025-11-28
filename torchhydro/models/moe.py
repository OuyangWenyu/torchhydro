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
    A Mixture of Experts (MoE) model supporting both Dense and Sparse routing.
    """

    def __init__(
        self,
        n_experts: int,
        experts_cfgs: list,
        gating_cfgs: dict,
        gating_mode: str = "full",
        top_k: int = None,
        capacity_factor: float = None,
        load_balance_loss_coef: float = 0.0,
        noisy_gating: bool = False,
        expert_devices: list = None,
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
        top_k
            If not None, use Sparse MoE with Top-K gating.
            If None, use Dense MoE (all experts run).
        capacity_factor
            Used for Hard Balancing in Sparse MoE.
            Capacity = tokens_per_batch * capacity_factor / n_experts.
            If None, defaults to 1.0 (if top_k is set).
        load_balance_loss_coef
            Coefficient for the auxiliary load balancing loss (Soft Balancing).
        noisy_gating
            If True, add noise to gating logits (for Sparse MoE).
        expert_devices
            A list of device strings (e.g., ['cuda:0', 'cuda:1']) for each expert.
            If None, all experts are on the same device as the model.
        """
        super(MoE, self).__init__()
        # to avoid circular import, import model_dict_function here
        from torchhydro.models.model_dict_function import pytorch_model_dict

        assert len(experts_cfgs) == n_experts
        self.n_experts = n_experts
        self.gating_mode = gating_mode
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.load_balance_loss_coef = load_balance_loss_coef
        self.noisy_gating = noisy_gating
        self.aux_loss = 0.0

        if expert_devices is not None:
            if len(expert_devices) != n_experts:
                # Round-robin assignment if lengths don't match
                self.expert_devices = [
                    expert_devices[i % len(expert_devices)] for i in range(n_experts)
                ]
            else:
                self.expert_devices = expert_devices
        else:
            self.expert_devices = [None] * n_experts

        self.experts = nn.ModuleList(
            [
                self._create_expert(
                    expert_cfg, pytorch_model_dict, device=self.expert_devices[i]
                )
                for i, expert_cfg in enumerate(experts_cfgs)
            ]
        )

        gating_model_name = gating_cfgs["model_name"]
        gating_hyperparam = gating_cfgs["model_hyperparam"]
        # The output of gating network is number of experts
        gating_hyperparam["ny"] = n_experts
        gating_cls = pytorch_model_dict[gating_model_name]
        self.gating_network = gating_cls(**gating_hyperparam)

        if self.noisy_gating:
            self.w_noise = nn.Parameter(torch.zeros(n_experts, n_experts))
            self.softplus = nn.Softplus()

    def _create_expert(
        self,
        expert_cfg: Dict[str, Any],
        pytorch_model_dict: Mapping[str, Type[nn.Module]],
        device: str = None,
    ) -> nn.Module:
        """
        Helper function to create an expert model from its configuration.
        """
        model_name = expert_cfg["model_name"]
        model_hyperparam = expert_cfg["model_hyperparam"]
        cls = pytorch_model_dict[model_name]
        model = cls(**model_hyperparam)
        if device is not None:
            model.to(device)
        return model

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

        # Get gating logits
        logits = self.gating_network(gating_input)

        if self.noisy_gating and self.training:
            # Add noise to logits (simplified noisy gating)
            # Standard noisy gating usually involves a separate trainable weight matrix for noise
            # Here we just add random noise for simplicity if w_noise is not fully implemented
            # or use the standard mechanism: logits + standard_normal * softplus(w_noise @ x)
            # For now, let's keep it simple or assume gating_network handles it if configured.
            # But since we added w_noise in init, let's use it if possible, or just standard noise.
            noise = torch.randn_like(logits)
            logits = logits + noise

        # Soft Balancing: Load Balancing Loss
        # We calculate this regardless of Dense/Sparse if coef > 0
        if self.load_balance_loss_coef > 0:
            probs = F.softmax(logits, dim=-1)
            # importance: sum of probs over batch
            importance = probs.sum(0)
            # load: count of tokens assigned to each expert (approximate with probs for soft)
            # For sparse, load is actual count. For dense/soft, we can use probs or hard assignment.
            # Switch Transformer definition:
            # auxiliary_loss = num_experts * sum(importance * load) / (sum(importance) * sum(load))
            # Simplified: mean(importance * load) * num_experts^2
            # Let's use the standard auxiliary loss from Switch Transformer paper
            # loss = alpha * N * \sum_{i=1}^N f_i * P_i
            # where f_i is fraction of tokens dispatched to expert i
            # and P_i is fraction of router probability allocated to expert i

            # P_i
            P = probs.mean(0)

            # f_i
            # f_i calculation is deferred to _forward_sparse and _forward_dense
            pass

        if self.top_k is None:
            return self._forward_dense(x, logits)
        else:
            return self._forward_sparse(x, logits)

    def _forward_dense(self, x: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        gating_weights = F.softmax(logits, dim=-1)

        # Soft Balancing Loss for Dense (penalize variance in expert usage)
        if self.load_balance_loss_coef > 0:
            # P_i = fraction of probability mass
            P = gating_weights.mean(0)
            # We want P to be uniform (1/N)
            # Loss = MSE(P, 1/N) or KL divergence
            target = torch.full_like(P, 1.0 / self.n_experts)
            loss = F.mse_loss(P, target)
            self.aux_loss = (
                self.load_balance_loss_coef * loss * self.n_experts
            )  # scale up
        else:
            self.aux_loss = 0.0

        expert_outputs = []
        for i, expert in enumerate(self.experts):
            # Move input to expert device
            inp = x
            if self.expert_devices[i] is not None:
                inp = inp.to(self.expert_devices[i])

            output = expert(inp)
            # Some models like SimpleLSTM return a tuple (output, state)
            if isinstance(output, tuple):
                output = output[0]

            # Move output back to original device
            if self.expert_devices[i] is not None:
                output = output.to(x.device)

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

    def _forward_sparse(self, x: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        # Top-K Gating
        # logits: [batch_size, n_experts]
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)  # Normalize over top-k

        batch_size = logits.size(0)

        # Hard Balancing: Capacity
        if self.capacity_factor is None:
            capacity = batch_size  # No limit effectively
        else:
            capacity = int(
                self.capacity_factor * batch_size / self.n_experts * self.top_k
            )
            capacity = max(capacity, 1)  # At least 1

        # Assign tokens to experts
        # We need to route x to experts.
        # x shape: [batch_size, ...] or [batch_size, seq_len, ...]
        # If input is sequence but gating is 'last', we still route the whole sequence?
        # Usually MoE routes token-by-token or sequence-by-sequence.
        # Here, x is the input to experts.
        # If experts are RNNs (SimpleLSTM), they expect [batch, seq, feat].
        # So we route the whole sequence as one "token".

        # Store outputs
        # We'll use a zero-filled tensor for output
        # output shape should match expert output shape.
        # We need to run one expert to know the shape? Or assume from config.
        # Let's run one dummy pass or infer.
        # Better: collect inputs for each expert, run them, scatter back.

        expert_inputs = [[] for _ in range(self.n_experts)]
        expert_indices = [[] for _ in range(self.n_experts)]  # Original indices

        # Count load for soft balancing
        load = torch.zeros(self.n_experts, device=x.device)

        # Naive routing loop (can be optimized with scatter/gather)
        # For top_k > 1, a sample goes to multiple experts.
        # We need to handle accumulation.

        # Flatten top_k assignments
        # indices: [batch_size, top_k]

        # Check capacity
        expert_counts = torch.zeros(self.n_experts, dtype=torch.int32, device=x.device)

        final_outputs = None

        for k in range(self.top_k):
            indices_k = top_k_indices[:, k]  # [batch_size]
            weights_k = top_k_weights[:, k]  # [batch_size]

            for b in range(batch_size):
                expert_idx = indices_k[b].item()

                if expert_counts[expert_idx] < capacity:
                    expert_inputs[expert_idx].append(x[b])
                    expert_indices[expert_idx].append((b, weights_k[b]))
                    expert_counts[expert_idx] += 1
                else:
                    # Dropped due to capacity
                    pass

        # Calculate Soft Balancing Loss
        if self.load_balance_loss_coef > 0:
            # f_i = fraction of tokens dispatched
            # total tokens dispatched = batch_size * top_k (if no drop)
            # actually we should count dropped ones too for the "router's intent" or "actual load"?
            # Usually load balancing loss penalizes the router's *intent* (probs),
            # but Switch Transformer uses actual dispatch fraction?
            # "f_i is the fraction of tokens dispatched to expert i"
            # Let's use the counts before dropping? Or after?
            # Usually we want to encourage uniform routing, so we use the router's probability P
            # and the actual dispatch f.

            # P_i: mean probability of expert i across batch
            probs = F.softmax(logits, dim=-1)
            P = probs.mean(0)

            # f_i: fraction of samples assigned to expert i
            # We use the counts *before* capacity dropping to encourage router to spread load
            # But here we only have counts *after* dropping in expert_counts if we don't track separately.
            # Let's track intent.
            intent_counts = torch.zeros(self.n_experts, device=x.device)
            flat_indices = top_k_indices.flatten()
            intent_counts.scatter_add_(
                0, flat_indices, torch.ones_like(flat_indices, dtype=torch.float)
            )
            f = intent_counts / (batch_size * self.top_k)

            aux_loss = self.n_experts * torch.sum(P * f)
            self.aux_loss = self.load_balance_loss_coef * aux_loss
        else:
            self.aux_loss = 0.0

        # Run experts
        results = []
        for i, expert in enumerate(self.experts):
            if len(expert_inputs[i]) > 0:
                # Stack inputs: [batch_assigned, ...]
                inp = torch.stack(expert_inputs[i])

                # Move to expert device
                if self.expert_devices[i] is not None:
                    inp = inp.to(self.expert_devices[i])

                out = expert(inp)
                if isinstance(out, tuple):
                    out = out[0]

                # Move back to original device
                if self.expert_devices[i] is not None:
                    out = out.to(x.device)

                results.append(out)
            else:
                results.append(None)

        # Scatter back
        # Find first non-None result to determine shape
        out_shape = None
        for r in results:
            if r is not None:
                out_shape = r.shape[1:]
                break

        if out_shape is not None:
            final_output = torch.zeros((batch_size, *out_shape), device=x.device)

            for i in range(self.n_experts):
                if results[i] is not None:
                    out = results[i]
                    # indices list of (batch_idx, weight)
                    for j, (b_idx, weight) in enumerate(expert_indices[i]):
                        final_output[b_idx] += out[j] * weight

            return final_output
        else:
            # Fallback if absolutely nothing ran (unlikely unless batch=0)
            # Try running expert 0 on x[0:1] to get shape
            dummy_out = self.experts[0](
                x[0:1].to(
                    self.expert_devices[0] if self.expert_devices[0] else x.device
                )
            )
            if isinstance(dummy_out, tuple):
                dummy_out = dummy_out[0]
            return torch.zeros((batch_size, *dummy_out.shape[1:]), device=x.device)
