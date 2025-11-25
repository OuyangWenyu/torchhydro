"""
Author: Wenyu Ouyang
Date: 2025-11-25
LastEditTime: 2025-11-25
LastEditors: Wenyu Ouyang
Description: Mixture of Experts (MoE) model
FilePath: \torchhydro\torchhydro\models\moe.py
Copyright (c) 2021-2025 Wenyu Ouyang. All rights reserved.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MoE(nn.Module):
    """
    A simple Mixture of Experts (MoE) model.
    """

    def __init__(self, n_experts, experts_cfgs, gating_cfgs, gating_mode="full", **kwargs):
        """
        __init__ method for the MoE model.

        Parameters
        ----------
        n_experts
            Number of expert models.
        experts_cfgs
            A list of configurations for each expert model. Each configuration is a
            dictionary with 'model_name' and 'model_hyperparam'.
        gating_cfgs
            Configuration for the gating network. It is a dictionary with
            'model_name' and 'model_hyperparam'.
        gating_mode
            'full' or 'last'. If 'full', the full input is passed to the gating network.
            If 'last', only the last time step is used (for sequence data).
        """
        super(MoE, self).__init__()
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
        self.gating_network = pytorch_model_dict[gating_model_name](**gating_hyperparam)

    def _create_expert(self, expert_cfg, pytorch_model_dict):
        """
        Helper function to create an expert model from its configuration.
        """
        model_name = expert_cfg["model_name"]
        model_hyperparam = expert_cfg["model_hyperparam"]
        return pytorch_model_dict[model_name](**model_hyperparam)
        
    def forward(self, x):
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
        if self.gating_mode == 'last':
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

