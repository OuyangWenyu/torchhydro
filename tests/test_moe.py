"""
Author: Wenyu Ouyang
Date: 2025-11-25
LastEditTime: 2025-11-25 17:42:43
LastEditors: Wenyu Ouyang
Description: Test function for MoE model
FilePath: \torchhydro\tests\test_moe.py
Copyright (c) 2023-2025 Wenyu Ouyang. All rights reserved.
"""

import torch
from torchhydro.models.moe import MoE


def test_moe_model():
    """
    Test the forward pass of the MoE model.
    """
    # Define hyperparameters
    nx = 10
    ny = 5
    n_experts = 4
    batch_size = 32

    # Configuration for the expert models (using SimpleAnn for simplicity)
    experts_cfgs = [
        {
            "model_name": "SimpleAnn",
            "model_hyperparam": {"nx": nx, "ny": ny, "hidden_size": [20, 20]},
        }
        for _ in range(n_experts)
    ]

    # Configuration for the gating network (also using SimpleAnn)
    gating_cfgs = {
        "model_name": "SimpleAnn",
        "model_hyperparam": {"nx": nx, "ny": n_experts, "hidden_size": [15]},
    }

    # Instantiate the MoE model
    model = MoE(
        n_experts=n_experts,
        experts_cfgs=experts_cfgs,
        gating_cfgs=gating_cfgs,
    )

    # Create a random input tensor
    # The input shape for SimpleAnn is (batch_size, nx)
    input_tensor = torch.randn(batch_size, nx)

    # Perform a forward pass
    output = model(input_tensor)

    # Check the output shape
    assert output.shape == (batch_size, ny)


def test_moe_model_with_sequence():
    """
    Test the forward pass of the MoE model with sequence data.
    """
    # Define hyperparameters
    nx = 10
    ny = 5
    n_experts = 2
    batch_size = 32
    seq_len = 15

    # Configuration for the expert models (using SimpleLSTM for simplicity)
    experts_cfgs = [
        {
            "model_name": "SimpleLSTM",
            "model_hyperparam": {
                "input_size": nx,
                "output_size": ny,
                "hidden_size": 20,
            },
        }
        for _ in range(n_experts)
    ]

    # The gating network acts on the last time step's features.
    gating_cfgs = {
        "model_name": "SimpleAnn",
        "model_hyperparam": {"nx": nx, "ny": n_experts, "hidden_size": [10]},
    }

    model = MoE(
        n_experts=n_experts,
        experts_cfgs=experts_cfgs,
        gating_cfgs=gating_cfgs,
        gating_mode="last",
    )

    # Create a random input tensor
    input_tensor = torch.randn(batch_size, seq_len, nx)

    # Perform a forward pass
    output = model(input_tensor)

    # Check the output shape
    # The output of SimpleLSTM is [batch_size, seq_len, n_output_features]
    assert output.shape == (batch_size, seq_len, ny)
