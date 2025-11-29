"""
Author: Wenyu Ouyang
Date: 2025-11-25
LastEditTime: 2025-11-25 18:51:13
LastEditors: Wenyu Ouyang
Description: Test function for MoE model
FilePath: \torchhydro\tests\test_moe.py
Copyright (c) 2023-2025 Wenyu Ouyang. All rights reserved.
"""

import pytest
import torch
from torchhydro.models.moe import MoE


def test_moe_model() -> None:
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


def test_moe_model_with_sequence() -> None:
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


def test_moe_sparse_top1() -> None:
    """
    Test Sparse MoE with Top-K=1.
    """
    nx = 10
    ny = 5
    n_experts = 4
    batch_size = 32
    top_k = 1

    experts_cfgs = [
        {
            "model_name": "SimpleAnn",
            "model_hyperparam": {"nx": nx, "ny": ny, "hidden_size": [20, 20]},
        }
        for _ in range(n_experts)
    ]

    gating_cfgs = {
        "model_name": "SimpleAnn",
        "model_hyperparam": {"nx": nx, "ny": n_experts, "hidden_size": [15]},
    }

    model = MoE(
        n_experts=n_experts,
        experts_cfgs=experts_cfgs,
        gating_cfgs=gating_cfgs,
        top_k=top_k,
    )

    input_tensor = torch.randn(batch_size, nx)
    output = model(input_tensor)

    assert output.shape == (batch_size, ny)


def test_moe_sparse_top2() -> None:
    """
    Test Sparse MoE with Top-K=2.
    """
    nx = 10
    ny = 5
    n_experts = 4
    batch_size = 32
    top_k = 2

    experts_cfgs = [
        {
            "model_name": "SimpleAnn",
            "model_hyperparam": {"nx": nx, "ny": ny, "hidden_size": [20, 20]},
        }
        for _ in range(n_experts)
    ]

    gating_cfgs = {
        "model_name": "SimpleAnn",
        "model_hyperparam": {"nx": nx, "ny": n_experts, "hidden_size": [15]},
    }

    model = MoE(
        n_experts=n_experts,
        experts_cfgs=experts_cfgs,
        gating_cfgs=gating_cfgs,
        top_k=top_k,
    )

    input_tensor = torch.randn(batch_size, nx)
    output = model(input_tensor)

    assert output.shape == (batch_size, ny)


def test_moe_capacity_factor() -> None:
    """
    Test Sparse MoE with Capacity Factor (Hard Balancing).
    """
    nx = 10
    ny = 5
    n_experts = 2
    batch_size = 10
    top_k = 1
    # Very low capacity to force dropping
    capacity_factor = 0.1

    experts_cfgs = [
        {
            "model_name": "SimpleAnn",
            "model_hyperparam": {"nx": nx, "ny": ny, "hidden_size": [20, 20]},
        }
        for _ in range(n_experts)
    ]

    gating_cfgs = {
        "model_name": "SimpleAnn",
        "model_hyperparam": {"nx": nx, "ny": n_experts, "hidden_size": [15]},
    }

    model = MoE(
        n_experts=n_experts,
        experts_cfgs=experts_cfgs,
        gating_cfgs=gating_cfgs,
        top_k=top_k,
        capacity_factor=capacity_factor,
    )

    input_tensor = torch.randn(batch_size, nx)
    output = model(input_tensor)

    assert output.shape == (batch_size, ny)


def test_moe_soft_balancing() -> None:
    """
    Test Soft Balancing (Load Balancing Loss).
    """
    nx = 10
    ny = 5
    n_experts = 4
    batch_size = 32
    top_k = 1
    load_balance_loss_coef = 0.1

    experts_cfgs = [
        {
            "model_name": "SimpleAnn",
            "model_hyperparam": {"nx": nx, "ny": ny, "hidden_size": [20, 20]},
        }
        for _ in range(n_experts)
    ]

    gating_cfgs = {
        "model_name": "SimpleAnn",
        "model_hyperparam": {"nx": nx, "ny": n_experts, "hidden_size": [15]},
    }

    model = MoE(
        n_experts=n_experts,
        experts_cfgs=experts_cfgs,
        gating_cfgs=gating_cfgs,
        top_k=top_k,
        load_balance_loss_coef=load_balance_loss_coef,
    )

    input_tensor = torch.randn(batch_size, nx)
    output = model(input_tensor)

    assert output.shape == (batch_size, ny)
    assert model.aux_loss > 0
    assert isinstance(model.aux_loss, torch.Tensor)


def test_moe_multi_device() -> None:
    """
    Test MoE with expert_devices argument.
    Since we might not have multiple GPUs, we test with 'cpu' for all devices
    to ensure the logic of moving data works without error.
    """
    nx = 10
    ny = 5
    n_experts = 2
    batch_size = 10
    top_k = 1
    # Simulate multi-device by passing explicit device strings (even if all cpu)
    expert_devices = ["cpu", "cpu"]

    experts_cfgs = [
        {
            "model_name": "SimpleAnn",
            "model_hyperparam": {"nx": nx, "ny": ny, "hidden_size": [20, 20]},
        }
        for _ in range(n_experts)
    ]

    gating_cfgs = {
        "model_name": "SimpleAnn",
        "model_hyperparam": {"nx": nx, "ny": n_experts, "hidden_size": [15]},
    }

    model = MoE(
        n_experts=n_experts,
        experts_cfgs=experts_cfgs,
        gating_cfgs=gating_cfgs,
        top_k=top_k,
        expert_devices=expert_devices,
    )

    input_tensor = torch.randn(batch_size, nx)
    output = model(input_tensor)

    assert output.shape == (batch_size, ny)


def test_moe_device_distribution() -> None:
    """
    Test automatic round-robin device distribution.
    """
    nx = 10
    ny = 5
    n_experts = 4
    batch_size = 10
    # Provide fewer devices than experts
    expert_devices = ["cpu"]

    experts_cfgs = [
        {
            "model_name": "SimpleAnn",
            "model_hyperparam": {"nx": nx, "ny": ny, "hidden_size": [20, 20]},
        }
        for _ in range(n_experts)
    ]

    gating_cfgs = {
        "model_name": "SimpleAnn",
        "model_hyperparam": {"nx": nx, "ny": n_experts, "hidden_size": [15]},
    }

    model = MoE(
        n_experts=n_experts,
        experts_cfgs=experts_cfgs,
        gating_cfgs=gating_cfgs,
        expert_devices=expert_devices,
    )

    # Check if devices were assigned in round-robin fashion
    # Since we only provided ["cpu"], all should be "cpu"
    expected_devices = ["cpu", "cpu", "cpu", "cpu"]
    assert model.expert_devices == expected_devices

    # Also verify forward pass runs without error
    input_tensor = torch.randn(batch_size, nx)
    output = model(input_tensor)
    assert output.shape == (batch_size, ny)


def test_moe_device_persistence() -> None:
    """
    Test that experts stay on their assigned devices even after model.to() is called.
    """
    nx = 10
    ny = 5
    n_experts = 2
    # Simulate different devices (using cpu for test environment)
    expert_devices = ["cpu", "cpu"]

    experts_cfgs = [
        {
            "model_name": "SimpleAnn",
            "model_hyperparam": {"nx": nx, "ny": ny, "hidden_size": [20, 20]},
        }
        for _ in range(n_experts)
    ]

    gating_cfgs = {
        "model_name": "SimpleAnn",
        "model_hyperparam": {"nx": nx, "ny": n_experts, "hidden_size": [15]},
    }

    model = MoE(
        n_experts=n_experts,
        experts_cfgs=experts_cfgs,
        gating_cfgs=gating_cfgs,
        expert_devices=expert_devices,
    )

    # Call model.to("cpu") - this mimics what the trainer does
    model.to("cpu")

    # Check if experts are still on the expected devices (which are "cpu" in this test)
    # In a real multi-gpu env, we would check against cuda:0, cuda:1 etc.
    # Here we verify the logic executes without error and devices match config.
    for i, expert in enumerate(model.experts):
        # Check device of first parameter
        param = next(expert.parameters())
        assert str(param.device) == "cpu"
        # Also verify our stored config matches
        assert model.expert_devices[i] == expert_devices[i]
