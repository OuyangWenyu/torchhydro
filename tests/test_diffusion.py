"""
Author: Wenyu Ouyang
Date: 2025-11-25 17:43:01
LastEditTime: 2025-11-25 20:37:01
LastEditors: Wenyu Ouyang
Description: Test functions for Diffusion model
FilePath: \torchhydro\tests\test_diffusion.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import torch
import pytest

from torchhydro.models.diffusion import ConditionalDiffusionModel


@pytest.fixture
def dummy_cfgs() -> dict:
    """Provides a dummy configuration for the LSTM diffusion model."""
    return {
        "model": {
            "name": "ConditionalDiffusionModel",
            "diffusion": {
                "num_timesteps": 50,
                "data_dim": 1,
                "condition_dim": 1,
                "time_emb_dim": 16,  # Dimension for time embedding
                "beta_schedule": "linear",
            },
            "denoise_net": {
                "hidden_size": 32,  # LSTM hidden size
                "num_layers": 2,  # Number of LSTM layers
            },
        }
    }


def test_diffusion_model_instantiation(dummy_cfgs: dict) -> None:
    """Tests if the model can be instantiated without errors."""
    try:
        model = ConditionalDiffusionModel(dummy_cfgs)
        assert model is not None
    except Exception as e:
        pytest.fail(f"Failed to instantiate ConditionalDiffusionModel: {e}")


def test_diffusion_model_dimensions(dummy_cfgs: dict) -> None:
    """
    Tests the input/output dimensions of the forward (training) and sample (inference) methods.
    """
    batch_size = 2
    seq_len = 64

    model = ConditionalDiffusionModel(dummy_cfgs)

    # Input shapes are (batch_size, num_features, seq_len) for Conv1d
    runoff = torch.randn(
        batch_size, dummy_cfgs["model"]["diffusion"]["data_dim"], seq_len
    )
    rainfall = torch.randn(
        batch_size, dummy_cfgs["model"]["diffusion"]["condition_dim"], seq_len
    )

    # --- Test forward pass (training) ---
    output = model(runoff, rainfall)
    assert "loss" in output, "Output dictionary should contain a 'loss' key"
    loss = output["loss"]
    assert isinstance(loss, torch.Tensor), "Loss should be a torch.Tensor"
    assert loss.shape == torch.Size(
        []
    ), f"Loss should be a scalar tensor, but got shape {loss.shape}"
    assert not torch.isnan(loss), "Loss should not be NaN"
    assert not torch.isinf(loss), "Loss should not be infinite"

    # --- Test sample pass (inference) ---
    # The model should be in eval mode for sampling
    model.eval()
    generated_runoff = model.sample(rainfall)
    assert isinstance(
        generated_runoff, torch.Tensor
    ), "Sampled output should be a torch.Tensor"
    assert (
        generated_runoff.shape == runoff.shape
    ), f"Shape of generated data should be {runoff.shape}, but got {generated_runoff.shape}"
