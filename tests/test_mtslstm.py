"""
Author: Wenyu Ouyang
Date: 2025-11-06 15:35:33
LastEditTime: 2025-11-06 15:41:43
LastEditors: Wenyu Ouyang
Description:
FilePath: \torchhydro\tests\test_mtslstm.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

import torch
from torchhydro.models.mtslstm import MTSLSTM


def test_mtslstm_two_timescales() -> None:
    """
    Test the MTSLSTM model with two time scales (daily and hourly)
    of random data.
    """
    # Model config
    # This is a simplified config for testing purposes
    nf = 2  # Number of frequencies: daily and hourly
    input_sizes = [2, 3]  # Number of features for daily and hourly inputs
    hidden_sizes = [64, 128]
    output_size = 1

    # Instantiate the model
    model = MTSLSTM(
        input_sizes=input_sizes,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
    )

    # Create mock data
    batch_size = 10
    # Daily data (e.g., 3 days, 2 features)
    seq_len_day = 3
    x_day = torch.randn(seq_len_day, batch_size, input_sizes[0])

    # Hourly data (e.g., 3*24=72 hours, 3 features)
    seq_len_hour = 72
    x_hour = torch.randn(seq_len_hour, batch_size, input_sizes[1])

    # Forward pass
    # The model expects inputs from lower frequency to higher frequency
    # So, daily data first, then hourly data
    output = model(x_day, x_hour)

    # Assertions
    # By default, the model returns the output of the highest frequency branch
    assert isinstance(output, torch.Tensor)

    # The output shape should be (seq_len_hour, batch_size, output_size)
    expected_shape = (seq_len_hour, batch_size, output_size)
    assert (
        output.shape == expected_shape
    ), f"Expected output shape {expected_shape}, but got {output.shape}"

    # Test with return_all=True
    all_outputs = model(x_day, x_hour, return_all=True)
    assert isinstance(all_outputs, dict)
    assert len(all_outputs) == nf
    assert f"f0" in all_outputs  # Daily output
    assert f"f1" in all_outputs  # Hourly output

    # Check shapes of all outputs
    expected_day_shape = (seq_len_day, batch_size, output_size)
    expected_hour_shape = (seq_len_hour, batch_size, output_size)
    assert all_outputs["f0"].shape == expected_day_shape
    assert all_outputs["f1"].shape == expected_hour_shape
