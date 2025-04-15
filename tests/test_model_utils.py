"""
Author: Wenyu Ouyang
Date: 2025-04-15 13:07:20
LastEditTime: 2025-04-15 13:09:25
LastEditors: Wenyu Ouyang
Description:
FilePath: /torchhydro/tests/test_model_utils.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import pytest
import torch
from torchhydro.models.model_utils import get_the_device


@pytest.mark.parametrize(
    "device_num, expected_device",
    [
        (-1, "cpu"),
        ([-1], "cpu"),
        (["-1"], "cpu"),
    ],
)
def test_get_the_device_cpu(device_num, expected_device):
    device = get_the_device(device_num)
    assert device.type == expected_device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize(
    "device_num, expected_device",
    [
        (0, "cuda"),
        ([0], "cuda"),
        (1, "cuda"),
    ],
)
def test_get_the_device_cuda(device_num, expected_device):
    device = get_the_device(device_num)
    assert device.type == expected_device
    assert device.index == (
        device_num[0] if isinstance(device_num, list) else device_num
    )


@pytest.mark.skipif(
    not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available(),
    reason="MPS is not available",
)
@pytest.mark.parametrize(
    "device_num, expected_device",
    [
        (0, "mps"),
        ([0], "mps"),
        (1, "mps"),  # Should warn and default to mps:0
    ],
)
def test_get_the_device_mps(device_num, expected_device):
    if device_num != 0:
        with pytest.warns(UserWarning, match="MPS only supports device 0"):
            device = get_the_device(device_num)
    else:
        device = get_the_device(device_num)
    assert device.type == expected_device
    assert device.index == 0
