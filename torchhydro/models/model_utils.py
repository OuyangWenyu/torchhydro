"""
Author: Wenyu Ouyang
Date: 2021-08-09 10:19:13
LastEditTime: 2025-04-15 12:59:35
LastEditors: Wenyu Ouyang
Description: Some util functions for modeling
FilePath: /torchhydro/torchhydro/models/model_utils.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""

import contextlib
from typing import Union
import warnings
import torch


def get_the_device(device_num: Union[list, int]):
    """
    Get device for torch according to its name

    Parameters
    ----------
    device_num : Union[list, int]
        number of the device -- -1 means "cpu" or 0, 1, ... means "cuda:x" or "mps:x"
    """
    if device_num in [[-1], -1, ["-1"]]:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return (
            torch.device(f"cuda:{str(device_num)}")
            if type(device_num) is not list
            else torch.device(f"cuda:{str(device_num[0])}")
        )
    # Check for MPS (MacOS)
    mps_available = False
    with contextlib.suppress(AttributeError):
        mps_available = torch.backends.mps.is_available()
    if mps_available:
        if device_num != 0:
            warnings.warn(
                f"MPS only supports device 0. Using 'mps:0' instead of {device_num}."
            )
        return torch.device("mps:0")
    if device_num not in [[-1], -1, ["-1"]]:
        warnings.warn("You don't have GPU, so have to choose cpu for models")
    return torch.device("cpu")
