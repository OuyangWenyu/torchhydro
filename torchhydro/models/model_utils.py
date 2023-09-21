"""
Author: Wenyu Ouyang
Date: 2021-08-09 10:19:13
LastEditTime: 2023-09-21 16:47:05
LastEditors: Wenyu Ouyang
Description: Some util functions for modeling
FilePath: /torchhydro/torchhydro/models/model_utils.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""

from typing import Union
import warnings
import torch


def get_the_device(device_num: Union[list, int]):
    """
    Get device for torch according to its name

    Parameters
    ----------
    device_num : Union[list, int]
        number of the device -- -1 means "cpu" or 0, 1, ... means "cuda:x"
    """
    if device_num in [[-1], -1, ["-1"]]:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return (
            torch.device(f"cuda:{str(device_num)}")
            if type(device_num) is not list
            else torch.device(f"cuda:{str(device_num[0])}")
        )
    if device_num not in [[-1], -1, ["-1"]]:
        warnings.warn("You don't have GPU, so have to choose cpu for models")
    return torch.device("cpu")
