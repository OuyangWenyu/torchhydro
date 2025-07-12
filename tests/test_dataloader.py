"""
Author: Wenyu Ouyang
Date: 2025-01-06 14:21:00
LastEditTime: 2025-01-25 09:19:55
LastEditors: Wenyu Ouyang
Description: 
FilePath: /torchhydro/tests/test_dataloader.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import pandas as pd
import xarray as xr
import numpy as np
from torchhydro.configs.config import default_config_file, update_cfg
from torchhydro.trainers.deep_hydro import DeepHydro
import os


def test_get_dataloader_train_eval(dataloader_args):
    """
    Test DataLoader generation in training mode.
    """
    config_data = default_config_file()
    update_cfg(config_data, dataloader_args)
    if not os.path.exists(config_data["data_cfgs"]["validation_path"]):
        os.makedirs(config_data["data_cfgs"]["validation_path"])
    deep_hydro = DeepHydro(config_data)

    dataloader, valid_dataloader = deep_hydro._get_dataloader(
        config_data["training_cfgs"],
        config_data["evaluation_cfgs"],
        config_data["data_cfgs"],
        mode="train",
    )

    # Assertions
    assert dataloader.batch_size == config_data["training_cfgs"]["batch_size"]
    assert valid_dataloader.batch_size == config_data["evaluation_cfgs"]["batch_size"]
    # add other assertions here if needed


def test_get_dataloader_test(dataloader_args):
    """
    Test DataLoader generation in training mode.
    """
    config_data = default_config_file()
    update_cfg(config_data, dataloader_args)
    if not os.path.exists(config_data["data_cfgs"]["validation_path"]):
        os.makedirs(config_data["data_cfgs"]["validation_path"])
    deep_hydro = DeepHydro(config_data)

    dataloader = deep_hydro._get_dataloader(
        config_data["training_cfgs"],
        config_data["evaluation_cfgs"],
        config_data["data_cfgs"],
        mode="infer",
    )

    # Assertions
    assert dataloader.batch_size == config_data["training_cfgs"]["batch_size"]
    # add other assertions here if needed
