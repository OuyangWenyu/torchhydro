"""
Author: Wenyu Ouyang
Date: 2025-01-06 16:19:26
LastEditTime: 2025-11-08 11:05:48
LastEditors: Wenyu Ouyang
Description: Test the selfmadehydrodataset models
FilePath: \torchhydro\tests\test_selfmadehydrodataset_models.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import pytest
from torchhydro.configs.config import default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate

@pytest.mark.requires_data
def test_selfmadehydrodataset_simplelstm(selfmadehydrodataset_args):
    config_data = default_config_file()
    update_cfg(config_data, selfmadehydrodataset_args)
    train_and_evaluate(config_data)


@pytest.mark.requires_data
@pytest.mark.skip(reason="TODO: This is an early-version test function, need to be updated, and then run it, now skipping.")
def test_selfmadehydrodataset_dpllstm(selfmadehydrodataset_dpl4xaj_args):
    config_data = default_config_file()
    update_cfg(config_data, selfmadehydrodataset_dpl4xaj_args)
    train_and_evaluate(config_data)
