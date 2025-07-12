"""
Author: Wenyu Ouyang
Date: 2025-01-06 16:19:26
LastEditTime: 2025-07-03 17:06:27
LastEditors: Wenyu Ouyang
Description: Test the selfmadehydrodataset models
FilePath: /torchhydro/tests/test_selfmadehydrodataset_models.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

from torchhydro.configs.config import default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate


def test_selfmadehydrodataset_simplelstm(selfmadehydrodataset_args):
    config_data = default_config_file()
    update_cfg(config_data, selfmadehydrodataset_args)
    train_and_evaluate(config_data)


def test_selfmadehydrodataset_dpllstm(selfmadehydrodataset_dpl4xaj_args):
    config_data = default_config_file()
    update_cfg(config_data, selfmadehydrodataset_dpl4xaj_args)
    train_and_evaluate(config_data)


def test_selfmadehydrodataset_transfer_learning(selfmadehydrodataset_tlargs):
    config_data = default_config_file()
    update_cfg(config_data, selfmadehydrodataset_tlargs)
    train_and_evaluate(config_data)
