"""
Author: Wenyu Ouyang
Date: 2024-05-20 10:40:46
LastEditTime: 2024-05-24 13:54:51
LastEditors: Wenyu Ouyang
Description: 
FilePath: \torchhydro\tests\test_train_with_gpm.py
Copyright (c) 2021-2024 Wenyu Ouyang. All rights reserved.
"""

from torchhydro.configs.config import update_cfg
from torchhydro.trainers.trainer import train_and_evaluate


def test_train_evaluate(s2s_args, config_data):
    update_cfg(config_data, s2s_args)
    train_and_evaluate(config_data)
