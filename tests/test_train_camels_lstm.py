"""
Author: Wenyu Ouyang
Date: 2023-07-25 16:47:19
LastEditTime: 2024-04-10 21:00:10
LastEditors: Wenyu Ouyang
Description: Test a full training and evaluating process
FilePath: \torchhydro\tests\test_train_camels_lstm.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

from torchhydro.configs.config import update_cfg
from torchhydro.trainers.trainer import train_and_evaluate


def test_train_evaluate(args, config_data):
    update_cfg(config_data, args)
    train_and_evaluate(config_data)
