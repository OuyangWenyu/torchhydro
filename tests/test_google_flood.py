"""
Author: Wenyu Ouyang
Date: 2025-01-06 16:19:26
LastEditTime: 2025-01-25 09:19:36
LastEditors: Wenyu Ouyang
Description: 
FilePath: /torchhydro/tests/test_google_flood.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

from torchhydro.configs.config import default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate


def test_google_flood(google_flood_args):
    config_data = default_config_file()
    update_cfg(config_data, google_flood_args)
    train_and_evaluate(config_data)
