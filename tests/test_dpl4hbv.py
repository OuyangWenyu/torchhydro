"""
Author: Wenyu Ouyang
Date: 2024-11-17 13:20:00
LastEditTime: 2024-11-17 16:52:54
LastEditors: Wenyu Ouyang
Description: Test funcs for hbv4dpl model
FilePath: \torchhydro\tests\test_dpl4hbv.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

from torchhydro.configs.config import default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate


def test_dpl_selfmadehydrodataset(dpl4hbv_selfmadehydrodataset_args):
    cfg = default_config_file()
    update_cfg(cfg, dpl4hbv_selfmadehydrodataset_args)
    train_and_evaluate(cfg)
    print("All processes are finished!")
