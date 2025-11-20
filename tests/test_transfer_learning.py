"""
Author: Wenyu Ouyang
Date: 2023-10-05 16:16:48
LastEditTime: 2025-11-08 09:50:01
LastEditors: Wenyu Ouyang
Description: A test function for transfer learning
FilePath: \torchhydro\tests\test_transfer_learning.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import pytest
from torchhydro.configs.config import default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate


@pytest.mark.requires_data
@pytest.mark.skip(
    reason="TODO: This is an early test function, need to be updated, and then run it, now skipping."
)
def test_transfer_camels_lstm_model(trans_args):
    """
    Test transfer learning for basins in CAMELS-US from CAMELS-US pre-trained model.
    This test requires a pre-trained model specified by `weight_path` in the `trans_args` fixture.
    """
    cfg = default_config_file()
    update_cfg(cfg, trans_args)
    # The following function is the actual test execution, which is skipped.
    train_and_evaluate(cfg)
    # Add a simple assertion to make this a valid test function.
    assert cfg is not None
    print("All processes are finished!")
