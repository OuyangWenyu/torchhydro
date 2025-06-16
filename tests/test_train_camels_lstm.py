"""
Author: Wenyu Ouyang
Date: 2023-07-25 16:47:19
LastEditTime: 2025-06-16 16:06:10
LastEditors: Wenyu Ouyang
Description: Test a full training and evaluating process
FilePath: \torchhydro\tests\test_train_camels_lstm.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import os
from torchhydro.configs.config import update_cfg
from torchhydro.trainers.trainer import train_and_evaluate


def test_train_evaluate(args, config_data):
    update_cfg(config_data, args)
    train_and_evaluate(config_data)


def test_train_evaluate_varied_seq(args, config_data):
    """set varied seq length for training"""
    args.variable_length_cfgs = {
        "use_variable_length": True,
        "variable_length_type": "fixed",  # Use fixed length mode
        "fixed_lengths": [
            30,
            60,
            90,
        ],  # [30,60,90] means 30, 60, 90 periods
        "pad_strategy": "Pad",
    }
    args.model_name = "SimpleLSTM"
    args.model_hyperparam = {
        "input_size": 23,
        "output_size": 1,
        "hidden_size": 128,
    }
    update_cfg(config_data, args)
    train_and_evaluate(config_data)


def test_train_evaluate_continue(args, config_data):
    """We test the training and evaluation process with the continue_train
      flag set to 1 and the start_epoch set to 2. This will load a pretrained
      model and continue training.
      This pattern is useful for training a model
      when its training is interrupted

    Parameters
    ----------
    args
        basic args in conftest.py
    config_data
        default config data
    """
    args.continue_train = 1
    args.start_epoch = 2
    args.train_mode = 1
    update_cfg(config_data, args)
    config_data["model_cfgs"]["weight_path"] = os.path.join(
        config_data["data_cfgs"]["case_dir"], "model_Ep1.pth"
    )
    train_and_evaluate(config_data)
