"""
Author: Yicheng Sun
Date: 2024-11-01 08:28:51
LastEditTime: 2025-07-01 18:19:32
LastEditors: Wenyu Ouyang
Description: Test BALSTM model
FilePath: \torchhydro\tests\test_balstm.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""
import pytest
import torch
from torchhydro.models.balstm_model import BALSTM
import logging
from torchhydro.trainers.trainer import train_and_evaluate

logging.basicConfig(level=logging.INFO)
for logger_name in logging.root.manager.loggerDict:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)


@pytest.mark.requires_data
def test_balstm(BALSTM_config):
    train_and_evaluate(BALSTM_config)


@pytest.fixture
def model():
    return BALSTM(
        input_size_sta=195,
        input_size_dyn=13,
        input_size_glo=106,
        hidden_size=64,
        output_size=1,
        num_layers=1,
    )


def test_forward_balstm(model):
    batch_size = 32
    seq_length = 12
    input_size_sta = 195
    input_size_dyn = 13
    input_size_glo = 106

    x_s = torch.randn(batch_size, input_size_sta)
    x_d = torch.randn(batch_size, seq_length, input_size_dyn)
    x_g = torch.randn(batch_size, seq_length, input_size_glo)
    outputs = model(x_s, x_d, x_g)
    assert outputs.shape == (batch_size, seq_length, 1)
