"""
Author: Wenyu Ouyang
Date: 2024-04-17 12:55:24
LastEditTime: 2024-11-05 10:40:35
LastEditors: Wenyu Ouyang
Description: Test funcs for seq2seq model
FilePath: \torchhydro\tests\test_seq2seq.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import pytest
import torch
from torchhydro.models.seq2seq import GeneralSeq2Seq

import logging
from torchhydro.trainers.trainer import train_and_evaluate

logging.basicConfig(level=logging.INFO)
for logger_name in logging.root.manager.loggerDict:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

gage_id = [
    "songliao_21401050",
    "songliao_21401550",
]


def test_seq2seq(seq2seq_config):
    # world_size = len(config["training_cfgs"]["device"])
    # mp.spawn(train_worker, args=(world_size, config), nprocs=world_size, join=True)
    train_and_evaluate(seq2seq_config)
    # ensemble_train_and_evaluate(config)


@pytest.fixture
def model():
    return GeneralSeq2Seq(
        en_input_size=2,
        de_input_size=3,
        output_size=2,
        hidden_size=20,
        forecast_length=5,
        hindcast_output_window=10,
        teacher_forcing_ratio=0.5,
    )


def test_forward_no_teacher_forcing(model):
    src1 = torch.randn(3, 10, 2)
    src2 = torch.randn(3, 5, 1)
    outputs = model(src1, src2)
    assert outputs.shape == (3, 6, 2)


def test_forward_with_teacher_forcing(model):
    src1 = torch.randn(3, 10, 2)
    src2 = torch.randn(3, 5, 1)
    trgs = torch.randn(3, 15, 2)
    outputs = model(src1, src2, trgs)
    assert outputs.shape == (3, 6, 2)
