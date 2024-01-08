"""
Author: Xinzhuo Wu
Date: 2023-09-30 1:20:18
LastEditTime: 2023-12-29 11:05:57
LastEditors: Xinzhuo Wu
Description: Test a full training and evaluating process with Spp_Lstm
FilePath: \torchhydro\tests\test_spp_lstm.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
import os
import pytest
import hydrodataset as hds
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate, ensemble_train_and_evaluate
import warnings

warnings.filterwarnings("ignore")


@pytest.fixture()
def config():
    project_name = "test_spp_lstm/ex3"
    config_data = default_config_file()
    args = cmd(
        sub=project_name,
        source="GPM_GFS",
        source_path=os.path.join(hds.ROOT_DIR, "gpm_gfs_data"),
        source_region="US",
        download=0,
        ctx=[2],
        model_name="SPPLSTM2",
        model_hyperparam={
            "seq_length": 168,
            "forecast_length": 24,
            "n_output": 1,
            "n_hidden_states": 60,
            "dropout": 0.25,
        },
        gage_id=["1_02051500", "86_21401550"],
        batch_size=256,
        var_t=["tp"],
        var_out=["streamflow"],
        dataset="GPM_GFS_Dataset",
        sampler="WuSampler",
        scaler="GPM_GFS_Scaler",
        train_epoch=50,
        save_epoch=1,
        te=50,
        train_period=[
            {"start": "2017-07-01", "end": "2017-09-29"},
            {"start": "2018-07-01", "end": "2018-09-29"},
            {"start": "2019-07-01", "end": "2019-09-29"},
            {"start": "2020-07-01", "end": "2020-09-29"},
        ],
        test_period=[
            {"start": "2021-07-01", "end": "2021-09-29"},
        ],
        valid_period=[
            {"start": "2021-07-01", "end": "2021-09-29"},
        ],
        loss_func="RMSESum",
        opt="Adam",
        lr_scheduler={1: 1e-3},
        lr_factor=0.5,
        lr_patience=3,
        weight_decay=1e-5,  # L2正则化衰减权重
        lr_val_loss=True,  # False则用NSE作为指标，而不是val loss,来更新lr、model、早退
        which_first_tensor="sequence",
        early_stopping=True,
        patience=10,
        rolling=False,  # evaluate 不采用滚动预测
        ensemble=True,
        ensemble_items={
            "kfold": 5,
            "batch_sizes": [256],
        },
    )
    update_cfg(config_data, args)
    return config_data


def test_spp_lstm(config):
    # train_and_evaluate(config)
    ensemble_train_and_evaluate(config)
