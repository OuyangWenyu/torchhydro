"""
Author: Wenyu Ouyang
Date: 2023-07-25 16:47:19
LastEditTime: 2023-11-29 17:27:01
LastEditors: Wenyu Ouyang
Description: Test a full training and evaluating process
FilePath: \torchhydro\tests\test_train_camels_lstm.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import os
import pytest
import hydrodataset as hds
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate
import pandas as pd

# show = pd.read_csv("data/gage_id.csv", dtype={'id':str})
# gage_id = show['id'].values.tolist()


@pytest.fixture()
def config():
    project_name = "test_mean_lstm/exp01"
    config_data = default_config_file()
    args = cmd(
        sub=project_name,
        source_path=os.path.join(hds.ROOT_DIR),
        streamflow_source_path="/ftproot/LSTM_data/nldas_hourly.nc",
        rainfall_source_path="/ftproot/LSTM_data/nldas_hourly.nc",
        attributes_path="/ftproot/camelsus_attributes_us.nc",
        ctx=[0],
        model_name="SimpleLSTMForecast",
        model_hyperparam={
            "input_size": 1,
            "output_size": 1,
            "hidden_size": 256,
            "forecast_length": 24,
        },
        gage_id=["02051500"],
        batch_size=64,
        rho=168,
        var_t=["total_precipitation"],
        var_c=["elev_mean"],
        var_out=["streamflow"],
        dataset="MEAN_Dataset",
        sampler="WuSampler",
        scaler="DapengScaler",
        train_epoch=1,
        save_epoch=1,
        te=1,
        train_period=[
            {"start": "2014-07-08", "end": "2014-09-28"},
            {"start": "2015-07-08", "end": "2015-09-28"},
            {"start": "2016-07-08", "end": "2016-09-28"},
            {"start": "2017-07-08", "end": "2017-09-28"},
        ],
        test_period=[
            {"start": "2018-07-08", "end": "2018-09-28"},
        ],
        valid_period=[
            {"start": "2018-07-08", "end": "2018-09-28"},
        ],
        loss_func="RMSESum",
        opt="Adam",
        lr_scheduler={1: 1e-3},
        which_first_tensor="sequence",
        lr_factor=0.5,
        lr_patience=1,
        weight_decay=1e-5,
        lr_val_loss=True,
        early_stopping=True,
        patience=4,
        is_tensorboard=False,
        ensemble=True,
        ensemble_items={
            "kfold": 5,
            "batch_sizes": [256],
        },
        user="zxw",
    )
    update_cfg(config_data, args)
    return config_data


def test_mean_lstm(config):
    train_and_evaluate(config)
    # ensemble_train_and_evaluate(config)
