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

# import xarray as xr
# show = xr.open_dataset("/ftproot/camels_hourly/data/usgs-streamflow-nldas_hourly.nc")
import pandas as pd
show = pd.read_csv("data/gage_id.csv", dtype={'id':str})
gage_id = show['id'].values.tolist()

@pytest.fixture()
def config():
    project_name = "test_mean_lstm/exp13"
    config_data = default_config_file()
    args = cmd(
        sub=project_name,
        source="Mean",
        source_path=os.path.join(hds.ROOT_DIR, "lstm_data"),
        source_region="US",
        download=0,
        ctx=[0],
        model_name="SimpleLSTMForecast",
        model_hyperparam={
            # "n_input_features": 1,
            "input_size": 2,
            "output_size": 1,
            "hidden_size": 256,
            "forecast_length": 24,
        },
        # gage_id=[
        #     "21401550"
        # ],
        gage_id=gage_id,
        batch_size=512,
        rho=168, # seq_length forecast_history forecast_length=1 linearIn
        forecast_length=24,
        # var_t=["p_mean"],
        var_t=["total_precipitation", "streamflow"],
        # var_c=["None"],
        var_out=["streamflow"],
        dataset="MEAN_Dataset",
        # dataset = "GPM_GFS_Dataset",
        sampler = "WuSampler",
        scaler="MeanScaler",
        train_epoch=50,
        save_epoch=1,
        te=49,
        #1979
        train_period=[            
            ["2014-07-01T00:00:00", "2014-09-30T00:00:00"],
            # ["2015-07-01T00:00:00", "2015-09-30T00:00:00"],
            # ["2016-07-01T00:00:00", "2016-09-30T00:00:00"],
            ],
        valid_period=[["2018-07-01T00:00:00", "2018-09-30T00:00:00"]],
        test_period=[["2017-07-01T00:00:00", "2017-09-30T00:00:00"]],

        loss_func="RMSESum",
        opt="Adam",
        # key is epoch, start from 1
        lr_scheduler={1: 1e-2, 2: 5e-3, 3: 1e-3},
        which_first_tensor="sequence",
        is_tensorboard=False,
        user = "zxw"
    )
    update_cfg(config_data, args)
    return config_data


def test_train_evaluate(config):
    train_and_evaluate(config)
