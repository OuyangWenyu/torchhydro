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

# @pytest.fixture()
def config():
    project_name = "test_mean_lstm/exp11"
    config_data = default_config_file()
    args = cmd(
        sub=project_name,
        source="Mean",
        source_path=os.path.join(hds.ROOT_DIR, "lstm_data"),
        source_region="US",
        download=0,
        ctx=[0],
        model_name="KuaiLSTM",
        model_hyperparam={
            "n_input_features": 516,
            "n_output_features": 1,
            "n_hidden_states": 256,
        },
        # gage_id=[
        #     "21401550"
        # ],
        gage_id=gage_id,
        batch_size=1024,
        rho=1, # seq_length forecast_history forecast_length=1 linearIn
        # var_t=["p_mean"],
        var_t=["total_precipitation"],
        # var_c=["None"],
        var_out=["streamflow"],
        dataset="MEAN_Dataset",
        sampler="KuaiSampler",
        scaler="DapengScaler",
        train_epoch=100,
        save_epoch=99,
        te=99,
        #1979
        train_period=["2018-02-01", "2019-02-01"],
        valid_period=["2018-07-01", "2018-09-30"],
        test_period=["2018-07-01", "2018-09-30"],

        loss_func="RMSESum",
        opt="Adam",
        # key is epoch, start from 1
        lr_scheduler={1: 1e-2, 2: 5e-3, 3: 1e-3},
        which_first_tensor="sequence",
        is_tensorboard=False
    )
    update_cfg(config_data, args)
    return config_data


def test_train_evaluate(config):
    train_and_evaluate(config)

test_train_evaluate(config())