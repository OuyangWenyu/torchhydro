"""
Author: Wenyu Ouyang
Date: 2023-07-25 16:47:19
LastEditTime: 2023-12-15 16:59:46
LastEditors: Wenyu Ouyang
Description: Test a full training and evaluating process
FilePath: /torchhydro/tests/test_train_camels_lstm.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""
import os
import pytest
from torchhydro import DATASOURCE_SETTINGS
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate


@pytest.fixture()
def config():
    project_name = "test_camels/exp1"
    data_dir = DATASOURCE_SETTINGS["datasets-origin"]
    source_path = os.path.join(
        data_dir, "camels", "camels_us"
    )
    config_data = default_config_file()
    args = cmd(
        sub=project_name,
        source="CAMELS",
        source_path=source_path,
        source_region="US",
        download=0,
        ctx=[-1],
        model_name="CpuLSTM",
        model_hyperparam={
            "n_input_features": 23,
            "n_output_features": 1,
            "n_hidden_states": 256,
        },
        gage_id=[
            "01013500",
            "01022500",
            "01030500",
            "01031500",
            "01047000",
            "01052500",
            "01054200",
            "01055000",
            "01057000",
            "01170100",
        ],
        batch_size=8,
        rho=20,
        var_t=["dayl", "prcp", "srad", "tmax", "tmin", "vp"],
        # var_c=["None"],
        var_out=["streamflow"],
        dataset="StreamflowDataset",
        sampler="KuaiSampler",
        scaler="DapengScaler",
        train_epoch=2,
        save_epoch=1,
        te=2,
        train_period=["2000-10-01", "2001-10-01"],
        valid_period=["2001-10-01", "2002-10-01"],
        test_period=["2002-10-01", "2003-10-01"],
        loss_func="RMSESum",
        opt="Adam",
        # key is epoch, start from 1
        lr_scheduler={1: 1e-2, 2: 5e-3, 3: 1e-3},
        which_first_tensor="sequence",
    )
    update_cfg(config_data, args)
    return config_data


def test_train_evaluate(config):
    train_and_evaluate(config)
