"""
Author: Wenyu Ouyang
Date: 2023-09-24 14:28:48
LastEditTime: 2023-12-18 09:11:55
LastEditors: Wenyu Ouyang
Description: A test for federated learning
FilePath: \torchhydro\tests\test_federated_learning.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""
import os
import pytest

from torchhydro import SETTING
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate


@pytest.fixture()
def config():
    project_name = "test_camels/exp3"
    config_data = default_config_file()
    args = cmd(
        sub=project_name,
        source="CAMELS",
        source_path=os.path.join(
            SETTING["local_data_path"]["datasets-origin"], "camels", "camels_us"
        ),
        source_region="US",
        ctx=[-1],
        model_type="FedLearn",
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
        forecast_history=0,
        forecast_length=20,
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
