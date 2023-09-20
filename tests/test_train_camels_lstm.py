"""
Author: Wenyu Ouyang
Date: 2023-07-25 16:47:19
LastEditTime: 2023-09-20 16:49:45
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


@pytest.fixture()
def config():
    project_name = "test_camels/exp1"
    config_data = default_config_file()
    args = cmd(
        sub=project_name,
        source="CAMELS",
        source_path=os.path.join(hds.ROOT_DIR, "camels", "camels_us"),
        source_region="US",
        download=0,
        ctx=[-1],
        model_name="CpuLSTM",
        model_param={
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
        batch_size=50,
        rho=20,
        var_t=["dayl", "prcp", "srad", "tmax", "tmin", "vp"],
        # var_c=["None"],
        var_out=["streamflow"],
        data_loader="StreamflowDataset",
        scaler="DapengScaler",
        scaler_params={
            "prcp_norm_cols": ["streamflow"],
            "gamma_norm_cols": [
                "prcp",
                "pr",
                "total_precipitation",
                "potential_evaporation",
                "ET",
                "PET",
                "ET_sum",
                "ssm",
            ],
        },
        train_epoch=5,
        save_epoch=1,
        te=5,
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
