"""
Author: Wenyu Ouyang
Date: 2023-07-25 16:47:19
LastEditTime: 2025-11-08 10:03:04
LastEditors: Wenyu Ouyang
Description: Test a full training and evaluating process
FilePath: \torchhydro\tests\test_caravan_train.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import os
import pytest
from hydrodataset.hydro_dataset import StandardVariable
from torchhydro import SETTING
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate


@pytest.fixture()
def var_c():
    return ["area", "p_mean", "pet_mean", "aridity", "frac_snow", "moisture_index"]


@pytest.fixture()
def var_t():
    return [
        StandardVariable.PRECIPITATION,
        StandardVariable.POTENTIAL_EVAPOTRANSPIRATION,
        StandardVariable.TEMPERATURE_MAX,
        StandardVariable.TEMPERATURE_MIN,
        StandardVariable.SOLAR_RADIATION,
    ]


@pytest.fixture()
def config(var_c, var_t):
    project_name = os.path.join("test_caravan", "exp5")
    config_data = default_config_file()
    args = cmd(
        sub=project_name,
        source_cfgs={
            "source_name": "caravan",
            "source_path": SETTING["local_data_path"]["datasets-origin"],
        },
        ctx=[-1],
        model_name="CpuLSTM",
        model_hyperparam={
            "n_input_features": len(var_t) + len(var_c),
            "n_output_features": 1,
            "n_hidden_states": 256,
        },
        gage_id=[
            "camels_01022500",
            "camels_01031500",
            "camels_01047000",
            "camels_01052500",
            "camels_01054200",
            "camels_01055000",
            "camels_01057000",
            "camelsaus_102101A",
            "camelsaus_108003A",
            "hysets_06444000",
        ],
        batch_size=8,
        hindcast_length=0,
        forecast_length=20,
        var_t=var_t,
        var_c=var_c,
        var_out=["streamflow"],
        dataset="StreamflowDataset",
        sampler="KuaiSampler",
        scaler="DapengScaler",
        train_epoch=2,
        save_epoch=1,
        train_period=["2000-10-01", "2001-10-01"],
        valid_period=["2001-10-01", "2002-10-01"],
        test_period=["2002-10-01", "2003-10-01"],
        loss_func="RMSESum",
        opt="Adam",
        # key is epoch, start from 1
        lr_scheduler={1: 1e-2, 2: 5e-3, 3: 1e-3},
        which_first_tensor="sequence",
        model_loader={"load_way": "specified", "test_epoch": 2},
    )
    update_cfg(config_data, args)
    return config_data


@pytest.mark.requires_data
def test_train_evaluate(config):
    train_and_evaluate(config)
