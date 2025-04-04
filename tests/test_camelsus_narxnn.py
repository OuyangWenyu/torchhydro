"""
Author: Wenyu Ouyang
Date: 2022-09-09 14:47:42
LastEditTime: 2024-11-11 18:33:10
LastEditors: Wenyu Ouyang
Description: a script to run experiments for LSTM - CAMELS
FilePath: \torchhydro\experiments\run_camelslstm_experiments.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import os

from torchhydro import SETTING
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate
import pytest

@pytest.fixture
def var_c():
    return [
        "elev_mean",
        "slope_mean",
        "area_gages2",
        "frac_forest",
        # "lai_max",
        # "lai_diff",
        # "dom_land_cover_frac",
        # "dom_land_cover",
        # "root_depth_50",
        # "soil_depth_statsgo",
        # "soil_porosity",
        # "soil_conductivity",
        # "max_water_content",
        # "geol_1st_class",
        # "geol_2nd_class",
        # "geol_porostiy",
        # "geol_permeability",
    ]

@pytest.fixture
def var_t():
    return [
        # NOTE: prcp must be the first variable
        "prcp",
        "PET",
        # "dayl",
        # "srad",
        # "swe",
        # "tmax",
        # "tmin",
        # "vp",
        "streamflow",
    ]

@pytest.fixture
def camelsusnarxnn_arg(var_c,var_t):
    project_name = os.path.join("test_camels", "narxnn_camelsus")
    # camels-us time_range: ["1980-01-01", "2014-12-31"]
    train_period = ["1985-10-01", "1995-10-01"]
    valid_period = ["1995-10-01", "2000-10-01"]
    test_period = ["2000-10-01", "2010-10-01"]
    config_data = default_config_file()
    args = cmd(
        sub=project_name,
        source_cfgs={
            "source_name": "camels_us",
            "source_path": os.path.join(
                SETTING["local_data_path"]["datasets-origin"], "camels", "camels_us"
            ),
        },
        ctx=[-1],
        # model_name="KuaiLSTM",
        model_name="Narxnn",
        model_hyperparam={
            "n_input_features": len(var_c) + len(var_t),  # 17 + 9 = 26
            "n_output_features": 1,
            "n_hidden_states": 64,
            "input_delay":[1,1],
            "feedback_delay":[1],
        },
        loss_func="RMSESum",
        sampler="KuaiSampler",
        dataset="StreamflowDataset",
        scaler="DapengScaler",
        batch_size=10,
        forecast_history=0,
        forecast_length=366,
        var_t=var_t,
        var_c=var_c,
        var_out=["streamflow"],
        train_period=train_period,
        valid_period=valid_period,
        test_period=test_period,
        opt="Adadelta",
        rs=1234,
        train_epoch=10,
        save_epoch=1,
        model_loader={
            "load_way": "specified",
            "test_epoch": 10,
        },
        # the gage_id.txt file is set by the user, it must be the format like:
        # GAUGE_ID
        # 01013500
        # 01022500
        # ......
        # Then it can be read by pd.read_csv(gage_id_file, dtype={0: str}).iloc[:, 0].values to get the gage_id list
        gage_id_file="D:\\minio\\waterism\\datasets-origin\\camels\\camels_us\\gage_id.txt",
        which_first_tensor="sequence",
    )
    update_cfg(config_data, args)
    return config_data

def test_camelsusnarxnn(camelsusnarxnn_arg):
    train_and_evaluate(camelsusnarxnn_arg)
    print("All processes are finished!")
