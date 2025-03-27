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
import pytest
from torchhydro import SETTING
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate

@pytest.fixture
def var_c():
    return [
        "elev_mean",
        "slope_mean",
        "area",
        "forest_frac",
        "crop_frac",
        "nf_frac",
        "dom_land_cover_frac",
        "dom_land_cover",
        "grass_frac",
        "shrub_frac",
        "wet_frac",
        "imp_frac",
        "fp_frac",
        "geol_class_1st",
        "geol_class_1st_frac",
        "geol_class_2nd",
        "carb_rocks_frac",
    ]

@pytest.fixture
def var_t():
    return [
        "precip_cr2met",
        "precip_chirps",
        "precip_mswep",
        "precip_tmpa",
        "tmin_cr2met",
        "tmax_cr2met",
        "tmean_cr2met",
        "pet_8d_modis",
        "pet_hargreaves",
        "swe",
    ]

@pytest.fixture
def camelscllsmt_args(var_c,var_t):
    project_name = os.path.join("test_camels", "lstm_camelscl"),
    # camels-cl time_range: ["1970-10-01", "2015-09-30"]
    train_period = ["2011-10-01", "2012-10-01"]
    valid_period = ["2012-10-01", "2013-10-01"]
    test_period = ["2013-10-01", "2014-10-01"]
    config_data = default_config_file()
    args = cmd(
        sub=project_name,
        source_cfgs={
            "source_name": "camels_cl",
            "source_path": os.path.join(
                SETTING["local_data_path"]["datasets-origin"], "camels", "camels_cl"
            ),
        },
        ctx=[-1],
        # model_name="KuaiLSTM",
        model_name="CpuLSTM",
        model_hyperparam={
            "n_input_features": len(var_c) + len(var_t),
            "n_output_features": 1,
            "n_hidden_states": 256,
        },
        loss_func="RMSESum",
        sampler="KuaiSampler",
        dataset="StreamflowDataset",
        scaler="DapengScaler",
        batch_size=50,
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
        train_epoch=1,
        save_epoch=1,
        model_loader={
            "load_way": "specified",
            "test_epoch": 1,
        },
        # the gage_id.txt file is set by the user, it must be the format like:
        # GAUGE_ID
        # 01013500
        # 01022500
        # ......
        # Then it can be read by pd.read_csv(gage_id_file, dtype={0: str}).iloc[:, 0].values to get the gage_id list
        gage_id_file="D:\\minio\\waterism\\datasets-origin\\camels\\camels_cl\\gage_id.txt",
        which_first_tensor="sequence",
    )
    update_cfg(config_data, args)
    return config_data


def test_camelscllstm(camelscllsmt_args):
    train_and_evaluate(camelscllsmt_args)
    print("All processes are finished!")
