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
        "scrub_perc",  # note: this field in original data file is different with its in data description pdf file, choose the former for convience.
        "mixed_wood_perc",  # note: this field in original data file is different with its in data description pdf file, choose the former for convience.
        "rock_perc",
        "dom_land_cover",
        "crop_perc",
        "root_depth_50",
        "root_depth",
        "porosity",
        "conductivity",
        "tot_avail_water",
        "unconsol_sediments",
        "siliciclastic_sedimentary",
        "geo_porosity",
        "geo_log10_permeability",
    ]

@pytest.fixture
def var_t():
    return [
        "precipitation",
        "waterlevel",
        "temperature_min",
        "temperature_mean",
        "temperature_max",
        "rel_sun_dur",
        "swe",
    ]

@pytest.fixture
def camelschlsmt_args(var_c, var_t):
    project_name = os.path.join("test_camels", "lstm_camelsch")
    # camels-ch time_range: ["1981-01-01", "2020-12-31"]
    train_period = ["2017-10-01", "2018-10-01"]
    valid_period = ["2018-10-01", "2019-10-01"]
    test_period = ["2019-10-01", "2020-10-01"]
    config_data = default_config_file()
    args = cmd(
        sub=project_name,
        source_cfgs={
            "source_name": "camels_ch",
            "source_path": os.path.join(
                SETTING["local_data_path"]["datasets-origin"], "camels", "camels_ch"
            ),
        },
        ctx=[-1],
        # model_name="KuaiLSTM",
        model_name="CpuLSTM",
        model_hyperparam={
            "n_input_features": len(var_t) + len(var_c),  # 7 + 17,
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
        # gage_id_file="D:\\minio\\waterism\\datasets-origin\\camels\\camels_ch\\gage_id.txt",
        gage_id_file=r"/mnt/d/minio/waterism/datasets-origin/camels/camels_ch/gage_id.txt",
        which_first_tensor="sequence",
    )
    update_cfg(config_data, args)
    return config_data


def test_camelschlstm(camelschlsmt_args):
    train_and_evaluate(camelschlsmt_args)
    print("All processes are finished!")
