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

VAR_C_CHOSEN_FROM_CAMELS_IND = [
    "elev_mean",
    "slope_mean",
    "cwc_area",
    "trees_frac",
    "lai_max",
    "lai_diff",
    "dom_land_cover",
    "dom_land_cover_frac",
    "crops_frac",
    "soil_depth",
    "soil_awc_top",
    "soil_conductivity_top",
    "water_frac",
    "geol_class_1st",
    "geol_class_2nd",
    "geol_porosity",
    "geol_permeability",
]
VAR_T_CHOSEN_FROM_IND = [
    "prcp",
    "pet",
    "tmax",
    "tmin",
    "tavg",
    "srad_lw",
    "srad_sw",
    "wind_u",
    "wind_v",
    "wind",
    "rel_hum",
    "pet_gleam",
    "aet_gleam",
    "evap_canopy",
    "evap_surface",
    "sm_lvl1",
    "sm_lvl2",
    "sm_lvl3",
    "sm_lvl4",
]

@pytest.fixture
def camelsindlsmt_args(
    var_c = VAR_C_CHOSEN_FROM_CAMELS_IND,
    var_t = VAR_T_CHOSEN_FROM_IND,
):
    project_name = os.path.join("test_camels", "lstm_camelsind"),
    # camels-ind time_range: ["1981-01-01", "2020-12-31"]
    train_period = ["2017-10-01", "2018-10-01"]
    valid_period = ["2018-10-01", "2019-10-01"]
    test_period = ["2019-10-01", "2020-10-01"]
    config_data = default_config_file()
    args = cmd(
        sub=project_name,
        source_cfgs={
            "source_name": "camels_ind",
            "source_path": os.path.join(
                SETTING["local_data_path"]["datasets-origin"], "camels", "camels_ind"
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
        gage_id_file="D:\\minio\\waterism\\datasets-origin\\camels\\camels_ind\\gage_id.txt",
        which_first_tensor="sequence",
    )
    update_cfg(config_data, args)
    return config_data


def test_camelsindlstm(camelsindlsmt_args):
    train_and_evaluate(camelsindlsmt_args)
    print("All processes are finished!")
