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

VAR_C_CHOSEN_FROM_CAMELS_AUS = [
    "elev_mean",
    "mean_slope_pct",
    "catchment_area",
    "prop_forested",
    "nvis_grasses_n",
    "lc19_shrbsca",
    "lc01_extracti",
    "lc03_waterbo",
    "nvis_nodata_n",
    "carbnatesed",
    "metamorph",
    "oldrock",
    "lc11_wetlands",
    "claya",
    "sanda",
    "geol_prim",
    "geol_prim_prop",
]
VAR_T_CHOSEN_FROM_AUS = [
    "precipitation_AWAP",
    "et_morton_actual_SILO",
    "et_morton_point_SILO",
    "et_morton_wet_SILO",
    "et_short_crop_SILO",
    "et_tall_crop_SILO",
    "evap_morton_lake_SILO",
    "evap_pan_SILO",
    "evap_syn_SILO",
    "solarrad_AWAP",
    "tmax_AWAP",
    "tmin_AWAP",
    "vprp_AWAP",
    "mslp_SILO",
    "radiation_SILO",
    "rh_tmax_SILO",
    "rh_tmin_SILO",
    "tmax_SILO",
    "tmin_SILO",
    "vp_deficit_SILO",
    "vp_SILO",
]

@pytest.fixture
def camelsauslsmt_args(
    var_c = VAR_C_CHOSEN_FROM_CAMELS_AUS,
    var_t = VAR_T_CHOSEN_FROM_AUS,
):
    project_name = os.path.join("test_camels", "lstm_camelsaus"),
    # camels-aus time_range: ["1990-01-01", "2010-01-01"]
    train_period = ["2006-10-01", "2007-10-01"]
    valid_period = ["2007-10-01", "2008-10-01"]
    test_period = ["2008-10-01", "2009-10-01"]
    config_data = default_config_file()
    args = cmd(
        sub=project_name,
        source_cfgs={
            "source_name": "camels_aus",
            "source_path": os.path.join(
                SETTING["local_data_path"]["datasets-origin"], "camels", "camels_aus"
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
        gage_id_file="D:\\minio\\waterism\\datasets-origin\\camels\\camels_aus\\gage_id.txt",
        which_first_tensor="sequence",
    )
    update_cfg(config_data, args)
    return config_data


def test_camelsauslstm(camelsauslsmt_args):
    train_and_evaluate(camelsauslsmt_args)
    print("All processes are finished!")
