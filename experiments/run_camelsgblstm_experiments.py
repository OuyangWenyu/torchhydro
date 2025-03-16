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

VAR_C_CHOSEN_FROM_CAMELS_GB = [
    "elev_mean",
    "slope_fdc",
    "area",
    "shrub_perc",
    "dwood_perc",
    "organic_perc",
    "dom_land_cover",
    "dom_land_cover",
    "root_depth_50",
    "root_depth",
    "porosity_cosby",
    "conductivity_cosby_50",
    "soil_depth_pelletier",
    "urban_perc",
    "inwater_perc",
    "inter_mod_perc",
    "frac_mod_perc",
]
VAR_T_CHOSEN_FROM_GB= [
    "precipitation",
    "pet",
    "temperature",
    "peti",
    "humidity",
    "shortwave_rad",
    "longwave_rad",
    "windspeed",
]


def run_normal_dl(
    project_name,
    gage_id_file,
    var_c=VAR_C_CHOSEN_FROM_CAMELS_GB,
    var_t=VAR_T_CHOSEN_FROM_GB,
    train_period=None,
    valid_period=None,
    test_period=None,
):
    if train_period is None:  # camels-gb time_range: ["1970-10-01", "2015-09-30"]
        train_period = ["2011-10-01", "2012-10-01"]
    if valid_period is None:
        valid_period = ["2012-10-01", "2013-10-01"]
    if test_period is None:
        test_period = ["2013-10-01", "2014-10-01"]
    config_data = default_config_file()
    args = cmd(
        sub=project_name,
        source_cfgs={
            "source_name": "camels_gb",
            "source_path": os.path.join(
                SETTING["local_data_path"]["datasets-origin"], "camels", "camels_gb"
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
        batch_size=512,
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
        gage_id_file=gage_id_file,
        which_first_tensor="sequence",
    )
    update_cfg(config_data, args)
    train_and_evaluate(config_data)
    print("All processes are finished!")


# the gage_id.txt file is set by the user, it must be the format like:
# GAUGE_ID
# 01013500
# 01022500
# ......
# Then it can be read by pd.read_csv(gage_id_file, dtype={0: str}).iloc[:, 0].values to get the gage_id list
run_normal_dl(os.path.join("test_camels", "lstm_camelsgb"), "D:\\minio\\waterism\\datasets-origin\\camels\\camels_gb\\8344e4f3-d2ea-44f5-8afa-86d2987543a9\\8344e4f3-d2ea-44f5-8afa-86d2987543a9\\data\\gage_id.txt")
