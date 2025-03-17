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

VAR_C_CHOSEN_FROM_CAMELS_FR = [
    "top_altitude_mean",
    "top_slo_mean",
    "sta_area_snap",
    "top_drainage_density",
    "clc_2018_lvl1_1",
    "clc_2018_lvl2_11",
    "clc_2018_lvl3_111",
    "clc_1990_lvl1_1",
    "clc_2018_lvl1_2",
    "top_slo_ori_n",
    "top_slo_ori_ne",
    "top_slo_ori_e",
    "top_slo_flat",
    "top_slo_gentle",
    "top_slo_moderate",
    "top_slo_ori_se",
    "geo_py",
    "geo_pa",
]
VAR_T_CHOSEN_FROM_FR = [
    "tsd_prec",
    "tsd_pet_ou",
    "tsd_prec_solid_frac",
    "tsd_temp",
    "tsd_pet_pe",
    "tsd_pet_pm",
    "tsd_wind",
    "tsd_humid",
    "tsd_rad_dli",
    "tsd_rad_ssi",
    "tsd_swi_gr",
    "tsd_swi_isba",
    "tsd_swe_isba",
    "tsd_temp_min",
    "tsd_temp_max",
]


def run_normal_dl(
    project_name,
    gage_id_file,
    var_c=VAR_C_CHOSEN_FROM_CAMELS_FR,
    var_t=VAR_T_CHOSEN_FROM_FR,
    train_period=None,
    valid_period=None,
    test_period=None,
):
    if train_period is None:  # camels-fr time_range: ["1970-01-01", "2022-01-01"]
        train_period = ["2017-10-01", "2018-10-01"]
    if valid_period is None:
        valid_period = ["2018-10-01", "2019-10-01"]
    if test_period is None:
        test_period = ["2019-10-01", "2020-10-01"]
    config_data = default_config_file()
    args = cmd(
        sub=project_name,
        source_cfgs={
            "source_name": "camels_fr",
            "source_path": os.path.join(
                SETTING["local_data_path"]["datasets-origin"], "camels", "camels_fr"
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
run_normal_dl(os.path.join("test_camels", "lstm_camelsfr"), "D:\\minio\\waterism\\datasets-origin\\camels\\camels_fr\\gage_id.txt")
