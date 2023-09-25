"""
Author: Wenyu Ouyang
Date: 2022-09-09 14:47:42
LastEditTime: 2023-09-21 20:13:44
LastEditors: Wenyu Ouyang
Description: a script to run experiments for LSTM - CAMELS
FilePath: /torchhydro/experiments/run_camelslstm_experiments.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""
import os
import hydrodataset as hds
from configs.config import cmd, default_config_file, update_cfg
from trainers.trainer import train_and_evaluate

VAR_C_CHOSEN_FROM_CAMELS_US = [
    "elev_mean",
    "slope_mean",
    "area_gages2",
    "frac_forest",
    "lai_max",
    "lai_diff",
    "dom_land_cover_frac",
    "dom_land_cover",
    "root_depth_50",
    "soil_depth_statsgo",
    "soil_porosity",
    "soil_conductivity",
    "max_water_content",
    "geol_1st_class",
    "geol_2nd_class",
    "geol_porostiy",
    "geol_permeability",
]
VAR_T_CHOSEN_FROM_DAYMET = [
    "dayl",
    "prcp",
    "srad",
    "swe",
    "tmax",
    "tmin",
    "vp",
]


def run_normal_dl(
    project_name,
    var_c=VAR_C_CHOSEN_FROM_CAMELS_US,
    var_t=VAR_T_CHOSEN_FROM_DAYMET,
    train_period=None,
    valid_period=None,
    test_period=None,
    gage_id_file="/mnt/sdc/owen/code/HydroFL/scripts/camels531.csv",
):
    if train_period is None:
        train_period = ["1985-10-01", "1995-10-01"]
    if valid_period is None:
        valid_period = ["1995-10-01", "2000-10-01"]
    if test_period is None:
        test_period = ["2000-10-01", "2010-10-01"]
    config_data = default_config_file()
    args = cmd(
        sub=project_name,
        source="CAMELS",
        source_region="US",
        source_path=os.path.join(hds.ROOT_DIR, "camels", "camels_us"),
        download=0,
        ctx=[0],
        model_name="KuaiLSTM",
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
        rho=366,
        var_t=var_t,
        var_c=var_c,
        var_out=["streamflow"],
        train_period=train_period,
        valid_period=valid_period,
        test_period=test_period,
        opt="Adadelta",
        rs=1234,
        train_epoch=20,
        save_epoch=1,
        te=20,
        gage_id_file=gage_id_file,
        which_first_tensor="sequence",
    )
    update_cfg(config_data, args)
    train_and_evaluate(config_data)
    print("All processes are finished!")


run_normal_dl("ndl/explstm")
