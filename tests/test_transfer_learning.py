"""
Author: Wenyu Ouyang
Date: 2023-10-05 16:16:48
LastEditTime: 2024-02-14 16:12:50
LastEditors: Wenyu Ouyang
Description: A test function for transfer learning
FilePath: \torchhydro\tests\test_transfer_learning.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""
import os
import pytest
from hydroutils.hydro_file import get_lastest_file_in_a_dir

from torchhydro import SETTING
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate


@pytest.fixture()
def var_c_target():
    return [
        "elev_mean",
        "slope_mean",
        "area_gages2",
        "frac_forest",
        "lai_max",
        "lai_diff",
    ]


@pytest.fixture()
def var_c_source():
    return [
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


@pytest.fixture()
def var_t_target():
    return ["dayl", "prcp", "srad"]


@pytest.fixture()
def var_t_source():
    return ["dayl", "prcp", "srad", "tmax", "tmin", "vp"]


def test_transfer_gages_lstm_model(
    var_c_source, var_c_target, var_t_source, var_t_target
):
    weight_dir = os.path.join(
        os.getcwd(),
        "results",
        "test_camels",
        "exp1",
    )
    weight_path = get_lastest_file_in_a_dir(weight_dir)
    project_name = "test_camels/exp4"
    args = cmd(
        sub=project_name,
        source="CAMELS",
        source_path=os.path.join(
            SETTING["local_data_path"]["datasets-origin"], "camels", "camels_us"
        ),
        source_region="US",
        ctx=[0],
        model_type="TransLearn",
        model_name="KaiLSTM",
        model_hyperparam={
            "linear_size": len(var_c_target) + len(var_t_target),
            "n_input_features": len(var_c_source) + len(var_t_source),
            "n_output_features": 1,
            "n_hidden_states": 256,
        },
        opt="Adadelta",
        loss_func="RMSESum",
        batch_size=5,
        forecast_history=0,
        forecast_length=20,
        rs=1234,
        train_period=["2010-10-01", "2011-10-01"],
        test_period=["2011-10-01", "2012-10-01"],
        scaler="DapengScaler",
        sampler="KuaiSampler",
        dataset="StreamflowDataset",
        weight_path=weight_path,
        weight_path_add={
            "freeze_params": ["lstm.b_hh", "lstm.b_ih", "lstm.w_hh", "lstm.w_ih"]
        },
        continue_train=True,
        train_epoch=20,
        te=20,
        save_epoch=10,
        var_t=var_t_target,
        var_c=var_c_target,
        var_out=["streamflow"],
        gage_id=[
            "01055000",
            "01057000",
            "01170100",
        ],
    )
    cfg = default_config_file()
    update_cfg(cfg, args)
    train_and_evaluate(cfg)
    print("All processes are finished!")
