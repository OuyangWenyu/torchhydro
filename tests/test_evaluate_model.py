"""
Author: Wenyu Ouyang
Date: 2023-09-18 14:34:53
LastEditTime: 2023-11-29 17:55:05
LastEditors: Wenyu Ouyang
Description: A simple evaluate model test
FilePath: \torchhydro\tests\test_evaluate_model.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""
import os
import hydrodataset as hds
import pytest
from hydroutils.hydro_file import get_lastest_file_in_a_dir
from hydroutils.hydro_plot import plot_ts
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.datasets.data_dict import data_sources_dict
from torchhydro.trainers.deep_hydro import DeepHydro
from torchhydro.trainers.trainer import set_random_seed


@pytest.fixture()
def config_data():
    project_name = "test_camels/exp1"
    weight_dir = os.path.join(
        os.getcwd(),
        "results",
        "test_camels",
        "exp1",
    )
    weight_path = get_lastest_file_in_a_dir(weight_dir)
    args = cmd(
        sub=project_name,
        download=0,
        source_path=os.path.join(hds.ROOT_DIR, "camels", "camels_us"),
        source_region="US",
        ctx=[-1],
        model_name="CpuLSTM",
        model_hyperparam={
            "n_input_features": 23,
            "n_output_features": 1,
            "n_hidden_states": 256,
        },
        gage_id=[
            "01013500",
            "01022500",
            "01030500",
            "01031500",
            "01047000",
            "01052500",
            "01054200",
            "01055000",
            "01057000",
            "01170100",
        ],
        batch_size=5,
        rho=20,  # batch_size=100, rho=365,
        var_t=["dayl", "prcp", "srad", "tmax", "tmin", "vp"],
        var_out=["streamflow"],
        dataset="StreamflowDataset",
        scaler="DapengScaler",
        train_epoch=5,
        save_epoch=1,
        te=5,
        train_period=["2000-10-01", "2001-10-01"],
        test_period=["2001-10-01", "2002-10-01"],
        loss_func="RMSESum",
        opt="Adadelta",
        which_first_tensor="sequence",
        weight_path=weight_path,
        continue_train=False,
    )
    config_data = default_config_file()
    update_cfg(config_data, args)
    return config_data


def test_evaluate_model(config_data):
    random_seed = config_data["training_cfgs"]["random_seed"]
    set_random_seed(random_seed)
    data_cfgs = config_data["data_cfgs"]
    data_source_name = data_cfgs["data_source_name"]
    data_source = data_sources_dict[data_source_name](
        data_cfgs["data_path"], data_cfgs["download"]
    )
    model = DeepHydro(data_source, config_data)
    eval_log, preds_xr, obss_xr = model.model_evaluate()
    print(eval_log)
    plot_ts(
        [preds_xr["time"].values, obss_xr["time"].values],
        [
            preds_xr["streamflow"].sel(basin="01013500").values,
            obss_xr["streamflow"].sel(basin="01013500").values,
        ],
        leg_lst=["pred", "obs"],
        fig_size=(6, 4),
        xlabel="Date",
        ylabel="Streamflow",
    )
