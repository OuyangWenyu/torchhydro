"""
Author: Wenyu Ouyang
Date: 2023-09-18 14:34:53
LastEditTime: 2024-04-11 09:06:09
LastEditors: Wenyu Ouyang
Description: A simple evaluate model test
FilePath: \torchhydro\tests\test_evaluate_model.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import pytest

from hydroutils.hydro_plot import plot_ts
from torchhydro.configs.config import update_cfg, default_config_file, cmd
from torchhydro.trainers.deep_hydro import DeepHydro
from torchhydro.trainers.trainer import set_random_seed


@pytest.fixture()
def config_data():
    return default_config_file()

@pytest.fixture()
def args(config_data):
    # A basic config, but it's not enough as it needs a real model path
    project_name = "test_evaluate_model/camels_lstm"
    args = cmd(sub=project_name, train_mode=False)
    return args

@pytest.fixture()
def mtl_args(config_data):
    # A basic config, but it's not enough as it needs a real model path
    project_name = "test_evaluate_model/camels_mtl_lstm"
    args = cmd(sub=project_name, train_mode=False)
    return args

@pytest.fixture()
def camelslstm_config(args, config_data):
    args.model_loader = {"load_way": "latest"}
    args.train_mode = False
    update_cfg(config_data, args)
    return config_data


@pytest.fixture()
def camelsmtllstm_config(mtl_args, config_data):
    mtl_args.model_loader = {"load_way": "latest"}
    mtl_args.train_mode = False
    update_cfg(config_data, mtl_args)
    return config_data


@pytest.fixture
def _config(request):
    return request.getfixturevalue(request.param)


@pytest.mark.skip(reason="TODO: Integration test requiring pre-trained models, skipping.")
@pytest.mark.parametrize(
    "_config", ["camelslstm_config", "camelsmtllstm_config"], indirect=True
)
def test_evaluate_model(_config):
    random_seed = _config["training_cfgs"]["random_seed"]
    set_random_seed(random_seed)
    model = DeepHydro(_config)
    preds_xr, obss_xr = model.model_evaluate()
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