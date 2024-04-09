"""
Author: Wenyu Ouyang
Date: 2023-04-06 14:45:34
LastEditTime: 2024-04-06 19:08:32
LastEditors: Wenyu Ouyang
Description: Test the multioutput model
FilePath: \torchhydro\tests\test_train_camelspro_mtl.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""

import numpy as np
import torch
import os

from torchhydro import SETTING
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.models.crits import MultiOutLoss, RMSELoss
from torchhydro.trainers.trainer import train_and_evaluate


def test_cuda_available():
    assert torch.cuda.is_available()


def test_multiout_loss_nan_gap():
    data0 = torch.tensor([2.0]).repeat(8, 3, 1)
    data1 = torch.tensor(np.full((8, 3, 1), np.nan))
    data1[0, 0, :] = 1.0
    data1[3, 0, :] = 2.0
    data1[6, 0, :] = 3.0
    data1[1, 1, :] = 4.0
    data1[4, 1, :] = 5.0
    data1[7, 1, :] = 6.0
    data1[2, 2, :] = 7.0
    data1[5, 2, :] = 8.0
    targ = torch.cat((data0, data1), dim=-1)
    pred = torch.tensor(np.full((8, 3, 2), 1.0))
    # device = torch.device('cpu')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    targ = targ.to(device)
    pred = pred.to(device)
    rmse = RMSELoss()
    r = MultiOutLoss(rmse, data_gap=[0, 1], device=[0], item_weight=[1, 1])
    # for sum, we ignore last interval
    expect_value = rmse(
        torch.tensor(np.array([1.0, 2.0, 4.0, 5.0, 7.0]).astype(float)),
        torch.tensor([3.0, 3.0, 3.0, 3.0, 3.0]),
    ) + rmse(data0, torch.tensor(np.full((8, 3, 1), 1.0)))
    np.testing.assert_almost_equal(
        r(pred, targ).cpu().detach().numpy(), expect_value.cpu().detach().numpy()
    )


def test_flow_et_mtl():
    """
    Test for data augmentation of flow -> et with KuaiLSTM

    Parameters
    ----------
    config_data

    Returns
    -------

    """
    project_name = "test_camels/expmtl001"
    data_origin_dir = SETTING["local_data_path"]["datasets-origin"]
    data_interim_dir = SETTING["local_data_path"]["datasets-interim"]
    args = cmd(
        sub=project_name,
        source_cfgs={
            "source_names": [
                "usgs4camels",
                "modiset4camels",
                "nldas4camels",
                "smap4camels",
            ],
            "source_paths": [
                os.path.join(data_origin_dir, "camels", "camels_us"),
                os.path.join(data_interim_dir, "camels_us", "modiset4camels"),
                os.path.join(data_interim_dir, "camels_us", "nldas4camels"),
                os.path.join(data_interim_dir, "camels_us", "smap4camels"),
            ],
        },
        ctx=[0],
        model_type="MTL",
        model_name="KuaiLSTMMultiOut",
        model_hyperparam={
            "n_input_features": 23,
            "n_output_features": 2,
            "n_hidden_states": 64,
            "layer_hidden_size": 32,
        },
        loss_func="MultiOutLoss",
        loss_param={
            "loss_funcs": "RMSESum",
            "data_gap": [0, 2],
            "device": [0],
            "item_weight": [1.0, 0.0],
            "limit_part": [1],
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
        rho=30,  # batch_size=100, rho=365,
        var_t=[
            "temperature",
            "specific_humidity",
            "shortwave_radiation",
            "potential_energy",
            "potential_evaporation",
            "total_precipitation",
        ],
        var_t_type=["nldas"],
        var_out=["streamflow", "ET"],
        var_to_source_map={
            "temperature": "nldas4camels",
            "specific_humidity": "nldas4camels",
            "shortwave_radiation": "nldas4camels",
            "potential_energy": "nldas4camels",
            "potential_evaporation": "nldas4camels",
            "total_precipitation": "nldas4camels",
            "streamflow": "usgs4camels",
            "ET": "modiset4camels",
            "elev_mean": "usgs4camels",
            "slope_mean": "usgs4camels",
            "area_gages2": "usgs4camels",
            "frac_forest": "usgs4camels",
            "lai_max": "usgs4camels",
            "lai_diff": "usgs4camels",
            "dom_land_cover_frac": "usgs4camels",
            "dom_land_cover": "usgs4camels",
            "root_depth_50": "usgs4camels",
            "soil_depth_statsgo": "usgs4camels",
            "soil_porosity": "usgs4camels",
            "soil_conductivity": "usgs4camels",
            "max_water_content": "usgs4camels",
            "geol_1st_class": "usgs4camels",
            "geol_2nd_class": "usgs4camels",
            "geol_porostiy": "usgs4camels",
            "geol_permeability": "usgs4camels",
        },
        train_period=["2015-04-01", "2016-04-01"],
        test_period=["2016-04-01", "2017-04-01"],
        dataset="FlexDataset",
        sampler="KuaiSampler",
        scaler="DapengScaler",
        n_output=2,
        train_epoch=20,
        te=20,
        fill_nan=["no", "mean"],
    )
    config_data = default_config_file()
    update_cfg(config_data, args)
    train_and_evaluate(config_data)
