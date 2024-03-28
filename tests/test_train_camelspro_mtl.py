"""
Author: Wenyu Ouyang
Date: 2023-04-06 14:45:34
LastEditTime: 2024-03-28 19:22:41
LastEditors: Wenyu Ouyang
Description: Test the multioutput model
FilePath: \torchhydro\tests\test_mtl.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""

import numpy as np
import torch
import os

from torchhydro import SETTING
from torchhydro.configs.config import cmd, update_cfg
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


def test_flow_et_mtl(config_data):
    """
    Test for data augmentation of flow -> et with KuaiLSTM

    Parameters
    ----------
    config_data

    Returns
    -------

    """
    project_name = "test_camels/expmtl001"
    data_dir = SETTING["local_data_path"]["datasets-origin"]
    args = cmd(
        sub=project_name,
        source="CAMELS_FLOW_ET",
        source_path=[
            os.path.join(data_dir, "camelsflowet"),
            os.path.join(data_dir, "modiset4camels"),
            os.path.join(data_dir, "camels", "camels_us"),
            os.path.join(data_dir, "nldas4camels"),
            os.path.join(data_dir, "smap4camels"),
        ],
        download=0,
        ctx=[0],
        model_type="MTL",
        model_name="KuaiLSTMMultiOut",
        model_param={
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
        cache_write=1,
        cache_read=1,
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
        var_out=["usgsFlow", "ET"],
        train_period=["2015-04-01", "2016-04-01"],
        test_period=["2016-04-01", "2017-04-01"],
        data_loader="StreamflowDataModel",
        scaler="DapengScaler",
        n_output=2,
        train_epoch=20,
        te=20,
        fill_nan=["no", "mean"],
    )
    update_cfg(config_data, args)
    train_and_evaluate(config_data)
