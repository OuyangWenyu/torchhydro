"""
Author: Wenyu Ouyang
Date: 2023-04-06 14:45:34
LastEditTime: 2024-04-10 20:23:36
LastEditors: Wenyu Ouyang
Description: Test the multioutput model
FilePath: \torchhydro\tests\test_train_camelspro_mtl.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""

import numpy as np
import torch

from torchhydro.configs.config import update_cfg
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


def test_flow_et_mtl(mtl_args, config_data):
    """
    Test for data augmentation of flow -> et with KuaiLSTM

    Parameters
    ----------
    config_data

    Returns
    -------

    """
    update_cfg(config_data, mtl_args)
    train_and_evaluate(config_data)
