"""
Author: Wenyu Ouyang
Date: 2023-06-02 17:17:51
LastEditTime: 2023-06-02 17:21:47
LastEditors: Wenyu Ouyang
Description: Test for dpl4xaj
FilePath: /hydro-model-xaj/test/test_dpl4xaj.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""
import os

import pytest
import torch
from hydroutils.hydro_file import get_lastest_file_in_a_dir

from torchhydro.datasets.config import cmd, default_config_file, update_cfg
from torchhydro.datasets.data_dict import data_sources_dict
from torchhydro.models.dpl4xaj import DplLstmXaj
import hydrodataset as hds

from torchhydro.trainers.trainer import set_random_seed


@pytest.fixture()
def config_data():
    project_name = "test_camels/exp2"
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
        ctx=[0],
        model_name="KuaiLSTM",
        model_param={
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
        data_loader="KuaiDataset",
        scaler="DapengScaler",
        scaler_params={
            "prcp_norm_cols": ["streamflow"],
            "gamma_norm_cols": [
                "prcp",
                "pr",
                "total_precipitation",
                "potential_evaporation",
                "ET",
                "PET",
                "ET_sum",
                "ssm",
            ],
        },
        train_epoch=5,
        save_epoch=1,
        te=5,
        train_period=["2000-10-01", "2001-10-01"],
        test_period=["2001-10-01", "2002-10-01"],
        loss_func="RMSESum",
        opt="Adadelta",
        which_first_tensor="sequence",
        weight_path=weight_path,
    )
    config_data = default_config_file()
    update_cfg(config_data, args)
    return config_data


@pytest.fixture()
def device():
    return "cuda:0" if torch.cuda.is_available() else "cpu"


@pytest.fixture()
def dpl(device):
    dpl_ = DplLstmXaj(5, 15, 64, kernel_size=15, warmup_length=5)
    return dpl_.to(device)


def test_dpl_lstm_xaj(config_data, device, dpl):
    # sequence-first tensor: time_sequence, batch, feature_size (assume that they are p, pet, srad, tmax, tmin)
    random_seed = config_data["training_params"]["random_seed"]
    set_random_seed(random_seed)
    data_params = config_data["data_params"]
    data_source_name = data_params["data_source_name"]
    data_source = data_sources_dict[data_source_name](
        data_params["data_path"], data_params["download"]
    )
    q = dpl(
        config_data["model_params"]["model_name"], data_source, config_data
    )
    print(q)
    '''
    x = torch.rand(20, 10, 5).to(device)
    z = torch.rand(20, 10, 5).to(device)
    q = dpl(x, z)
    assert len(q.shape) == 3
    assert q.shape == (15, 10, 1)
    assert type(q) == torch.Tensor
    '''


'''
def test_uh_gamma():
    # batch = 10
    tempa = torch.Tensor(np.full(10, [2.5]))
    tempb = torch.Tensor(np.full(10, [3.5]))
    # repeat for 20 periods and add one dim as feature: time_seq, batch, feature
    routa = tempa.repeat(20, 1).unsqueeze(-1)
    routb = tempb.repeat(20, 1).unsqueeze(-1)
    uh = uh_gamma(routa, routb, len_uh=15)
    np.testing.assert_almost_equal(
        uh.numpy()[:, 0, :],
        np.array(
            [
                [0.0069],
                [0.0314],
                [0.0553],
                [0.0738],
                [0.0860],
                [0.0923],
                [0.0939],
                [0.0919],
                [0.0875],
                [0.0814],
                [0.0744],
                [0.0670],
                [0.0597],
                [0.0525],
                [0.0459],
            ]
        ),
        decimal=3,
    )


def test_uh():
    uh_from_gamma = torch.full((5, 3, 1), 1.0)
    rf = torch.Tensor(np.arange(30).reshape(10, 3, 1) / 100)
    qs = uh_conv(rf, uh_from_gamma)
    np.testing.assert_almost_equal(
        np.array(
            [
                [0.0000, 0.0100, 0.0200],
                [0.0300, 0.0500, 0.0700],
                [0.0900, 0.1200, 0.1500],
                [0.1800, 0.2200, 0.2600],
                [0.3000, 0.3500, 0.4000],
                [0.4500, 0.5000, 0.5500],
                [0.6000, 0.6500, 0.7000],
                [0.7500, 0.8000, 0.8500],
                [0.9000, 0.9500, 1.0000],
                [1.0500, 1.1000, 1.1500],
            ]
        ),
        qs.numpy()[:, :, 0],
        decimal=3,
    )
'''
