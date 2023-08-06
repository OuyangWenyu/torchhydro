"""
Uncompleted, Please don't run it
Author: Wenyu Ouyang
Date: 2023-06-02 17:17:51
LastEditTime: 2023-06-02 17:21:47
LastEditors: Wenyu Ouyang
Description: Test for dpl4xaj
FilePath: /hydro-model-xaj/test/test_dpl4xaj.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""
import os

import hydrodataset as hds
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchhydro.datasets.config import cmd, default_config_file, update_cfg
from torchhydro.datasets.data_dict import data_sources_dict
from torchhydro.datasets.data_sets import KuaiDataset
from torchhydro.models.dpl4xaj import DplLstmXaj
from torchhydro.trainers.trainer import set_random_seed, train_and_evaluate
import HydroErr as he


@pytest.fixture()
def config_data():
    project_name = "test_camels/exp3"
    args = cmd(
        sub=project_name,
        download=0,
        source_path=os.path.join(hds.ROOT_DIR, "camels", "camels_us"),
        source_region="US",
        ctx=[0],
        model_name="DplLstmXaj",
        model_param={
            "n_input_features": 23,
            "n_output_features": 1,
            "n_hidden_states": 32,
        },
        gage_id=[
            "01013500",
        ],
        batch_size=5,
        rho=18,  # batch_size=100, rho=365,
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
        # loss_func="RMSESum",
        which_first_tensor="sequence",
    )
    config_data = default_config_file()
    update_cfg(config_data, args)
    return config_data


@pytest.fixture()
def device():
    return "cuda:0" if torch.cuda.is_available() else "cpu"


@pytest.fixture()
def dpl(device):
    dpl_ = DplLstmXaj(5, 15, 18, kernel_size=15, warmup_length=0)
    return dpl_.to(device)


def test_dpl_lstm_xaj(config_data, device, dpl):
    train_and_evaluate(config_data)
    '''
    # sequence-first tensor: time_sequence, batch, feature_size (assume that they are p, pet, srad, tmax, tmin)
    random_seed = config_data["training_params"]["random_seed"]
    set_random_seed(random_seed)
    data_params = config_data["data_params"]
    data_source_name = data_params["data_source_name"]
    if data_source_name in ["CAMELS", "CAMELS_SERIES"]:
        # there are many different regions for CAMELS datasets
        data_source = data_sources_dict[data_source_name](
            data_params["data_path"],
            data_params["download"],
            data_params["data_region"],
        )
    else:
        data_source = data_sources_dict[data_source_name](
            data_params["data_path"], data_params["download"]
        )
    train_kuai_ds = KuaiDataset(data_source, data_params, 'train')
    train_batch_size = data_params['batch_size']
    tr_loader = DataLoader(train_kuai_ds, batch_size=train_batch_size, shuffle=True)
    val_batch_size = 5*train_batch_size
    val_loader = DataLoader(train_kuai_ds, batch_size=val_batch_size)
    learning_rate = 1e-5
    optimizer = torch.optim.Adam(dpl.parameters(), lr=learning_rate)
    n_epochs = 5  # Number of training epochs
    loss_func = nn.MSELoss()
    for i in range(n_epochs):
        train_epoch(dpl, optimizer, tr_loader, loss_func, i + 1)
        obs, preds = eval_model(dpl, val_loader)
        preds_df = preds.squeeze(-1).cpu().numpy()
        obs_df = obs.squeeze(-1).cpu().numpy()
        # obs = obs.cpu().numpy().reshape(basins_num, -1)
        # preds = preds.values.reshape(basins_num, -1)
        print(preds_df)
        print(obs_df)
        nse = np.array([he.nse(preds_df[i], obs_df[i]) for i in range(obs.shape[0])])
        tqdm.write(f"Validation NSE mean: {nse.mean():.2f}")
    print('_________________________________________')


def train_epoch(model, optimizer, loader, loss_func, epoch):
    """Train model for a single epoch"""
    # set model to train mode (important for dropout)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.train()
    pbar = tqdm(loader)
    pbar.set_description(f"Epoch {epoch}")
    # request mini-batch of data from the loader
    for xs, ys in pbar:
        # delete previously stored gradients from the model
        optimizer.zero_grad()
        # push data to GPU (if available)
        xs, ys = xs.to(device), ys.to(device)
        # get model predictions
        y_hat = model(xs, ys)
        # calculate loss
        loss = loss_func(y_hat, ys)
        # calculate gradients
        loss.backward()
        # update the weights
        optimizer.step()
        # write current loss in the progress bar
        pbar.set_postfix_str(str(loss))


def eval_model(model, loader):
    """Evaluate the model"""
    # set model to eval mode (important for dropout)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.eval()
    obs = []
    preds = []
    # in inference mode, we don't need to store intermediate steps for
    # backprob
    with torch.no_grad():
        # request mini-batch of data from the loader
        for xs, ys in loader:
            # push data to GPU (if available)
            xs, ys = xs.to(device), ys.to(device)
            # get model predictions
            y_hat = model(xs, ys)
            obs.append(ys)
            preds.append(y_hat)
    return torch.cat(obs), torch.cat(preds)
'''
