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

import HydroErr as he
import hydrodataset as hds
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchhydro.datasets.config import cmd, default_config_file, update_cfg
from torchhydro.datasets.data_dict import data_sources_dict
from torchhydro.datasets.data_sets import DplDataset
from torchhydro.models.dpl4xaj import DplLstmXaj
from torchhydro.trainers.trainer import set_random_seed


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
        data_loader="DplDataset",
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
    )
    config_data = default_config_file()
    update_cfg(config_data, args)
    return config_data


@pytest.fixture()
def config_data1():
    args = cmd(gage_id=[
        "01078000",
    ],
        var_t=["dayl", "prcp", "srad", "tmax", "tmin", "vp"],
        var_out=["streamflow"],
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
        }, )
    config_data1 = default_config_file()
    update_cfg(config_data1, args)
    return config_data1


@pytest.fixture()
def device():
    return "cuda:0" if torch.cuda.is_available() else "cpu"


@pytest.fixture()
def dpl(device):
    dpl_ = DplLstmXaj(5, 15, 18, kernel_size=15, warmup_length=0)
    return dpl_.to(device)


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


def test_train_model(config_data, dpl, config_data1):
    seed = int(config_data["training_params"]["random_seed"])
    set_random_seed(seed)
    data_source_name = config_data['data_params']['data_source_name']
    data_source = data_sources_dict[data_source_name]()
    # Sample hyper-parameters for learning rate, batch size, seed and a few other HPs
    num_epochs = int(config_data["training_params"]["epochs"])
    train_dataloader, eval_dataloader = DataLoader(
        DplDataset(data_source, config_data['data_params'], 'train'),
        batch_size=config_data["training_params"]["batch_size"],
        shuffle=True,
        sampler=None,
        batch_sampler=None,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
    ), DataLoader(
        DplDataset(data_source, config_data1['data_params'], 'test'),
        batch_size=config_data1["training_params"]["batch_size"],
        shuffle=True,
        sampler=None,
        batch_sampler=None,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
    )
    # Instantiate the model (we build the model here so that the seed also control new weights initialization)
    optimizer = config_data["training_params"]["optimizer"]
    loss_func = config_data["training_params"]["criterion"]
    # Now we train the model
    for epoch in range(num_epochs):
        train_epoch(dpl, optimizer, train_dataloader, loss_func, epoch)
        obs, preds = eval_model(dpl, eval_dataloader)
        preds = eval_dataloader.dataset.local_denormalization(
            preds.cpu().numpy(), variable="streamflow"
        )
        obs = obs.cpu().numpy().reshape(2, -1)
        preds = preds.reshape(2, -1)
        nse = np.array([he.nse(preds[i], obs[i]) for i in range(obs.shape[0])])
        tqdm.write(f"epoch {epoch} -- Validation NSE mean: {nse.mean():.2f}")
