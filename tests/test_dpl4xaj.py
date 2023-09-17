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
from functools import reduce

import HydroErr as he
import hydrodataset as hds
import numpy as np
import pytest
import torch
from numpy import isnan
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.datasets.data_dict import data_sources_dict
from torchhydro.datasets.data_sets import DplDataset
from torchhydro.models.dpl4xaj import DplLstmXaj
from torchhydro.models.model_dict_function import pytorch_opt_dict, pytorch_criterion_dict
from torchhydro.trainers.evaluator import evaluate_model
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
            "n_output_features": 15,
            "n_hidden_states": 64,
        },
        gage_id=[
            "01013500",
        ],
        batch_size=16,
        warmup_length=8,
        rho=5,  # batch_size=100, rho=365,
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
def device():
    return "cuda:0" if torch.cuda.is_available() else "cpu"


@pytest.fixture()
def dpl(device, config_data):
    dpl_ = DplLstmXaj(23, 15, 64, kernel_size=15, warmup_length=8)
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
        if type(xs) is list:
            xs = [data_tmp.permute(1, 0, 2) for data_tmp in xs]
            # Will bring error
            for data_tmp_np in xs:
                data_tmp_np[isnan(data_tmp_np)] = 0
            xs = [data_tmp.to(device) for data_tmp in xs]
        else:
            xs = xs.to(device)
        ys = ys.permute(1, 0, 2).to(device)
        # get model predictions
        output = model(*xs)
        # y_hat = model(xs, ys)
        # calculate loss
        ys_rho = ys[model.pb_model.warmup_length:, :, :]
        loss = loss_func(output, ys_rho)
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
            if type(xs) is list:
                xs = [data_tmp.permute(1, 0, 2) for data_tmp in xs]
                for data_tmp_np in xs:
                    data_tmp_np[isnan(data_tmp_np)] = 0
                xs = [data_tmp.to(device) for data_tmp in xs]
            else:
                xs = xs.to(device)
            # push data to GPU (if available)
            ys = ys.permute(1, 0, 2).to(device)
            # get model predictions
            output = model(*xs)
            # y_hat = model(xs, ys)
            ys_rho = ys[model.pb_model.warmup_length:, :, :]
            obs.append(ys_rho)
            preds.append(output)
    return torch.cat(obs[:-1]), torch.cat(preds[:-1])


def test_train_model(config_data, dpl):
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
        DplDataset(data_source, config_data['data_params'], 'test'),
        batch_size=config_data["training_params"]["batch_size"],
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
    )
    # Instantiate the model (we build the model here so that the seed also control new weights initialization)
    optimizer = pytorch_opt_dict[config_data["training_params"]["optimizer"]](params=dpl.parameters())
    loss_func = pytorch_criterion_dict[config_data["training_params"]["criterion"]]()
    # Now we train the model
    for epoch in range(num_epochs):
        train_epoch(dpl, optimizer, train_dataloader, loss_func, epoch)
        obs, preds = eval_model(dpl, eval_dataloader)
        preds = preds.cpu().numpy().reshape(2, -1)
        obs = obs.cpu().numpy().reshape(2, -1)
        nse = np.array([he.nse(preds[i], obs[i]) for i in range(obs.shape[0])])
        tqdm.write(f"epoch {epoch} -- Validation NSE mean: {nse.mean():.2f}")


def test_evaluate_model(config_data, dpl, device):
    data_source = data_sources_dict['CAMELS']()
    dataloader = DataLoader(
        DplDataset(data_source, config_data['data_params'], 'test'),
        batch_size=config_data["training_params"]["batch_size"],
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
    )
    preds, obs = generate_predictions(dpl, dataloader, True, torch.device(device), config_data['data_params'])
    eval_log, preds_xr, obss_xr = evaluate_model(dpl)
    print(eval_log, preds_xr, obss_xr)


def generate_predictions(
    model,
    test_dataloader,
    seq_first: bool,
    device: torch.device,
    data_params: dict,
    return_cell_state: bool = False,
) -> np.ndarray:
    """Perform Evaluation on the test (or valid) data.

    Parameters
    ----------
    model : DplLstmXaj
        _description_
    test_model : TestDataModel
        _description_
    seq_first
        _description_
    device : torch.device
        _description_
    data_params : dict
        _description_
    return_cell_state : bool, optional
        if True, time-loop evaluation for cell states, by default False
        NOTE: ONLY for LSTM models

    Returns
    -------
    np.ndarray
        _description_
    """
    model.train(mode=False)
    # here the batch is just an index of lookup table, so any batch size could be chosen
    test_preds = []
    obss = []
    with torch.no_grad():
        for i_batch, (xs, ys) in enumerate(test_dataloader):
            # here the a batch doesn't mean a basin; it is only an index in lookup table
            # for NtoN mode, only basin is index in lookup table, so the batch is same as basin
            # for Nto1 mode, batch is only an index
            if seq_first:
                xs = xs.transpose(0, 1)
                ys = ys.transpose(0, 1)
            xs = xs.to(device)
            ys = ys.to(device)
            output = model(xs)
            if type(output) is tuple:
                others = output[1:]
                # Convention: y_p must be the first output of model
                output = output[0]
            if seq_first:
                output = output.transpose(0, 1)
                ys = ys.transpose(0, 1)
            test_preds.append(output.cpu().numpy())
            obss.append(ys.cpu().numpy())
        pred = reduce(lambda x, y: np.vstack((x, y)), test_preds)
        obs = reduce(lambda x, y: np.vstack((x, y)), obss)
    if pred.ndim == 2:
        # TODO: check
        # the ndim is 2 meaning we use an Nto1 mode
        # as lookup table is (basin 1's all time length, basin 2's all time length, ...)
        # params of reshape should be (basin size, time length)
        pred = pred.flatten().reshape(test_dataloader.test_data.y.shape[0], -1, 1)
        obs = obs.flatten().reshape(test_dataloader.test_data.y.shape[0], -1, 1)
    return pred, obs
