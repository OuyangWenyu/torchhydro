"""
Author: Wenyu Ouyang
Date: 2023-09-21 15:06:12
LastEditTime: 2023-10-03 18:04:22
LastEditors: Wenyu Ouyang
Description: Some basic functions for training
FilePath: \torchhydro\torchhydro\trainers\train_utils.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""


import copy
import os
from functools import reduce
from hydroutils.hydro_stat import stat_error
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import xarray as xr

from torchhydro.models.crits import GaussianLoss


def model_infer(seq_first, device, model, xs, ys):
    """_summary_

    Parameters
    ----------
    seq_first : _type_
        _description_
    device : _type_
        _description_
    model : _type_
        _description_
    xs : list or tensor
        xs is always batch first
    ys : tensor
        _description_

    Returns
    -------
    _type_
        _description_
    """
    if type(xs) is list:
        xs = [
            data_tmp.permute([1, 0, 2]).to(device)
            if seq_first and data_tmp.ndim == 3
            else data_tmp.to(device)
            for data_tmp in xs
        ]
    else:
        xs = [
            xs.permute([1, 0, 2]).to(device)
            if seq_first and xs.ndim == 3
            else xs.to(device)
        ]
    ys = (
        ys.permute([1, 0, 2]).to(device)
        if seq_first and ys.ndim == 3
        else ys.to(device)
    )
    output = model(*xs)
    if type(output) is tuple:
        # Convention: y_p must be the first output of model
        output = output[0]
    if seq_first:
        output = output.transpose(0, 1)
        ys = ys.transpose(0, 1)
    return ys, output


def denormalize4eval(validation_data_loader, output, labels):
    target_scaler = validation_data_loader.dataset.target_scaler
    target_data = target_scaler.data_target
    # the units are dimensionless for pure DL models
    units = {k: "dimensionless" for k in target_data.attrs["units"].keys()}
    if target_scaler.pbm_norm:
        units = {**units, **target_data.attrs["units"]}
    # need to remove data in the warmup period
    warmup_length = validation_data_loader.dataset.warmup_length
    selected_time_points = target_data.coords["time"][warmup_length:]
    selected_data = target_data.sel(time=selected_time_points)
    preds_xr = target_scaler.inverse_transform(
        xr.DataArray(
            output.transpose(2, 0, 1),
            dims=selected_data.dims,
            coords=selected_data.coords,
            attrs={"units": units},
        )
    )
    obss_xr = target_scaler.inverse_transform(
        xr.DataArray(
            labels.transpose(2, 0, 1),
            dims=selected_data.dims,
            coords=selected_data.coords,
            attrs={"units": units},
        )
    )

    return preds_xr, obss_xr


class EarlyStopper(object):
    def __init__(
        self,
        patience: int,
        min_delta: float = 0.0,
        cumulative_delta: bool = False,
    ):
        """
        EarlyStopping handler can be used to stop the training if no improvement after a given number of events.

        Parameters
        ----------
        patience
            Number of events to wait if no improvement and then stop the training.
        min_delta
            A minimum increase in the score to qualify as an improvement,
            i.e. an increase of less than or equal to `min_delta`, will count as no improvement.
        cumulative_delta
            It True, `min_delta` defines an increase since the last `patience` reset, otherwise,
        it defines an increase after the last event. Default value is False.
        """

        if patience < 1:
            raise ValueError("Argument patience should be positive integer.")

        if min_delta < 0.0:
            raise ValueError("Argument min_delta should not be a negative number.")

        self.patience = patience
        self.min_delta = min_delta
        self.cumulative_delta = cumulative_delta
        self.counter = 0
        self.best_score = None

    def check_loss(self, model, validation_loss) -> bool:
        score = validation_loss
        if self.best_score is None:
            self.save_model_checkpoint(model)
            self.best_score = score

        elif score + self.min_delta >= self.best_score:
            if not self.cumulative_delta and score > self.best_score:
                self.best_score = score
            self.counter += 1
            print(self.counter)
            if self.counter >= self.patience:
                return False
        else:
            self.save_model_checkpoint(model)
            self.best_score = score
            self.counter = 0
        return True

    def save_model_checkpoint(self, model):
        torch.save(model.state_dict(), "checkpoint.pth")


def evaluate_validation(
    validation_data_loader, output, labels, evaluation_metrics, fill_nan, target_col
):
    """
    calculate metrics for validation

    Parameters
    ----------
    output
        model output
    labels
        model target
    evaluation_metrics
        metrics to evaluate
    fill_nan
        how to fill nan
    target_col
        target columns

    Returns
    -------
    tuple
        metrics
    """
    if type(fill_nan) is list and len(fill_nan) != len(target_col):
        raise ValueError("length of fill_nan must be equal to target_col's")
    eval_log = {}
    # renormalization to get real metrics
    preds_xr, obss_xr = denormalize4eval(validation_data_loader, output, labels)
    for i in range(len(target_col)):
        obs_xr = obss_xr[list(obss_xr.data_vars.keys())[i]]
        pred_xr = preds_xr[list(preds_xr.data_vars.keys())[i]]
        if type(fill_nan) is str:
            inds = stat_error(
                obs_xr.to_numpy(),
                pred_xr.to_numpy(),
                fill_nan,
            )
        else:
            inds = stat_error(
                obs_xr.to_numpy(),
                pred_xr.to_numpy(),
                fill_nan[i],
            )
        for evaluation_metric in evaluation_metrics:
            eval_log[f"{evaluation_metric} of {target_col[i]}"] = inds[
                evaluation_metric
            ].tolist()
    return eval_log


def compute_loss(
    labels: torch.Tensor, output: torch.Tensor, criterion, **kwargs
) -> float:
    """
    Function for computing the loss

    Parameters
    ----------
    labels
        The real values for the target. Shape can be variable but should follow (batch_size, time)
    output
        The output of the model
    criterion
        loss function
    validation_dataset
        Only passed when unscaling of data is needed.
    m
        defaults to 1

    Returns
    -------
    float
        the computed loss
    """
    if isinstance(criterion, GaussianLoss):
        if len(output[0].shape) > 2:
            g_loss = GaussianLoss(output[0][:, :, 0], output[1][:, :, 0])
        else:
            g_loss = GaussianLoss(output[0][:, 0], output[1][:, 0])
        return g_loss(labels)
    if (
        isinstance(output, torch.Tensor)
        and len(labels.shape) != len(output.shape)
        and len(labels.shape) > 1
    ):
        if labels.shape[1] == output.shape[1]:
            labels = labels.unsqueeze(2)
        else:
            labels = labels.unsqueeze(0)
    assert labels.shape == output.shape
    return criterion(output, labels.float())


def torch_single_train(
    model,
    opt: optim.Optimizer,
    criterion,
    data_loader: DataLoader,
    device=None,
    **kwargs,
) -> float:
    """
    Training function for one epoch

    Parameters
    ----------
    model
        a PyTorch model inherit from nn.Module
    opt
        optimizer function from PyTorch optim.Optimizer
    criterion
        loss function
    data_loader
        object for loading data to the model
    device
        where we put the tensors and models

    Returns
    -------
    tuple(float, int)
        loss of this epoch and number of all iterations

    Raises
    --------
    ValueError
        if nan exits, raise a ValueError
    """
    # we will set model.eval() in the validation function so here we should set model.train()
    model.train()
    n_iter_ep = 0
    running_loss = 0.0
    which_first_tensor = kwargs["which_first_tensor"]
    seq_first = which_first_tensor != "batch"
    pbar = tqdm(data_loader)

    for _, (src, trg) in enumerate(pbar):
        # iEpoch starts from 1, iIter starts from 0, we hope both start from 1
        trg, output = model_infer(seq_first, device, model, src, trg)
        loss = compute_loss(trg, output, criterion, **kwargs)
        if loss > 100:
            print("Warning: high loss detected")
        loss.backward()
        opt.step()
        model.zero_grad()
        if torch.isnan(loss) or loss == float("inf"):
            raise ValueError(
                "Error infinite or NaN loss detected. Try normalizing data or performing interpolation"
            )
        running_loss += loss.item()
        n_iter_ep += 1
    total_loss = running_loss / float(n_iter_ep)
    return total_loss, n_iter_ep


def compute_validation(
    model,
    criterion,
    data_loader: DataLoader,
    device: torch.device = None,
    **kwargs,
) -> float:
    """
    Function to compute the validation loss metrics

    Parameters
    ----------
    model
        the trained model
    criterion
        torch.nn.modules.loss
    dataloader
        The data-loader of either validation or test-data
    device
        torch.device

    Returns
    -------
    tuple
        validation observations (numpy array), predictions (numpy array) and the loss of validation
    """
    model.eval()
    seq_first = kwargs["which_first_tensor"] != "batch"
    obs = []
    preds = []
    with torch.no_grad():
        for src, trg in data_loader:
            trg, output = model_infer(seq_first, device, model, src, trg)
            obs.append(trg)
            preds.append(output)
        # first dim is batch
        obs_final = torch.cat(obs, dim=0)
        pred_final = torch.cat(preds, dim=0)
        valid_loss = compute_loss(obs_final, pred_final, criterion)
    y_obs = obs_final.detach().cpu().numpy()
    y_pred = pred_final.detach().cpu().numpy()
    return y_obs, y_pred, valid_loss


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def cellstates_when_inference(seq_first, data_cfgs, pred):
    """get cell states when inference"""
    cs_out = (
        cs_cat_lst.detach().cpu().numpy().swapaxes(0, 1)
        if seq_first
        else cs_cat_lst.detach().cpu().numpy()
    )
    cs_out_lst = [cs_out]
    cell_state = reduce(lambda a, b: np.vstack((a, b)), cs_out_lst)
    np.save(os.path.join(data_cfgs["test_path"], "cell_states.npy"), cell_state)
    # model.zero_grad()
    torch.cuda.empty_cache()
    return pred, cell_state
