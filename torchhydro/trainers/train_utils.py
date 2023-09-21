"""
Author: Wenyu Ouyang
Date: 2023-09-21 15:06:12
LastEditTime: 2023-09-21 16:47:36
LastEditors: Wenyu Ouyang
Description: Some basic functions for training
FilePath: /torchhydro/torchhydro/trainers/train_utils.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""


import torch
import xarray as xr


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
