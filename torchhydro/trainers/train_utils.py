"""
Author: Wenyu Ouyang
Date: 2023-09-21 15:06:12
LastEditTime: 2023-09-21 16:04:40
LastEditors: Wenyu Ouyang
Description: Some basic functions for training
FilePath: /torchhydro/torchhydro/trainers/train_utils.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""


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
