"""
Author: Wenyu Ouyang
Date: 2021-12-31 11:08:29
LastEditTime: 2023-07-30 10:48:57
LastEditors: Wenyu Ouyang
Description: Testing functions for hydroDL models
FilePath: \HydroTL\hydrotl\models\evaluator.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
import os
from typing import Dict, Tuple
from functools import reduce
import numpy as np
import xarray as xr
import torch
from torch.utils.data import DataLoader
from hydroutils.hydro_stat import stat_error

from torchhydro.trainers.time_model import PyTorchForecast
from torchhydro.models.model_utils import get_the_device


def evaluate_model(model: PyTorchForecast) -> Tuple[Dict, np.array, np.array]:
    """
    A function to evaluate a model, called at end of training.

    Parameters
    ----------
    model
        the DL model class

    Returns
    -------
    tuple[dict, np.array, np.array]
        eval_log, denormalized predictions and observations
    """
    data_params = model.params["data_params"]
    # types of observations
    target_col = model.params["data_params"]["target_cols"]
    evaluation_metrics = model.params["evaluate_params"]["metrics"]
    # fill_nan: "no" means ignoring the NaN value;
    #           "sum" means calculate the sum of the following values in the NaN locations.
    #           For example, observations are [1, nan, nan, 2], and predictions are [0.3, 0.3, 0.3, 1.5].
    #           Then, "no" means [1, 2] v.s. [0.3, 1.5] while "sum" means [1, 2] v.s. [0.3 + 0.3 + 0.3, 1.5].
    #           If it is a str, then all target vars use same fill_nan method;
    #           elif it is a list, each for a var
    fill_nan = model.params["evaluate_params"]["fill_nan"]
    # save result here
    eval_log = {}

    # test the trained model
    test_epoch = model.params["evaluate_params"]["test_epoch"]
    train_epoch = model.params["training_params"]["epochs"]
    if test_epoch != train_epoch:
        # Generally we use same epoch for train and test, but sometimes not
        # TODO: better refactor this part, because sometimes we save multi models for multi hyperparameters
        model_filepath = model.params["data_params"]["test_path"]
        model.model = model.load_model(
            model.params["model_params"]["model_name"],
            model.params["model_params"],
            weight_path=os.path.join(model_filepath, f"model_Ep{str(test_epoch)}.pth"),
        )
    pred, obs, test_data = infer_on_torch_model(model)
    print("Un-transforming data")
    preds_xr = test_data.target_scaler.inverse_transform(pred)
    obss_xr = test_data.target_scaler.inverse_transform(obs)

    #  Then evaluate the model metrics
    if type(fill_nan) is list and len(fill_nan) != len(target_col):
        raise Exception("length of fill_nan must be equal to target_col's")
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
            ]

    # Finally, try to explain model behaviour using shap
    # TODO: SHAP has not been tested
    is_shap = False
    if is_shap:
        deep_explain_model_summary_plot(
            model, test_data, data_params["t_range_test"][0]
        )
        deep_explain_model_heatmap(model, test_data, data_params["t_range_test"][0])

    return eval_log, preds_xr, obss_xr


def infer_on_torch_model(
    model: PyTorchForecast,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Function to handle both test evaluation and inference on a test data-frame.
    """
    data_params = model.params["data_params"]
    training_params = model.params["training_params"]
    device = get_the_device(model.params["training_params"]["device"])
    test_dataloader = DataLoader(
        model.test_data,
        batch_size=training_params["batch_size"],
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
    )
    seq_first = training_params["which_first_tensor"] == "sequence"
    model.model.eval()
    pred, obs = generate_predictions(
        model,
        test_dataloader,
        seq_first=seq_first,
        device=device,
        data_params=data_params,
    )
    # transform to xarray dataarray
    pred_xr = xr.DataArray(
        pred.transpose(2, 0, 1),
        dims=model.test_data.y.dims,
        coords=model.test_data.y.coords,
    )
    obs_xr = xr.DataArray(
        obs.transpose(2, 0, 1),
        dims=model.test_data.y.dims,
        coords=model.test_data.y.coords,
    )
    return pred_xr, obs_xr, model.test_data


def generate_predictions(
    ts_model: PyTorchForecast,
    test_dataloader,
    seq_first: bool,
    device: torch.device,
    data_params: dict,
    return_cell_state: bool = False,
) -> np.ndarray:
    """Perform Evaluation on the test (or valid) data.

    Parameters
    ----------
    ts_model : PyTorchForecast
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
    model = ts_model.model
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
    # TODO: not support return_cell_states yet
    if return_cell_state:
        return _cellstates_from_generate_predictions_(seq_first, data_params, pred)
    return pred, obs


def _cellstates_from_generate_predictions_(seq_first, data_params, pred):
    cs_out = (
        cs_cat_lst.detach().cpu().numpy().swapaxes(0, 1)
        if seq_first
        else cs_cat_lst.detach().cpu().numpy()
    )
    cs_out_lst = [cs_out]
    cell_state = reduce(lambda a, b: np.vstack((a, b)), cs_out_lst)
    np.save(os.path.join(data_params["test_path"], "cell_states.npy"), cell_state)
    # model.zero_grad()
    torch.cuda.empty_cache()
    return pred, cell_state
