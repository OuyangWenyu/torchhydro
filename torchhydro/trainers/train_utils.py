"""
Author: Wenyu Ouyang
Date: 2024-04-08 18:16:26
LastEditTime: 2025-04-18 08:46:07
LastEditors: Wenyu Ouyang
Description: Some basic functions for training
FilePath: /HydroForecastEval/mnt/disk1/owen/code/torchhydro/torchhydro/trainers/train_utils.py
Copyright (c) 2024-2024 Wenyu Ouyang. All rights reserved.
"""

import copy
import fnmatch
import os
import re
import shutil
from functools import reduce
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import numpy as np
import xarray as xr
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from hydroutils.hydro_stat import stat_error
from hydroutils.hydro_file import (
    get_lastest_file_in_a_dir,
    unserialize_json,
    get_latest_file_in_a_lst,
)

from torchhydro.models.crits import GaussianLoss


def _rolling_once_evaluate(
    batch_shape,
    rho,
    forecast_length,
    rolling_stride,
    hindcast_output_window,
    the_array,
):
    """
    Perform rolling evaluation to restore the prediction results to the original time series length.

    This function is used to restore the rolling prediction results of the model to the original time series length.
    It assumes that the length of the rolling window is equal to the forecast length, and that there is only one prediction value for each time step.
    The function calculates the restored prediction results based on the given batch shape, historical window length, forecast length, rolling window length,
    hindcast output window length, and prediction result array.

    Parameters
    ----------
    batch_shape : tuple
        The shape of the batch, containing three elements (ngrid, nt, nf), representing the number of grids, the number of time steps, and the number of features, respectively.
    rho : int
        The length of the historical window.
    forecast_length : int
        The length of the forecast.
    rolling_stride : int
        The stride of the rolling
    hindcast_output_window : int
        The length of the hindcast output window.
    the_array : np.ndarray
        The prediction result array, with the shape (samples, window_size, nf), where samples represent the number of samples,
        window_size represents the size of the window, and nf represents the number of features.

    Returns
    -------
    np.ndarray
        The restored prediction result array, with the shape (ngrid, recover_len, nf), where recover_len represents the length of the restored time steps.

    Raises
    ------
    NotImplementedError
        Raised when the rolling window length is not equal to the forecast length.
    """
    ngrid, nt, nf = batch_shape
    if rolling_stride != forecast_length:
        # TODO: now we only guarantee each time has only one value,
        # so we directly reshape the data rather than a real rolling
        raise NotImplementedError(
            "rolling should be equal to forecast_length in data_cfgs now, others are not supported yet"
        )
    window_size = hindcast_output_window + forecast_length
    recover_len = nt - rho + hindcast_output_window
    samples = int(the_array.shape[0] / ngrid)
    the_array_ = np.full((ngrid, recover_len, nf), np.nan)
    # recover the_array to pred_
    the_array_4d = the_array.reshape(ngrid, samples, window_size, nf)
    for i in range(ngrid):
        for j in range(0, recover_len - window_size + 1, window_size):
            the_array_[i, j : j + window_size, :] = the_array_4d[i, j, :, :]
    return the_array_.reshape(ngrid, recover_len, nf)


def model_infer(seq_first, device, model, xs, ys):
    """_summary_

    Parameters
    ----------
    seq_first : bool
        if True, the input data is sequence first
    device : torch.device
        cpu or gpu
    model : torch.nn.Module
        the model
    xs : list or tensor
        xs is always batch first
    ys : tensor
        observed data

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        first is the observed data, second is the predicted data;
        both tensors are batch first
    """
    if type(xs) is list:
        xs = [
            (
                data_tmp.permute([1, 0, 2]).to(device)
                if seq_first and data_tmp.ndim == 3
                else data_tmp.to(device)
            )
            for data_tmp in xs
        ]
    else:
        xs = [
            (
                xs.permute([1, 0, 2]).to(device)
                if seq_first and xs.ndim == 3
                else xs.to(device)
            )
        ]
    if ys is not None:
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
        if ys is not None:
            ys = ys.transpose(0, 1)
    return ys, output


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

    def check_loss(self, model, validation_loss, save_dir) -> bool:
        score = validation_loss
        if self.best_score is None:
            self.save_model_checkpoint(model, save_dir)
            self.best_score = score

        elif score + self.min_delta >= self.best_score:
            self.counter += 1
            print("Epochs without Model Update:", self.counter)
            if self.counter >= self.patience:
                return False
        else:
            self.save_model_checkpoint(model, save_dir)
            print("Model Update")
            self.best_score = score
            self.counter = 0
        return True

    def save_model_checkpoint(self, model, save_dir):
        torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))


def calculate_and_record_metrics(
    obs, pred, evaluation_metrics, target_col, fill_nan, eval_log
):
    fill_nan_value = fill_nan
    inds = stat_error(obs, pred, fill_nan_value)

    for evaluation_metric in evaluation_metrics:
        eval_log[f"{evaluation_metric} of {target_col}"] = inds[
            evaluation_metric
        ].tolist()

    return eval_log


def get_evaluation(
    valorte_data_loader,
    evaluation_cfgs,
    output,
    labels,
):
    """
    Get evaluation results:
    the denormalized data without metrics by different eval ways

    Parameters
    ----------
    valorte_data_loader : DataLoader
        validation or test data loader
    evaluation_cfgs : dict
        evaluation configs
    output : np.ndarray
        model output
    labels : np.ndarray
        model target

    Returns
    -------
    tuple
        _description_
    """
    evaluator = evaluation_cfgs["evaluator"]
    batch_size = valorte_data_loader.batch_size
    target_scaler = valorte_data_loader.dataset.target_scaler
    target_data = target_scaler.data_target
    horizon = valorte_data_loader.dataset.horizon
    hindcast_output_window = target_scaler.data_cfgs["hindcast_output_window"]
    nf = valorte_data_loader.dataset.noutputvar  # number of features
    nt = valorte_data_loader.dataset.nt  # number of time steps
    basin_num = len(target_data.basin)
    if evaluator["eval_way"] == "once":
        stride = evaluator["stride"]
        if stride < 1:
            # means we directly use the data to perform denorm and metrics cal
            # as the results may be got by performing rolling prediction
            # then there are many results for each timestep and we flatten them
            # so, to perform denormlization with xarray, we need to reset time
            # and we also need to reshape the data to (basin_num, times, nf)
            is_real_time = True
            if labels.shape[0] > nt:
                # 0 dim means time dim
                is_real_time = False
                # TODO: to be implemented
                raise NotImplementedError(
                    "we only support the case that the length of labels is equal to nt"
                )
            obss_xr = valorte_data_loader.dataset.denormalize(
                labels.reshape(basin_num, -1, nf), is_real_time
            )
            preds_xr = valorte_data_loader.dataset.denormalize(
                output.reshape(basin_num, -1, nf), is_real_time
            )
        else:
            # TODO: need more test
            raise NotImplementedError(
                "we only support the case that the stride is equal to 1 now, others need to be implemented"
            )
            obss_xr = _rolling_once_evaluate(
                (basin_num, horizon, nf),
                target_scaler.rho,
                evaluation_cfgs["forecast_length"],
                stride,
                hindcast_output_window,
                target_data.reshape(basin_num, horizon, nf),
            )
            preds_xr = _rolling_once_evaluate(
                (basin_num, horizon, nf),
                target_scaler.rho,
                evaluation_cfgs["forecast_length"],
                stride,
                hindcast_output_window,
                output.reshape(batch_size, horizon, nf),
            )
    elif evaluator["eval_way"] == "1pace":
        pace_idx = evaluator["pace_idx"]
        # for 1pace with pace_idx meaning which value of output was chosen to show
        # 1st, we need to transpose data to 4-dim to show the whole data
        pred = _recover_samples_to_basin(output, valorte_data_loader, pace_idx)
        obs = _recover_samples_to_basin(labels, valorte_data_loader, pace_idx)
        valte_dataset = valorte_data_loader.dataset
        preds_xr = valte_dataset.denormalize(pred)
        obss_xr = valte_dataset.denormalize(obs)
    elif evaluator["eval_way"] == "rolling":
        # TODO: to be implemented
        raise NotImplementedError(
            "we will implement this function in the future, please choose 1pace or once now"
        )
    else:
        raise ValueError("eval_way should be rolling or 1pace")
    return obss_xr, preds_xr


def _recover_samples_to_basin(arr_3d, valorte_data_loader, pace_idx):
    """Reorganize the 3D prediction results by basin

    Parameters
    ----------
    arr_3d : np.ndarray
        A 3D prediction array with the shape (total number of samples, number of time steps, number of features).
    valorte_data_loader: DataLoader
        The corresponding data loader used to obtain the basin-time index mapping.
    pace_idx: int
        Which time step was chosen to show.

    Returns
        -------
        np.ndarray
            The reorganized 3D array with the shape (number of basins, length of time, number of features).
    """
    dataset = valorte_data_loader.dataset
    basin_num = len(dataset.t_s_dict["sites_id"])
    nt = dataset.nt
    rho = dataset.rho
    warmup_len = dataset.warmup_length
    horizon = dataset.horizon
    nf = dataset.noutputvar

    basin_array = np.full((basin_num, nt, nf), np.nan)

    for sample_idx in range(arr_3d.shape[0]):
        # Get the basin and start time index corresponding to this sample
        basin, start_time = dataset.lookup_table[sample_idx]
        # Take the value at the last time step of this sample (at the position of rho + horizon)
        value = arr_3d[sample_idx, pace_idx, :]
        # Calculate the time position in the result array
        result_time_idx = start_time + warmup_len + rho + horizon + pace_idx
        # Fill in the corresponding position
        basin_array[basin, result_time_idx, :] = value

    return basin_array


def evaluate_validation(
    validation_data_loader,
    output,
    labels,
    evaluation_cfgs,
    target_col,
):
    """
    calculate metrics for validation

    Parameters
    ----------
    output
        model output
    labels
        model target
    evaluation_cfgs
        evaluation configs
    target_col
        target columns

    Returns
    -------
    tuple
        metrics
    """
    fill_nan = evaluation_cfgs["fill_nan"]
    if isinstance(fill_nan, list) and len(fill_nan) != len(target_col):
        raise ValueError("Length of fill_nan must be equal to length of target_col.")
    eval_log = {}
    evaluation_metrics = evaluation_cfgs["metrics"]
    obss_xr, preds_xr = get_evaluation(
        validation_data_loader,
        evaluation_cfgs,
        output,
        labels,
    )
    for i, col in enumerate(target_col):
        obs = obss_xr[col].to_numpy()
        pred = preds_xr[col].to_numpy()
        # eval_log will be updated rather than completely replaced, no need to use eval_log["key"]
        eval_log = calculate_and_record_metrics(
            obs,
            pred,
            evaluation_metrics,
            col,
            fill_nan[i] if isinstance(fill_nan, list) else fill_nan,
            eval_log,
        )
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
    # a = np.sum(output.cpu().detach().numpy(),axis=1)/len(output)
    # b=[]
    # for i in a:
    #     b.append([i.tolist()])
    # output = torch.tensor(b, requires_grad=True).to(torch.device("cuda"))

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
        trg, output = model_infer(seq_first, device, model, src, trg)

        loss = compute_loss(trg, output, criterion, **kwargs)
        if loss > 100:
            print("Warning: high loss detected")
        if torch.isnan(loss):
            raise ValueError("nan loss detected")
            # continue
        loss.backward()  # Backpropagate to compute the current gradient
        opt.step()  # Update network parameters based on gradients
        model.zero_grad()  # clear gradient
        if loss == float("inf"):
            raise ValueError(
                "Error infinite loss detected. Try normalizing data or performing interpolation"
            )
        running_loss += loss.item()
        n_iter_ep += 1
    if n_iter_ep == 0:
        raise ValueError(
            "All batch computations of loss result in NAN. Please check the data."
        )
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
    valid_loss = 0.0
    with torch.no_grad():
        for src, trg in data_loader:
            trg, output = model_infer(seq_first, device, model, src, trg)
            obs.append(trg)
            preds.append(output)
            valid_loss_ = compute_loss(trg, output, criterion)
            valid_loss = valid_loss + valid_loss_.item()
            # clear memory to save GPU memory
            torch.cuda.empty_cache()
        # first dim is batch
        obs_final = torch.cat(obs, dim=0)
        pred_final = torch.cat(preds, dim=0)
    valid_loss = valid_loss / len(data_loader)
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


def _find_min_validation_loss_epoch(data):
    """
    Find the epoch with the minimum validation loss from the training log data.

    Parameters
    ----------
    data : list of dict
        A list of dictionaries containing training information, where each dictionary corresponds to an epoch.

    Returns
    -------
    tuple
        (min_epoch, min_val_loss) The epoch number with the minimum validation loss and its corresponding loss value.
        If the data is empty or no valid validation loss can be found, returns (None, None).
    """
    if not data:
        print("Input data is empty.")
        return None, None

    df = pd.DataFrame(data)

    if "epoch" not in df.columns or "validation_loss" not in df.columns:
        print("Input data is missing 'epoch' or 'validation_loss' fields.")
        return None, None

    # Define a function to extract the numerical value from `validation_loss`
    def extract_val_loss(val_loss_str):
        """
        Extract the numerical part of the validation loss from the string.

        Parameters
        ----------
        val_loss_str : str
            A string in the form "tensor(4.1230, device='cuda:2')".

        Returns
        -------
        float
            The extracted validation loss value. If extraction fails, returns positive infinity.
        """
        match = re.search(r"tensor\(([\d\.]+)", val_loss_str)
        if match:
            try:
                return float(match[1])
            except ValueError:
                return float("inf")
        return float("inf")

    # Apply function to extract the numerical part
    df["validation_loss_value"] = df["validation_loss"].apply(extract_val_loss)

    # Check if there are valid validation losses
    if df["validation_loss_value"].isnull().all():
        print("All 'validation_loss' values cannot be parsed.")
        return None, None

    # Find the minimum validation loss and the corresponding epoch
    min_idx = df["validation_loss_value"].idxmin()
    min_row = df.loc[min_idx]

    min_epoch = min_row["epoch"]
    min_val_loss = min_row["validation_loss_value"]

    return min_epoch, min_val_loss


def read_pth_from_model_loader(model_loader, model_pth_dir):
    if model_loader["load_way"] == "specified":
        test_epoch = model_loader["test_epoch"]
        weight_path = os.path.join(model_pth_dir, f"model_Ep{str(test_epoch)}.pth")
    elif model_loader["load_way"] == "best":
        weight_path = os.path.join(model_pth_dir, "best_model.pth")
        if not os.path.exists(weight_path):
            # read log file and find the best model
            log_json = read_torchhydro_log_json_file(model_pth_dir)
            if "run" not in log_json:
                raise ValueError(
                    "No best model found. You have to train the model first."
                )
            min_epoch, min_val_loss = _find_min_validation_loss_epoch(log_json["run"])
            try:
                shutil.copy2(
                    os.path.join(model_pth_dir, f"model_Ep{str(min_epoch)}.pth"),
                    os.path.join(model_pth_dir, "best_model.pth"),
                )
            except FileNotFoundError:
                # TODO: add a recursive call to find the saved best model
                raise FileNotFoundError(
                    f"The best model's weight file {os.path.join(model_pth_dir, f'model_Ep{str(min_epoch)}.pth')} does not exist."
                )
    elif model_loader["load_way"] == "latest":
        weight_path = get_lastest_file_in_a_dir(model_pth_dir)
    elif model_loader["load_way"] == "pth":
        weight_path = model_loader["pth_path"]
    else:
        raise ValueError("Invalid load_way")
    if not os.path.exists(weight_path):
        raise ValueError(f"Model file {weight_path} does not exist.")
    return weight_path


def get_lastest_logger_file_in_a_dir(dir_path):
    """Get the last logger file in a directory

    Parameters
    ----------
    dir_path : str
        the directory

    Returns
    -------
    str
        the path of the logger file
    """
    pattern = r"^\d{1,2}_[A-Za-z]+_\d{6}_\d{2}(AM|PM)\.json$"
    pth_files_lst = [
        os.path.join(dir_path, file)
        for file in os.listdir(dir_path)
        if re.match(pattern, file)
    ]
    return get_latest_file_in_a_lst(pth_files_lst)


def cellstates_when_inference(seq_first, data_cfgs, pred):
    """get cell states when inference"""
    cs_out = (
        cs_cat_lst.detach().cpu().numpy().swapaxes(0, 1)
        if seq_first
        else cs_cat_lst.detach().cpu().numpy()
    )
    cs_out_lst = [cs_out]
    cell_state = reduce(lambda a, b: np.vstack((a, b)), cs_out_lst)
    np.save(os.path.join(data_cfgs["case_dir"], "cell_states.npy"), cell_state)
    # model.zero_grad()
    torch.cuda.empty_cache()
    return pred, cell_state


def read_torchhydro_log_json_file(cfg_dir):
    json_files_lst = []
    json_files_ctime = []
    for file in os.listdir(cfg_dir):
        if (
            fnmatch.fnmatch(file, "*.json")
            and "_stat" not in file  # statistics json file
            and "_dict" not in file  # data cache json file
        ):
            json_files_lst.append(os.path.join(cfg_dir, file))
            json_files_ctime.append(os.path.getctime(os.path.join(cfg_dir, file)))
    sort_idx = np.argsort(json_files_ctime)
    cfg_file = json_files_lst[sort_idx[-1]]
    return unserialize_json(cfg_file)


def get_latest_pbm_param_file(param_dir):
    """Get the latest parameter file of physics-based models in the current directory.

    Parameters
    ----------
    param_dir : str
        The directory of parameter files.

    Returns
    -------
    str
        The latest parameter file.
    """
    param_file_lst = [
        os.path.join(param_dir, f)
        for f in os.listdir(param_dir)
        if f.startswith("pb_params") and f.endswith(".csv")
    ]
    param_files = [Path(f) for f in param_file_lst]
    param_file_names_lst = [param_file.stem.split("_") for param_file in param_files]
    ctimes = [
        int(param_file_names[param_file_names.index("params") + 1])
        for param_file_names in param_file_names_lst
    ]
    return param_files[ctimes.index(max(ctimes))] if ctimes else None


def get_latest_tensorboard_event_file(log_dir):
    """Get the latest event file in the log_dir directory.

    Parameters
    ----------
    log_dir : str
        The directory where the event files are stored.

    Returns
    -------
    str
        The latest event file.
    """
    event_file_lst = [
        os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.startswith("events")
    ]
    event_files = [Path(f) for f in event_file_lst]
    event_file_names_lst = [event_file.stem.split(".") for event_file in event_files]
    ctimes = [
        int(event_file_names[event_file_names.index("tfevents") + 1])
        for event_file_names in event_file_names_lst
    ]
    return event_files[ctimes.index(max(ctimes))]
