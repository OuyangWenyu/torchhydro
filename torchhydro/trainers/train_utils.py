"""
Author: Wenyu Ouyang
Date: 2024-04-08 18:16:26
LastEditTime: 2025-11-06 13:48:48
LastEditors: Wenyu Ouyang
Description: Some basic functions for training
FilePath: \torchhydro\torchhydro\trainers\train_utils.py
Copyright (c) 2024-2024 Wenyu Ouyang. All rights reserved.
"""

import copy
import fnmatch
import itertools
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
from torch.utils.data.dataloader import default_collate
from torch.nn.utils.rnn import pad_sequence

from hydroutils.hydro_stat import stat_error
from hydroutils.hydro_file import (
    get_lastest_file_in_a_dir,
    unserialize_json,
    get_latest_file_in_a_lst,
)
from torchhydro.datasets.data_sets import FloodEventDataset
from torchhydro.models.crits import FloodBaseLoss, GaussianLoss


def _rolling_preds_for_once_eval(
    batch_shape,
    rho,
    forecast_length,
    rolling_stride,
    hindcast_output_window,
    the_array,
):
    """
    Get predictions to perform rolling evaluation: restore the prediction results to the original time series length.

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


def model_infer(
    seq_first, device, model, batch, variable_length_cfgs=None, return_key=None
):
    """
    Unified model inference function with variable length support

    Parameters
    ----------
    seq_first : bool
        if True, the input data is sequence first
    device : torch.device
        cpu or gpu
    model : torch.nn.Module
        the model
    batch : tuple or list
        batch data from collate_fn or dataset
    variable_length_cfgs : dict, optional
        variable length configuration containing mask settings
    return_key : str, optional
        when model returns a dict, choose which key (e.g., "f2") to return.
        if None, defaults to the last frequency (max key).
    """
    result = get_masked_tensors(variable_length_cfgs, batch, seq_first)

    # --- unpack inputs ---
    if len(result) == 9:
        (
            xs,
            ys,
            edge_index,
            edge_weight,
            batch_vector,
            xs_mask,
            ys_mask,
            xs_lens,
            ys_lens,
        ) = result
    elif len(result) == 8:
        xs, ys, edge_index, edge_weight, xs_mask, ys_mask, xs_lens, ys_lens = result
        batch_vector = None
    else:
        xs, ys, xs_mask, ys_mask, xs_lens, ys_lens = result
        edge_index = edge_weight = batch_vector = None

    # --- move xs to device ---
    if isinstance(xs, list):
        xs = [
            (
                x.permute(1, 0, 2).to(device)
                if seq_first and x.ndim == 3
                else x.to(device)
            )
            for x in xs
        ]
    else:
        xs = [
            (
                xs.permute(1, 0, 2).to(device)
                if seq_first and xs.ndim == 3
                else xs.to(device)
            )
        ]

    # --- move ys to device ---
    if ys is not None:
        ys = (
            ys.permute(1, 0, 2).to(device)
            if seq_first and ys.ndim == 3
            else ys.to(device)
        )

    # --- move graph data ---
    if edge_index is not None:
        edge_index = edge_index.to(device)
    if edge_weight is not None:
        edge_weight = edge_weight.to(device)
    if batch_vector is not None:
        batch_vector = batch_vector.to(device)

    # --- forward ---
    if xs_mask is not None and ys_mask is not None:
        if edge_index is not None and edge_weight is not None:
            output = model(
                *xs,
                edge_index=edge_index,
                edge_weight=edge_weight,
                batch_vector=batch_vector,
                mask=xs_mask,
                seq_lengths=xs_lens,
            )
        else:
            output = model(*xs, mask=xs_mask, seq_lengths=xs_lens)
    else:
        if edge_index is not None and edge_weight is not None:
            output = model(
                *xs,
                edge_index=edge_index,
                edge_weight=edge_weight,
                batch_vector=batch_vector,
            )
        else:
            output = model(*xs)

    # --- handle model outputs ---
    if isinstance(output, tuple):
        output = output[0]

    if isinstance(output, dict):
        # 默认取最高频的输出
        if return_key is None:
            return_key = sorted(output.keys())[-1]  # e.g., "f2"
        if return_key not in output:
            raise KeyError(
                f"Model returned keys {list(output.keys())}, but return_key='{return_key}' not found"
            )
        output = output[return_key]

    if ys_mask is not None:
        ys = ys.masked_fill(ys_mask == 0, torch.nan)

    # --- seq_first transpose back ---
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
    obs, pred, evaluation_metrics, target_col, fill_nan, eval_log, horizon=1
):
    fill_nan_value = fill_nan
    inds = stat_error(obs, pred, fill_nan_value)

    for evaluation_metric in evaluation_metrics:
        if horizon == 1:
            eval_log[f"{evaluation_metric} of {target_col}"] = inds[
                evaluation_metric
            ].tolist()
        eval_log[f"{evaluation_metric} of {target_col} in horizon {horizon}"] = inds[
            evaluation_metric
        ].tolist()

    return eval_log


def get_preds_to_be_eval(
    valorte_data_loader,
    evaluation_cfgs,
    output,
    labels,
):
    """
    Get prediction results prepared for evaluation:
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
    # this test_rolling means how we perform prediction during testing
    test_rolling = evaluation_cfgs["rolling"]
    batch_size = valorte_data_loader.batch_size
    target_scaler = valorte_data_loader.dataset.target_scaler
    target_data = target_scaler.data_target
    rho = valorte_data_loader.dataset.rho
    horizon = valorte_data_loader.dataset.horizon
    warmup_length = valorte_data_loader.dataset.warmup_length
    hindcast_output_window = target_scaler.data_cfgs["hindcast_output_window"]
    nf = valorte_data_loader.dataset.noutputvar  # number of features
    # number of time steps after warmup as outputs typically don't include warmup period
    nt = valorte_data_loader.dataset.nt - warmup_length
    basin_num = len(target_data.basin)
    data_shape = (basin_num, nt, nf)
    if evaluator["eval_way"] == "once":
        stride = evaluator["stride"]
        if stride > 0:
            if horizon != stride:
                raise NotImplementedError(
                    "horizon should be equal to stride in evaluator if you chose eval_way to be once, or else you need to change the eval_way to be 1pace or rolling"
                )
            obs = _rolling_preds_for_once_eval(
                (basin_num, horizon, nf),
                rho,
                evaluation_cfgs["forecast_length"],
                stride,
                hindcast_output_window,
                target_data.reshape(basin_num, horizon, nf),
            )
            pred = _rolling_preds_for_once_eval(
                (basin_num, horizon, nf),
                rho,
                evaluation_cfgs["forecast_length"],
                stride,
                hindcast_output_window,
                output.reshape(batch_size, horizon, nf),
            )
        else:
            if test_rolling > 0:
                raise RuntimeError(
                    "please set rolling to 0 when you chose eval way as once and stride=0"
                )
            obs = labels.reshape(basin_num, -1, nf)
            pred = output.reshape(basin_num, -1, nf)
    elif evaluator["eval_way"] == "1pace":
        if test_rolling < 1:
            raise NotImplementedError(
                "rolling should be larger than 0 if you chose eval_way to be 1pace"
            )
        pace_idx = evaluator["pace_idx"]
        # stride = evaluator.get("stride", 1)
        # for 1pace with pace_idx meaning which value of output was chosen to show
        # 1st, we need to transpose data to 4-dim to show the whole data

        # TODO:check should we select which def
        pred = _recover_samples_to_basin(output, valorte_data_loader, pace_idx)
        obs = _recover_samples_to_basin(labels, valorte_data_loader, pace_idx)

    elif evaluator["eval_way"] == "rolling":
        # 获取滚动预测所需的参数
        stride = evaluator.get("stride", 1)
        if stride != 1:
            raise NotImplementedError(
                "if stride is not equal to 1, we think it is meaningless"
            )
        # 重组预测结果和观测值
        basin_num = len(target_data.basin)

        # 新增：根据配置选择不同的数据组织方式
        recover_mode = evaluator.get("recover_mode", "bybasins")
        stride = evaluator.get("stride", 1)
        data_shape = (basin_num, nt, nf)

        if recover_mode == "bybasins":

            pred = _recover_samples_to_4d_by_basins(
                data_shape,
                valorte_data_loader,
                stride,
                hindcast_output_window,
                output,
            )
            obs = _recover_samples_to_4d_by_basins(
                data_shape,
                valorte_data_loader,
                stride,
                hindcast_output_window,
                labels,
            )
        elif recover_mode == "byforecast":
            pred = _recover_samples_to_4d_by_forecast(
                data_shape,
                valorte_data_loader,
                stride,
                hindcast_output_window,
                output,  # samples, seq_length, nf
            )
            obs = _recover_samples_to_4d_by_forecast(
                data_shape,
                valorte_data_loader,
                stride,
                hindcast_output_window,
                labels,
            )
        elif recover_mode == "byensembles":
            pred = _recover_samples_to_3d_by_4d_ensembles(
                data_shape,
                valorte_data_loader,
                stride,
                hindcast_output_window,
                output,
            )
            obs = _recover_samples_to_3d_by_4d_ensembles(
                data_shape,
                valorte_data_loader,
                stride,
                hindcast_output_window,
                labels,
            )
        else:
            raise ValueError(
                f"Unsupported recover_mode: {recover_mode}, must be 'bybasins' or 'byforecast' or 'byensembles'"
            )
    elif evaluator["eval_way"] == "floodevent":
        # For flood event evaluation, stride is not typically used, but we set it to 1 for consistency
        stride = evaluator.get("stride", 1)
        pred = _recover_samples_to_continuous_by_floodevent(
            data_shape,
            valorte_data_loader,
            stride,
            hindcast_output_window,
            output,
        )
        obs = _recover_samples_to_continuous_by_floodevent(
            data_shape,
            valorte_data_loader,
            stride,
            hindcast_output_window,
            labels,
        )
    else:
        raise ValueError("eval_way should be rolling or 1pace")

    # pace_idx = np.nan
    recover_mode = evaluator.get("recover_mode")
    valte_dataset = valorte_data_loader.dataset
    # 检查数据维度并进行适当处理
    if pred.ndim == 4:
        # 如果是四维数据，需要根据评估方式选择合适的处理方法
        if evaluator["eval_way"] == "1pace" and "pace_idx" in evaluator:
            # 对于1pace模式，选择特定的预测步长
            pace_idx = evaluator["pace_idx"]
            # 选择特定预测步长的数据
            pred_3d = pred[:, :, pace_idx, :]
            obs_3d = obs[:, :, pace_idx, :]
            preds_xr = valte_dataset.denormalize(pred_3d, pace_idx)
            obss_xr = valte_dataset.denormalize(obs_3d, pace_idx)
        elif evaluator["eval_way"] == "rolling" and recover_mode == "byforecast":
            # 对于byforecast模式，需要特殊处理
            # 创建一个列表存储每个预测步长的结果
            preds_xr_list = []
            obss_xr_list = []
            for i in range(pred.shape[2]):
                pred_3d = pred[:, :, i, :]
                obs_3d = obs[:, :, i, :]
                the_array_pred_ = np.full(target_data.shape, np.nan)
                the_array_obs_ = np.full(target_data.shape, np.nan)
                start = rho + i  # TODO:need check
                end = start + pred_3d.shape[1]
                assert end <= the_array_pred_.shape[1]
                the_array_pred_[:, start:end, :] = pred_3d
                the_array_obs_[:, start:end, :] = obs_3d
                preds_xr_list.append(valte_dataset.denormalize(the_array_pred_, i))
                obss_xr_list.append(valte_dataset.denormalize(the_array_obs_, i))
            # 合并结果
            # preds_xr = xr.concat(preds_xr_list, dim="horizon")
            # obss_xr = xr.concat(obss_xr_list, dim="horizon")
            return obss_xr_list, preds_xr_list
        elif evaluator["eval_way"] == "rolling" and recover_mode == "bybasins":
            # 对于其他情况，可以考虑将四维数据转换为三维
            # 例如，取最后一个预测步长
            preds_xr_list = []
            obss_xr_list = []
            for i in range(pred.shape[0]):
                pred_3d = pred[i, :, :, :]
                obs_3d = obs[i, :, :, :]
                selected_data = target_scaler.data_target
                the_array_pred_ = np.full(selected_data.shape, np.nan)
                the_array_obs_ = np.full(selected_data.shape, np.nan)
                start = rho  # TODO:need check
                end = start + pred_3d.shape[1]  # 自动计算填充的结束位置

                # 检查是否越界（可选）
                assert end <= the_array_pred_.shape[1]  # "填充范围超出目标数组的边界"

                # 执行填充
                the_array_pred_[:, start:end, :] = pred_3d
                the_array_obs_[:, start:end, :] = obs_3d

                preds_xr = valte_dataset.denormalize(the_array_pred_, -1)
                obss_xr = valte_dataset.denormalize(the_array_obs_, -1)
    else:
        # for 3d data, directly process
        # TODO: maybe need more test for the pace_idx case
        preds_xr = valte_dataset.denormalize(pred)
        obss_xr = valte_dataset.denormalize(obs)

    def _align_and_order(_obs, _pred):
        # 对齐到公共 (basin,time,variable) 的交集，避免 outer 引入 NaN
        _obs, _pred = xr.align(_obs, _pred, join="inner")
        # time 维为空（无交集）时直接抛错，避免进入 nanmean
        if _obs.sizes.get("time", 0) == 0:
            raise ValueError(
                "No overlapping timestamps between observations and predictions "
                f"(obs.time len={_obs.sizes.get('time',0)}, pred.time len={_pred.sizes.get('time',0)})."
            )
        # 按时间排序（保险）
        if "time" in _obs.dims:
            _obs = _obs.sortby("time")
        if "time" in _pred.dims:
            _pred = _pred.sortby("time")
        # 规范维度顺序（若存在）
        wanted = [d for d in ("basin", "time", "variable") if d in _obs.dims]
        _obs = _obs.transpose(*wanted, missing_dims="ignore")
        _pred = _pred.transpose(*wanted, missing_dims="ignore")
        return _obs, _pred

    # 单对象 vs 列表分别处理
    if preds_xr is not None and obss_xr is not None:
        obss_xr, preds_xr = _align_and_order(obss_xr, preds_xr)
        return obss_xr, preds_xr

    elif preds_xr_list is not None and obss_xr_list is not None:
        obss_aligned, preds_aligned = [], []
        for _o, _p in zip(obss_xr_list, preds_xr_list):
            _o2, _p2 = _align_and_order(_o, _p)
            obss_aligned.append(_o2)
            preds_aligned.append(_p2)
        return obss_aligned, preds_aligned

    else:
        # 理论不应走到这
        raise RuntimeError("Failed to build preds_xr / obss_xr for evaluation.")


def _recover_samples_to_4d_by_forecast(
    data_shape,
    valorte_data_loader,
    stride,
    hindcast_output_window,
    arr_3d,
):
    """
    将模型输出的3D预测结果重组为4D数组，并针对每个预测步长进行处理

    Parameters
    ----------
    data_shape : tuple
        数据形状，包含三个元素 (basin_num, nt, nf)，分别表示流域数量、时间步数和特征数量
    valorte_data_loader : DataLoader
        验证或测试数据加载器，用于获取流域-时间索引映射
    stride : int
        滚动步长
    hindcast_output_window : int
        历史输出窗口长度
    arr_3d : np.ndarray
        3D预测数组，形状为 (样本数量, 时间步数, 特征数量)

    Returns
    -------
    np.ndarray
        重组后的4D数组，形状为 (basin_num, i_e_time_length, forecast_length, nf)
    """
    # 从数据集获取必要参数
    basin_num, nt, nf = data_shape
    dataset = valorte_data_loader.dataset
    basin_num = len(dataset.t_s_dict["sites_id"])
    forecast_length = dataset.horizon

    # 将arr_3d重塑为所需的形状
    output = arr_3d.reshape(basin_num, -1, arr_3d.shape[1], nf)
    i_e_time_length = output.shape[1]

    # 创建结果数组，初始化为NaN
    result = np.full((basin_num, i_e_time_length, forecast_length, nf), np.nan)

    # 提取最后forecast_length个时间步的预测结果
    forecast_output = output[:, :, -forecast_length:, :]

    # 并行处理所有流域
    # results = []
    # for j in range(forecast_length):
    #     # 计算有效索引范围
    #     # valid_indices = np.arange(i_e_time_length - j)

    #     # 对所有流域同时处理
    #     for basin_idx in range(basin_num):
    #         # 填充结果数组
    #         result[basin_idx, :, j, :] = forecast_output[basin_idx, :, j, :]
    #         # result[j, basin_idx, :, nf] = forecast_output[basin_idx, :, j, :]
    result = forecast_output
    return result
    # dict / array
    # [第几个预见期，流域，总长，变量个数]
    # {“1”: [流域，总长，变量个数], "2":……}


def _recover_samples_to_4d_by_basins(
    data_shape,
    valorte_data_loader,
    stride,
    hindcast_output_window,
    arr_3d,
):
    """
    将模型输出按照流域重组为4D数组，便于计算每个流域的所有样本预见期的指标

    Parameters
    ----------
    data_shape : tuple
        数据形状，包含三个元素 (basin_num, nt, nf)，分别表示流域数量、时间步数和特征数量
    valorte_data_loader : DataLoader
        验证或测试数据加载器，用于获取流域-时间索引映射
    stride : int
        滚动步长
    hindcast_output_window : int
        历史输出窗口长度
    arr_3d : np.ndarray
        3D预测数组，形状为 (样本数量, 时间步数, 特征数量)

    Returns
    -------
    np.ndarray
        重组后的4D数组，形状为 (basin_num, i_e_time_length, forecast_length, nf)
        第一个维度表示流域，便于计算每个流域的所有样本预见期的指标
    """
    # 从数据集获取必要参数
    basin_num, nt, nf = data_shape
    dataset = valorte_data_loader.dataset
    basin_num = len(dataset.t_s_dict["sites_id"])
    forecast_length = dataset.horizon

    # 将arr_3d重塑为所需的形状
    output = arr_3d.reshape(basin_num, -1, arr_3d.shape[1], nf)
    i_e_time_length = output.shape[1]

    # 创建结果数组，初始化为NaN
    result = np.full((basin_num, i_e_time_length, forecast_length, nf), np.nan)

    # 提取最后forecast_length个时间步的预测结果
    forecast_output = output[:, :, -forecast_length:, :]

    # 对每个流域进行处理
    for basin_idx in range(basin_num):
        # 对每个预测步长进行处理
        for j in range(forecast_length):
            # 填充结果数组，注意这里将流域放在第一个维度
            result[basin_idx, :, j, :] = forecast_output[basin_idx, :, j, :]

    return result


def _recover_samples_to_4d(
    data_shape,
    valorte_data_loader,
    stride,
    hindcast_output_window,
    arr_3d,
):
    """Reorganize the 3D prediction results to 4D

    Prepare rolling result for the following two ways to calculate rolling evaluation results:
    1. We can organize data according to forecast horizons, with each horizon having a set of evaluation results
    2. For each rolling prediction result, calculate a set of metrics, with each basin having one set of metrics, and all basins stored in a 2D array containing all metrics.

    TODO: to be finished

    Parameters
    ----------
    arr_3d : np.ndarray
        A 3D prediction array with the shape (total number of samples, number of time steps, number of features).
    valorte_data_loader: DataLoader
        The corresponding data loader used to obtain the basin-time index mapping.
    stride: int
        The stride of the rolling.

    Returns
        -------
        np.ndarray
            The reorganized 4D array with the shape (number of basins, length of time, forecast steps, number of features).
    """
    basin_num, nt, nf = data_shape
    dataset = valorte_data_loader.dataset
    batch_size = valorte_data_loader.batch_size
    basin_num = len(dataset.t_s_dict["sites_id"])
    nt = dataset.nt
    rho = dataset.rho
    warmup_len = dataset.warmup_length
    horizon = dataset.horizon
    nf = dataset.noutputvar

    # Initialize the 4D array with NaN values
    basin_array = np.full((basin_num, nt - warmup_len - rho, horizon, nf), np.nan)

    for sample_idx in range(arr_3d.shape[0]):
        # Get the basin and start time index corresponding to this sample
        basin, start_time = dataset.lookup_table[sample_idx]
        # Take the value at the last time step of this sample (at the position of rho + horizon)
        value = arr_3d[sample_idx, warmup_len + rho :, :]
        # Calculate the time position in the result array
        result_time_idx = start_time + warmup_len + stride * (sample_idx % batch_size)
        # Fill in the corresponding position
        basin_array[basin, result_time_idx, :, :] = value

    return basin_array


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
        -1 means we chose the final value for one prediction
        positive values means we chose the results during horzion periods
        we ignore 0, because it may lead to confusion. 1 means the 1st horizon period
        TODO: when hindcast_output is not None, this part need to be modified.

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
        basin, start_time, _ = dataset.lookup_table[sample_idx]
        # Calculate the time position in the result array
        if pace_idx < 0:
            value = arr_3d[sample_idx, pace_idx, :]
            result_time_idx = start_time + warmup_len + rho + horizon + pace_idx
        else:
            value = arr_3d[sample_idx, pace_idx - 1, :]
            result_time_idx = start_time + warmup_len + rho + pace_idx - 1
        # Fill in the corresponding position
        basin_array[basin, result_time_idx, :] = value

    return basin_array


def _recover_samples_to_3d_by_4d_ensembles(
    data_shape,
    valorte_data_loader,
    stride,
    hindcast_output_window,
    arr_3d,
):
    """
    Merge the rolling prediction results and calculate the average value for the same time period.

    Parameters
    ----------
    data_shape : tuple
        The shape of the data, containing three elements (basin_num, nt, nf), representing the number of basins, the number of time steps, and the number of features, respectively.
    valorte_data_loader : DataLoader
        The corresponding data loader used to obtain the basin-time index mapping.
    stride : int
        The stride of the rolling.
    hindcast_output_window : int
        The length of the historical output window.
    arr_3d : np.ndarray
        The 3D prediction array with the shape (samples, time_steps, features).

    Returns
    -------
    np.ndarray
        The reorganized 3D array with the shape (basin_num, time_length, nf).
        The value of each time period is the average of all predictions.
    """
    basin_num, nt, nf = data_shape
    dataset = valorte_data_loader.dataset
    basin_num = len(dataset.t_s_dict["sites_id"])
    forecast_length = dataset.horizon
    rho = dataset.rho

    # Calculate the actual time length
    actual_time_length = nt - rho + hindcast_output_window

    # Create the result array and count array
    result = np.full((basin_num, actual_time_length, nf), np.nan)
    count_array = np.zeros((basin_num, actual_time_length, nf))

    # Reshape arr_3d to the required shape
    output = arr_3d.reshape(basin_num, -1, arr_3d.shape[1], nf)
    samples_per_basin = output.shape[1]

    # Process each basin
    for basin_idx, sample_idx in itertools.product(
        range(basin_num), range(samples_per_basin)
    ):
        # Calculate the starting position of the current sample on the time axis
        sample_start_time = sample_idx * stride

        # Extract the prediction sequence (the last forecast_length time steps)
        prediction_sequence = output[basin_idx, sample_idx, -forecast_length:, :]

        # Assign the prediction values to the corresponding time positions
        for horizon_idx in range(forecast_length):
            target_time = sample_start_time + horizon_idx

            # Ensure that the time range is not exceeded
            if target_time < actual_time_length:
                # Accumulate the prediction values
                if np.isnan(result[basin_idx, target_time, :]).all():
                    result[basin_idx, target_time, :] = prediction_sequence[
                        horizon_idx, :
                    ]
                    count_array[basin_idx, target_time, :] = 1
                else:
                    # If there is a value, accumulate
                    valid_mask = ~np.isnan(prediction_sequence[horizon_idx, :])
                    result[basin_idx, target_time, valid_mask] += prediction_sequence[
                        horizon_idx, valid_mask
                    ]
                    count_array[basin_idx, target_time, valid_mask] += 1

    # Calculate the average value
    for basin_idx in range(basin_num):
        for time_idx in range(actual_time_length):
            for feature_idx in range(nf):
                if count_array[basin_idx, time_idx, feature_idx] > 1:
                    result[basin_idx, time_idx, feature_idx] /= count_array[
                        basin_idx, time_idx, feature_idx
                    ]

    return result


def _recover_samples_to_continuous_by_floodevent(
    data_shape,
    valorte_data_loader,
    stride,
    hindcast_output_window,
    arr_3d,
):
    """
    Turn the independent flood events to continuous time series

    The flood event data is a series of independent flood events, which are not continuous in time.
    This function puts these data in the corresponding time positions of the original time series length,
    so that it is convenient to perform the inverse normalization operation.

    For FloodEventDataset, this function handles the flood_event column properly:
    - If arr_3d is predictions missing flood_event column, adds a placeholder column
    - If arr_3d is observations with flood_event column, keeps it as is
    This ensures dimensional consistency for subsequent denormalization.

    Parameters
    ----------
    data_shape : tuple
        The shape of the data, containing three elements (basin_num, nt, nf), representing the number of basins, the number of time steps, and the number of features, respectively.
    valorte_data_loader : DataLoader
        The corresponding data loader used to obtain the basin-time index mapping.
    stride : int
        The stride of the rolling.
    hindcast_output_window : int
        The length of the historical output window.
    arr_3d : np.ndarray
        The 3D prediction array with the shape (samples, time_steps, features).
        For flood datasets, may or may not include flood_event column.

    Returns
    -------
    np.ndarray
        The reorganized 3D array with the shape (basin_num, time_length, nf).
        The independent flood event data is placed in the correct position of the original time series.
        For flood datasets, ensures consistent dimensionality with target columns.
    """
    basin_num, nt, nf = data_shape
    dataset = valorte_data_loader.dataset

    # Get the actual number of basins
    basin_num = len(dataset.t_s_dict["sites_id"])

    # Check if this is a FloodEventDataset and handle flood_event column properly
    is_flood_dataset = isinstance(dataset, FloodEventDataset)

    # For flood datasets, we need to ensure consistency between predictions and observations
    # If arr_3d is predictions and missing flood_event column, we need to add it
    # If arr_3d is observations with flood_event column, we keep it as is
    if is_flood_dataset:
        expected_features = len(dataset.data_cfgs["target_cols"])
        current_features = arr_3d.shape[-1]

        if current_features < expected_features:
            # This is likely predictions missing flood_event column, add it
            # We need to get the actual flood_event data from the dataset for each sample
            flood_event_data = np.zeros((arr_3d.shape[0], arr_3d.shape[1], 1))

            # Try to get the actual flood_event data from the dataset
            for sample_idx in range(arr_3d.shape[0]):
                basin_idx, start_time, actual_length = dataset.lookup_table[sample_idx]
                end_time = start_time + actual_length
                if end_time > dataset.nt:
                    end_time = dataset.nt
                    actual_length = end_time - start_time

                # Get the actual flood_event data from the dataset
                if hasattr(dataset, "y") and dataset.flood_event_idx is not None:
                    sample_length = min(arr_3d.shape[1], actual_length)
                    actual_flood_events = dataset.y[
                        basin_idx,
                        start_time : start_time + sample_length,
                        dataset.flood_event_idx,
                    ]
                    flood_event_data[sample_idx, :sample_length, 0] = (
                        actual_flood_events
                    )

            arr_3d_processed = np.concatenate([arr_3d, flood_event_data], axis=-1)
            # Update nf to match the expected number of features
            nf = expected_features
        else:
            # This is likely observations with flood_event column, keep as is
            arr_3d_processed = arr_3d
            nf = current_features
    else:
        arr_3d_processed = arr_3d

    # Create the result array, initialized with NaN
    result = np.full((basin_num, nt, nf), np.nan)

    # Iterate over each sample
    for sample_idx in range(arr_3d_processed.shape[0]):
        # Get the basin, start time, and actual length of the current sample from lookup_table
        basin_idx, start_time, actual_length = dataset.lookup_table[sample_idx]

        # Get the prediction result of the current sample
        sample_prediction = arr_3d_processed[sample_idx, :, :]  # [time_steps, features]

        # Calculate the end time
        end_time = start_time + actual_length

        # Ensure that the time range is not exceeded
        if end_time > nt:
            end_time = nt
            actual_length = end_time - start_time

        # Ensure that the length of the prediction result matches the actual length
        pred_length = min(sample_prediction.shape[0], actual_length)

        # Place the prediction result in the correct position of the original time series
        # Note: here we need to handle the possible length mismatch problem
        if pred_length > 0:
            # If the prediction length is greater than the actual length, take the first part
            if pred_length > actual_length:
                pred_to_use = sample_prediction[:actual_length, :]
            else:
                pred_to_use = sample_prediction[:pred_length, :]

            # Fill the prediction result into the corresponding position
            end_fill = start_time + pred_to_use.shape[0]
            result[basin_idx, start_time:end_fill, :] = pred_to_use

    return result


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
    obss_xr, preds_xr = get_preds_to_be_eval(
        validation_data_loader,
        evaluation_cfgs,
        output,
        labels,
    )
    # obss_xr_list
    # preds_xr_list
    # if type()
    # for i in range(obs.shape[0]): # 第几个预见期
    ## obs_ = obs[i]
    if isinstance(obss_xr, list):
        obss_xr_list = obss_xr
        preds_xr_list = preds_xr
        for horizon_idx in range(len(obss_xr_list)):
            obss_xr = obss_xr_list[horizon_idx]
            preds_xr = preds_xr_list[horizon_idx]
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
                    horizon_idx + 1,
                )
        return eval_log

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
) -> torch.Tensor:
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
    torch.Tensor
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
    if isinstance(criterion, FloodBaseLoss):
        # labels has one more column than output, which is the flood mask
        # so we need to remove the last column of labels to get targets
        flood_mask = labels[:, :, -1:]  # Extract flood mask from last column
        targets = labels[:, :, :-1]  # Extract targets (remove last column)
        return criterion(output, targets, flood_mask)
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


def varied_length_collate_fn(batch):
    """Collate function for variable length training

    This function is automatically used by DataLoader when variable_length_cfgs["use_variable_length"] is True.
    It pads sequences to the same length and generates corresponding masks.

    Parameters
    ----------
    batch : list of tuples
        The batch data after the dataset __getitem__ method

    Returns
    -------
    list
        [xs_pad, ys_pad, xs_lens, ys_lens, xs_mask, ys_mask]
        - xs_pad: padded input sequences [batch, max_seq_len, input_dim]
        - ys_pad: padded output sequences [batch, max_seq_len, output_dim]
        - xs_lens: original sequence lengths for input
        - ys_lens: original sequence lengths for output
        - xs_mask: valid position mask for input [batch, max_seq_len]
        - ys_mask: valid position mask for output [batch, max_seq_len]
    """

    xs, ys = zip(*batch)
    # sometimes x is a tuple like in dpl dataset, then we can get the shape of the first element as the length
    xs_lens = [x[0].shape[0] if type(x) in [tuple, list] else x.shape[0] for x in xs]
    ys_lens = [y[0].shape[0] if type(y) in [tuple, list] else y.shape[0] for y in ys]
    # if all ys_lens are the same, use default collate_fn to create tensors
    if len(set(ys_lens)) == 1 and len(set(xs_lens)) == 1:
        xs_tensor = default_collate(xs)
        ys_tensor = default_collate(ys)
        return [xs_tensor, ys_tensor, None, None, None, None]

    # pad the batch data with padding value 0
    xs_pad = pad_sequence(xs, batch_first=True, padding_value=0)
    ys_pad = pad_sequence(ys, batch_first=True, padding_value=0)

    # generate the mask for the batch data
    # xs_mask: [batch_size, max_seq_len] or [batch_size, max_seq_len, 1]
    batch_size = len(xs_lens)
    max_xs_len = max(xs_lens)
    max_ys_len = max(ys_lens)

    # create the mask for the input sequence (True for valid positions, False for padding positions)
    xs_mask = torch.zeros(batch_size, max_xs_len, dtype=torch.bool)
    for i, length in enumerate(xs_lens):
        xs_mask[i, :length] = True

    # create the mask for the output sequence
    ys_mask = torch.zeros(batch_size, max_ys_len, dtype=torch.bool)
    for i, length in enumerate(ys_lens):
        ys_mask[i, :length] = True

    return [
        xs_pad,
        ys_pad,
        xs_lens,
        ys_lens,
        xs_mask,
        ys_mask,
    ]


def gnn_collate_fn(batch):
    """
    Custom collate function for GNN datasets that handles variable-sized graphs

    Each sample in batch is a tuple: (sxc, y, edge_index, edge_weight)
    where:
    - sxc: [num_stations_i, seq_length, feature_dim] (variable num_stations)
    - y: [forecast_length, output_dim]
    - edge_index: [2, num_edges_i] (variable num_edges)
    - edge_weight: [num_edges_i] (variable num_edges)

    Returns:
    --------
    list
        [batched_sxc, batched_y, batched_edge_index, batched_edge_weight]
        where batched_sxc has shape [batch_size, max_num_nodes, seq_length, feature_dim]
    """
    import torch

    if len(batch) == 0:
        return []

    # Unpack the batch
    sxc_list, y_list, edge_index_list, edge_weight_list = zip(*batch)

    # Batch the target values (y) - these should have the same shape
    batched_y = torch.stack(y_list, dim=0)  # [batch_size, forecast_length, output_dim]

    # Find the maximum number of nodes in this batch
    max_num_nodes = max(sxc.shape[0] for sxc in sxc_list)

    # Get dimensions
    batch_size = len(sxc_list)
    seq_length = sxc_list[0].shape[1]
    feature_dim = sxc_list[0].shape[2]

    # Create padded tensor for node features
    batched_sxc = torch.zeros(batch_size, max_num_nodes, seq_length, feature_dim)

    # Create batched edge indices and weights
    # For each graph in the batch, we need to offset node indices
    batched_edge_index = []
    batched_edge_weight = []
    batch_vector = []
    node_offset = 0
    for i, (sxc, edge_index, edge_weight) in enumerate(
        zip(sxc_list, edge_index_list, edge_weight_list)
    ):
        num_nodes = sxc.shape[0]
        # Fill the padded tensor with actual node features
        batched_sxc[i, :num_nodes] = sxc
        # For edge indices, we need to offset by node_offset to make them unique across batch
        if edge_index.numel() > 0:
            if edge_index.max() >= num_nodes:
                print(
                    f"Warning: Graph {i} has edge indices {edge_index.max().item()} >= num_nodes {num_nodes}"
                )
                valid_mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
                edge_index = edge_index[:, valid_mask]
                edge_weight = edge_weight[valid_mask]
            if edge_index.numel() > 0:
                offset_edge_index = edge_index + node_offset
                batched_edge_index.append(offset_edge_index)
                batched_edge_weight.append(edge_weight)
        # batch_vector: for each node in this graph, assign batch index i
        batch_vector.append(torch.full((num_nodes,), i, dtype=torch.long))
        node_offset += num_nodes
    # Concatenate edge indices and weights if they exist
    if batched_edge_index:
        batched_edge_index = torch.cat(batched_edge_index, dim=1)  # [2, total_edges]
        batched_edge_weight = torch.cat(batched_edge_weight, dim=0)  # [total_edges]
    else:
        batched_edge_index = torch.empty((2, 0), dtype=torch.long)
        batched_edge_weight = torch.empty(0)
    batch_vector = torch.cat(batch_vector, dim=0)  # [total_nodes]
    return [
        batched_sxc,
        batched_y,
        batched_edge_index,
        batched_edge_weight,
        batch_vector,
    ]


def get_masked_tensors(variable_length_cfgs, batch, seq_first):
    """Get the mask for the data

    Parameters
    ----------
    variable_length_cfgs : dict
        The variable length configuration
    batch : tuple or list
        The batch data from collate_fn or dataset
    seq_first : bool
        Whether the data is in sequence first format

    Returns
    -------
    tuple
        For standard datasets: (xs, ys, xs_mask, ys_mask, xs_lens, ys_lens)
        For GNN datasets: (xs, ys, edge_index, edge_weight, xs_mask, ys_mask, xs_lens, ys_lens)
        For GNN with batch vector: (xs, ys, edge_index, edge_weight, batch_vector, xs_mask, ys_mask, xs_lens, ys_lens)
    """
    xs_mask = None
    ys_mask = None
    xs_lens = None
    ys_lens = None
    edge_index = None
    edge_weight = None

    if variable_length_cfgs is None:
        # Check batch length to determine format
        if len(batch) == 5:
            # GNN batch with batch_vector: [sxc, y, edge_index, edge_weight, batch_vector]
            xs, ys, edge_index, edge_weight, batch_vector = batch
            return (
                xs,
                ys,
                edge_index,
                edge_weight,
                batch_vector,
                xs_mask,
                ys_mask,
                xs_lens,
                ys_lens,
            )
        elif len(batch) == 4:
            # GNN batch: [sxc, y, edge_index, edge_weight]
            xs, ys, edge_index, edge_weight = batch
            return xs, ys, edge_index, edge_weight, xs_mask, ys_mask, xs_lens, ys_lens
        else:
            # Standard batch: [xs, ys]
            xs, ys = batch[0], batch[1]
            return xs, ys, xs_mask, ys_mask, xs_lens, ys_lens

    if variable_length_cfgs.get("use_variable_length", False):
        # When using variable length training, batch comes from varied_length_collate_fn
        # which returns [xs_pad, ys_pad, xs_lens, ys_lens, xs_mask, ys_mask]
        if len(batch) >= 6:
            xs, ys, xs_lens, ys_lens, xs_mask_bool, ys_mask_bool = batch[:6]
        else:
            # Fallback: treat as regular batch with first two elements
            xs, ys = batch[0], batch[1]
            xs_lens = ys_lens = xs_mask_bool = ys_mask_bool = None

        if xs_mask_bool is None and ys_mask_bool is None:
            # sometime even you choose to use variable length training, the batch data may still be fixed length
            # so we need to return the batch data directly
            return xs, ys, xs_mask_bool, ys_mask_bool, xs_lens, ys_lens
        # Convert masks to the format expected by model (float tensor with shape [..., 1])
        xs_mask = xs_mask_bool.unsqueeze(-1).float()  # [batch, seq, 1]
        ys_mask = ys_mask_bool.unsqueeze(-1).float()  # [batch, seq, 1]

        # Convert to appropriate format for model if needed
        if seq_first:
            xs_mask = xs_mask.transpose(0, 1)  # [seq, batch, 1]
            ys_mask = ys_mask.transpose(0, 1)  # [seq, batch, 1]
    else:
        # Check batch length to determine format
        if len(batch) == 5:
            # GNN batch with batch_vector: [sxc, y, edge_index, edge_weight, batch_vector]
            xs, ys, edge_index, edge_weight, batch_vector = batch
        elif len(batch) == 4:
            # GNN batch: [sxc, y, edge_index, edge_weight]
            xs, ys, edge_index, edge_weight = batch
        else:
            # Standard batch: [xs, ys]
            xs, ys = batch[0], batch[1]

    # Return appropriate format based on what we have
    if edge_index is not None and edge_weight is not None:
        return (
            xs,
            ys,
            edge_index,
            edge_weight,
            batch_vector,
            xs_mask,
            ys_mask,
            xs_lens,
            ys_lens,
        )
    else:
        return xs, ys, xs_mask, ys_mask, xs_lens, ys_lens


def torch_single_train(
    model,
    opt: optim.Optimizer,
    criterion,
    data_loader: DataLoader,
    device=None,
    **kwargs,
):
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
    tuple(torch.Tensor, int)
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
    variable_length_cfgs = data_loader.dataset.training_cfgs.get(
        "variable_length_cfgs", None
    )
    pbar = tqdm(data_loader)

    for _, batch in enumerate(pbar):
        # mask handling is already done inside model_infer function
        trg, output = model_infer(seq_first, device, model, batch, variable_length_cfgs)
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
):
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
    obs_final = None
    pred_final = None
    with torch.no_grad():
        iter_num = 0
        for batch in tqdm(data_loader, desc="Evaluating", total=len(data_loader)):
            trg, output = model_infer(seq_first, device, model, batch)
            obs.append(trg)
            preds.append(output)
            valid_loss_ = compute_loss(trg, output, criterion)
            if torch.isnan(valid_loss_):
                # for not-train mode, we may get all nan data for trg
                # so we skip this batch
                continue
                print("NAN loss detected, skipping this batch")
            valid_loss = valid_loss + valid_loss_.item()
            iter_num = iter_num + 1

            # For flood datasets, remove the flood_mask column from observations
            # to match the prediction dimensions for evaluation
            trg_for_eval = (
                trg[:, :, :-1] if isinstance(criterion, FloodBaseLoss) else trg
            )
            # clear memory to save GPU memory
            if obs_final is None:
                obs_final = trg_for_eval.detach().cpu()
                pred_final = output.detach().cpu()
            else:
                obs_final = torch.cat([obs_final, trg_for_eval.detach().cpu()], dim=0)
                pred_final = torch.cat([pred_final, output.detach().cpu()], dim=0)
            del trg, output
            torch.cuda.empty_cache()
    valid_loss = valid_loss / iter_num
    y_obs = obs_final.numpy()
    y_pred = pred_final.numpy()
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
