"""
Author: Wenyu Ouyang
Date: 2021-12-05 11:21:58
LastEditTime: 2023-07-30 14:57:09
LastEditors: Wenyu Ouyang
Description: Main function for training and testing
FilePath: \torchhydro\torchhydro\models\trainer.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
import fnmatch
import os
import random

import numpy as np
from typing import Dict, Tuple, Union
import pandas as pd
import torch
from hydroutils.hydro_stat import stat_error
from hydroutils.hydro_file import serialize_numpy, unserialize_numpy
from torchhydro.datasets.data_dict import data_sources_dict
from torchhydro.trainers.evaluator import evaluate_model
from torchhydro.trainers.pytorch_training import model_train, save_model_params_log
from torchhydro.trainers.time_model import PyTorchForecast


def set_random_seed(seed):
    """
    Set a random seed to guarantee the reproducibility

    Parameters
    ----------
    seed
        a number

    Returns
    -------
    None
    """
    print("Random seed:", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_and_evaluate(params: Dict):
    """
    Function to train and test a Model

    Parameters
    ----------
    params
        Dictionary containing all the parameters needed to run the model

    Returns
    -------
    None
    """
    random_seed = params["training_params"]["random_seed"]
    set_random_seed(random_seed)
    data_params = params["data_params"]
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
    model = PyTorchForecast(params["model_params"]["model_name"], data_source, params)
    if params["training_params"]["train_mode"]:
        if (
            "weight_path" in params["model_params"]
            and params["model_params"]["continue_train"]
        ) or ("weight_path" not in params["model_params"]):
            model_train(model)
        test_acc = evaluate_model(model)
        print("summary test_accuracy", test_acc[0])
        # save the results
        save_result(
            data_params["test_path"],
            params["evaluate_params"]["test_epoch"],
            test_acc[1],
            test_acc[2],
        )
    param_file_exist = any(
        (
            fnmatch.fnmatch(file, "*.json")
            and "_stat" not in file  # statistics json file
            and "_dict" not in file  # data cache json file
        )
        for file in os.listdir(data_params["test_path"])
    )
    if not param_file_exist:
        # although we save params log during training, but sometimes we directly evaluate a model
        # so here we still save params log if param file does not exist
        # no param file was saved yet, here we save data and params setting
        save_param_log_path = params["data_params"]["test_path"]
        save_model_params_log(params, save_param_log_path)


def save_result(save_dir, epoch, pred, obs, pred_name="flow_pred", obs_name="flow_obs"):
    """
    save the pred value of testing period and obs value

    Parameters
    ----------
    save_dir
        directory where we save the results
    epoch
        in this epoch, we save the results
    pred
        predictions
    obs
        observations
    pred_name
        the file name of predictions
    obs_name
        the file name of observations

    Returns
    -------
    None
    """
    flow_pred_file = os.path.join(save_dir, f"epoch{str(epoch)}" + pred_name)
    flow_obs_file = os.path.join(save_dir, f"epoch{str(epoch)}" + obs_name)
    pred.to_netcdf(flow_pred_file + ".nc")
    obs.to_netcdf(flow_obs_file + ".nc")


def load_result(
    save_dir, epoch, pred_name="flow_pred", obs_name="flow_obs", not_only_1out=False
) -> Tuple[np.array, np.array]:
    """load the pred value of testing period and obs value

    Parameters
    ----------
    save_dir : _type_
        _description_
    epoch : _type_
        _description_
    pred_name : str, optional
        _description_, by default "flow_pred"
    obs_name : str, optional
        _description_, by default "flow_obs"
    not_only_1out : bool, optional
        Sometimes our model give multiple output and we will load all of them,
        then we set this parameter True, by default False

    Returns
    -------
    Tuple[np.array, np.array]
        _description_
    """
    flow_pred_file = os.path.join(save_dir, f"epoch{str(epoch)}" + pred_name + ".npy")
    flow_obs_file = os.path.join(save_dir, f"epoch{str(epoch)}" + obs_name + ".npy")
    pred = unserialize_numpy(flow_pred_file)
    obs = unserialize_numpy(flow_obs_file)
    if not_only_1out:
        return pred, obs
    if obs.ndim == 3 and obs.shape[-1] == 1:
        if pred.shape[-1] != obs.shape[-1]:
            # TODO: for convenient, now we didn't process this special case for MTL
            pred = pred[:, :, 0]
        pred = pred.reshape(pred.shape[0], pred.shape[1])
        obs = obs.reshape(obs.shape[0], obs.shape[1])
    return pred, obs


def stat_result_for1out(var_name, unit, pred, obs, fill_nan, basin_area=None):
    """
    show the statistics result for 1 output
    """
    inds = stat_error(obs, pred, fill_nan=fill_nan)
    inds_df = pd.DataFrame(inds)
    return inds_df, obs, pred


def stat_result(
    save_dirs: str,
    test_epoch: int,
    return_value: bool = False,
    fill_nan: Union[str, list, tuple] = "no",
    unit="m3/s",
    basin_area=None,
    var_name=None,
) -> Tuple[pd.DataFrame, np.array, np.array]:
    """
    Show the statistics result

    Parameters
    ----------
    save_dirs : str
        where we read results
    test_epoch : int
        the epoch of test
    return_value : bool, optional
        if True, returen pred and obs data, by default False
    fill_nan : Union[str, list, tuple], optional
        how to deal with nan in obs, by default "no"
    unit : str, optional
        unit of flow, by default "m3/s"
        if m3/s, then didn't transform; else transform to m3/s

    Returns
    -------
    Tuple[pd.DataFrame, np.array, np.array]
        statistics results, 3-dim predicitons, 3-dim observations
    """
    pred, obs = load_result(save_dirs, test_epoch)
    if type(unit) is list:
        inds_df_lst = []
        pred_lst = []
        obs_lst = []
        for i in range(len(unit)):
            inds_df_, pred_, obs_ = stat_result_for1out(
                var_name[i],
                unit[i],
                pred[:, :, i],
                obs[:, :, i],
                fill_nan[i],
                basin_area=basin_area,
            )
            inds_df_lst.append(inds_df_)
            pred_lst.append(pred_)
            obs_lst.append(obs_)
        return inds_df_lst, pred_lst, obs_lst if return_value else inds_df_lst
    else:
        inds_df_, pred_, obs_ = stat_result_for1out(
            var_name, unit, pred, obs, fill_nan, basin_area=basin_area
        )
        return (inds_df_, pred_, obs_) if return_value else inds_df_


def load_ensemble_result(
    save_dirs, test_epoch, flow_unit="m3/s", basin_areas=None
) -> Tuple[np.array, np.array]:
    """
    load ensemble mean value

    Parameters
    ----------
    save_dirs
    test_epoch
    flow_unit
        default is m3/s, if it is not m3/s, transform the results
    basin_areas
        if unit is mm/day it will be used, default is None

    Returns
    -------

    """
    preds = []
    obss = []
    for save_dir in save_dirs:
        pred_i, obs_i = load_result(save_dir, test_epoch)
        if pred_i.ndim == 3 and pred_i.shape[-1] == 1:
            pred_i = pred_i.reshape(pred_i.shape[0], pred_i.shape[1])
            obs_i = obs_i.reshape(obs_i.shape[0], obs_i.shape[1])
        preds.append(pred_i)
        obss.append(obs_i)
    preds_np = np.array(preds)
    obss_np = np.array(obss)
    pred_mean = np.mean(preds_np, axis=0)
    obs_mean = np.mean(obss_np, axis=0)
    if flow_unit == "mm/day":
        if basin_areas is None:
            raise ArithmeticError("No basin areas we cannot calculate")
        basin_areas = np.repeat(basin_areas, obs_mean.shape[1], axis=0).reshape(
            obs_mean.shape
        )
        obs_mean = obs_mean * basin_areas * 1e-3 * 1e6 / 86400
        pred_mean = pred_mean * basin_areas * 1e-3 * 1e6 / 86400
    elif flow_unit == "m3/s":
        pass
    elif flow_unit == "ft3/s":
        obs_mean = obs_mean / 35.314666721489
        pred_mean = pred_mean / 35.314666721489
    return pred_mean, obs_mean


def stat_ensemble_result(
    save_dirs, test_epoch, return_value=False, flow_unit="m3/s", basin_areas=None
) -> Tuple[np.array, np.array]:
    """calculate statistics for ensemble results

    Parameters
    ----------
    save_dirs : _type_
        where the results save
    test_epoch : _type_
        we name the results files with the test_epoch
    return_value : bool, optional
        if True, return (inds_df, pred_mean, obs_mean), by default False
    flow_unit : str, optional
        arg for load_ensemble_result, by default "m3/s"
    basin_areas : _type_, optional
        arg for load_ensemble_result, by default None

    Returns
    -------
    Tuple[np.array, np.array]
        inds_df or (inds_df, pred_mean, obs_mean)
    """
    pred_mean, obs_mean = load_ensemble_result(
        save_dirs, test_epoch, flow_unit=flow_unit, basin_areas=basin_areas
    )
    inds = stat_error(obs_mean, pred_mean)
    inds_df = pd.DataFrame(inds)
    return (inds_df, pred_mean, obs_mean) if return_value else inds_df
