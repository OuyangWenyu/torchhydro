"""
Author: Wenyu Ouyang
Date: 2021-12-05 11:21:58
LastEditTime: 2023-11-25 18:20:24
LastEditors: Wenyu Ouyang
Description: Main function for training and testing
FilePath: \torchhydro\torchhydro\trainers\trainer.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
import copy
from datetime import datetime
import fnmatch
import os
from pathlib import Path
import random

import numpy as np
from typing import Dict, Tuple, Union
import pandas as pd
from sklearn.model_selection import KFold
import torch
from hydroutils.hydro_stat import stat_error
from hydroutils.hydro_file import unserialize_numpy
from torchhydro.datasets.data_dict import data_sources_dict
from torchhydro.trainers.train_logger import save_model_params_log
from torchhydro.trainers.deep_hydro import model_type_dict


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


def train_and_evaluate(cfgs: Dict):
    """
    Function to train and test a Model

    Parameters
    ----------
    cfgs
        Dictionary containing all configs needed to run the model

    Returns
    -------
    None
    """
    random_seed = cfgs["training_cfgs"]["random_seed"]
    set_random_seed(random_seed)
    result_dir = cfgs["data_cfgs"]["test_path"]
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    data_source = _get_datasource(cfgs)
    deephydro = _get_deep_hydro(cfgs, data_source)
    if cfgs["training_cfgs"]["train_mode"]:
        if (
            deephydro.weight_path is not None
            and deephydro.cfgs["model_cfgs"]["continue_train"]
        ) or (deephydro.weight_path is None):
            deephydro.model_train()
        test_acc = deephydro.model_evaluate()
        print("summary test_accuracy", test_acc[0])
        # save the results
        save_result(
            cfgs["data_cfgs"]["test_path"],
            cfgs["evaluation_cfgs"]["test_epoch"],
            test_acc[1],
            test_acc[2],
        )
    param_file_exist = any(
        (
            fnmatch.fnmatch(file, "*.json")
            and "_stat" not in file  # statistics json file
            and "_dict" not in file  # data cache json file
        )
        for file in os.listdir(cfgs["data_cfgs"]["test_path"])
    )
    if not param_file_exist:
        # although we save params log during training, but sometimes we directly evaluate a model
        # so here we still save params log if param file does not exist
        # no param file was saved yet, here we save data and params setting
        save_param_log_path = cfgs["data_cfgs"]["test_path"]
        save_model_params_log(cfgs, save_param_log_path)


def _get_deep_hydro(cfgs, data_source):
    model_type = cfgs["model_cfgs"]["model_type"]
    return model_type_dict[model_type](data_source, cfgs)


def _get_datasource(cfgs):
    data_cfgs = cfgs["data_cfgs"]
    data_source_name = data_cfgs["data_source_name"]
    return (
        data_sources_dict[data_source_name](
            data_cfgs["data_path"],
            data_cfgs["download"],
            data_cfgs["data_region"],
        )
        if data_source_name in ["CAMELS", "Caravan"]
        else data_sources_dict[data_source_name](
            data_cfgs["data_path"], data_cfgs["download"]
        )
    )


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


def _update_cfg_with_1ensembleitem(cfg, key, value):
    """update a dict with key and value

    Parameters
    ----------
    my_dict : _type_
        _description_
    key : _type_
        _description_
    value : _type_
        _description_
    """
    new_cfg = copy.deepcopy(cfg)
    if key == "kfold":
        new_cfg["data_cfgs"]["t_range_train"] = value[0]
        new_cfg["data_cfgs"]["t_range_valid"] = None
        new_cfg["data_cfgs"]["t_range_test"] = value[1]
    elif key == "batch_sizes":
        new_cfg["training_cfgs"]["batch_size"] = value
        new_cfg["data_cfgs"]["batch_size"] = value
    elif key == "seeds":
        new_cfg["training_cfgs"]["random_seed"] = value
    elif key == "expdir":
        project_dir = new_cfg["data_cfgs"]["test_path"]
        project_path = Path(project_dir)
        subset = project_path.parent.name
        subexp = f"{project_path.name}_{value}"
        new_cfg["data_cfgs"]["test_path"] = os.path.join(
            project_path.parent.parent, subset, subexp
        )
    else:
        raise ValueError(f"key {key} is not supported")
    return new_cfg


def _create_kfold_periods(train_period, valid_period, test_period, kfold):
    """
    Create k folds from the complete time period defined by the earliest start date and latest end date
    among train, valid, and test periods, ignoring any period that is None or has NaN values.

    Parameters:
    - train_period: List with 2 elements [start_date, end_date] for training period.
    - valid_period: List with 2 elements [start_date, end_date] for validation period.
    - test_period: List with 2 elements [start_date, end_date] for testing period.
    - kfold: Number of folds to split the data into.

    Returns:
    - A list of k elements, each element is a tuple containing two lists: train period and test period.
    """

    # Collect periods, ignoring None or NaN
    periods = [train_period, valid_period, test_period]
    periods = [p for p in periods if p and not pd.isna(p[0]) and not pd.isna(p[1])]

    # Convert string dates to datetime objects and find the earliest start and latest end dates
    dates = [
        datetime.strptime(date, "%Y-%m-%d") for period in periods for date in period
    ]
    start_date = min(dates)
    end_date = max(dates)

    # Create a continuous period
    full_period = pd.date_range(start=start_date, end=end_date)
    periods = np.array(range(len(full_period)))

    # Apply KFold
    kf = KFold(n_splits=kfold)
    folds = []
    for train_index, test_index in kf.split(periods):
        train_start = full_period[train_index[0]].strftime("%Y-%m-%d")
        train_end = full_period[train_index[-1]].strftime("%Y-%m-%d")
        test_start = full_period[test_index[0]].strftime("%Y-%m-%d")
        test_end = full_period[test_index[-1]].strftime("%Y-%m-%d")
        folds.append(([train_start, train_end], [test_start, test_end]))

    return folds


def _nested_loop_train_and_evaluate(keys, index, my_dict, update_dict, perform_count=0):
    """a recursive function to update the update_dict and perform train_and_evaluate

    Parameters
    ----------
    keys : list
        a list of keys
    index : int
        the loop index
    my_dict : dict
        the dict we want to loop
    update_dict : dict
        the dict we want to update
    perform_count : int, optional
        a counter for naming different experiments, by default 0
    """
    if index == len(keys):
        return perform_count
    current_key = keys[index]
    for value in my_dict[current_key]:
        # update the update_dict
        cfg = _update_cfg_with_1ensembleitem(update_dict, current_key, value)
        # for final key, perform train and evaluate
        if index == len(keys) - 1:
            cfg = _update_cfg_with_1ensembleitem(cfg, "expdir", perform_count)
            train_and_evaluate(cfg)
            # print(perform_count)
            perform_count += 1
        # recursive
        perform_count = _nested_loop_train_and_evaluate(
            keys, index + 1, my_dict, cfg, perform_count
        )
    return perform_count


def _trans_kfold_to_periods(update_dict, my_dict, current_key="kfold"):
    # set train and test period
    train_period = update_dict["data_cfgs"]["t_range_train"]
    valid_period = update_dict["data_cfgs"]["t_range_valid"]
    test_period = update_dict["data_cfgs"]["t_range_test"]
    kfold = my_dict[current_key]
    kfold_periods = _create_kfold_periods(
        train_period, valid_period, test_period, kfold
    )
    my_dict[current_key] = kfold_periods


def ensemble_train_and_evaluate(cfgs: Dict):
    """
    Function to train and test for ensemble models

    Parameters
    ----------
    cfgs
        Dictionary containing all configs needed to run the model

    Returns
    -------
    None
    """
    # for basins and models
    ensemble = cfgs["training_cfgs"]["ensemble"]
    if not ensemble:
        raise ValueError(
            "ensemble should be True, otherwise should use train_and_evaluate rather than ensemble_train_and_evaluate"
        )
    ensemble_items = cfgs["training_cfgs"]["ensemble_items"]
    number_of_items = len(ensemble_items)
    if number_of_items == 0:
        raise ValueError("ensemble_items should not be empty")
    keys_list = list(ensemble_items.keys())
    if "kfold" in keys_list:
        _trans_kfold_to_periods(cfgs, ensemble_items, "kfold")
    _nested_loop_train_and_evaluate(keys_list, 0, ensemble_items, cfgs)


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
