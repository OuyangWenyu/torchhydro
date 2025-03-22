"""
Author: Wenyu Ouyang
Date: 2021-12-05 11:21:58
LastEditTime: 2024-05-23 15:21:17
LastEditors: Wenyu Ouyang
Description: Main function for training and testing
FilePath: \torchhydro\torchhydro\trainers\trainer.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""

import copy
from datetime import datetime
import os
from pathlib import Path
import random

import numpy as np
from typing import Dict
import pandas as pd
from sklearn.model_selection import KFold, TimeSeriesSplit
import torch
from torchhydro.trainers.deep_hydro import model_type_dict
from torchhydro.trainers.resulter import Resulter


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
    # print("Random seed:", seed)
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
    resulter = Resulter(cfgs)
    deephydro = _get_deep_hydro(cfgs)
    if cfgs["training_cfgs"]["train_mode"] and (
        (
            deephydro.weight_path is not None
            and deephydro.cfgs["model_cfgs"]["continue_train"]
        )
        or (deephydro.weight_path is None)
    ):
        deephydro.model_train()
    preds, obss = deephydro.model_evaluate()
    resulter.save_cfg(deephydro.cfgs)
    resulter.save_result(preds, obss)
    resulter.eval_result(preds, obss)


def _get_deep_hydro(cfgs):
    model_type = cfgs["model_cfgs"]["model_type"]
    return model_type_dict[model_type](cfgs)


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
        # although cross_validation contain the name "valid",
        # it is actually the test period for our model's evaluating
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


def _create_kfold_periods(train_period, test_period, kfold):
    """
    Create k folds from the complete time period defined by the earliest start date and latest end date
    among train, valid, and test periods, ignoring any period that is None or has NaN values.

    Parameters:
    - train_period: List with 2 elements [start_date, end_date] for training period.
    - test_period: List with 2 elements [start_date, end_date] for testing period.
    - kfold: Number of folds to split the data into.

    Returns:
    - A list of k elements, each element is a tuple containing two lists: train period and test period.
    """

    # Collect periods, ignoring None or NaN
    periods = [train_period, test_period]
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
    # kf = KFold(n_splits=kfold)
    kf = TimeSeriesSplit(n_splits=kfold)
    folds = []
    for train_index, test_index in kf.split(periods):
        train_start = full_period[train_index[0]].strftime("%Y-%m-%d")
        train_end = full_period[train_index[-1]].strftime("%Y-%m-%d")
        test_start = full_period[test_index[0]].strftime("%Y-%m-%d")
        test_end = full_period[test_index[-1]].strftime("%Y-%m-%d")
        folds.append(([train_start, train_end], [test_start, test_end]))

    return folds


def _create_kfold_discontinuous_periods(train_period, test_period, kfold):
    periods = train_period + test_period
    periods = sorted(periods, key=lambda x: x[0])
    cross_validation_sets = []

    for i in range(kfold):
        valid = [periods[i]]
        train = [p for j, p in enumerate(periods) if j != i]
        cross_validation_sets.append((train, valid))
    return cross_validation_sets


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


def _trans_kfold_to_periods(update_dict, ensemble_items, current_key="kfold"):
    # set train and test period
    train_period = update_dict["data_cfgs"]["t_range_train"]
    test_period = update_dict["data_cfgs"]["t_range_test"]
    kfold = ensemble_items[current_key]
    if ensemble_items["kfold_continuous"]:
        kfold_periods = _create_kfold_periods(train_period, test_period, kfold)
    else:
        kfold_periods = _create_kfold_discontinuous_periods(
            train_period, test_period, kfold
        )
    ensemble_items[current_key] = kfold_periods


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
