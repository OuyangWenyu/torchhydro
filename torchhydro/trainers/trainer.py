"""
Author: Wenyu Ouyang
Date: 2021-12-05 11:21:58
LastEditTime: 2023-11-21 21:57:17
LastEditors: Wenyu Ouyang
Description: Main function for training and testing
FilePath: /torchhydro/torchhydro/trainers/trainer.py
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
    basin_ids = ["61561", "62618"]
    # lstm, tl-lstm
    exp61561 = ["gages/exp615610", "gages/exp615611"]
    exp62618 = ["gages/exp626180", "gages/exp626181"]
    all_exps = [exp61561, exp62618]
    train_periods = [
        [["2018-10-01", "2021-10-01"], ["2015-10-01", "2018-10-01"]],
        [["2018-10-01", "2021-10-01"], ["2015-10-01", "2018-10-01"]],
        [["1986-10-01", "1989-10-01"], ["1983-10-01", "1986-10-01"]],
    ]
    valid_periods = [
        [["2015-10-01", "2018-10-01"], ["2018-10-01", "2021-10-01"]],
        [["2015-10-01", "2018-10-01"], ["2018-10-01", "2021-10-01"]],
        [["1983-10-01", "1986-10-01"], ["1986-10-01", "1989-10-01"]],
    ]
    kfold = 2
    # for basins and models
    best_batchsize = [[50, 200], [100, 20], [50, 100]]
    best_bs_dir = []
    for bs in best_batchsize:
        bs_dir = [f"opt_Adadelta_lr_1.0_bsize_{str(b)}/training_params" for b in bs]
        best_bs_dir.append(bs_dir)
    lstm_train = False
    lstm_valid = False
    if lstm_train:
        camesl523_exp = "exp311"
        for i, j in itertools.product(range(len(basin_ids)), range(kfold)):
            train_and_evaluate(
                # "00": first zero means the first exp for lstm, second zero means the first fold
                "exp" + basin_ids[i] + "00" + str(j),
                random_seed=1234,
                opt="Adadelta",
                batch_size=best_batchsize[i],
                epoch=100,
                save_epoch=1,
                gage_id=[basin_ids[i]],
                data_loader="StreamflowDataset",
                num_workers=4,
                train_period=train_periods[i][j],
                valid_period=valid_periods[i][j],
                test_period=valid_periods[i][j],
                # only one basin, we don't need attribute
                var_c=[],
            )

    # for basins and models
    best_bs_dir = []
    for bs in best_batchsize:
        bs_dir = [f"opt_Adadelta_lr_1.0_bsize_{str(b)}/training_params" for b in bs]
        best_bs_dir.append(bs_dir)
    lstm_train = False
    lstm_valid = False
    if lstm_train:
        camesl523_exp = "exp311"
        for i, j in itertools.product(range(len(basin_ids)), range(kfold)):
            camels_cc_lstm_model(
                # "00": first zero means the first exp for lstm, second zero means the first fold
                "exp" + basin_ids[i] + "00" + str(j),
                random_seed=1234,
                opt="Adadelta",
                batch_size=best_batchsize[i],
                epoch=100,
                save_epoch=1,
                gage_id=[basin_ids[i]],
                data_loader="StreamflowDataset",
                num_workers=4,
                train_period=train_periods[i][j],
                valid_period=valid_periods[i][j],
                test_period=valid_periods[i][j],
                # only one basin, we don't need attribute
                var_c=[],
            )
            transfer_gages_lstm_model_to_camelscc(
                camesl523_exp,
                "exp" + basin_ids[i] + "10" + str(j),
                random_seed=1234,
                freeze_params=None,
                opt="Adadelta",
                batch_size=best_batchsize[i],
                epoch=100,
                save_epoch=1,
                gage_id=[basin_ids[i]],
                data_loader="StreamflowDataset",
                device=[1],
                train_period=train_periods[i][j],
                valid_period=valid_periods[i][j],
                test_period=valid_periods[i][j],
                var_c_target=[],
                num_workers=0,
            )
    if lstm_valid:
        best_epoch = [[58, 56], [86, 26], [85, 69]]
        for i, j in itertools.product(range(len(basin_ids)), range(kfold)):
            # the first evaluate_a_model is for training, the second is for validation
            evaluate_a_model(
                "exp" + basin_ids[i] + "00" + str(j),
                example="gages",
                epoch=best_epoch[i][0],
                train_period=valid_periods[i][j],
                test_period=train_periods[i][j],
                save_result_name=f"fold{str(j)}train",
                sub_exp=best_bs_dir[i][0],
                device=[0],
            )

            evaluate_a_model(
                "exp" + basin_ids[i] + "00" + str(j),
                example="gages",
                epoch=best_epoch[i][0],
                train_period=train_periods[i][j],
                test_period=valid_periods[i][j],
                save_result_name=f"fold{str(j)}valid",
                sub_exp=best_bs_dir[i][0],
                device=[0],
            )
            # transfer learning model
            evaluate_a_model(
                "exp" + basin_ids[i] + "10" + str(j),
                example="gages",
                epoch=best_epoch[i][1],
                train_period=valid_periods[i][j],
                test_period=train_periods[i][j],
                save_result_name=f"fold{str(j)}train",
                is_tl=True,
                sub_exp=best_bs_dir[i][1],
                device=[0],
            )

            evaluate_a_model(
                "exp" + basin_ids[i] + "10" + str(j),
                example="gages",
                epoch=best_epoch[i][1],
                train_period=train_periods[i][j],
                test_period=valid_periods[i][j],
                save_result_name=f"fold{str(j)}valid",
                is_tl=True,
                sub_exp=best_bs_dir[i][1],
                device=[0],
            )


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
