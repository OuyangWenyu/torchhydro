"""
Author: Wenyu Ouyang
Date: 2021-12-31 11:08:29
LastEditTime: 2023-11-24 18:19:55
LastEditors: Wenyu Ouyang
Description: Config for hydroDL
FilePath: /torchhydro/torchhydro/configs/config.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
import argparse
import fnmatch
import json
import logging
import os
import pandas as pd
import numpy as np
from hydroutils import hydro_file

DAYMET_NAME = "daymet"
SSM_SMAP_NAME = "ssm"
ET_MODIS_NAME = "ET"
Q_CAMELS_US_NAME = "usgsFlow"
# Q_CAMELS_CC_NAME = "Q_fix"
Q_CAMELS_CC_NAME = "Q"
PRCP_DAYMET_NAME = "prcp"
PRCP_NLDAS_NAME = "total_precipitation"
PET_MODIS_NAME = "PET"
PET_NLDAS_NAME = "potential_evaporation"
NLDAS_NAME = "nldas"
ERA5LAND_NAME = "era5land"
ET_ERA5LAND_NAME = "total_evaporation"
PRCP_ERA5LAND_NAME = "total_precipitation"
PET_DAYMET_NAME = "PET"
PET_ERA5LAND_NAME = "potential_evaporation"


def default_config_file():
    """
    Default config file for all models/data/training parameters in this repo

    Returns
    -------
    dict
        configurations
    """

    return {
        "model_cfgs": {
            # model_type including normal deep learning (Normal), federated learning (FedLearn), transfer learing (TransLearn), multi-task learning, etc.
            "model_type": "Normal",
            # supported models can be seen in hydroDL/model_dict_function.py
            "model_name": "LSTM",
            # the details of model parameters for the "model_name" model
            "model_hyperparam": {
                # the rho in LSTM
                "seq_length": 30,
                # the size of input (feature number)
                "input_size": 24,
                # the length of output time-sequence (feature number)
                "output_size": 1,
                "hidden_size": 20,
                "num_layers": 1,
                "bias": True,
                "batch_size": 100,
            },
            "weight_path": None,
            "continue_train": True,
            # the name of the model's wrapper class
            "model_wrapper": None,
            # the wrapper class's parameters
            "model_wrapper_param": None,
            # federated learning parameters
            "fl_hyperparam": {
                # sampling for federated learning
                "fl_sample": "basin",
                # number of users for federated learning
                # TODO: we don't use this parameter now, but we may use it in the future
                "fl_num_users": 10,
                # the number of local epochs
                "fl_local_ep": 5,
                # local batch size
                "fl_local_bs": 6,
                # the fraction of clients
                "fl_frac": 0.1,
            },
            "tl_hyperparam": {
                # part of transfer learning in a model: a list of layers' names, such as ["lstm"]
                "tl_part": None,
            },
        },
        "data_cfgs": {
            "data_source_name": "CAMELS",
            "data_path": "../../example/camels_us",
            "data_region": None,
            "download": True,
            "validation_path": None,
            "test_path": None,
            "batch_size": 100,
            # the rho in LSTM
            "forecast_history": 30,
            "forecast_length": 1,
            # modeled objects
            "object_ids": "ALL",
            # modeling time range
            "t_range_train": ["1992-01-01", "1993-01-01"],
            "t_range_valid": None,
            "t_range_test": ["1993-01-01", "1994-01-01"],
            # For physics-based models, we need warmup; default is 0 as DL models generally don't need it
            "warmup_length": 0,
            # the output
            "target_cols": [Q_CAMELS_US_NAME],
            "target_rm_nan": False,
            # only for cases in which target data will be used as input:
            # data assimilation -- use streamflow from period 0 to t-1 (TODO: not included now)
            # for physics-based model -- use streamflow to calibrate models
            "target_as_input": False,
            # the time series input
            # TODO: now we only support one forcing type
            "relevant_types": [DAYMET_NAME],
            "relevant_cols": [
                "dayl",
                PRCP_DAYMET_NAME,
                "srad",
                "swe",
                "tmax",
                "tmin",
                "vp",
            ],
            "relevant_rm_nan": True,
            # the attribute input
            "constant_cols": [
                "elev_mean",
                "slope_mean",
                "area_gages2",
                "frac_forest",
                "lai_max",
                "lai_diff",
                "dom_land_cover_frac",
                "dom_land_cover",
                "root_depth_50",
                "soil_depth_statsgo",
                "soil_porosity",
                "soil_conductivity",
                "max_water_content",
                "geol_1st_class",
                "geol_2nd_class",
                "geol_porostiy",
                "geol_permeability",
            ],
            "constant_rm_nan": True,
            # if constant_only, we will only use constant data as DL models' input: this is only for dpl models now
            "constant_only": False,
            # more other cols, use dict to express!
            "other_cols": None,
            # only numerical scaler: for categorical vars, they are transformed to numerical vars when reading them
            "scaler": "StandardScaler",
            # Some parameters for the chosen scaler function, default is DapengScaler's
            "scaler_params": {
                "prcp_norm_cols": [
                    Q_CAMELS_US_NAME,
                    "streamflow",
                    Q_CAMELS_CC_NAME,
                    "qobs",
                ],
                "gamma_norm_cols": [
                    PRCP_DAYMET_NAME,
                    "pr",
                    # PRCP_ERA5LAND_NAME is same as PRCP_NLDAS_NAME
                    PRCP_NLDAS_NAME,
                    "pre",
                    # pet may be negative, but we set negative as 0 because of gamma_norm_cols
                    # https://earthscience.stackexchange.com/questions/12031/does-negative-reference-evapotranspiration-make-sense-using-fao-penman-monteith
                    "pet",
                    # PET_ERA5LAND_NAME is same as PET_NLDAS_NAME
                    PET_NLDAS_NAME,
                    ET_MODIS_NAME,
                    "LE",
                    PET_MODIS_NAME,
                    "PLE",
                    "GPP",
                    "Ec",
                    "Es",
                    "Ei",
                    "ET_water",
                    "ET_sum",
                    SSM_SMAP_NAME,
                    "susm",
                    "smp",
                    "ssma",
                    "susma",
                ],
                "pbm_norm": False,
            },
            "stat_dict_file": None,
            # dataset for pytorch dataset
            "dataset": "StreamflowDataset",
            # sampler for pytorch dataloader, here we mainly use it for Kuai Fang's sampler in all his DL papers
            "sampler": None,
        },
        "training_cfgs": {
            # if train_mode is False, don't train and evaluate
            "train_mode": True,
            "criterion": "RMSE",
            "criterion_params": None,
            "optimizer": "Adam",
            "optim_params": {
                "lr": 0.001,
            },
            "lr_scheduler": None,
            "epochs": 20,
            # save_epoch ==0 means only save once in the final epoch
            "save_epoch": 0,
            # save_iter ==0 means we don't save model during training in a epoch
            "save_iter": 0,
            # when we train a model for long time, some accidents may interrupt our training.
            # Then we need retrain the model with saved weights, and the start_epoch is not 1 yet.
            "start_epoch": 1,
            "batch_size": 100,
            "random_seed": 1234,
            "device": [0],
            "multi_targets": 1,
            "num_workers": 0,
            # sometimes we want to directly use the trained model in each epoch during training,
            # for example, we want to save each epoch's log again, and in this time, we will set train_but_not_real to True
            "train_but_not_real": False,
            "which_first_tensor": "sequence",
            # for ensemble exp:
            # basically we set kfold/seeds/hyper_params for trianing such as batch_sizes
            "ensemble": False,
            "ensemble_items": {
                # kfold means a time cross validation,
                # concatenate train ,valid, and test data together,
                # then split them into k folds
                "kfold": None,
                "batch_sizes": None,
                # if seeds is not None,
                # we will use different seeds for different sub-exps
                "seeds": None,
            },
        },
        # For evaluation
        "evaluation_cfgs": {
            "metrics": ["NSE"],
            "fill_nan": "no",
            "test_epoch": 20,
            "explainer": None,
        },
    }


def cmd(
    sub=None,
    source="CAMELS",
    source_path=None,
    source_region=None,
    download=0,
    scaler=None,
    scaler_params=None,
    dataset=None,
    sampler=None,
    fl_sample=None,
    fl_num_users=None,
    fl_local_ep=None,
    fl_local_bs=None,
    fl_frac=None,
    ctx=None,
    rs=None,
    gage_id_file=None,
    gage_id=None,
    train_period=None,
    valid_period=None,
    test_period=None,
    opt=None,
    lr_scheduler=None,
    opt_param=None,
    batch_size=None,
    rho=None,
    train_mode=None,
    train_epoch=None,
    save_epoch=None,
    save_iter=None,
    te=None,
    model_type=None,
    model_name=None,
    weight_path=None,
    continue_train=None,
    var_c=None,
    c_rm_nan=1,
    var_t=None,
    t_rm_nan=1,
    n_output=None,
    loss_func=None,
    model_hyperparam=None,
    weight_path_add=None,
    var_t_type=None,
    var_o=None,
    var_out=None,
    out_rm_nan=0,
    target_as_input=0,
    constant_only=0,
    gage_id_screen=None,
    loss_param=None,
    metrics=None,
    fill_nan=None,
    warmup_length=0,
    start_epoch=1,
    stat_dict_file=None,
    model_wrapper=None,
    model_wrapper_param=None,
    num_workers=None,
    train_but_not_real=None,
    which_first_tensor=None,
    ensemble=0,
    ensemble_items=None,
):
    """input args from cmd"""
    parser = argparse.ArgumentParser(
        description="Train a Time-Series Deep Learning Model for Basins"
    )
    parser.add_argument(
        "--sub", dest="sub", help="subset and sub experiment", default=sub, type=str
    )
    parser.add_argument(
        "--source",
        dest="source",
        help="name of data source such as CAMELS",
        default=source,
        type=str,
    )
    parser.add_argument(
        "--source_path",
        dest="source_path",
        help="directory of data source",
        default=source_path,
        nargs="+",
    )
    parser.add_argument(
        "--source_region",
        dest="source_region",
        help="region(s) of data source such as US, or ['US','CE']",
        default=source_region,
        nargs="+",
    )
    parser.add_argument(
        "--download",
        dest="download",
        help="Do we need to download data",
        default=download,
        type=int,
    )
    parser.add_argument(
        "--scaler",
        dest="scaler",
        help="Choose a Scaler function",
        default=scaler,
        type=str,
    )
    parser.add_argument(
        "--scaler_params",
        dest="scaler_params",
        help="Parameters of the chosen Scaler function",
        default=scaler_params,
        type=json.loads,
    )
    parser.add_argument(
        "--dataset",
        dest="dataset",
        help="Choose a dataset class for PyTorch",
        default=dataset,
        type=str,
    )
    parser.add_argument(
        "--sampler",
        dest="sampler",
        help="None or KuaiSampler",
        default=sampler,
        type=str,
    )
    parser.add_argument(
        "--fl_sample",
        dest="fl_sample",
        help="sampling method for federated learning",
        default=fl_sample,
        type=str,
    )
    parser.add_argument(
        "--fl_num_users",
        dest="fl_num_users",
        help="number of users for federated learning",
        default=fl_num_users,
        type=int,
    )
    parser.add_argument(
        "--fl_local_ep",
        dest="fl_local_ep",
        help="number of local epochs for federated learning",
        default=fl_local_ep,
        type=int,
    )
    parser.add_argument(
        "--fl_local_bs",
        dest="fl_local_bs",
        help="local batch size for federated learning",
        default=fl_local_bs,
        type=float,
    )
    parser.add_argument(
        "--fl_frac",
        dest="fl_frac",
        help="the fraction of clients for federated learning",
        default=fl_frac,
        type=float,
    )
    parser.add_argument(
        "--ctx",
        dest="ctx",
        help="Running Context -- gpu num or cpu. E.g `--ctx 0 1` means run code in gpu 0 and 1; -1 means cpu",
        default=ctx,
        nargs="+",
    )
    parser.add_argument("--rs", dest="rs", help="random seed", default=rs, type=int)
    parser.add_argument("--te", dest="te", help="test epoch", default=te, type=int)
    # There is something wrong with "bool", so I used 1 as True, 0 as False
    parser.add_argument(
        "--train_mode",
        dest="train_mode",
        help="If 0, no train or test, just read data; else train + test",
        default=train_mode,
        type=int,
    )
    parser.add_argument(
        "--train_epoch",
        dest="train_epoch",
        help="epoches of training period",
        default=train_epoch,
        type=int,
    )
    parser.add_argument(
        "--save_epoch",
        dest="save_epoch",
        help="save for every save_epoch epoches",
        default=save_epoch,
        type=int,
    )
    parser.add_argument(
        "--save_iter",
        dest="save_iter",
        help="save for every save_iter in save_epoches",
        default=save_iter,
        type=int,
    )
    parser.add_argument(
        "--loss_func",
        dest="loss_func",
        help="choose loss function",
        default=loss_func,
        type=str,
    )
    parser.add_argument(
        "--loss_param",
        dest="loss_param",
        help="choose parameters of loss function",
        default=loss_param,
        type=json.loads,
    )
    parser.add_argument(
        "--train_period",
        dest="train_period",
        help="The training period",
        default=train_period,
        nargs="+",
    )
    parser.add_argument(
        "--valid_period",
        dest="valid_period",
        help="The validating period",
        default=valid_period,
        nargs="+",
    )
    parser.add_argument(
        "--test_period",
        dest="test_period",
        help="The test period",
        default=test_period,
        nargs="+",
    )
    parser.add_argument(
        "--batch_size",
        dest="batch_size",
        help="batch_size",
        default=batch_size,
        type=int,
    )
    parser.add_argument(
        "--rho",
        dest="rho",
        help="length of time sequence when training",
        default=rho,
        type=int,
    )
    parser.add_argument(
        "--model_type",
        dest="model_type",
        help="The type of DL model",
        default=model_type,
        type=str,
    )
    parser.add_argument(
        "--model_name",
        dest="model_name",
        help="The name of DL model. now in the zoo",
        default=model_name,
        type=str,
    )
    parser.add_argument(
        "--weight_path",
        dest="weight_path",
        help="The weights of trained model",
        default=weight_path,
        type=str,
    )
    parser.add_argument(
        "--weight_path_add",
        dest="weight_path_add",
        help="More info about the weights of trained model",
        default=weight_path_add,
        type=json.loads,
    )
    parser.add_argument(
        "--continue_train",
        dest="continue_train",
        help="Continue to train the model from weight_path when continue_train>0",
        default=continue_train,
        type=int,
    )
    parser.add_argument(
        "--gage_id",
        dest="gage_id",
        help="just select some sites",
        default=gage_id,
        nargs="+",
    )
    parser.add_argument(
        "--gage_id_screen",
        dest="gage_id_screen",
        help="the criterion to chose some gages",
        default=gage_id_screen,
        type=json.loads,
    )
    parser.add_argument(
        "--gage_id_file",
        dest="gage_id_file",
        help="select some sites from a file",
        default=gage_id_file,
        type=str,
    )
    parser.add_argument(
        "--opt", dest="opt", help="choose an optimizer", default=opt, type=str
    )
    parser.add_argument(
        "--opt_param",
        dest="opt_param",
        help="the optimizer parameters",
        default=opt_param,
        type=json.loads,
    )
    parser.add_argument(
        "--var_c", dest="var_c", help="types of attributes", default=var_c, nargs="+"
    )
    parser.add_argument(
        "--c_rm_nan",
        dest="c_rm_nan",
        help="if true, we remove NaN value for var_c data when scaling",
        default=c_rm_nan,
        type=int,
    )
    parser.add_argument(
        "--var_t", dest="var_t", help="types of forcing", default=var_t, nargs="+"
    )
    parser.add_argument(
        "--t_rm_nan",
        dest="t_rm_nan",
        help="if true, we remove NaN value for var_t data when scaling",
        default=t_rm_nan,
        type=int,
    )
    parser.add_argument(
        "--var_t_type",
        dest="var_t_type",
        help="types of forcing data_source",
        default=var_t_type,
        nargs="+",
    )
    parser.add_argument(
        "--var_o",
        dest="var_o",
        help="more other inputs except for var_c and var_t",
        default=var_o,
        type=json.loads,
    )
    parser.add_argument(
        "--var_out", dest="var_out", help="type of outputs", default=var_out, nargs="+"
    )
    parser.add_argument(
        "--out_rm_nan",
        dest="out_rm_nan",
        help="if true, we remove NaN value for var_out data when scaling",
        default=out_rm_nan,
        type=int,
    )
    parser.add_argument(
        "--target_as_input",
        dest="target_as_input",
        help="if true, we will use target data as input for data assimilation or physics-based models",
        default=target_as_input,
        type=int,
    )
    parser.add_argument(
        "--constant_only",
        dest="constant_only",
        help="if true, we will only use attribute data as input for deep learning models; "
        "now it is only for dpl models and it is only used when target_as_input is False",
        default=constant_only,
        type=int,
    )
    parser.add_argument(
        "--n_output",
        dest="n_output",
        help="the number of output features",
        default=n_output,
        type=int,
    )
    parser.add_argument(
        "--model_hyperparam",
        dest="model_hyperparam",
        help="the model_hyperparam in model_cfgs",
        default=model_hyperparam,
        type=json.loads,
    )
    parser.add_argument(
        "--metrics",
        dest="metrics",
        help="The evaluating metrics",
        default=metrics,
        nargs="+",
    )
    parser.add_argument(
        "--fill_nan",
        dest="fill_nan",
        help="how to fill nan values when evaluating",
        default=fill_nan,
        nargs="+",
    )
    parser.add_argument(
        "--warmup_length",
        dest="warmup_length",
        help="Physical hydro models need warmup",
        default=warmup_length,
        type=int,
    )
    parser.add_argument(
        "--start_epoch",
        dest="start_epoch",
        help="The index of epoch when starting training, default is 1. "
        "When retraining after an interrupt, it will be larger than 1",
        default=start_epoch,
        type=int,
    )
    parser.add_argument(
        "--stat_dict_file",
        dest="stat_dict_file",
        help="for testing sometimes such as pub cases, we need stat_dict_file from trained dataset",
        default=stat_dict_file,
        type=str,
    )
    parser.add_argument(
        "--model_wrapper",
        dest="model_wrapper",
        help="Sometimes we need a wrapper for the DL models to add some functions",
        default=model_wrapper,
        type=str,
    )
    parser.add_argument(
        "--model_wrapper_param",
        dest="model_wrapper_param",
        help="The parameters for model_wrapper",
        default=model_wrapper_param,
        type=json.loads,
    )
    parser.add_argument(
        "--num_workers",
        dest="num_workers",
        help="The number of workers used in Dataloader",
        default=num_workers,
        type=int,
    )
    parser.add_argument(
        "--train_but_not_real",
        dest="train_but_not_real",
        help="If true, we will enter the training function but not really train the model and just use the trained model during training",
        default=train_but_not_real,
        type=int,
    )
    parser.add_argument(
        "--which_first_tensor",
        dest="which_first_tensor",
        help="sequence_first or batch_first",
        default=which_first_tensor,
        type=str,
    )
    parser.add_argument(
        "--lr_scheduler",
        dest="lr_scheduler",
        help="The learning rate scheduler",
        default=lr_scheduler,
        type=json.loads,
    )
    parser.add_argument(
        "--ensemble",
        dest="ensemble",
        help="ensemble config",
        default=ensemble,
        type=int,
    )
    parser.add_argument(
        "--ensemble_items",
        dest="ensemble_items",
        help="ensemble config",
        default=ensemble_items,
        type=json.loads,
    )
    # To make pytest work in PyCharm, here we use the following code instead of "args = parser.parse_args()":
    # https://blog.csdn.net/u014742995/article/details/100119905
    args, unknown = parser.parse_known_args()
    return args


def update_nested_dict(d, keys, value):
    """update nested dict

    Parameters
    ----------
    d
        the dict to be updated
    keys
        the keys of the dict
    value
        the updated value of the dict
    """
    if len(keys) == 1:
        d[keys[0]] = value
    else:
        update_nested_dict(d[keys[0]], keys[1:], value)


def update_cfg(cfg_file, new_args):
    """
    Update default config with new arguments

    Parameters
    ----------
    cfg_file
        default config
    new_args
        new arguments

    Returns
    -------
    None
        in-place operation for cfg_file
    """
    print("update config file")
    project_dir = os.getcwd()
    result_dir = os.path.join(project_dir, "results")
    if os.path.exists(result_dir) is False:
        os.makedirs(result_dir)
    if new_args.sub is not None:
        subset, subexp = new_args.sub.split("/")
        cfg_file["data_cfgs"]["validation_path"] = os.path.join(
            project_dir, "results", subset, subexp
        )
        cfg_file["data_cfgs"]["test_path"] = os.path.join(result_dir, subset, subexp)
    if new_args.source is not None:
        cfg_file["data_cfgs"]["data_source_name"] = new_args.source
    if new_args.source_path is not None:
        cfg_file["data_cfgs"]["data_path"] = new_args.source_path
        if type(new_args.source_path) == list and len(new_args.source_path) == 1:
            cfg_file["data_cfgs"]["data_path"] = new_args.source_path[0]
    if new_args.source_region is not None:
        cfg_file["data_cfgs"]["data_region"] = new_args.source_region
        if len(new_args.source_region) == 1:
            cfg_file["data_cfgs"]["data_region"] = new_args.source_region[0]
    if new_args.download is not None:
        if new_args.download == 0:
            cfg_file["data_cfgs"]["download"] = False
        else:
            cfg_file["data_cfgs"]["download"] = True
    if new_args.scaler is not None:
        cfg_file["data_cfgs"]["scaler"] = new_args.scaler
    if new_args.scaler_params is not None:
        cfg_file["data_cfgs"]["scaler_params"] = new_args.scaler_params
    if new_args.dataset is not None:
        cfg_file["data_cfgs"]["dataset"] = new_args.dataset
    if new_args.sampler is not None:
        cfg_file["data_cfgs"]["sampler"] = new_args.sampler
    if new_args.fl_sample is not None:
        if new_args.fl_sample not in ["basin", "region"]:
            # basin means each client is a basin
            raise ValueError("fl_sample must be 'basin' or 'region'")
        cfg_file["model_cfgs"]["fl_hyperparam"]["fl_sample"] = new_args.fl_sample
    if new_args.fl_num_users is not None:
        cfg_file["model_cfgs"]["fl_hyperparam"]["fl_num_users"] = new_args.fl_num_users
    if new_args.fl_local_ep is not None:
        cfg_file["model_cfgs"]["fl_hyperparam"]["fl_local_ep"] = new_args.fl_local_ep
    if new_args.fl_local_bs is not None:
        cfg_file["model_cfgs"]["fl_hyperparam"]["fl_local_bs"] = new_args.fl_local_bs
    if new_args.fl_frac is not None:
        cfg_file["model_cfgs1"]["fl_hyperparam"]["fl_frac"] = new_args.fl_frac
    if new_args.ctx is not None:
        cfg_file["training_cfgs"]["device"] = new_args.ctx
    if new_args.rs is not None:
        cfg_file["training_cfgs"]["random_seed"] = new_args.rs
    if new_args.train_mode is not None:
        if new_args.train_mode > 0:
            cfg_file["training_cfgs"]["train_mode"] = True
        else:
            cfg_file["training_cfgs"]["train_mode"] = False
    if new_args.loss_func is not None:
        cfg_file["training_cfgs"]["criterion"] = new_args.loss_func
        if new_args.loss_param is not None:
            cfg_file["training_cfgs"]["criterion_params"] = new_args.loss_param
    if new_args.train_period is not None:
        cfg_file["data_cfgs"]["t_range_train"] = new_args.train_period
    if new_args.valid_period is not None:
        cfg_file["data_cfgs"]["t_range_valid"] = new_args.valid_period
    if new_args.test_period is not None:
        cfg_file["data_cfgs"]["t_range_test"] = new_args.test_period
    if new_args.gage_id is not None or new_args.gage_id_file is not None:
        if new_args.gage_id_file is not None:
            gage_id_lst = (
                pd.read_csv(new_args.gage_id_file, dtype={0: str}).iloc[:, 0].values
            )
            cfg_file["data_cfgs"]["object_ids"] = gage_id_lst.tolist()
        else:
            cfg_file["data_cfgs"]["object_ids"] = new_args.gage_id
    if new_args.opt is not None:
        cfg_file["training_cfgs"]["optimizer"] = new_args.opt
        if new_args.opt_param is not None:
            cfg_file["training_cfgs"]["optim_params"] = new_args.opt_param
        else:
            cfg_file["training_cfgs"]["optim_params"] = {}
    if new_args.var_c is not None:
        # I don't find a method to receive empty list for argparse, so if we input "None" or "" or " ", we treat it as []
        if (
            new_args.var_c == ["None"]
            or new_args.var_c == [""]
            or new_args.var_c == [" "]
        ):
            cfg_file["data_cfgs"]["constant_cols"] = []
        else:
            cfg_file["data_cfgs"]["constant_cols"] = new_args.var_c
    if new_args.c_rm_nan == 0:
        cfg_file["data_cfgs"]["constant_rm_nan"] = False
    else:
        cfg_file["data_cfgs"]["constant_rm_nan"] = True
    if new_args.var_t is not None:
        cfg_file["data_cfgs"]["relevant_cols"] = new_args.var_t
    if new_args.var_t_type is not None:
        cfg_file["data_cfgs"]["relevant_types"] = new_args.var_t_type
    if new_args.t_rm_nan == 0:
        cfg_file["data_cfgs"]["relevant_rm_nan"] = False
    else:
        cfg_file["data_cfgs"]["relevant_rm_nan"] = True
    if new_args.var_o is not None:
        cfg_file["data_cfgs"]["other_cols"] = new_args.var_o
    if new_args.var_out is not None:
        cfg_file["data_cfgs"]["target_cols"] = new_args.var_out
    if new_args.out_rm_nan == 0:
        cfg_file["data_cfgs"]["target_rm_nan"] = False
    else:
        cfg_file["data_cfgs"]["target_rm_nan"] = True
    if new_args.target_as_input == 0:
        cfg_file["data_cfgs"]["target_as_input"] = False
        if new_args.constant_only == 0:
            cfg_file["data_cfgs"]["constant_only"] = False
        else:
            cfg_file["data_cfgs"]["constant_only"] = True
    else:
        cfg_file["data_cfgs"]["target_as_input"] = True
    if new_args.train_epoch is not None:
        cfg_file["training_cfgs"]["epochs"] = new_args.train_epoch
    if new_args.save_epoch is not None:
        cfg_file["training_cfgs"]["save_epoch"] = new_args.save_epoch
    if new_args.save_iter is not None:
        cfg_file["training_cfgs"]["save_iter"] = new_args.save_iter
    if new_args.model_type is not None:
        cfg_file["model_cfgs"]["model_type"] = new_args.model_type
    if new_args.model_name is not None:
        cfg_file["model_cfgs"]["model_name"] = new_args.model_name
    if new_args.weight_path is not None:
        cfg_file["model_cfgs"]["weight_path"] = new_args.weight_path
        if new_args.continue_train is None or new_args.continue_train == 0:
            continue_train = False
        else:
            continue_train = True
        cfg_file["model_cfgs"]["continue_train"] = continue_train
    if new_args.weight_path_add is not None:
        cfg_file["model_cfgs"]["weight_path_add"] = new_args.weight_path_add
    if new_args.n_output is not None:
        cfg_file["training_cfgs"]["multi_targets"] = new_args.n_output
        if len(cfg_file["data_cfgs"]["target_cols"]) != new_args.n_output:
            raise AttributeError(
                "Please make sure size of vars in data_cfgs/target_cols is same as n_output"
            )
    if new_args.model_hyperparam is None:
        if new_args.batch_size is not None:
            batch_size = new_args.batch_size
            cfg_file["model_cfgs"]["model_hyperparam"]["batch_size"] = batch_size
            cfg_file["data_cfgs"]["batch_size"] = batch_size
            cfg_file["training_cfgs"]["batch_size"] = batch_size
        if new_args.rho is not None:
            rho = new_args.rho
            cfg_file["model_cfgs"]["model_hyperparam"]["seq_length"] = rho
            cfg_file["data_cfgs"]["forecast_history"] = rho
        if new_args.n_output is not None:
            cfg_file["model_cfgs"]["model_hyperparam"][
                "output_seq_len"
            ] = new_args.n_output
    else:
        cfg_file["model_cfgs"]["model_hyperparam"] = new_args.model_hyperparam
        if "batch_size" in new_args.model_hyperparam.keys():
            cfg_file["data_cfgs"]["batch_size"] = new_args.model_hyperparam[
                "batch_size"
            ]
            cfg_file["training_cfgs"]["batch_size"] = new_args.model_hyperparam[
                "batch_size"
            ]
        elif new_args.batch_size is not None:
            batch_size = new_args.batch_size
            cfg_file["data_cfgs"]["batch_size"] = batch_size
            cfg_file["training_cfgs"]["batch_size"] = batch_size
        else:
            raise NotImplemented("Please set the batch_size!!!")
        if "seq_length" in new_args.model_hyperparam.keys():
            cfg_file["data_cfgs"]["forecast_history"] = new_args.model_hyperparam[
                "seq_length"
            ]
        elif "forecast_history" in new_args.model_hyperparam.keys():
            cfg_file["data_cfgs"]["forecast_history"] = new_args.model_hyperparam[
                "forecast_history"
            ]
        elif new_args.rho is not None:
            cfg_file["data_cfgs"]["forecast_history"] = new_args.rho
        else:
            raise NotImplemented(
                "Please set the time_sequence length in a batch when training!!!"
            )
        if (
            "output_seq_len" in new_args.model_hyperparam.keys()
            and new_args.n_output is not None
        ):
            assert new_args.model_hyperparam["output_seq_len"] == new_args.n_output
    if new_args.metrics is not None:
        cfg_file["evaluation_cfgs"]["metrics"] = new_args.metrics
    if new_args.fill_nan is not None:
        cfg_file["evaluation_cfgs"]["fill_nan"] = new_args.fill_nan
    if new_args.te is not None:
        cfg_file["evaluation_cfgs"]["test_epoch"] = new_args.te
        if new_args.train_epoch is not None and new_args.te > new_args.train_epoch:
            raise RuntimeError("testing epoch cannot be larger than training epoch")
    if new_args.warmup_length > 0:
        cfg_file["data_cfgs"]["warmup_length"] = new_args.warmup_length
        if "warmup_length" in new_args.model_hyperparam.keys() and (
            not cfg_file["data_cfgs"]["warmup_length"]
            == new_args.model_hyperparam["warmup_length"]
        ):
            raise RuntimeError(
                "Please set same warmup_length in model_cfgs and data_cfgs"
            )
    if new_args.start_epoch > 1:
        cfg_file["training_cfgs"]["start_epoch"] = new_args.start_epoch
    if new_args.stat_dict_file is not None:
        cfg_file["data_cfgs"]["stat_dict_file"] = new_args.stat_dict_file

    if new_args.model_wrapper is not None:
        cfg_file["model_cfgs"]["model_wrapper"] = new_args.model_wrapper
    if new_args.model_wrapper_param is not None:
        cfg_file["model_cfgs"]["model_wrapper_param"] = new_args.model_wrapper_param
    if new_args.num_workers is not None and new_args.num_workers > 0:
        cfg_file["training_cfgs"]["num_workers"] = new_args.num_workers
    if new_args.train_but_not_real is not None and new_args.train_but_not_real > 0:
        cfg_file["training_cfgs"]["train_but_not_real"] = True
    if new_args.which_first_tensor is not None:
        cfg_file["training_cfgs"]["which_first_tensor"] = new_args.which_first_tensor
    if new_args.lr_scheduler is not None:
        cfg_file["training_cfgs"]["lr_scheduler"] = new_args.lr_scheduler
    if new_args.ensemble == 0:
        cfg_file["training_cfgs"]["ensemble"] = False
    else:
        cfg_file["training_cfgs"]["ensemble"] = True
    if new_args.ensemble_items is not None:
        cfg_file["training_cfgs"]["ensemble_items"] = new_args.ensemble_items
    # print("the updated config:\n", json.dumps(cfg_file, indent=4, ensure_ascii=False))


def get_config_file(cfg_dir):
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
    cfg = hydro_file.unserialize_json(cfg_file)
    return cfg
