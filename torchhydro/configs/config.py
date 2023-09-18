"""
Author: Wenyu Ouyang
Date: 2021-12-31 11:08:29
LastEditTime: 2023-07-30 14:41:09
LastEditors: Wenyu Ouyang
Description: Config for hydroDL
FilePath: \torchhydro\torchhydro\datasets\config.py
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
from torchhydro.datasets.data_constant import (
    DAYMET_NAME,
    ET_MODIS_NAME,
    PET_MODIS_NAME,
    PET_NLDAS_NAME,
    PRCP_NLDAS_NAME,
    Q_CAMELS_CC_NAME,
    SSM_SMAP_NAME,
    PRCP_DAYMET_NAME,
    Q_CAMELS_US_NAME,
)


def default_config_file():
    """
    Default config file for all models/data/training parameters in this repo

    Returns
    -------
    dict
        configurations
    """

    config_default = {
        "model_params": {
            # now only PyTorch is supported
            "model_type": "PyTorch",
            # supported models can be seen in hydroDL/model_dict_function.py
            "model_name": "LSTM",
            # the details of model parameters for the "model_name" model
            "model_param": {
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
            # the name of the model's wrapper class
            "model_wrapper": None,
            # the wrapper class's parameters
            "model_wrapper_param": None,
        },
        "data_params": {
            "data_source_name": "CAMELS",
            "data_path": "../../example/camels_us",
            "data_region": None,
            "download": True,
            "cache_read": True,
            "cache_write": False,
            "cache_path": None,
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
            # data_loader for loading data to models
            "data_loader": "StreamflowDataset",
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
            },
            "stat_dict_file": None,
        },
        "training_params": {
            # if train_mode is False, don't train and evaluate
            "train_mode": True,
            "criterion": "RMSE",
            "criterion_params": None,
            "optimizer": "Adam",
            "optim_params": {
                "lr": 0.001,
            },
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
        },
        # For evaluation
        "evaluate_params": {"metrics": ["NSE"], "fill_nan": "no", "test_epoch": 20},
    }
    return config_default


def cmd(
    sub=None,
    source="CAMELS",
    source_path=None,
    source_region=None,
    download=0,
    scaler=None,
    scaler_params=None,
    data_loader=None,
    ctx=None,
    rs=None,
    gage_id_file=None,
    gage_id=None,
    train_period=None,
    valid_period=None,
    test_period=None,
    opt=None,
    cache_read=None,
    cache_write=None,
    cache_path=None,
    opt_param=None,
    batch_size=None,
    rho=None,
    train_mode=None,
    train_epoch=None,
    save_epoch=None,
    save_iter=None,
    te=None,
    model_name=None,
    weight_path=None,
    continue_train=None,
    var_c=None,
    c_rm_nan=1,
    var_t=None,
    t_rm_nan=1,
    n_output=None,
    loss_func=None,
    model_param=None,
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
        "--data_loader",
        dest="data_loader",
        help="Choose a data loader class",
        default=data_loader,
        type=str,
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
        "--cache_read",
        dest="cache_read",
        help="read binary file",
        default=cache_read,
        type=int,
    )
    parser.add_argument(
        "--cache_write",
        dest="cache_write",
        help="write binary file",
        default=cache_write,
        type=int,
    )
    parser.add_argument(
        "--cache_path",
        dest="cache_path",
        help="specify the directory of data cache files",
        default=cache_path,
        type=str,
    )
    parser.add_argument(
        "--model_param",
        dest="model_param",
        help="the model_param in model_params",
        default=model_param,
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
    # To make pytest work in PyCharm, here we use the following code instead of "args = parser.parse_args()":
    # https://blog.csdn.net/u014742995/article/details/100119905
    args, unknown = parser.parse_known_args()
    return args


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
        if not os.path.exists(os.path.join(result_dir, subset, subexp)):
            os.makedirs(os.path.join(result_dir, subset, subexp))
        cfg_file["data_params"]["validation_path"] = os.path.join(
            project_dir, "results", subset, subexp
        )
        cfg_file["data_params"]["test_path"] = os.path.join(result_dir, subset, subexp)
        if new_args.cache_path is not None:
            cfg_file["data_params"]["cache_path"] = new_args.cache_path
        else:
            cfg_file["data_params"]["cache_path"] = os.path.join(
                result_dir, subset, subexp
            )
    if new_args.source is not None:
        cfg_file["data_params"]["data_source_name"] = new_args.source
    if new_args.source_path is not None:
        cfg_file["data_params"]["data_path"] = new_args.source_path
        if len(new_args.source_path) == 1:
            cfg_file["data_params"]["data_path"] = new_args.source_path[0]
    if new_args.source_region is not None:
        cfg_file["data_params"]["data_region"] = new_args.source_region
        if len(new_args.source_region) == 1:
            cfg_file["data_params"]["data_region"] = new_args.source_region[0]
    if new_args.download is not None:
        if new_args.download == 0:
            cfg_file["data_params"]["download"] = False
        else:
            cfg_file["data_params"]["download"] = True
    if new_args.scaler is not None:
        cfg_file["data_params"]["scaler"] = new_args.scaler
    if new_args.scaler_params is not None:
        cfg_file["data_params"]["scaler_params"] = new_args.scaler_params
    if new_args.data_loader is not None:
        cfg_file["data_params"]["data_loader"] = new_args.data_loader
    if new_args.ctx is not None:
        cfg_file["training_params"]["device"] = new_args.ctx
    if new_args.rs is not None:
        cfg_file["training_params"]["random_seed"] = new_args.rs
    if new_args.train_mode is not None:
        if new_args.train_mode > 0:
            cfg_file["training_params"]["train_mode"] = True
        else:
            cfg_file["training_params"]["train_mode"] = False
    if new_args.loss_func is not None:
        cfg_file["training_params"]["criterion"] = new_args.loss_func
        if new_args.loss_param is not None:
            cfg_file["training_params"]["criterion_params"] = new_args.loss_param
    if new_args.train_period is not None:
        cfg_file["data_params"]["t_range_train"] = new_args.train_period
    if new_args.valid_period is not None:
        cfg_file["data_params"]["t_range_valid"] = new_args.valid_period
    if new_args.test_period is not None:
        cfg_file["data_params"]["t_range_test"] = new_args.test_period
    if new_args.gage_id is not None or new_args.gage_id_file is not None:
        if new_args.gage_id_file is not None:
            gage_id_lst = (
                pd.read_csv(new_args.gage_id_file, dtype={0: str}).iloc[:, 0].values
            )
            cfg_file["data_params"]["object_ids"] = gage_id_lst.tolist()
        else:
            cfg_file["data_params"]["object_ids"] = new_args.gage_id
    if new_args.opt is not None:
        cfg_file["training_params"]["optimizer"] = new_args.opt
        if new_args.opt_param is not None:
            cfg_file["training_params"]["optim_params"] = new_args.opt_param
        else:
            cfg_file["training_params"]["optim_params"] = {}
    if new_args.var_c is not None:
        # I don't find a method to receive empty list for argparse, so if we input "None" or "" or " ", we treat it as []
        if (
            new_args.var_c == ["None"]
            or new_args.var_c == [""]
            or new_args.var_c == [" "]
        ):
            cfg_file["data_params"]["constant_cols"] = []
        else:
            cfg_file["data_params"]["constant_cols"] = new_args.var_c
    if new_args.c_rm_nan == 0:
        cfg_file["data_params"]["constant_rm_nan"] = False
    else:
        cfg_file["data_params"]["constant_rm_nan"] = True
    if new_args.var_t is not None:
        cfg_file["data_params"]["relevant_cols"] = new_args.var_t
    if new_args.var_t_type is not None:
        cfg_file["data_params"]["relevant_types"] = new_args.var_t_type
    if new_args.t_rm_nan == 0:
        cfg_file["data_params"]["relevant_rm_nan"] = False
    else:
        cfg_file["data_params"]["relevant_rm_nan"] = True
    if new_args.var_o is not None:
        cfg_file["data_params"]["other_cols"] = new_args.var_o
    if new_args.var_out is not None:
        cfg_file["data_params"]["target_cols"] = new_args.var_out
    if new_args.out_rm_nan == 0:
        cfg_file["data_params"]["target_rm_nan"] = False
    else:
        cfg_file["data_params"]["target_rm_nan"] = True
    if new_args.target_as_input == 0:
        cfg_file["data_params"]["target_as_input"] = False
        if new_args.constant_only == 0:
            cfg_file["data_params"]["constant_only"] = False
        else:
            cfg_file["data_params"]["constant_only"] = True
    else:
        cfg_file["data_params"]["target_as_input"] = True
    if new_args.train_epoch is not None:
        cfg_file["training_params"]["epochs"] = new_args.train_epoch
    if new_args.save_epoch is not None:
        cfg_file["training_params"]["save_epoch"] = new_args.save_epoch
    if new_args.save_iter is not None:
        cfg_file["training_params"]["save_iter"] = new_args.save_iter
    if new_args.cache_read is not None:
        if new_args.cache_read > 0:
            cfg_file["data_params"]["cache_read"] = True
        else:
            cfg_file["data_params"]["cache_read"] = False
    if new_args.cache_write is not None:
        if new_args.cache_write > 0:
            cfg_file["data_params"]["cache_write"] = True
            if not cfg_file["data_params"]["cache_read"]:
                logging.warning(
                    "Since you have chosen cache_write, please read data from cache after it is saved"
                )
        else:
            cfg_file["data_params"]["cache_write"] = False
    if new_args.model_name is not None:
        cfg_file["model_params"]["model_name"] = new_args.model_name
    if new_args.weight_path is not None:
        cfg_file["model_params"]["weight_path"] = new_args.weight_path
        if new_args.continue_train is None or new_args.continue_train == 0:
            continue_train = False
        else:
            continue_train = True
        cfg_file["model_params"]["continue_train"] = continue_train
    if new_args.weight_path_add is not None:
        cfg_file["model_params"]["weight_path_add"] = new_args.weight_path_add
    if new_args.n_output is not None:
        cfg_file["training_params"]["multi_targets"] = new_args.n_output
        if len(cfg_file["data_params"]["target_cols"]) != new_args.n_output:
            raise AttributeError(
                "Please make sure size of vars in data_params/target_cols is same as n_output"
            )
    if new_args.model_param is None:
        if new_args.batch_size is not None:
            batch_size = new_args.batch_size
            cfg_file["model_params"]["model_param"]["batch_size"] = batch_size
            cfg_file["data_params"]["batch_size"] = batch_size
            cfg_file["training_params"]["batch_size"] = batch_size
        if new_args.rho is not None:
            rho = new_args.rho
            cfg_file["model_params"]["model_param"]["seq_length"] = rho
            cfg_file["data_params"]["forecast_history"] = rho
        if new_args.n_output is not None:
            cfg_file["model_params"]["model_param"][
                "output_seq_len"
            ] = new_args.n_output
    else:
        cfg_file["model_params"]["model_param"] = new_args.model_param
        if "batch_size" in new_args.model_param.keys():
            cfg_file["data_params"]["batch_size"] = new_args.model_param["batch_size"]
            cfg_file["training_params"]["batch_size"] = new_args.model_param[
                "batch_size"
            ]
        elif new_args.batch_size is not None:
            batch_size = new_args.batch_size
            cfg_file["data_params"]["batch_size"] = batch_size
            cfg_file["training_params"]["batch_size"] = batch_size
        else:
            raise NotImplemented("Please set the batch_size!!!")
        if "seq_length" in new_args.model_param.keys():
            cfg_file["data_params"]["forecast_history"] = new_args.model_param[
                "seq_length"
            ]
        elif "forecast_history" in new_args.model_param.keys():
            cfg_file["data_params"]["forecast_history"] = new_args.model_param[
                "forecast_history"
            ]
        elif new_args.rho is not None:
            cfg_file["data_params"]["forecast_history"] = new_args.rho
        else:
            raise NotImplemented(
                "Please set the time_sequence length in a batch when training!!!"
            )
        if (
            "output_seq_len" in new_args.model_param.keys()
            and new_args.n_output is not None
        ):
            assert new_args.model_param["output_seq_len"] == new_args.n_output
    if new_args.metrics is not None:
        cfg_file["evaluate_params"]["metrics"] = new_args.metrics
    if new_args.fill_nan is not None:
        cfg_file["evaluate_params"]["fill_nan"] = new_args.fill_nan
    if new_args.te is not None:
        cfg_file["evaluate_params"]["test_epoch"] = new_args.te
        if new_args.train_epoch is not None and new_args.te > new_args.train_epoch:
            raise RuntimeError("testing epoch cannot be larger than training epoch")
    if new_args.warmup_length > 0:
        cfg_file["data_params"]["warmup_length"] = new_args.warmup_length
        if "warmup_length" in new_args.model_param.keys() and (
            not cfg_file["data_params"]["warmup_length"]
            == new_args.model_param["warmup_length"]
        ):
            raise RuntimeError(
                "Please set same warmup_length in model_params and data_params"
            )
    if new_args.start_epoch > 1:
        cfg_file["training_params"]["start_epoch"] = new_args.start_epoch
    if new_args.stat_dict_file is not None:
        cfg_file["data_params"]["stat_dict_file"] = new_args.stat_dict_file

    if new_args.model_wrapper is not None:
        cfg_file["model_params"]["model_wrapper"] = new_args.model_wrapper
    if new_args.model_wrapper_param is not None:
        cfg_file["model_params"]["model_wrapper_param"] = new_args.model_wrapper_param
    if new_args.num_workers is not None and new_args.num_workers > 0:
        cfg_file["training_params"]["num_workers"] = new_args.num_workers
    if new_args.train_but_not_real is not None and new_args.train_but_not_real > 0:
        cfg_file["training_params"]["train_but_not_real"] = True
    if new_args.which_first_tensor is not None:
        cfg_file["training_params"]["which_first_tensor"] = new_args.which_first_tensor
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
