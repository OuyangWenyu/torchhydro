"""
Author: Wenyu Ouyang
Date: 2023-11-21 07:20:41
LastEditTime: 2023-11-24 09:44:14
LastEditors: Wenyu Ouyang
Description: Test weight analysis
FilePath: \torchhydro\tests\test_weight_analysis.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""
import itertools
import os
import matplotlib
import numpy as np
import pandas as pd
import pytest

import hydrodataset as hds

from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.explainers.weight_anlysis import (
    plot_param_hist_model_fold,
)
from torchhydro.trainers.trainer import ensemble_train_and_evaluate


@pytest.fixture()
def hist_stat_dir():
    hist_stat_dir = os.path.join(
        "results",
        "hist_statistic",
    )
    if not os.path.exists(hist_stat_dir):
        os.makedirs(hist_stat_dir)
    return hist_stat_dir


@pytest.fixture()
def var_c_target():
    return [
        "p_mean",
        "pet_mean",
        "Area",
        "geol_class_1st",
        "elev",
        "SNDPPT",
    ]


@pytest.fixture()
def var_c_source():
    return [
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
    ]


@pytest.fixture()
def var_t_target():
    # mainly from ERA5LAND
    return ["total_precipitation", "potential_evaporation", "temperature_2m"]


@pytest.fixture()
def var_t_source():
    return ["dayl", "prcp", "srad", "tmax", "tmin", "vp"]


@pytest.fixture()
def gage_id():
    return [
        "61561",
        "62618",
    ]


def test_run_lstm_cross_val(var_c_target, var_t_target):
    config = default_config_file()
    project_name = "camels/expcccv"
    all_exps = ["exp61561", "exp62618"]
    kfold = 2
    train_period = ["2018-10-01", "2021-10-01"]
    valid_period = ["2015-10-01", "2018-10-01"]
    args = cmd(
        sub=project_name,
        source="CAMELS",
        source_path=os.path.join(
            hds.ROOT_DIR, "waterism", "datasets-interim", "camels_cc"
        ),
        download=0,
        ctx=[0],
        model_name="KuaiLSTM",
        model_hyperparam={
            "n_input_features": len(var_c_target) + len(var_t_target),
            "n_output_features": 1,
            "n_hidden_states": 256,
        },
        opt="Adadelta",
        # opt_param=opt_param,
        loss_func="RMSESum",
        train_period=train_period,
        test_period=valid_period,
        batch_size=20,
        rho=365,
        scaler="DapengScaler",
        dataset="StreamflowDataset",
        continue_train=True,
        warmup_length=0,
        train_epoch=100,
        var_t=var_t_target,
        var_t_type="era5land",
        var_c=var_c_target,
        var_out="streamflow",
        gage_id=gage_id,
        ensemble_cfgs={
            "ensemble": True,
            "model_names": ["lstm", "tl-lstm"],
            "exps": all_exps,
            "kfold": kfold,
            "batch_sizes": [20, 50, 100, 200, 300],
        },
    )
    update_cfg(config, args)
    ensemble_train_and_evaluate(config)
    print("All processes are finished!")


def test_run_cross_val_tlcamelsus2cc(
    var_c_source, var_c_target, var_t_source, var_t_target
):
    weight_dir = os.path.join(
        os.getcwd(),
        "results",
        "test_camels",
        "exp1",
    )
    weight_path = get_lastest_file_in_a_dir(weight_dir)
    project_name = "test_camels/exptl4cccv"
    args = cmd(
        sub=project_name,
        source="SelfMadeCAMELS",
        # cc means China continent
        source_path=os.path.join(
            hds.ROOT_DIR, "waterism", "datasets-interim", "camels_cc"
        ),
        download=0,
        ctx=[0],
        model_type="TransLearn",
        model_name="KaiLSTM",
        model_hyperparam={
            "linear_size": len(var_c_target) + len(var_t_target),
            "n_input_features": len(var_c_source) + len(var_t_source),
            "n_output_features": 1,
            "n_hidden_states": 256,
        },
        opt="Adadelta",
        loss_func="RMSESum",
        batch_size=5,
        rho=20,
        rs=1234,
        train_period=["2014-10-01", "2019-10-01"],
        test_period=["2019-10-01", "2021-10-01"],
        scaler="DapengScaler",
        sampler="KuaiSampler",
        dataset="StreamflowDataset",
        weight_path=weight_path,
        weight_path_add={
            "freeze_params": ["lstm.b_hh", "lstm.b_ih", "lstm.w_hh", "lstm.w_ih"]
        },
        continue_train=True,
        train_epoch=20,
        te=20,
        save_epoch=10,
        var_t=var_t_target,
        var_c=var_c_target,
        var_out=["streamflow"],
        gage_id=[
            "61561",
            "62618",
        ],
        ensemble_cfgs={
            "ensemble": True,
            "model_names": ["lstm", "tl-lstm"],
            "exps": all_exps,
            "kfold": 2,
            "batch_sizes": [20, 50, 100, 200, 300],
            "t_range_train": train_periods,
            "t_range_valid": None,
            "t_range_test": valid_periods,
            "other_cfg": best_batchsize,
        },
    )
    cfg = default_config_file()
    update_cfg(cfg, args)
    ensemble_train_and_evaluate(cfg)
    print("All processes are finished!")


def test_weight_analysis(hist_stat_dir):
    basin_ids = [
        "61561",
        "62618",
    ]
    # NOTICE: THE ORDER CANNOT BE MODIFIED WITHOUT DEBUGGING THE CODE IN plot_param_hist_model_fold
    chosen_layer_for_hist = [
        "former_linear.bias",
        "former_linear.weight",
        "linearIn.bias",
        "linearIn.weight",
        "linearOut.bias",
        "linearOut.weight",
        "lstm.b_hh",
        "lstm.b_ih",
        "lstm.w_hh",
        "lstm.w_ih",
    ]
    # too many figures lead to "Fail to allocate bitmap"
    matplotlib.use("Agg")
    show_hist_bs = [20, 50, 100, 200]
    model_names = ["local_model", "transfer_learning_model"]
    kfold = [2, 2]
    exp61561 = ["exp6156100", "exp6156110"]
    exp62618 = ["exp6261800", "exp6261810"]
    exps = [exp61561, exp62618]
    best_epoch = [[58, 56], [86, 26], [85, 69]]
    for i, basin_id in enumerate(basin_ids):
        for j, fold, show_hist_b in itertools.product(
            range(len(model_names)), range(kfold), show_hist_bs
        ):
            _, chosen_layers_consine = plot_param_hist_model_fold(
                model_names,
                exps[i],
                best_epoch[i],
                show_hist_b,
                chosen_layer_for_hist,
                basin_id,
                fold,
            )
            pd.DataFrame(chosen_layers_consine).to_csv(
                os.path.join(
                    hist_stat_dir,
                    f"{basin_id}_bs{show_hist_b}_fold{fold}_chosen_layer_consine.csv",
                )
            )

    # integrate to one file for each basin
    row_names = [
        "linearIn.weight",
        "linearIn.bias",
        "lstm.w_ih",
        "lstm.b_ih",
        "lstm.w_hh",
        "lstm.b_hh",
        "linearOut.weight",
        "linearOut.bias",
        "former_linear.weight",
        "former_linear.bias",
    ]
    # NOTICE: ONLY SUPPORT two fold, two models
    fold_num = 2
    model_num = 2
    for basin_id in basin_ids:
        basin_mat = np.full(
            (len(row_names) * fold_num, len(show_hist_bs) * model_num), np.nan
        )
        for j, fold, show_hist_b in itertools.product(
            range(len(model_names)), range(kfold), show_hist_bs
        ):
            cosine_sim = pd.read_csv(
                os.path.join(
                    hist_stat_dir,
                    f"{basin_id}_bs{show_hist_b}_fold{fold}_chosen_layer_consine.csv",
                ),
                index_col=0,
            )
            for i in range(len(row_names)):
                mat_row = i * fold_num + fold
                mat_col = show_hist_bs.index(show_hist_b) * model_num + j
                basin_mat[mat_row, mat_col] = cosine_sim[row_names[i]][model_names[j]]
        pd.DataFrame(basin_mat).round(3).to_csv(
            os.path.join(hist_stat_dir, f"{basin_id}_basin_mat.csv")
        )
