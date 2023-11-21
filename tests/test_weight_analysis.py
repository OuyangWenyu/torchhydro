"""
Author: Wenyu Ouyang
Date: 2023-11-21 07:20:41
LastEditTime: 2023-11-21 08:55:01
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

from explainers.weight_anlysis import (
    plot_param_hist_model_fold,
)


@pytest.fixture()
def hist_stat_dir():
    hist_stat_dir = os.path.join(
        "results",
        "hist_statistic",
    )
    if not os.path.exists(hist_stat_dir):
        os.makedirs(hist_stat_dir)
    return hist_stat_dir

def camels_cc_lstm_model(
    target_exp,
    target_dir="gages",
    opt="Adadelta",
    opt_param=None,
    random_seed=1234,
    batch_size=10,
    epoch=100,
    save_epoch=10,
    data_loader="StreamflowDataModel",
    var_c=VAR_C_CHOSEN_FROM_CAMELS_CC,
    var_t=VAR_T_CHOSEN_FROM_ERA5LAND,
    train_period=["2014-10-01", "2019-10-01"],
    test_period=["2020-10-01", "2021-10-01"],
    valid_period=["2019-10-01", "2020-10-01"],
    gage_id_file=os.path.join(
        definitions.ROOT_DIR,
        "hydroSPB",
        "example",
        "camels",
        "camels_cc_2014_2020_flow_screen.csv",
    ),
    gage_id=None,
    save_iter=0,
    num_workers=4,
):
    config = default_config_file()
    project_name = target_dir + "/" + target_exp
    if gage_id is None:
        if gage_id_file is None:
            gage_id = "ALL"
        else:
            gage_id = (
                pd.read_csv(gage_id_file, dtype={0: str}).iloc[:, 0].values.tolist()
            )
    args = cmd(
        sub=project_name,
        source="CAMELS",
        source_path=os.path.join(definitions.DATASET_DIR, "camels", "camels_cc"),
        source_region="CC",
        cache_write=1,
        cache_read=1,
        download=0,
        ctx=[0],
        model_name="KuaiLSTM",
        model_param={
            "n_input_features": len(var_c) + len(var_t),
            "n_output_features": 1,
            "n_hidden_states": 256,
        },
        opt=opt,
        opt_param=opt_param,
        loss_func="RMSESum",
        batch_size=batch_size,
        rho=365,
        rs=random_seed,
        train_period=train_period,
        test_period=test_period,
        valid_period=valid_period,
        scaler="DapengScaler",
        data_loader=data_loader,
        continue_train=True,
        warmup_length=0,
        train_epoch=epoch,
        te=epoch,
        save_epoch=save_epoch,
        var_t=var_t,
        var_t_type=[ERA5LAND_NAME],
        var_c=var_c,
        var_out=[Q_CAMELS_CC_NAME],
        gage_id=gage_id,
        save_iter=save_iter,
        num_workers=num_workers,
    )
    update_cfg(config, args)
    train_and_evaluate(config)
    print("All processes are finished!")
def transfer_gages_lstm_model_to_camelscc(
    source_exp,
    target_exp,
    source_dir="gages",
    target_dir="gages",
    opt="Adadelta",
    opt_param=None,
    random_seed=1234,
    batch_size=10,
    epoch=100,
    save_epoch=10,
    data_loader="StreamflowDataModel",
    var_c_source=VAR_C_CHOSEN_FROM_GAGES_II,
    var_c_target=VAR_C_CHOSEN_FROM_CAMELS_CC,
    var_t_source=VAR_T_CHOSEN_FROM_DAYMET,
    var_t_target=VAR_T_CHOSEN_FROM_ERA5LAND,
    train_period=["2014-10-01", "2019-10-01"],
    test_period=["2020-10-01", "2021-10-01"],
    valid_period=["2019-10-01", "2020-10-01"],
    freeze_params=["lstm.b_hh", "lstm.b_ih", "lstm.w_hh", "lstm.w_ih"],
    gage_id_file=os.path.join(
        definitions.ROOT_DIR,
        "hydroSPB",
        "example",
        "camels",
        "camels_cc_2014_2020_flow_screen.csv",
    ),
    gage_id=None,
    device=[1],
    num_workers=4,
):
    config = default_config_file()
    weight_path_dir = os.path.join(
        definitions.ROOT_DIR, "hydroSPB", "example", source_dir, source_exp
    )
    weight_path = get_lastest_weight_path(weight_path_dir)
    project_name = target_dir + "/" + target_exp
    if freeze_params is None:
        weight_path_add = None
    else:
        weight_path_add = {"freeze_params": freeze_params}
    if gage_id is None:
        if gage_id_file is None:
            gage_id = "ALL"
        else:
            gage_id = (
                pd.read_csv(gage_id_file, dtype={0: str}).iloc[:, 0].values.tolist()
            )
    args = cmd(
        sub=project_name,
        source="CAMELS",
        source_path=os.path.join(definitions.DATASET_DIR, "camels", "camels_cc"),
        source_region="CC",
        cache_write=1,
        cache_read=1,
        download=0,
        ctx=device,
        model_name="KaiTlLSTM",
        model_param={
            "linear_size": len(var_c_target) + len(var_t_target),
            "model_name": "kai_tl",
            "n_input_features": len(var_c_source) + len(var_t_source),
            "n_output_features": 1,
            "n_hidden_states": 256,
        },
        opt=opt,
        opt_param=opt_param,
        loss_func="RMSESum",
        batch_size=batch_size,
        rho=365,
        rs=random_seed,
        train_period=train_period,
        test_period=test_period,
        valid_period=valid_period,
        scaler="DapengScaler",
        data_loader=data_loader,
        weight_path=weight_path,
        weight_path_add=weight_path_add,
        continue_train=True,
        warmup_length=0,
        train_epoch=epoch,
        te=epoch,
        save_epoch=save_epoch,
        var_t=var_t_target,
        var_t_type=[ERA5LAND_NAME],
        var_c=var_c_target,
        var_out=[Q_CAMELS_CC_NAME],
        gage_id=gage_id,
        num_workers=num_workers,
    )
    update_cfg(config, args)
    train_and_evaluate(config)
    print("All processes are finished!")
    
def test_run_cross_validation():
    basin_ids = ["61561", "62618"]
    camels_cc = Camels(
        os.path.join(definitions.DATASET_DIR, "camels", "camels_cc"), region="CC"
    )
    basin_areas = camels_cc.read_basin_area(basin_ids)
    # sce-ua, lstm, tl-lstm
    model_names = ["sceua_xaj", "lstm", "tl-lstm"]
    exp61561 = ["gages/exp61561", "gages/exp615610", "gages/exp615611"]
    exp62618 = ["gages/exp62618", "gages/exp626180", "gages/exp626181"]
    exp92354 = ["gages/exp92354", "gages/exp923540", "gages/exp923541"]
    all_exps = [exp61561, exp62618, exp92354]
    # all_exps = [exp61561, exp62618]
    sceua_xaj_comments = {
        "61561": "HFsourcesrep200000ngs1000",
        "62618": "HFsourcesrep10000ngs10000",
        "92354": "HFsourcesrep20000ngs50",
    }
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
    best_epoch = [[58, 56], [86, 26], [85, 69]]


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
        for i, j in itertools.product(range(len(basin


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
