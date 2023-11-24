"""
Author: Wenyu Ouyang
Date: 2023-11-21 07:20:41
LastEditTime: 2023-11-24 16:07:03
LastEditors: Wenyu Ouyang
Description: Test weight analysis
FilePath: /torchhydro/tests/test_weight_analysis.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""
import itertools
import os
import matplotlib
import numpy as np
import pandas as pd
import pytest


from torchhydro.explainers.weight_anlysis import (
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
