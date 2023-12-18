"""
Author: Wenyu Ouyang
Date: 2023-11-21 07:20:41
LastEditTime: 2023-12-18 09:16:10
LastEditors: Wenyu Ouyang
Description: Test weight analysis
FilePath: \torchhydro\tests\test_weight_analysis.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""
import os
import pandas as pd
import pytest


from torchhydro.explainers.weight_anlysis import (
    plot_param_hist_model,
)


@pytest.fixture()
def result_dir():
    project_dir = os.getcwd()
    return os.path.join(project_dir, "results")


def test_weight_analysis(result_dir):
    basin_id = "61561"
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
    show_hist_b = 20
    fold = 0
    exp61561_dir = os.path.join(result_dir, "test_camels", "expcccv61561_0")
    _, chosen_layers_consine = plot_param_hist_model(
        "lstm",
        exp61561_dir,
        show_hist_b,
        chosen_layer_for_hist,
        start_epoch=0,
        end_epoch=10,
        epoch_interval=1,
    )
    pd.DataFrame(chosen_layers_consine).to_csv(
        os.path.join(
            exp61561_dir,
            f"{basin_id}_bs{show_hist_b}_fold{fold}_chosen_layer_consine.csv",
        )
    )
