"""
Author: Wenyu Ouyang
Date: 2024-09-15 11:23:28
LastEditTime: 2024-09-15 17:07:04
LastEditors: Wenyu Ouyang
Description: Test the Resulter class
FilePath: \torchhydro\tests\test_resulter.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import os
import pytest
import xarray as xr
import numpy as np
from torchhydro.trainers.resulter import Resulter


@pytest.fixture
def mock_cfgs(tmpdir):
    return {
        "data_cfgs": {
            "test_path": str(tmpdir),
            "target_cols": ["streamflow", "total_evaporation_hourly"],
        },
        "evaluation_cfgs": {
            "model_loader": {"load_way": "latest"},
            "metrics": ["RMSE", "NSE"],
        },
        "training_cfgs": {"epochs": 10},
    }


@pytest.fixture
def mock_resulter(mock_cfgs):
    return Resulter(mock_cfgs)


def create_mock_netcdf(file_path, data):
    ds = xr.Dataset({"data": (("basin", "time"), data)})
    ds.to_netcdf(file_path)


def test_load_result(mock_resulter, tmpdir):
    # Create mock prediction and observation NetCDF files
    pred_data = np.random.rand(10, 10)
    obs_data = np.random.rand(10, 10)

    pred_file = os.path.join(tmpdir, "epoch10flow_pred.nc")
    obs_file = os.path.join(tmpdir, "epoch10flow_obs.nc")

    create_mock_netcdf(pred_file, pred_data)
    create_mock_netcdf(obs_file, obs_data)

    # Load the results using the method
    pred, obs = mock_resulter.load_result()

    # Check if the loaded data matches the mock data
    np.testing.assert_array_equal(pred["data"].values, pred_data)
    np.testing.assert_array_equal(obs["data"].values, obs_data)


def test_eval_result(mock_resulter):
    # Mock configuration for eval_result
    mock_resulter.cfgs["data_cfgs"]["target_cols"] = [
        "streamflow",
        "total_evaporation_hourly",
    ]
    mock_resulter.cfgs["evaluation_cfgs"]["metrics"] = ["RMSE", "NSE"]
    mock_resulter.cfgs["evaluation_cfgs"]["fill_nan"] = ["no", "no"]
    mock_resulter.cfgs["evaluation_cfgs"]["explainer"] = "none"

    # Create mock prediction and observation xarray datasets
    preds_data = {
        "streamflow": (("time", "basin"), np.random.rand(10, 10)),
        "total_evaporation_hourly": (("time", "basin"), np.random.rand(10, 10)),
    }
    obss_data = {
        "streamflow": (("time", "basin"), np.random.rand(10, 10)),
        "total_evaporation_hourly": (("time", "basin"), np.random.rand(10, 10)),
    }
    preds_xr = xr.Dataset(preds_data)
    obss_xr = xr.Dataset(obss_data)

    # Mock the calculate_and_record_metrics function
    def mock_calculate_and_record_metrics(obs, pred, metrics, col, fill_nan, eval_log):
        eval_log[col] = {metric: np.random.rand() for metric in metrics}
        return eval_log

    mock_resulter.calculate_and_record_metrics = mock_calculate_and_record_metrics

    # Evaluate the results
    eval_log = mock_resulter.eval_result(preds_xr, obss_xr)

    # Check if the evaluation log contains the expected metrics for each variable
    for col in mock_resulter.cfgs["data_cfgs"]["target_cols"]:
        for metric in mock_resulter.cfgs["evaluation_cfgs"]["metrics"]:
            assert f"{metric} of {col}" in eval_log
            assert isinstance(eval_log[f"{metric} of {col}"], list)