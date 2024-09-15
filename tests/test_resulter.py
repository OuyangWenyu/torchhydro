"""
Author: Wenyu Ouyang
Date: 2024-09-15 11:23:28
LastEditTime: 2024-09-15 11:23:31
LastEditors: Wenyu Ouyang
Description: Test the Resulter class
FilePath: \torchhydro\torchhydro\trainers\test_resulter.py
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
        "data_cfgs": {"test_path": str(tmpdir)},
        "evaluation_cfgs": {"model_loader": {"load_way": "latest"}},
        "training_cfgs": {"epochs": 10},
    }


@pytest.fixture
def mock_resulter(mock_cfgs):
    return Resulter(mock_cfgs)


def create_mock_netcdf(file_path, data):
    ds = xr.Dataset({"data": (("dim_0", "dim_1"), data)})
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
