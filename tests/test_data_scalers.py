"""
Author: Wenyu Ouyang
Date: 2023-07-25 16:47:19
LastEditTime: 2025-11-06 21:30:23
LastEditors: Wenyu Ouyang
Description: Test a full training and evaluating process
FilePath: \torchhydro\tests\test_data_scalers.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import os
import shutil

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from torchhydro.datasets.data_scalers import DapengScaler, ScalerHub
from hydrodatasource.reader.data_source import SelfMadeHydroDataset


@pytest.fixture
def sample_data(tmp_path):
    # Use pytest's tmp_path fixture for a clean, unique temporary directory
    case_dir = tmp_path / "scaler_test"
    case_dir.mkdir()

    # Create sample xarray DataArrays for testing
    target_vars = xr.DataArray(
        np.random.rand(2, 2, 2),
        dims=("basin", "time", "variable"),
        coords={
            "basin": ["songliao_21401050", "songliao_21401550"],
            "time": pd.to_datetime(["2000-01-01", "2000-01-02"]),
            "variable": ["streamflow", "et"],
        },
    )
    target_vars.attrs["units"] = {"streamflow": "m3/s", "et": "mm"}

    relevant_vars = xr.DataArray(
        np.random.rand(2, 2, 2),
        dims=("basin", "time", "variable"),
        coords={
            "basin": ["songliao_21401050", "songliao_21401550"],
            "time": pd.to_datetime(["2000-01-01", "2000-01-02"]),
            "variable": ["precipitationCal", "sm_surface"],
        },
    )
    relevant_vars.attrs["units"] = {"precipitationCal": "mm", "sm_surface": "m3/m3"}

    constant_vars = xr.DataArray(
        np.random.rand(2, 2),
        dims=("basin", "variable"),
        coords={
            "basin": ["songliao_21401050", "songliao_21401550"],
            "variable": ["area", "ele_mt_smn"],
        },
    )
    constant_vars.attrs["units"] = {"area": "km2", "ele_mt_smn": "m"}

    data_cfgs = {
        "case_dir": str(case_dir),
        "target_cols": ["streamflow"],
        "relevant_cols": ["precipitationCal", "sm_surface"],
        "constant_cols": ["area", "ele_mt_smn"],
        "object_ids": ["songliao_21401050", "songliao_21401550"],
        "scaler": "DapengScaler",
        "scaler_params": {
            "pbm_norm": False,
            "gamma_norm_cols": [],
            "prcp_norm_cols": [],
        },
        "t_range_train": ["2000-01-01", "2000-01-02"],
        "t_range_valid": ["2000-01-01", "2000-01-02"],
        "t_range_test": ["2000-01-01", "2000-01-02"],
        "stat_dict_file": None,
    }
    return target_vars, relevant_vars, constant_vars, data_cfgs


@pytest.fixture
def mock_data_source(mocker, sample_data):
    _, _, _, data_cfgs = sample_data
    mocker.patch(
        "hydrodatasource.reader.data_source.SelfMadeHydroDataset.__init__",
        return_value=None,
    )
    mock_instance = SelfMadeHydroDataset()
    # Create a DataArray WITHOUT a dimension named 'variable' to avoid collision
    mean_prcp_da = xr.DataArray(
        np.random.rand(2), dims=["basin"], coords={"basin": data_cfgs["object_ids"]}
    )
    mean_prcp_ds = xr.Dataset({"mean_prcp": mean_prcp_da})
    mocker.patch.object(mock_instance, "read_mean_prcp", return_value=mean_prcp_ds)
    return mock_instance


def test_dapeng_scaler_initialization(sample_data, mock_data_source):
    target_vars, relevant_vars, constant_vars, data_cfgs = sample_data
    vars_data_dict = {
        "target_cols": target_vars,
        "relevant_cols": relevant_vars,
        "constant_cols": constant_vars,
    }
    scaler = DapengScaler(
        vars_data=vars_data_dict,
        data_cfgs=data_cfgs,
        is_tra_val_te="train",
        data_source=mock_data_source,
    )
    assert scaler is not None
    assert os.path.exists(os.path.join(data_cfgs["case_dir"], "dapengscaler_stat.json"))


def test_dapeng_scaler_load_data_and_denorm(sample_data, mock_data_source):
    target_vars, relevant_vars, constant_vars, data_cfgs = sample_data
    vars_data_dict = {
        "target_cols": target_vars,
        "relevant_cols": relevant_vars,
        "constant_cols": constant_vars,
    }

    scaler = DapengScaler(
        vars_data_dict,
        data_cfgs=data_cfgs,
        is_tra_val_te="train",
        data_source=mock_data_source,
    )
    norm_data_dict = scaler.load_norm_data(vars_data_dict)
    denorm_data = scaler.inverse_transform(norm_data_dict["target_cols"])
    assert denorm_data is not None
    assert not np.all(np.isnan(denorm_data["streamflow"].to_numpy()))


def test_sklearn_scale_train_mode(sample_data):
    target_vars, relevant_vars, constant_vars, data_cfgs = sample_data
    data_cfgs["scaler"] = "StandardScaler"
    vars_data_dict = {
        "target_cols": target_vars,
        "relevant_cols": relevant_vars,
        "constant_cols": constant_vars,
    }
    scaler_hub = ScalerHub(
        vars_data=vars_data_dict,
        data_cfgs=data_cfgs,
        is_tra_val_te="train",
    )
    assert os.path.exists(os.path.join(data_cfgs["case_dir"], "target_cols_scaler.pkl"))
    assert os.path.exists(
        os.path.join(data_cfgs["case_dir"], "relevant_cols_scaler.pkl")
    )
    assert os.path.exists(
        os.path.join(data_cfgs["case_dir"], "constant_cols_scaler.pkl")
    )


def test_sklearn_scale_test_mode_with_existing_scaler(sample_data):
    target_vars, relevant_vars, constant_vars, data_cfgs = sample_data
    data_cfgs["scaler"] = "StandardScaler"
    vars_data_dict = {
        "target_cols": target_vars,
        "relevant_cols": relevant_vars,
        "constant_cols": constant_vars,
    }
    ScalerHub(
        vars_data=vars_data_dict,
        data_cfgs=data_cfgs,
        is_tra_val_te="train",
    )

    scaler_hub_test = ScalerHub(
        vars_data=vars_data_dict,
        data_cfgs=data_cfgs,
        is_tra_val_te="test",
    )
    assert "target_cols" in scaler_hub_test.norm_data
    assert scaler_hub_test.norm_data["target_cols"] is not None


def test_sklearn_scale_test_mode_without_scaler_file(sample_data):
    target_vars, relevant_vars, constant_vars, data_cfgs = sample_data
    data_cfgs["scaler"] = "StandardScaler"
    vars_data_dict = {
        "target_cols": target_vars,
        "relevant_cols": relevant_vars,
        "constant_cols": constant_vars,
    }
    shutil.rmtree(data_cfgs["case_dir"])
    os.makedirs(data_cfgs["case_dir"])

    with pytest.raises(FileNotFoundError):
        ScalerHub(
            vars_data=vars_data_dict,
            data_cfgs=data_cfgs,
            is_tra_val_te="test",
        )
