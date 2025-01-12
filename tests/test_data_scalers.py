import tempfile
import pandas as pd
import pytest
import numpy as np
import xarray as xr
import json
import os
from torchhydro.datasets.data_scalers import DapengScaler
from hydrodatasource.reader.data_source import SelfMadeHydroDataset


@pytest.fixture
def sample_data():
    target_vars = xr.DataArray(
        np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),  # 替换为固定值
        coords={
            "basin": ["songliao_21401050", "songliao_21401550"],
            "time": pd.date_range("2018-08-01 01", periods=2, freq="3H"),
            "variable": ["precipitationCal", "sm_surface"],
        },
        dims=["basin", "time", "variable"],
    )
    relevant_vars = xr.DataArray(
        np.array([[[10, 20], [30, 40]], [[50, 60], [70, 80]]]),  # 替换为固定值
        coords={
            "basin": ["songliao_21401050", "songliao_21401550"],
            "time": pd.date_range("2018-08-01 01", periods=2, freq="3H"),
            "variable": ["precipitationCal", "sm_surface"],
        },
        dims=["basin", "time", "variable"],
    )
    constant_vars = xr.DataArray(
        np.array([[10.0, 20.0], [30.0, 40.0]]),  # 替换为固定值
        coords={
            "basin": ["songliao_21401050", "songliao_21401550"],
            "variable": ["area", "ele_mt_smn"],
        },
        dims=["basin", "variable"],
    )
    # make a temporary directory
    test_path = os.path.join(os.path.dirname(__file__), "..", "tmp")
    os.makedirs(test_path, exist_ok=True)
    data_cfgs = {
        "scaler": "DapengScaler",
        "case_dir": test_path,  # Specify the path to save the json file if needed
        "stat_dict_file": None,
        "target_cols": ["precipitationCal", "sm_surface"],
        "relevant_cols": ["precipitationCal", "sm_surface"],
        "constant_cols": ["area", "ele_mt_smn"],
        "object_ids": ["songliao_21401050", "songliao_21401550"],
        "t_range_train": [
            ("2018-01-01 01", "2018-03-01 01")
        ],  # not used but need to specify
        "t_range_test": [
            ("2018-01-01 01", "2018-03-01 01")
        ],  # not used but need to specify
    }
    return target_vars, relevant_vars, constant_vars, data_cfgs


def test_dapeng_scaler_initialization(sample_data):
    target_vars, relevant_vars, constant_vars, data_cfgs = sample_data
    scaler = DapengScaler(
        target_vars=target_vars,
        relevant_vars=relevant_vars,
        constant_vars=constant_vars,
        data_cfgs=data_cfgs,
        is_tra_val_te="train",
    )
    assert scaler.data_target is not None
    assert scaler.data_forcing is not None
    assert scaler.data_attr is not None
    assert scaler.stat_dict is not None


def test_dapeng_scaler_cal_stat_all(sample_data):
    target_vars, relevant_vars, constant_vars, data_cfgs = sample_data
    scaler = DapengScaler(
        target_vars=target_vars,
        relevant_vars=relevant_vars,
        constant_vars=constant_vars,
        data_cfgs=data_cfgs,
        is_tra_val_te="train",
        data_source=SelfMadeHydroDataset(
            data_path="/ftproot/basins-interim", time_unit=["3h"]
        ),  # to satisfy the mean_prcp function
    )
    stat_dict = scaler.cal_stat_all()
    assert isinstance(stat_dict, dict)
    assert all(key in stat_dict for key in data_cfgs["target_cols"])
    assert all(key in stat_dict for key in data_cfgs["relevant_cols"])
    assert all(key in stat_dict for key in data_cfgs["constant_cols"])


def test_dapeng_scaler_load_data_and_denorm(sample_data):
    target_vars, relevant_vars, constant_vars, data_cfgs = sample_data
    scaler = DapengScaler(
        target_vars=target_vars,
        relevant_vars=relevant_vars,
        constant_vars=constant_vars,
        data_cfgs=data_cfgs,
        is_tra_val_te="train",
        data_source=SelfMadeHydroDataset(
            data_path="/ftproot/basins-interim", time_unit=["3h"]
        ),
    )
    x, y, c = scaler.load_data()
    assert x is not None
    assert y is not None
    assert c is not None

    # denormalizing y
    denorm_y = scaler.inverse_transform(y)

    # Check if the values of each variable are consistent
    target_dataset = target_vars.to_dataset("variable")
    for var in target_dataset.data_vars:
        np.testing.assert_allclose(
            target_dataset[var].values,
            denorm_y[var].values,
            err_msg=f"{var} is inconsistent",
        )

    # Check if the coordinates are consistent
    for coord in target_dataset.coords:
        np.testing.assert_array_equal(
            target_dataset.coords[coord].values,
            denorm_y.coords[coord].values,
            err_msg=f"{coord} is inconsistent",
        )
