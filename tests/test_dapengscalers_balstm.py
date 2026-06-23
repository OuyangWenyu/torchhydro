import tempfile
import pandas as pd
import pytest
import numpy as np
import xarray as xr
import json
import os
from torchhydro import SETTING
from torchhydro.datasets.data_scalers import DapengScaler
from hydrodatasource.reader.data_source import SelfMadeHydroDataset,LongTermDataset


@pytest.fixture
def sample_data():
    target_vars = xr.DataArray(
        np.array([[[1.0], [2.0]], [[3.0], [4.0]]]),  # 替换为固定值
        coords={
            "basin": ["Caravan_6978250", "Caravan_6976450"],
            "time": pd.date_range("2018-08-01", periods=2, freq="MS"),
            "variable": ["streamflow"],
        },
        dims=["basin", "time", "variable"],
    )
    relevant_vars = xr.DataArray(
        np.array([[[10, 20], [30, 40]], [[50, 60], [70, 80]]]),  # 替换为固定值
        coords={
            "basin": ["Caravan_6978250", "Caravan_6976450"],
            "time": pd.date_range("2018-08-01", periods=2, freq="MS"),
            "variable": ["tp", "d2m"],
        },
        dims=["basin", "time", "variable"],
    )
    constant_vars = xr.DataArray(
        np.array([[10.0, 20.0], [30.0, 40.0]]),  # 替换为固定值
        coords={
            "basin": ["Caravan_6978250", "Caravan_6976450"],
            "variable": ["area", "ele_mt_smn"],
        },
        dims=["basin", "variable"],
    )
    global_vars = xr.DataArray(
        np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
        coords={
            "basin": ["Caravan_6978250", "Caravan_6976450"],
            "time": pd.date_range("2018-08-01", periods=2, freq="MS"),
            "variable": ["A1", "A2"],
        },
        dims=["basin", "time", "variable"],
    )
    # make a temporary directory
    test_path = os.path.join(os.path.dirname(__file__), "..", "tmp")
    os.makedirs(test_path, exist_ok=True)
    data_cfgs = {
        "scaler": "DapengScaler",
        "test_path": test_path,  # Specify the path to save the json file if needed
        "stat_dict_file": None,
        "target_cols": ["streamflow"],
        "relevant_cols": ["tp", "d2m"],
        "constant_cols": ["area", "ele_mt_smn"],
        "global_cols": ["A1", "A2"],
        "object_ids": ["Caravan_6978250", "Caravan_6976450"],
        "t_range_train": [("2018-01-01", "2018-03-01")],  # not used but need to specify
        "t_range_test": [("2018-01-01", "2018-03-01")],  # not used but need to specify
    }
    return target_vars, relevant_vars, constant_vars, global_vars, data_cfgs


def test_dapeng_scaler_initialization(sample_data):
    target_vars, relevant_vars, constant_vars, global_vars, data_cfgs = sample_data
    scaler = DapengScaler(
        target_vars=target_vars,
        relevant_vars=relevant_vars,
        constant_vars=constant_vars,
        data_cfgs=data_cfgs,
        is_tra_val_te="train",
        data_source=LongTermDataset(
            data_path=SETTING["local_data_path"]["datasets-interim"], time_unit=["1MS"]
        ), 
        global_vars=global_vars,
    )
    assert scaler.data_target is not None
    assert scaler.data_forcing is not None
    assert scaler.data_attr is not None
    assert scaler.data_global is not None
    assert scaler.stat_dict is not None


def test_dapeng_scaler_cal_stat_all(sample_data):
    target_vars, relevant_vars, constant_vars, global_vars, data_cfgs = sample_data
    scaler = DapengScaler(
        target_vars=target_vars,
        relevant_vars=relevant_vars,
        constant_vars=constant_vars,
        data_cfgs=data_cfgs,
        is_tra_val_te="train",
        data_source=LongTermDataset(
            data_path=SETTING["local_data_path"]["datasets-interim"], time_unit=["1MS"]
        ), 
        global_vars=global_vars,
    )
    stat_dict = scaler.cal_stat_all()
    assert isinstance(stat_dict, dict)
    assert all(key in stat_dict for key in data_cfgs["target_cols"])
    assert all(key in stat_dict for key in data_cfgs["relevant_cols"])
    assert all(key in stat_dict for key in data_cfgs["constant_cols"])
    assert all(key in stat_dict for key in data_cfgs["global_cols"])


def test_dapeng_scaler_load_data_and_denorm(sample_data):
    target_vars, relevant_vars, constant_vars, global_vars, data_cfgs = sample_data
    scaler = DapengScaler(
        target_vars=target_vars,
        relevant_vars=relevant_vars,
        constant_vars=constant_vars,
        data_cfgs=data_cfgs,
        is_tra_val_te="train",
        data_source=LongTermDataset(
            data_path=SETTING["local_data_path"]["datasets-interim"], time_unit=["1MS"]
        ), 
        global_vars=global_vars,
    )
    x, y, c, g = scaler.load_data()
    assert x is not None
    assert y is not None
    assert c is not None
    assert g is not None
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
