"""
Author: Wenyu Ouyang
Date: 2024-05-27 13:33:08
LastEditTime: 2024-05-27 15:31:14
LastEditors: Wenyu Ouyang
Description: Unit test for datasets
FilePath: \torchhydro\tests\test_data_sets.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import pytest
import os
import numpy as np
import pandas as pd
import xarray as xr
import pickle
from sklearn.preprocessing import StandardScaler
from torchhydro.datasets.data_sets import BaseDataset
from torchhydro.datasets.data_sources import data_sources_dict


class MockDatasource:
    def __init__(self, source_cfgs):
        self.ngrid = 2
        self.nt = 366
        self.data_cfgs = source_cfgs

    def read_ts_xrdataset(self, basin_id, t_range, var_lst):
        basins = [f"{i:08d}" for i in range(1013500, 1013500 + self.ngrid)]

        # 创建时间序列
        start_date, end_date = t_range
        times = pd.date_range(start=start_date, end=end_date, freq="D")

        # 确保时间序列长度和 nt 一致
        if len(times) != self.nt:
            raise ValueError(
                "The generated time range does not match the expected length."
            )

        # 创建数据变量
        data_vars = {
            var: (["basin", "time"], np.random.rand(self.ngrid, self.nt))
            for var in var_lst
        }

        # 创建 xarray Dataset
        dataset = xr.Dataset(
            data_vars=data_vars, coords={"basin": basins, "time": times}
        )
        if "streamflow" in dataset.variables:
            dataset["streamflow"].attrs["units"] = "mm/day"

        if "prcp" in dataset.variables:
            dataset["prcp"].attrs["units"] = "mm/day"

        return dataset

    def read_attr_xrdataset(self, basin_id, var_lst, all_number=True):
        # 创建 basin ID 列表
        basins = [f"{i:08d}" for i in range(1013500, 1013500 + self.ngrid)]

        # 创建数据变量
        attr_lst = ["geol_1st_class", "geol_2nd_class"]
        data_vars = {attr: (["basin"], np.random.rand(self.ngrid)) for attr in attr_lst}

        # 添加整数类型的虚拟数据变量

        for var in attr_lst:
            data_vars[var] = (["basin"], np.random.randint(0, 10, size=self.ngrid))

        return xr.Dataset(data_vars=data_vars, coords={"basin": basins})


def test_create_lookup_table(tmp_path):
    temp_test_path = tmp_path / "test_datasets"
    os.makedirs(temp_test_path, exist_ok=True)
    data_sources_dict.update({"mockdatasource": MockDatasource})
    data_cfgs = {
        "source_cfgs": {
            "source_name": "mockdatasource",
            "source_path": str(temp_test_path),
        },
        "test_path": str(temp_test_path),
        "object_ids": [
            "01013500",
            "01013501",
        ],  # Add this line with the actual object IDs
        "t_range_train": [
            "2001-01-01",
            "2002-01-01",
        ],  # Add this line with the actual start and end dates for training.
        "t_range_test": [
            "2002-01-01",
            "2003-01-01",
        ],  # Add this line with the actual start and end dates for validation.
        "relevant_cols": [
            # List the relevant column names here.
            "prcp",
            "pet",
            # ... other relevant columns ...
        ],
        "target_cols": [
            # List the target column names here.
            "streamflow",
            "surface_sm",
            # ... other target columns ...
        ],
        "constant_cols": [
            # List the constant column names here.
            "geol_1st_class",
            "geol_2nd_class",
            # ... other constant columns ...
        ],
        "forecast_history": 7,
        "warmup_length": 14,
        "forecast_length": 1,
        "min_time_unit": "D",
        "min_time_interval": 1,
        "target_rm_nan": True,
        "relevant_rm_nan": True,
        "constant_rm_nan": True,
        "scaler": "StandardScaler",  # Add the scaler configuration here
        "stat_dict_file": None,  # Added the missing configuration
    }
    is_tra_val_te = "train"
    dataset = BaseDataset(data_cfgs, is_tra_val_te)
    lookup_table = dataset.lookup_table
    assert isinstance(lookup_table, dict)
    assert len(lookup_table) > 0
    assert all(
        isinstance(key, int) and isinstance(value, tuple)
        for key, value in lookup_table.items()
    )
    is_tra_val_te = "test"
    mock_data = np.random.rand(100, 2)  # Replace with relevant data.
    scaler = StandardScaler()
    scaler.fit(mock_data)
    scaler_file_path = tmp_path / "test_datasets_scaler.pkl"
    data_cfgs["stat_dict_file"] = str(scaler_file_path)
    with open(scaler_file_path, "wb") as file:
        pickle.dump(scaler, file)
    dataset = BaseDataset(data_cfgs, is_tra_val_te)
    lookup_table = dataset.lookup_table
    assert isinstance(lookup_table, dict)
    assert len(lookup_table) > 0
    assert all(
        isinstance(key, int) and isinstance(value, tuple)
        for key, value in lookup_table.items()
    )
