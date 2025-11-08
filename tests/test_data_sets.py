"""
Author: Wenyu Ouyang
Date: 2024-04-08 18:16:53
LastEditTime: 2025-11-06 21:41:45
LastEditors: Wenyu Ouyang
Description: Test cases for dataset
FilePath: \torchhydro\tests\test_data_sets.py
Copyright (c) 2024-2024 Wenyu Ouyang. All rights reserved.
"""

import os
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from torch.utils.data import DataLoader

from torchhydro.datasets.data_sets import BaseDataset
from torchhydro.datasets.data_sources import data_sources_dict


class MockDatasource:
    def __init__(self, data_path, **kwargs):
        self.data_path = data_path
        self.kwargs = kwargs

    def get_name(self):
        return "mockdatasource"

    def read_ts_xrdataset(self, object_ids, t_range, var_lst, **kwargs):
        times = pd.to_datetime(
            pd.date_range(start=t_range[0], end=t_range[1], freq="D")
        )
        data = np.random.rand(len(object_ids), len(times), len(var_lst))
        ds = xr.Dataset(
            {
                var: (("basin", "time"), data[:, :, i], {"units": "mock_unit"})
                for i, var in enumerate(var_lst)
            },
            coords={"basin": object_ids, "time": times},
        )
        return ds

    def read_attr_xrdataset(self, object_ids, var_lst, **kwargs):
        data = np.random.rand(len(object_ids), len(var_lst))
        ds = xr.Dataset(
            {
                var: (("basin"), data[:, i], {"units": "mock_unit"})
                for i, var in enumerate(var_lst)
            },
            coords={"basin": object_ids},
        )
        return ds

    def read_area(self, object_ids):
        return xr.DataArray(
            np.random.rand(len(object_ids)),
            coords={"basin": object_ids},
            dims=["basin"],
        )


def test_create_lookup_table(tmp_path):
    temp_test_path = tmp_path / "test_datasets"
    os.makedirs(temp_test_path, exist_ok=True)
    data_sources_dict.update({"mockdatasource": MockDatasource})
    data_cfgs = {
        "source_cfgs": {
            "source_name": "mockdatasource",
            "source_path": str(temp_test_path),
        },
        "case_dir": str(temp_test_path),
        "object_ids": [
            "01013500",
            "01013501",
        ],
        "t_range_train": [
            "2001-01-01",
            "2002-01-01",
        ],
        "t_range_test": [
            "2002-01-01",
            "2003-01-01",
        ],
        "relevant_cols": ["prcp", "pet"],
        "target_cols": ["streamflow", "surface_sm"],
        "constant_cols": ["geol_1st_class", "geol_2nd_class"],
        "min_time_unit": "D",
        "min_time_interval": 1,
        "target_rm_nan": True,
        "relevant_rm_nan": True,
        "constant_rm_nan": True,
        "scaler": "StandardScaler",
        "stat_dict_file": None,
        "scaler_params": {"pbm_norm": False},
    }
    training_cfgs = {
        "warmup_length": 14,
        "hindcast_length": 7,
        "forecast_length": 1,
        "valid_batch_mode": "test",
    }
    evaluation_cfgs = {"rolling": 0, "hrwin": None, "frwin": None, "evaluator": {}}
    cfgs = {
        "data_cfgs": data_cfgs,
        "training_cfgs": training_cfgs,
        "evaluation_cfgs": evaluation_cfgs,
    }
    is_tra_val_te = "train"
    dataset = BaseDataset(cfgs, is_tra_val_te)
    assert dataset.num_samples > 0
    assert len(dataset.lookup_table) == dataset.num_samples


@pytest.mark.skip(reason="This test is obsolete as SelfMadeForecastDataset is not integrated into the codebase yet.")
def test_read_forecast_xrdataset():
    pass
