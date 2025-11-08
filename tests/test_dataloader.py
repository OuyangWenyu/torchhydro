"""
Author: Wenyu Ouyang
Date: 2025-01-06 14:21:00
LastEditTime: 2025-01-25 09:19:55
LastEditors: Wenyu Ouyang
Description: Test cases for dataloader
FilePath: /torchhydro/tests/test_dataloader.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import os
import pytest
import pandas as pd
import xarray as xr
import numpy as np

from torchhydro.trainers.deep_hydro import DeepHydro
from torchhydro.datasets.data_sources import data_sources_dict


class MockDatasource:
    """A mock data source for testing purposes."""
    def __init__(self, data_path, **kwargs):
        self.data_path = data_path
        self.kwargs = kwargs

    def get_name(self):
        return "mockdatasource"

    def read_ts_xrdataset(self, object_ids, t_range, var_lst, **kwargs):
        times = pd.to_datetime(pd.date_range(start=t_range[0], end=t_range[1], freq="D"))
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
        return xr.DataArray(np.random.rand(len(object_ids)), coords={"basin": object_ids}, dims=["basin"])


@pytest.fixture
def complete_config(tmp_path):
    """Provides a complete and valid configuration dictionary for tests."""
    temp_test_path = tmp_path / "test_dataloader"
    temp_test_path.mkdir()

    data_cfgs = {
        "source_cfgs": {
            "source_name": "mockdatasource",
            "source_path": str(temp_test_path),
        },
        "case_dir": str(temp_test_path),
        "object_ids": ["01013500", "01013501"],
        "t_range_train": ["2001-01-01", "2002-01-01"],
        "t_range_valid": ["2002-01-01", "2003-01-01"],
        "t_range_test": ["2003-01-01", "2004-01-01"],
        "dataset": "StreamflowDataset",
        "relevant_cols": ["prcp", "pet"],
        "target_cols": ["streamflow"],
        "constant_cols": ["area"],
        "min_time_unit": "D",
        "min_time_interval": 1,
        "target_rm_nan": True,
        "relevant_rm_nan": True,
        "constant_rm_nan": True,
        "scaler": "StandardScaler",
        "scaler_params": {"pbm_norm": False},
        "stat_dict_file": None,
        "sampler": None,
    }

    training_cfgs = {
        "train_mode": True,
        "batch_size": 32,
        "warmup_length": 10,
        "hindcast_length": 20,
        "forecast_length": 5,
        "valid_batch_mode": "test",
        "variable_length_cfgs": {"use_variable_length": False},
    }

    evaluation_cfgs = {
        "batch_size": 128, # Use a different batch size for validation/testing
        "rolling": 0,
        "hrwin": None,
        "frwin": None,
        "evaluator": {},
    }

    model_cfgs = {
        "model_name": "SimpleLSTM",
        "weight_path": None,
        "model_hyperparam": {
            "input_size": len(data_cfgs["relevant_cols"]) + len(data_cfgs["constant_cols"]),
            "hidden_size": 64,
            "output_size": len(data_cfgs["target_cols"])
        }
    }

    data_sources_dict.update({"mockdatasource": MockDatasource})

    return {
        "data_cfgs": data_cfgs,
        "training_cfgs": training_cfgs,
        "evaluation_cfgs": evaluation_cfgs,
        "model_cfgs": model_cfgs,
    }


def test_get_dataloader_train_eval(complete_config):
    """
    Test DataLoader generation in training mode.
    """
    deep_hydro = DeepHydro(complete_config)

    dataloader, valid_dataloader = deep_hydro._get_dataloader(
        complete_config["training_cfgs"],
        complete_config["data_cfgs"],
        mode="train",
    )

    # Assertions
    assert dataloader.batch_size == complete_config["training_cfgs"]["batch_size"]
    assert valid_dataloader.batch_size == complete_config["training_cfgs"]["batch_size"]


def test_get_dataloader_test(complete_config):
    """
    Test DataLoader generation in inference mode.
    """
    deep_hydro = DeepHydro(complete_config)

    dataloader = deep_hydro._get_dataloader(
        complete_config["training_cfgs"],
        complete_config["data_cfgs"],
        mode="infer",
    )

    # Assertions
    assert dataloader.batch_size == complete_config["training_cfgs"]["batch_size"]