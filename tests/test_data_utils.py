"""
Author: Wenyu Ouyang
Date: 2024-04-09 21:16:45
LastEditTime: 2024-04-09 21:23:48
LastEditors: Wenyu Ouyang
Description: Test the data utilities
FilePath: \torchhydro\tests\test_data_utils.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""
import numpy as np
import pytest
import xarray as xr

from torchhydro.datasets.data_utils import warn_if_nan


def test_warn_if_nan_no_nan_values():
    # Create a dataarray without any NaN values
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    da = xr.DataArray(data)
    # Call the function and assert that it returns False
    assert not warn_if_nan(da)


def test_warn_if_nan_with_nan_values():
    # Create a dataarray with NaN values
    data = np.array([[1, np.nan, 3], [4, 5, np.nan], [7, 8, 9]])
    da = xr.DataArray(data)
    # Call the function and assert that it returns True
    assert warn_if_nan(da)


def test_warn_if_nan_all_nan_values():
    # Create a dataarray with all NaN values
    data = np.array(
        [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]]
    )
    da = xr.DataArray(data)
    # Call the function and assert that it raises a ValueError
    with pytest.raises(ValueError):
        warn_if_nan(da, nan_mode="all")


def test_warn_if_nan_max_display():
    # Create a dataarray with NaN values
    data = np.array([[1, np.nan, 3], [4, 5, np.nan], [7, np.nan, 9]])
    da = xr.DataArray(data)
    # Call the function with a max_display of 2
    # Assert that it raises a warning with the correct message
    with pytest.warns(
        UserWarning,
        match=r"The dataarray contains 3 NaN values! Here are the indices of the first 2 NaNs:",
    ):
        warn_if_nan(da, max_display=2)
