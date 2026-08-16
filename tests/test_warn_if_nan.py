import numpy as np
import pytest
import xarray as xr

from torchhydro.datasets.data_utils import warn_if_nan


def test_warn_if_nan_returns_false_without_nan():
    data = xr.DataArray(np.ones((2, 3), dtype=np.float32))

    assert warn_if_nan(data) is False


def test_warn_if_nan_returns_true_for_partial_nan():
    data = xr.DataArray(np.array([[1.0, np.nan], [2.0, 3.0]], dtype=np.float32))

    assert warn_if_nan(data, nan_mode="any") is True
    assert warn_if_nan(data, nan_mode="all") is True


def test_warn_if_nan_preserves_all_nan_modes():
    data = xr.DataArray(np.full((2, 3), np.nan, dtype=np.float32))

    with pytest.raises(ValueError, match="only NaN"):
        warn_if_nan(data, nan_mode="any")
    assert warn_if_nan(data, nan_mode="all") is True


def test_warn_if_nan_rejects_invalid_mode():
    data = xr.DataArray(np.ones((2, 3), dtype=np.float32))

    with pytest.raises(ValueError, match="nan_mode"):
        warn_if_nan(data, nan_mode="invalid")
