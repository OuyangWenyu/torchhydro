import numpy as np
import xarray as xr

from torchhydro.datasets.data_sets import _fill_gaps_da


def test_interpolate_fills_each_basin_and_variable_independently():
    values = np.array(
        [
            [[0.0, 10.0], [np.nan, 20.0], [2.0, np.nan], [3.0, 40.0]],
            [[4.0, 50.0], [6.0, np.nan], [np.nan, 70.0], [10.0, 80.0]],
        ],
        dtype=np.float32,
    )
    data = xr.DataArray(values, dims=("basin", "time", "variable"))

    actual = _fill_gaps_da(data, fill_nan="interpolate")

    expected = np.array(
        [
            [[0.0, 10.0], [1.0, 20.0], [2.0, 30.0], [3.0, 40.0]],
            [[4.0, 50.0], [6.0, 60.0], [8.0, 70.0], [10.0, 80.0]],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(actual.values, expected)


def test_interpolate_extrapolates_edges_and_preserves_all_nan_series():
    values = np.array(
        [
            [[np.nan, np.nan], [1.0, np.nan], [2.0, np.nan], [np.nan, np.nan]],
        ],
        dtype=np.float32,
    )
    data = xr.DataArray(values, dims=("basin", "time", "variable"))

    actual = _fill_gaps_da(data, fill_nan="interpolate")

    np.testing.assert_allclose(actual.values[0, :, 0], [0.0, 1.0, 2.0, 3.0])
    assert np.isnan(actual.values[0, :, 1]).all()
