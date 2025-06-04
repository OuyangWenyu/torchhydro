import os
import pytest
from torchhydro.trainers.train_utils import read_pth_from_model_loader
import numpy as np
from torchhydro.trainers.train_utils import _recover_samples_to_4d
from torchhydro.trainers.train_utils import _recover_samples_to_4d_by_forecast
from torchhydro.trainers.train_utils import _recover_samples_to_4d_by_forecast
from torchhydro.trainers.train_utils import _recover_samples_to_4d_by_forecast
from torchhydro.trainers.train_utils import _recover_samples_to_4d_by_forecast


def test_read_pth_from_model_loader_specified():
    model_loader = {"load_way": "specified", "test_epoch": 5}
    model_pth_dir = "/path/to/models"
    expected_path = os.path.join(model_pth_dir, "model_Ep5.pth")
    assert read_pth_from_model_loader(model_loader, model_pth_dir) == expected_path


def test_read_pth_from_model_loader_best():
    model_loader = {"load_way": "best"}
    model_pth_dir = "/path/to/models"
    expected_path = os.path.join(model_pth_dir, "best_model.pth")
    assert read_pth_from_model_loader(model_loader, model_pth_dir) == expected_path


def test_read_pth_from_model_loader_latest(mocker):
    model_loader = {"load_way": "latest"}
    model_pth_dir = "/path/to/models"
    latest_file = "latest_model.pth"
    mocker.patch(
        "torchhydro.trainers.train_utils.get_lastest_file_in_a_dir",
        return_value=latest_file,
    )
    expected_path = latest_file
    assert read_pth_from_model_loader(model_loader, model_pth_dir) == expected_path


def test_read_pth_from_model_loader_pth():
    model_loader = {"load_way": "pth", "pth_path": "/path/to/models/custom_model.pth"}
    model_pth_dir = "/path/to/models"
    expected_path = "/path/to/models/custom_model.pth"
    assert read_pth_from_model_loader(model_loader, model_pth_dir) == expected_path


def test_read_pth_from_model_loader_invalid():
    model_loader = {"load_way": "invalid"}
    model_pth_dir = "/path/to/models"
    with pytest.raises(ValueError, match="Invalid load_way"):
        read_pth_from_model_loader(model_loader, model_pth_dir)


class DummyDataset:
    def __init__(
        self,
        t_s_dict,
        lookup_table,
        nt,
        rho,
        warmup_length,
        horizon,
        noutputvar,
    ):
        self.t_s_dict = t_s_dict
        self.lookup_table = lookup_table
        self.nt = nt
        self.rho = rho
        self.warmup_length = warmup_length
        self.horizon = horizon
        self.noutputvar = noutputvar


class DummyDataLoader:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size


def test_recover_samples_to_4d_by_basins_basic():
    # Setup
    basin_num = 2
    nt = 6
    nf = 1
    forecast_length = 2
    stride = 1
    hindcast_output_window = 0
    i_e_time_length = 3  # arr_3d will be reshaped to (2, 3, time_steps, nf)
    arr_3d_time_steps = 4  # must be >= forecast_length

    class DummyDataset:
        def __init__(self):
            self.t_s_dict = {"sites_id": [0, 1]}
            self.horizon = forecast_length

    class DummyDataLoader:
        def __init__(self):
            self.dataset = DummyDataset()

    arr_3d = np.arange(basin_num * i_e_time_length * arr_3d_time_steps * nf).reshape(
        basin_num * i_e_time_length, arr_3d_time_steps, nf
    )

    data_shape = (basin_num, nt, nf)
    data_loader = DummyDataLoader()

    result = _recover_samples_to_4d_by_basins(
        data_shape, data_loader, stride, hindcast_output_window, arr_3d
    )

    # Check shape
    assert result.shape == (basin_num, i_e_time_length, forecast_length, nf)

    # Check that the filled values match the expected forecast_output
    output = arr_3d.reshape(basin_num, i_e_time_length, arr_3d_time_steps, nf)
    forecast_output = output[:, :, -forecast_length:, :]
    for j in range(forecast_length):
        valid_indices = np.arange(i_e_time_length - j)
        for basin_idx in range(basin_num):
            expected = forecast_output[basin_idx, valid_indices + j, j, :]
            actual = result[basin_idx, valid_indices, j, :]
            assert np.allclose(actual, expected)


def test_recover_samples_to_4d_by_basins_nan_fill():
    # Setup for a case where not all positions are filled
    basin_num = 1
    nt = 5
    nf = 1
    forecast_length = 2
    stride = 1
    hindcast_output_window = 0
    i_e_time_length = 2
    arr_3d_time_steps = 3

    class DummyDataset:
        def __init__(self):
            self.t_s_dict = {"sites_id": [0]}
            self.horizon = forecast_length

    class DummyDataLoader:
        def __init__(self):
            self.dataset = DummyDataset()

    arr_3d = np.ones((basin_num * i_e_time_length, arr_3d_time_steps, nf))

    data_shape = (basin_num, nt, nf)
    data_loader = DummyDataLoader()

    result = _recover_samples_to_4d_by_basins(
        data_shape, data_loader, stride, hindcast_output_window, arr_3d
    )

    # Only certain positions should be filled, others should be nan
    # For j=0, valid_indices = [0,1], for j=1, valid_indices = [0]
    assert not np.any(np.isnan(result[0, 0, 0, :]))
    assert not np.any(np.isnan(result[0, 1, 0, :]))
    assert not np.any(np.isnan(result[0, 0, 1, :]))
    assert np.all(np.isnan(result[0, 1, 1, :]))


def test_recover_samples_to_4d_by_basins_empty():
    basin_num = 5
    nt = 7
    nf = 2
    forecast_length = 4
    stride = 1
    hindcast_output_window = 0
    i_e_time_length = 0
    arr_3d_time_steps = 3

    class DummyDataset:
        def __init__(self):
            self.t_s_dict = {"sites_id": [0]}
            self.horizon = forecast_length

    class DummyDataLoader:
        def __init__(self):
            self.dataset = DummyDataset()

    arr_3d = np.empty((0, arr_3d_time_steps, nf))

    data_shape = (basin_num, nt, nf)
    data_loader = DummyDataLoader()

    result = _recover_samples_to_4d_by_basins(
        data_shape, data_loader, stride, hindcast_output_window, arr_3d
    )

    assert result.shape == (basin_num, 0, forecast_length, nf)


def test_recover_samples_to_4d_by_forecast_basic():
    basin_num = 2
    nt = 6
    nf = 1
    forecast_length = 2
    stride = 1
    hindcast_output_window = 0
    i_e_time_length = 3  # arr_3d will be reshaped to (2, 3, time_steps, nf)
    arr_3d_time_steps = 4  # must be >= forecast_length

    class DummyDataset:
        def __init__(self):
            self.t_s_dict = {"sites_id": [0, 1]}
            self.horizon = forecast_length

    class DummyDataLoader:
        def __init__(self):
            self.dataset = DummyDataset()

    arr_3d = np.arange(basin_num * i_e_time_length * arr_3d_time_steps * nf).reshape(
        basin_num * i_e_time_length, arr_3d_time_steps, nf
    )

    data_shape = (basin_num, nt, nf)
    data_loader = DummyDataLoader()

    result = _recover_samples_to_4d_by_forecast(
        data_shape, data_loader, stride, hindcast_output_window, arr_3d
    )

    # Check shape
    assert result.shape == (forecast_length, basin_num, i_e_time_length, nf)

    # Check that the filled values match the expected forecast_output
    output = arr_3d.reshape(basin_num, i_e_time_length, arr_3d_time_steps, nf)
    forecast_output = output[:, :, -forecast_length:, :]
    for j in range(forecast_length):
        valid_indices = np.arange(i_e_time_length - j)
        for basin_idx in range(basin_num):
            expected = forecast_output[basin_idx, valid_indices + j, j, :]
            actual = result[j, basin_idx, valid_indices, :]
            assert np.allclose(actual, expected)


def test_recover_samples_to_4d_by_forecast_nan_fill():
    basin_num = 1
    nt = 5
    nf = 1
    forecast_length = 2
    stride = 1
    hindcast_output_window = 0
    i_e_time_length = 2
    arr_3d_time_steps = 3

    class DummyDataset:
        def __init__(self):
            self.t_s_dict = {"sites_id": [0]}
            self.horizon = forecast_length

    class DummyDataLoader:
        def __init__(self):
            self.dataset = DummyDataset()

    arr_3d = np.ones((basin_num * i_e_time_length, arr_3d_time_steps, nf))

    data_shape = (basin_num, nt, nf)
    data_loader = DummyDataLoader()

    result = _recover_samples_to_4d_by_forecast(
        data_shape, data_loader, stride, hindcast_output_window, arr_3d
    )

    # Only certain positions should be filled, others should be nan
    # For j=0, valid_indices = [0,1], for j=1, valid_indices = [0]
    assert not np.any(np.isnan(result[0, 0, 0, :]))
    assert not np.any(np.isnan(result[0, 0, 1, :]))
    assert not np.any(np.isnan(result[1, 0, 0, :]))
    assert np.all(np.isnan(result[1, 0, 1, :]))


def test_recover_samples_to_4d_by_forecast_multiple_features():
    basin_num = 2
    nt = 6
    nf = 3
    forecast_length = 2
    stride = 1
    hindcast_output_window = 0
    i_e_time_length = 2
    arr_3d_time_steps = 4

    class DummyDataset:
        def __init__(self):
            self.t_s_dict = {"sites_id": [0, 1]}
            self.horizon = forecast_length

    class DummyDataLoader:
        def __init__(self):
            self.dataset = DummyDataset()

    arr_3d = np.arange(basin_num * i_e_time_length * arr_3d_time_steps * nf).reshape(
        basin_num * i_e_time_length, arr_3d_time_steps, nf
    )

    data_shape = (basin_num, nt, nf)
    data_loader = DummyDataLoader()

    result = _recover_samples_to_4d_by_forecast(
        data_shape, data_loader, stride, hindcast_output_window, arr_3d
    )

    # Check shape
    assert result.shape == (forecast_length, basin_num, i_e_time_length, nf)
    # Check that the values are correct for all features
    output = arr_3d.reshape(basin_num, i_e_time_length, arr_3d_time_steps, nf)
    forecast_output = output[:, :, -forecast_length:, :]
    for j in range(forecast_length):
        valid_indices = np.arange(i_e_time_length - j)
        for basin_idx in range(basin_num):
            expected = forecast_output[basin_idx, valid_indices + j, j, :]
            actual = result[j, basin_idx, valid_indices, :]
            assert np.allclose(actual, expected)


def test_recover_samples_to_4d_by_forecast_empty():
    basin_num = 1
    nt = 3
    nf = 1
    forecast_length = 2
    stride = 1
    hindcast_output_window = 0
    i_e_time_length = 0
    arr_3d_time_steps = 3

    class DummyDataset:
        def __init__(self):
            self.t_s_dict = {"sites_id": [0]}
            self.horizon = forecast_length

    class DummyDataLoader:
        def __init__(self):
            self.dataset = DummyDataset()

    arr_3d = np.empty((0, arr_3d_time_steps, nf))

    data_shape = (basin_num, nt, nf)
    data_loader = DummyDataLoader()

    result = _recover_samples_to_4d_by_forecast(
        data_shape, data_loader, stride, hindcast_output_window, arr_3d
    )

    assert result.shape == (forecast_length, basin_num, 0, nf)
    assert np.all(np.isnan(result))
