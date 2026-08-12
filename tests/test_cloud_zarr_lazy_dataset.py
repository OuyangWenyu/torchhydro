import csv
import json
import multiprocessing as mp
import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr
from torch.utils.data import DataLoader

from torchhydro.datasets.cloud_zarr_lazy_dataset import CloudZarrLazyDataset
from torchhydro.datasets.cloud_zarr_sampler import CloudZarrChunkBatchSampler
from torchhydro.models.crits import RMSELoss
from torchhydro.models.simple_lstm import SimpleLSTM
from torchhydro.trainers.train_utils import compute_loss, torch_single_train
from torchhydro.trainers.performance_monitor import TrainingPerformanceMonitor
from torchhydro.trainers.streaming_evaluation import evaluate_cloud_zarr_lazy_nse
from experiments.train_cloud_zarr_lazy_joint import build_cfg, infer_model_sizes


def _prebatched_collate_fn(batch):
    return batch


def _base_cfg(tmp_path, ts_uri, attr_uri, relevant_cols=None):
    return {
        "data_cfgs": {
            "dataset": "CloudZarrLazyDataset",
            "case_dir": str(tmp_path / "case"),
            "source_cfgs": {"dataset_id": "camels_us", "other_settings": {}},
            "object_ids": ["01013500", "01022500"],
            "t_range_train": ["2000-01-01", "2000-01-10"],
            "t_range_valid": None,
            "t_range_test": ["2000-01-01", "2000-01-10"],
            "target_cols": {"streamflow": "q_cms_obs"},
            "relevant_cols": relevant_cols or {"precipitation": "pcp_mm"},
            "constant_cols": {"area": "area_km2"},
            "target_rm_nan": False,
            "relevant_rm_nan": False,
            "constant_rm_nan": False,
            "scaler": "StandardScaler",
            "scaler_params": {
                "prcp_norm_cols": [],
                "gamma_norm_cols": [],
                "pbm_norm": False,
            },
            "stat_dict_file": None,
            "min_time_unit": "D",
            "min_time_interval": 1,
            "lazy_zarr": {
                "timeseries_uri": str(ts_uri),
                "attributes_uri": str(attr_uri),
                "use_hydrodataset_reader": False,
                "normalization": "standard",
                "stat_cache_dir": str(tmp_path / "stat_cache"),
                "skip_all_nan_target_windows": True,
            },
        },
        "training_cfgs": {
            "batch_size": 3,
            "hindcast_length": 3,
            "forecast_length": 1,
            "warmup_length": 0,
            "valid_batch_mode": "valid",
            "variable_length_cfgs": {"use_variable_length": False},
        },
        "evaluation_cfgs": {
            "rolling": 0,
            "hrwin": None,
            "frwin": None,
            "evaluator": {"recover_mode": "bybasins"},
        },
    }


@pytest.fixture()
def tiny_zarr_stores(tmp_path):
    pytest.importorskip("zarr")
    return _write_tiny_zarr(
        tmp_path,
        "camels_us",
        ["01013500", "01022500"],
        include_temperature=False,
    )


def _write_tiny_zarr(tmp_path, name, basins, include_temperature=False, offset=0.0):
    import zarr

    basins = np.array(basins)
    times = pd.date_range("2000-01-01", periods=10, freq="D")
    pcp = (
        np.arange(len(basins) * 10, dtype=np.float32).reshape(len(basins), 10) + offset
    )
    flow = (
        np.arange(100, 100 + len(basins) * 10, dtype=np.float32).reshape(
            len(basins), 10
        )
        + offset
    )
    data_vars = {
        "pcp_mm": (pcp, "mm/day"),
        "q_cms_obs": (flow, "m^3/s"),
    }
    if include_temperature:
        data_vars["temp_mean"] = (
            np.ones((len(basins), 10), dtype=np.float32) * (10.0 + offset),
            "degC",
        )
    ts_uri = tmp_path / f"{name}_timeseries.zarr"
    attr_uri = tmp_path / f"{name}_attributes.zarr"

    ts_root = zarr.open_group(ts_uri, mode="w", zarr_format=2)
    for var_name, (values, unit) in data_vars.items():
        array = ts_root.create_array(
            var_name,
            shape=values.shape,
            chunks=(min(2, len(basins)), 5),
            dtype="float32",
        )
        array[:] = values
        array.attrs["_ARRAY_DIMENSIONS"] = ["basin", "time"]
        array.attrs["units"] = unit
    basin_dtype = f"<U{max(len(str(value)) for value in basins)}"
    basin_array = ts_root.create_array(
        "basin",
        shape=basins.shape,
        chunks=basins.shape,
        dtype=basin_dtype,
    )
    basin_array[:] = basins.astype(basin_dtype)
    basin_array.attrs["_ARRAY_DIMENSIONS"] = ["basin"]
    time_array = ts_root.create_array(
        "time",
        shape=(len(times),),
        chunks=(5,),
        dtype="int64",
    )
    time_array[:] = times.asi8
    time_array.attrs.update(
        {
            "_ARRAY_DIMENSIONS": ["time"],
            "units": "nanoseconds since 1970-01-01",
            "calendar": "proleptic_gregorian",
        }
    )

    attr_root = zarr.open_group(attr_uri, mode="w", zarr_format=2)
    area = attr_root.create_array(
        "area_km2",
        shape=(len(basins),),
        chunks=(min(2, len(basins)),),
        dtype="float32",
    )
    area[:] = np.arange(len(basins), dtype=np.float32) + 10.0
    area.attrs["_ARRAY_DIMENSIONS"] = ["basin"]
    attr_basin = attr_root.create_array(
        "basin",
        shape=basins.shape,
        chunks=basins.shape,
        dtype=basin_dtype,
    )
    attr_basin[:] = basins.astype(basin_dtype)
    attr_basin.attrs["_ARRAY_DIMENSIONS"] = ["basin"]
    return ts_uri, attr_uri


def _target_qc_cfg(tmp_path, *, normalization="none"):
    import zarr

    ts_uri, attr_uri = _write_tiny_zarr(tmp_path, "target_qc", ["01013500"])
    target_values = np.array(
        [[1.0, 0.0, 2.0, 3.0, -1.0, 4.0, 5.0, np.nan, -9999.0, 6.0]],
        dtype=np.float32,
    )
    root = zarr.open_group(ts_uri, mode="r+")
    del root["q_cms_obs"]
    target = root.create_array(
        "q_cms_obs",
        shape=target_values.shape,
        chunks=(1, 5),
        dtype="float32",
        fill_value=-9999.0,
    )
    target[:] = target_values
    target.attrs.update(
        {
            "_ARRAY_DIMENSIONS": ["basin", "time"],
            "units": "m^3/s",
            "_FillValue": -9999.0,
            "missing_value": -9999.0,
        }
    )

    cfg = _base_cfg(tmp_path, ts_uri, attr_uri)
    cfg["data_cfgs"]["object_ids"] = ["01013500"]
    cfg["data_cfgs"]["lazy_zarr"].update(
        {
            "normalization": normalization,
            "target_nan_policy": "keep",
            "skip_all_nan_target_windows": False,
        }
    )
    return cfg, target_values


def _multi_source_cfg(tmp_path, *, normalization="standard"):
    ts_us, attr_us = _write_tiny_zarr(
        tmp_path,
        "camels_us",
        ["01013500", "01022500"],
        include_temperature=True,
    )
    ts_gb, attr_gb = _write_tiny_zarr(
        tmp_path,
        "camels_gb",
        ["gb_001"],
        include_temperature=False,
        offset=1000.0,
    )
    cfg = _base_cfg(tmp_path, ts_us, attr_us)
    cfg["data_cfgs"]["lazy_zarr"] = {
        "use_hydrodataset_reader": False,
        "normalization": normalization,
        "normalization_scope": "global",
        "stat_cache_dir": str(tmp_path / "stat_cache"),
        "input_schema": ["precipitation", "temperature_mean"],
        "target_schema": ["streamflow"],
        "constant_schema": ["area"],
        "missing_policy": "fill_and_mask",
        "worker_cache_size_mb": 1,
        "datasets": [
            {
                "dataset_id": "camels_us",
                "timeseries_uri": str(ts_us),
                "attributes_uri": str(attr_us),
                "basins": ["01013500", "01022500"],
                "time_range": ["2000-01-01", "2000-01-10"],
                "relevant_cols": {
                    "precipitation": "pcp_mm",
                    "temperature_mean": "temp_mean",
                },
                "target_cols": {"streamflow": "q_cms_obs"},
                "constant_cols": {"area": "area_km2"},
                "weight": 1.0,
            },
            {
                "dataset_id": "camels_gb",
                "timeseries_uri": str(ts_gb),
                "attributes_uri": str(attr_gb),
                "basins": ["gb_001"],
                "time_range": ["2000-01-01", "2000-01-10"],
                "relevant_cols": {"precipitation": "pcp_mm"},
                "target_cols": {"streamflow": "q_cms_obs"},
                "constant_cols": {"area": "area_km2"},
                "weight": 3.0,
            },
        ],
    }
    return cfg


def test_cloud_zarr_lazy_dataset_prebatched_shape(tmp_path, tiny_zarr_stores):
    ts_uri, attr_uri = tiny_zarr_stores
    dataset = CloudZarrLazyDataset(_base_cfg(tmp_path, ts_uri, attr_uri), "train")
    dataset._batch_level_dataloader_enabled = True

    loader = DataLoader(
        dataset,
        batch_size=3,
        shuffle=False,
        num_workers=0,
        collate_fn=_prebatched_collate_fn,
    )
    x, y = next(iter(loader))

    assert tuple(x.shape) == (3, 4, 2)
    assert tuple(y.shape) == (3, 4, 1)
    assert x.dtype == y.dtype
    assert len(dataset) == 14


def test_split_datasets_reuse_train_normalization_stats(tmp_path, tiny_zarr_stores):
    ts_uri, attr_uri = tiny_zarr_stores
    cfg = _base_cfg(tmp_path, ts_uri, attr_uri)
    cfg["data_cfgs"]["t_range_train"] = ["2000-01-01", "2000-01-05"]
    cfg["data_cfgs"]["t_range_valid"] = ["2000-01-06", "2000-01-10"]
    cfg["data_cfgs"]["t_range_test"] = ["2000-01-06", "2000-01-10"]
    cfg["data_cfgs"]["lazy_zarr"]["split_periods"] = {
        "train": ["2000-01-01", "2000-01-05"],
        "valid": ["2000-01-06", "2000-01-10"],
        "test": ["2000-01-06", "2000-01-10"],
    }

    train = CloudZarrLazyDataset(cfg, "train")
    valid = CloudZarrLazyDataset(cfg, "valid")
    test = CloudZarrLazyDataset(cfg, "test")
    try:
        train_stats = train.source_states[0].stats
        assert train.source_states[0].times[0] == pd.Timestamp("2000-01-01")
        assert valid.source_states[0].times[0] == pd.Timestamp("2000-01-06")
        assert test.source_states[0].times[-1] == pd.Timestamp("2000-01-10")
        assert valid.source_states[0].stats == train_stats
        assert test.source_states[0].stats == train_stats
        assert train_stats["relevant_cols"]["precipitation"]["mean"] < 10.0
    finally:
        train.close()
        valid.close()
        test.close()


def test_split_period_requires_full_zarr_coverage(tmp_path, tiny_zarr_stores):
    ts_uri, attr_uri = tiny_zarr_stores
    cfg = _base_cfg(tmp_path, ts_uri, attr_uri)
    cfg["data_cfgs"]["lazy_zarr"]["split_periods"] = {
        "train": ["1999-12-31", "2000-01-05"],
        "valid": None,
        "test": ["2000-01-01", "2000-01-10"],
    }

    with pytest.raises(ValueError, match="does not cover the requested train period"):
        CloudZarrLazyDataset(cfg, "train")


def test_validation_requires_persisted_training_stats(tmp_path, tiny_zarr_stores):
    ts_uri, attr_uri = tiny_zarr_stores
    cfg = _base_cfg(tmp_path, ts_uri, attr_uri)
    cfg["data_cfgs"]["lazy_zarr"]["split_periods"] = {
        "train": ["2000-01-01", "2000-01-05"],
        "valid": ["2000-01-06", "2000-01-10"],
        "test": ["2000-01-06", "2000-01-10"],
    }

    with pytest.raises(
        ValueError, match="Training normalization statistics are required"
    ):
        CloudZarrLazyDataset(cfg, "valid")


@pytest.mark.parametrize("split", ["valid", "test"])
def test_non_train_split_cannot_fit_normalization_stats(
    tmp_path, tiny_zarr_stores, split
):
    ts_uri, attr_uri = tiny_zarr_stores
    cfg = _base_cfg(tmp_path, ts_uri, attr_uri)
    cfg["data_cfgs"]["lazy_zarr"]["split_periods"] = {
        "train": ["2000-01-01", "2000-01-05"],
        "valid": ["2000-01-06", "2000-01-10"],
        "test": ["2000-01-06", "2000-01-10"],
    }

    with pytest.raises(
        ValueError, match="Training normalization statistics are required"
    ):
        CloudZarrLazyDataset(cfg, split)


def test_training_stats_signature_rejects_changed_variable_mapping(
    tmp_path, tiny_zarr_stores
):
    ts_uri, attr_uri = tiny_zarr_stores
    cfg = _base_cfg(tmp_path, ts_uri, attr_uri)
    cfg["data_cfgs"]["lazy_zarr"]["split_periods"] = {
        "train": ["2000-01-01", "2000-01-05"],
        "valid": ["2000-01-06", "2000-01-10"],
        "test": ["2000-01-06", "2000-01-10"],
    }
    train = CloudZarrLazyDataset(cfg, "train")
    train.close()

    changed_cfg = json.loads(json.dumps(cfg))
    changed_cfg["data_cfgs"]["lazy_zarr"]["datasets"] = [
        {
            "dataset_id": "camels_us",
            "timeseries_uri": str(ts_uri),
            "attributes_uri": str(attr_uri),
            "basins": ["01013500", "01022500"],
            "time_range": ["2000-01-06", "2000-01-10"],
            "relevant_cols": {"precipitation": {"physical": "pcp_mm", "scale": 2.0}},
            "target_cols": {"streamflow": "q_cms_obs"},
            "constant_cols": {"area": "area_km2"},
        }
    ]

    with pytest.raises(ValueError, match="do not match"):
        CloudZarrLazyDataset(changed_cfg, "valid")


def test_cloud_zarr_lazy_dataset_reopens_handles_after_fork(tmp_path, tiny_zarr_stores):
    ts_uri, attr_uri = tiny_zarr_stores
    dataset = CloudZarrLazyDataset(_base_cfg(tmp_path, ts_uri, attr_uri), "train")
    old_ts = dataset.source_states[0].ts_ds
    assert old_ts is not None

    dataset._open_pid = -1
    dataset._ensure_open(0)

    assert dataset._open_pid != -1
    assert dataset.source_states[0].ts_ds is not None
    assert dataset.source_states[0].ts_ds is not old_ts


def test_cloud_zarr_lazy_dataset_pickle_reopens_handles(tmp_path, tiny_zarr_stores):
    ts_uri, attr_uri = tiny_zarr_stores
    cfg = _base_cfg(tmp_path, ts_uri, attr_uri)
    cfg["data_cfgs"]["lazy_zarr"]["normalization"] = "none"
    cfg["data_cfgs"]["lazy_zarr"]["skip_all_nan_target_windows"] = False
    dataset = CloudZarrLazyDataset(cfg, "train")

    restored = pickle.loads(pickle.dumps(dataset))
    restored._batch_level_dataloader_enabled = True
    x, y = restored.__getitems__([0, 1])

    assert restored._open_pid == mp.current_process().pid
    assert tuple(x.shape) == (2, 4, 2)
    assert tuple(y.shape) == (2, 4, 1)


def test_cloud_zarr_lazy_dataset_opens_zarr_without_dask_chunks(
    tmp_path, tiny_zarr_stores
):
    ts_uri, attr_uri = tiny_zarr_stores
    dataset = CloudZarrLazyDataset(_base_cfg(tmp_path, ts_uri, attr_uri), "train")

    assert dataset.source_states[0].ts_ds["pcp_mm"].chunks is None


def test_cloud_zarr_lazy_dataset_reopens_reader_zarr_without_dask(
    tmp_path, tiny_zarr_stores, monkeypatch
):
    import xarray as xr

    ts_uri, attr_uri = tiny_zarr_stores

    class FakeReader:
        def read_object_ids(self):
            return np.asarray(["01013500", "01022500"])

        def _load_ts_dataset(self):
            return xr.open_zarr(ts_uri, consolidated=False, mask_and_scale=False)

        def _load_attr_dataset(self):
            return xr.open_zarr(attr_uri, consolidated=False, mask_and_scale=False)

    monkeypatch.setattr(
        "torchhydro.datasets.cloud_zarr_lazy_dataset.open_dataset",
        lambda *args, **kwargs: FakeReader(),
    )
    cfg = _base_cfg(tmp_path, ts_uri, attr_uri)
    cfg["data_cfgs"]["lazy_zarr"].pop("timeseries_uri")
    cfg["data_cfgs"]["lazy_zarr"].pop("attributes_uri")
    cfg["data_cfgs"]["lazy_zarr"]["use_hydrodataset_reader"] = True

    dataset = CloudZarrLazyDataset(cfg, "train")

    assert dataset.source_states[0].ts_ds["pcp_mm"].chunks is None


def test_cloud_zarr_lazy_dataset_optional_present_variable_has_no_missing_mask(
    tmp_path,
):
    ts_uri, attr_uri = _write_tiny_zarr(
        tmp_path,
        "camels_us_optional",
        ["01013500", "01022500"],
        include_temperature=True,
    )
    cfg = _base_cfg(
        tmp_path,
        ts_uri,
        attr_uri,
        relevant_cols={
            "precipitation": "pcp_mm",
            "temperature_mean": {
                "physical": "temp_mean",
                "required": False,
            },
        },
    )
    cfg["data_cfgs"]["lazy_zarr"]["missing_policy"] = "fill_and_mask"
    dataset = CloudZarrLazyDataset(cfg, "train")
    dataset._batch_level_dataloader_enabled = True
    x, _ = dataset.__getitems__([0, 1])

    assert dataset._mask_schema == []
    assert tuple(x.shape) == (2, 4, 3)


def test_cloud_zarr_lazy_dataset_missing_variable_mask(tmp_path, tiny_zarr_stores):
    ts_uri, attr_uri = tiny_zarr_stores
    cfg = _base_cfg(
        tmp_path,
        ts_uri,
        attr_uri,
        relevant_cols={
            "precipitation": "pcp_mm",
            "temperature_mean": {
                "physical": "missing_temperature",
                "required": False,
                "fill_value": 0.0,
            },
        },
    )
    cfg["data_cfgs"]["lazy_zarr"]["missing_policy"] = "fill_and_mask"
    dataset = CloudZarrLazyDataset(cfg, "train")
    dataset._batch_level_dataloader_enabled = True
    x, y = dataset.__getitems__([0, 1])

    assert tuple(x.shape) == (2, 4, 4)
    assert tuple(y.shape) == (2, 4, 1)
    assert np.allclose(x[:, :, 2].numpy(), 1.0)


def test_missing_required_target_reports_dataset_and_available_variables(
    tmp_path, tiny_zarr_stores
):
    ts_uri, attr_uri = tiny_zarr_stores
    cfg = _base_cfg(tmp_path, ts_uri, attr_uri)
    cfg["data_cfgs"]["target_cols"] = {"streamflow": "missing_streamflow"}

    with pytest.raises(ValueError) as exc_info:
        CloudZarrLazyDataset(cfg, "train")

    message = str(exc_info.value)
    assert "Dataset 'camels_us'" in message
    assert "missing_streamflow" in message
    assert "q_cms_obs" in message


def test_cloud_block_read_retries_transient_endpoint_error(
    tmp_path, tiny_zarr_stores, monkeypatch
):
    ts_uri, attr_uri = tiny_zarr_stores
    cfg = _base_cfg(tmp_path, ts_uri, attr_uri)
    cfg["data_cfgs"]["lazy_zarr"].update(
        {
            "normalization": "none",
            "skip_all_nan_target_windows": False,
            "cloud_read_max_retries": 3,
            "cloud_read_retry_base_seconds": 0.0,
            "cloud_read_retry_max_seconds": 0.0,
        }
    )
    dataset = CloudZarrLazyDataset(cfg, "train")
    original = xr.DataArray.to_numpy
    calls = {"count": 0}
    endpoint_error = type("EndpointConnectionError", (OSError,), {})

    def flaky_to_numpy(array):
        if array.name == "pcp_mm" and calls["count"] < 2:
            calls["count"] += 1
            raise endpoint_error("Timeout while contacting DNS servers")
        return original(array)

    monkeypatch.setattr(xr.DataArray, "to_numpy", flaky_to_numpy)

    with pytest.warns(RuntimeWarning, match="cloud read failed"):
        block = dataset._read_physical_block_with_retry(0, "pcp_mm", 0, 2, 0, 3)

    assert calls["count"] == 2
    assert block.shape == (2, 3)
    assert not dataset._is_retryable_cloud_error(ValueError("bad dimensions"))


def test_cloud_zarr_lazy_dataset_drop_any_nan_keeps_targets_finite(
    tmp_path, tiny_zarr_stores
):
    import zarr

    ts_uri, attr_uri = tiny_zarr_stores
    root = zarr.open_group(ts_uri, mode="r+")
    root["q_cms_obs"][0, 2] = np.nan
    cfg = _base_cfg(tmp_path, ts_uri, attr_uri)
    cfg["data_cfgs"]["lazy_zarr"]["normalization"] = "none"
    cfg["data_cfgs"]["lazy_zarr"]["target_nan_policy"] = "drop_any_nan"
    dataset = CloudZarrLazyDataset(cfg, "train")
    dataset._batch_level_dataloader_enabled = True

    _, y = dataset.__getitems__(list(range(len(dataset))))

    assert len(dataset) == 11
    assert torch.isfinite(y).all()


def test_negative_streamflow_is_invalid(tmp_path):
    cfg, _ = _target_qc_cfg(tmp_path)
    dataset = CloudZarrLazyDataset(cfg, "test")
    dataset._batch_level_dataloader_enabled = True

    _, y = dataset.__getitems__([1])

    assert torch.isnan(y[0, -1, 0])


def test_zero_streamflow_remains_valid(tmp_path):
    cfg, _ = _target_qc_cfg(tmp_path)
    dataset = CloudZarrLazyDataset(cfg, "test")
    dataset._batch_level_dataloader_enabled = True

    _, y = dataset.__getitems__([0])

    assert y[0, 1, 0].item() == pytest.approx(0.0)
    assert torch.isfinite(y[0, 1, 0])


def test_missing_target_is_not_converted_to_zero(tmp_path):
    cfg, _ = _target_qc_cfg(tmp_path)
    dataset = CloudZarrLazyDataset(cfg, "test")
    dataset._batch_level_dataloader_enabled = True

    _, y = dataset.__getitems__([5])

    assert torch.isnan(y[0, 2:, 0]).all()
    assert not torch.eq(y[0, 2:, 0], 0.0).any()


def test_target_qc_runs_after_scale_offset(tmp_path):
    cfg, _ = _target_qc_cfg(tmp_path)
    cfg["data_cfgs"]["target_cols"] = {
        "streamflow": {
            "physical": "q_cms_obs",
            "scale": 0.1,
            "offset": 2.0,
        }
    }
    dataset = CloudZarrLazyDataset(cfg, "test")
    dataset._batch_level_dataloader_enabled = True

    _, y = dataset.__getitems__([5])
    _, negative_window = dataset.__getitems__([1])

    assert y[0, :2, 0].numpy() == pytest.approx([2.4, 2.5])
    assert torch.isnan(y[0, 2:, 0]).all()
    assert negative_window[0, -1, 0].item() == pytest.approx(1.9)


def test_invalid_target_is_excluded_from_scaler_statistics(tmp_path):
    cfg, _ = _target_qc_cfg(tmp_path, normalization="standard")
    dataset = CloudZarrLazyDataset(cfg, "train")

    stat = dataset.source_states[0].stats["target_cols"]["streamflow"]

    assert stat["count"] == 7
    assert stat["mean"] == pytest.approx(3.0)


def test_training_lookup_excludes_invalid_target_windows(tmp_path):
    cfg, _ = _target_qc_cfg(tmp_path, normalization="none")
    cfg["data_cfgs"]["lazy_zarr"]["target_nan_policy"] = "drop_any_nan"
    dataset = CloudZarrLazyDataset(cfg, "train")
    dataset._batch_level_dataloader_enabled = True

    assert len(dataset) == 1
    _, y = dataset.__getitems__([0])
    assert torch.isfinite(y).all()
    assert y[0, :, 0].numpy() == pytest.approx([1.0, 0.0, 2.0, 3.0])


def test_training_loss_ignores_invalid_target():
    output = torch.tensor([[2.0, 100.0, 0.0]], requires_grad=True)
    target = torch.tensor([[1.0, float("nan"), 0.0]])

    loss = RMSELoss()(output, target)
    loss.backward()

    assert loss.item() == pytest.approx(np.sqrt(0.5))
    assert torch.isfinite(output.grad).all()


def test_generic_training_loss_ignores_invalid_target():
    output = torch.tensor([[2.0, 100.0, 0.0]], requires_grad=True)
    target = torch.tensor([[1.0, float("nan"), 0.0]])

    loss = compute_loss(target, output, torch.nn.MSELoss())
    loss.backward()

    assert loss.item() == pytest.approx(0.5)
    assert torch.isfinite(output.grad).all()


def test_validation_metric_ignores_missing_target(tmp_path):
    cfg, _ = _target_qc_cfg(tmp_path)
    dataset = CloudZarrLazyDataset(cfg, "test")

    class ZeroModel(torch.nn.Module):
        def forward(self, x):
            return x[..., :1] * 0.0

    result = evaluate_cloud_zarr_lazy_nse(
        ZeroModel(), dataset, torch.device("cpu"), batch_size=len(dataset)
    )

    assert result["global"]["streamflow"]["count"] == 4


def test_cloud_zarr_lazy_dataset_multi_source_global_schema(tmp_path):
    cfg = _multi_source_cfg(tmp_path)
    dataset = CloudZarrLazyDataset(cfg, "train")
    dataset._batch_level_dataloader_enabled = True

    first_us = int(np.flatnonzero(dataset._lookup_sources == 0)[0])
    first_gb = int(np.flatnonzero(dataset._lookup_sources == 1)[0])
    x, y = dataset.__getitems__([first_us, first_gb])

    assert tuple(x.shape) == (2, 4, 4)
    assert tuple(y.shape) == (2, 4, 1)
    assert x[0, :, 2].eq(0.0).all()
    assert x[1, :, 2].eq(1.0).all()

    sampler = CloudZarrChunkBatchSampler(
        dataset,
        batch_size=2,
        time_chunk_size=4,
        sampling="weighted_by_sample",
    )
    first_batch = next(iter(sampler))
    assert len(first_batch) == 2
    assert len(set(dataset._lookup_sources[first_batch].tolist())) == 1

    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=0,
        collate_fn=_prebatched_collate_fn,
    )
    xb, yb = next(iter(loader))
    assert tuple(xb.shape) == (2, 4, 4)
    assert tuple(yb.shape) == (2, 4, 1)


def test_cloud_zarr_lazy_dataset_uses_global_multi_source_stats(tmp_path):
    dataset = CloudZarrLazyDataset(_multi_source_cfg(tmp_path), "train")
    us_stat = dataset.source_states[0].stats["relevant_cols"]["precipitation"]
    gb_stat = dataset.source_states[1].stats["relevant_cols"]["precipitation"]
    expected_mean = np.mean(np.concatenate((np.arange(20), np.arange(1000, 1010))))

    assert us_stat == gb_stat
    assert us_stat["mean"] == pytest.approx(expected_mean)
    assert us_stat["count"] == 30


def test_cloud_zarr_lazy_dataset_applies_per_dataset_scale_and_offset(tmp_path):
    cfg = _multi_source_cfg(tmp_path, normalization="none")
    cfg["data_cfgs"]["lazy_zarr"]["datasets"][1]["relevant_cols"] = {
        "precipitation": {
            "physical": "pcp_mm",
            "scale": 0.1,
            "offset": 2.0,
            "unit": "mm/day",
        }
    }
    dataset = CloudZarrLazyDataset(cfg, "train")
    dataset._batch_level_dataloader_enabled = True
    first_gb = int(np.flatnonzero(dataset._lookup_sources == 1)[0])

    x, _ = dataset.__getitems__([first_gb])

    assert x[0, :, 0].numpy() == pytest.approx([102.0, 102.1, 102.2, 102.3])


def test_cloud_zarr_lazy_dataset_applies_scale_offset_before_fitting_stats(tmp_path):
    cfg = _multi_source_cfg(tmp_path)
    cfg["data_cfgs"]["lazy_zarr"]["normalization_scope"] = "per_dataset"
    cfg["data_cfgs"]["lazy_zarr"]["datasets"][1]["relevant_cols"] = {
        "precipitation": {
            "physical": "pcp_mm",
            "scale": 0.1,
            "offset": 2.0,
            "unit": "mm/day",
        }
    }

    dataset = CloudZarrLazyDataset(cfg, "train")
    try:
        stat = dataset.source_states[1].stats["relevant_cols"]["precipitation"]
        transformed = np.arange(1000.0, 1010.0) * 0.1 + 2.0

        assert stat["mean"] == pytest.approx(float(transformed.mean()))
        assert stat["std"] == pytest.approx(float(transformed.std()))
        assert stat["count"] == transformed.size

        dataset._batch_level_dataloader_enabled = True
        first_gb = int(np.flatnonzero(dataset._lookup_sources == 1)[0])
        x, _ = dataset.__getitems__([first_gb])
        expected = (transformed[:4] - transformed.mean()) / transformed.std()
        assert x[0, :, 0].numpy() == pytest.approx(expected, abs=5e-5)
    finally:
        dataset.close()


def test_cloud_zarr_lazy_dataset_fills_and_masks_unconfigured_static_variable(
    tmp_path,
):
    cfg = _multi_source_cfg(tmp_path)
    gb_entry = cfg["data_cfgs"]["lazy_zarr"]["datasets"][1]
    gb_entry["constant_cols"] = []
    gb_entry.pop("attributes_uri")
    dataset = CloudZarrLazyDataset(cfg, "train")
    dataset._batch_level_dataloader_enabled = True
    first_us = int(np.flatnonzero(dataset._lookup_sources == 0)[0])
    first_gb = int(np.flatnonzero(dataset._lookup_sources == 1)[0])

    x, _ = dataset.__getitems__([first_us, first_gb])

    assert tuple(x.shape) == (2, 4, 5)
    assert x[0, :, 2:4].eq(0.0).all()
    assert x[1, :, 1].eq(0.0).all()
    assert x[1, :, 2:4].eq(1.0).all()
    assert x[1, :, 4].eq(0.0).all()


def test_cloud_zarr_chunk_sampler_honors_dataset_weights(tmp_path):
    dataset = CloudZarrLazyDataset(
        _multi_source_cfg(tmp_path, normalization="none"), "train"
    )
    sampler = CloudZarrChunkBatchSampler(
        dataset,
        batch_size=2,
        sampling="weighted_by_dataset",
        max_batches_per_epoch=8,
        shuffle=False,
    )

    batches = list(sampler)
    source_counts = {0: 0, 1: 0}
    for batch in batches:
        source_counts[int(dataset._lookup_sources[batch[0]])] += 1

    assert len(sampler) == len(batches) == 8
    assert source_counts == {0: 2, 1: 6}


def test_cloud_zarr_chunk_sampler_keeps_chunk_runs_contiguous(tmp_path):
    dataset = CloudZarrLazyDataset(
        _multi_source_cfg(tmp_path, normalization="none"), "train"
    )
    sampler = CloudZarrChunkBatchSampler(
        dataset,
        batch_size=2,
        sampling="weighted_by_sample",
        shuffle=True,
        seed=7,
    )

    keys = [sampler.chunk_key(batch[0]) for batch in sampler]
    runs = [key for pos, key in enumerate(keys) if pos == 0 or key != keys[pos - 1]]

    assert len(runs) == len(set(runs))
    assert sampler.effective_chunk_shapes[0] == (2, 5)
    assert sampler.index_storage_nbytes <= len(dataset) * 8


def test_cloud_zarr_chunk_sampler_default_chunk_can_fill_batch(
    tmp_path, tiny_zarr_stores
):
    ts_uri, attr_uri = tiny_zarr_stores
    dataset = CloudZarrLazyDataset(_base_cfg(tmp_path, ts_uri, attr_uri), "train")
    sampler = CloudZarrChunkBatchSampler(
        dataset,
        batch_size=5,
        time_chunk_size=None,
        drop_last=True,
        shuffle=False,
    )

    first_batch = next(iter(sampler))
    assert sampler.time_chunk_size == 5
    assert len(first_batch) == 5
    assert len(sampler) == 2


def test_cloud_zarr_lazy_dataset_reads_grouped_windows(
    tmp_path, tiny_zarr_stores, monkeypatch
):
    ts_uri, attr_uri = tiny_zarr_stores
    cfg = _base_cfg(tmp_path, ts_uri, attr_uri)
    cfg["data_cfgs"]["lazy_zarr"]["normalization"] = "none"
    cfg["data_cfgs"]["lazy_zarr"]["skip_all_nan_target_windows"] = False
    dataset = CloudZarrLazyDataset(cfg, "train")
    dataset._batch_level_dataloader_enabled = True

    calls = []
    original = dataset._read_block_array

    def counted_read_block_array(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(dataset, "_read_block_array", counted_read_block_array)
    x, y = dataset.__getitems__([0, 1, 2, 3, 4])
    dataset.__getitems__([5, 6])

    assert tuple(x.shape) == (5, 4, 2)
    assert tuple(y.shape) == (5, 4, 1)
    assert len(calls) == 4


def test_cloud_zarr_lazy_dataset_reads_and_caches_multi_basin_blocks(
    tmp_path, tiny_zarr_stores, monkeypatch
):
    ts_uri, attr_uri = tiny_zarr_stores
    cfg = _base_cfg(tmp_path, ts_uri, attr_uri)
    cfg["data_cfgs"]["lazy_zarr"].update(
        {
            "normalization": "none",
            "skip_all_nan_target_windows": False,
            "worker_cache_size_mb": 1,
        }
    )
    dataset = CloudZarrLazyDataset(cfg, "train")
    dataset._batch_level_dataloader_enabled = True
    basin_zero = int(np.flatnonzero(dataset._lookup_basins == 0)[0])
    basin_one = int(np.flatnonzero(dataset._lookup_basins == 1)[0])

    calls = []
    original = dataset._read_block_array

    def counted_read_block_array(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(dataset, "_read_block_array", counted_read_block_array)
    first = dataset.__getitems__([basin_zero, basin_one])
    second = dataset.__getitems__([basin_zero, basin_one])

    assert tuple(first[0].shape) == tuple(second[0].shape) == (2, 4, 2)
    assert len(calls) == 2


@pytest.mark.skipif(
    "spawn" not in mp.get_all_start_methods(), reason="spawn unavailable"
)
def test_cloud_zarr_lazy_dataset_spawn_workers(tmp_path, tiny_zarr_stores):
    ts_uri, attr_uri = tiny_zarr_stores
    cfg = _base_cfg(tmp_path, ts_uri, attr_uri)
    cfg["data_cfgs"]["lazy_zarr"]["normalization"] = "none"
    cfg["data_cfgs"]["lazy_zarr"]["skip_all_nan_target_windows"] = False
    dataset = CloudZarrLazyDataset(cfg, "train")
    dataset._batch_level_dataloader_enabled = True
    loader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=1,
        multiprocessing_context="spawn",
        timeout=30,
        collate_fn=_prebatched_collate_fn,
    )

    x, y = next(iter(loader))

    assert tuple(x.shape) == (2, 4, 2)
    assert tuple(y.shape) == (2, 4, 1)


def test_cloud_zarr_lazy_dataset_torch_single_train(tmp_path, tiny_zarr_stores):
    ts_uri, attr_uri = tiny_zarr_stores
    cfg = _base_cfg(tmp_path, ts_uri, attr_uri)
    cfg["training_cfgs"].update(
        {
            "batch_size": 4,
            "which_first_tensor": "sequence",
            "num_workers": 0,
            "pin_memory": False,
        }
    )
    dataset = CloudZarrLazyDataset(cfg, "train")
    dataset._batch_level_dataloader_enabled = True
    sampler = CloudZarrChunkBatchSampler(
        dataset,
        batch_size=4,
        time_chunk_size=4,
        max_batches_per_epoch=2,
        drop_last=True,
        shuffle=False,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=0,
        collate_fn=_prebatched_collate_fn,
    )
    model = SimpleLSTM(
        input_size=dataset.x_feature_count,
        output_size=dataset.y_feature_count,
        hidden_size=8,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    monitor = TrainingPerformanceMonitor(tmp_path / "performance", torch.device("cpu"))
    monitor.start_epoch(1)

    loss, n_iter = torch_single_train(
        model,
        optimizer,
        RMSELoss(),
        loader,
        device=torch.device("cpu"),
        which_first_tensor="sequence",
        performance_monitor=monitor,
    )
    performance = monitor.finish_epoch(sampler=sampler, dataset=dataset)

    assert n_iter == 2
    assert np.isfinite(loss)
    assert performance["batches"] == 2
    assert performance["samples"] == 8
    assert performance["per_dataset_samples"] == {"camels_us": 8}


def test_streaming_nse_evaluation_uses_final_window_step(tmp_path, tiny_zarr_stores):
    ts_uri, attr_uri = tiny_zarr_stores
    cfg = _base_cfg(tmp_path, ts_uri, attr_uri)
    train = CloudZarrLazyDataset(cfg, "train")
    test = CloudZarrLazyDataset(cfg, "test")

    class ZeroModel(torch.nn.Module):
        def forward(self, x):
            return x[..., :1] * 0.0

    output_path = tmp_path / "streaming_metrics.json"
    try:
        result = evaluate_cloud_zarr_lazy_nse(
            ZeroModel(),
            test,
            torch.device("cpu"),
            batch_size=3,
            output_path=output_path,
        )
        stats = test.source_states[0].stats["target_cols"]["streamflow"]
        _, y = test._load_batch(list(range(len(test))))
        observed = y[:, -1, 0].numpy() * stats["std"] + stats["mean"]
        predicted = np.full_like(observed, stats["mean"], dtype=np.float64)
        expected_nse = 1.0 - np.square(observed - predicted).sum() / np.square(
            observed - observed.mean()
        ).sum()
        streamflow = result["global"]["streamflow"]
        assert result["prediction_policy"] == "last_step_per_window"
        assert streamflow["count"] == len(test)
        assert streamflow["nse"] == pytest.approx(expected_nse)
        basin_metrics = result["per_basin"]["camels_us"]
        assert set(basin_metrics) == {"01013500", "01022500"}
        assert sum(
            basin["streamflow"]["count"] for basin in basin_metrics.values()
        ) == len(test)
        assert json.loads(output_path.read_text())["global"] == result["global"]
        with output_path.with_suffix(".csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert {row["scope"] for row in rows} == {"global", "dataset", "basin"}
        assert sum(row["scope"] == "basin" for row in rows) == 2
    finally:
        train.close()
        test.close()


def test_streaming_nse_ignores_missing_raw_targets(tmp_path, tiny_zarr_stores):
    import zarr

    ts_uri, attr_uri = tiny_zarr_stores
    root = zarr.open_group(ts_uri, mode="r+")
    root["q_cms_obs"][0, 3] = np.nan
    cfg = _base_cfg(tmp_path, ts_uri, attr_uri)
    train = CloudZarrLazyDataset(cfg, "train")
    test = CloudZarrLazyDataset(cfg, "test")

    class ZeroModel(torch.nn.Module):
        def forward(self, x):
            return x[..., :1] * 0.0

    try:
        result = evaluate_cloud_zarr_lazy_nse(
            ZeroModel(), test, torch.device("cpu"), batch_size=3
        )
        raw_targets = test.load_final_target_values(list(range(len(test))))
        expected_count = int(np.isfinite(raw_targets[:, 0]).sum())
        assert expected_count < len(test)
        assert result["global"]["streamflow"]["count"] == expected_count
    finally:
        train.close()
        test.close()


def test_train_cloud_zarr_lazy_joint_dry_run_config(tmp_path, tiny_zarr_stores):
    ts_uri, attr_uri = tiny_zarr_stores
    args = type(
        "Args",
        (),
        {
            "datasets": "camels_us",
            "source": "cloud",
            "timeseries_uri": str(ts_uri),
            "attributes_uri": str(attr_uri),
            "use_hydrodataset_reader": False,
            "case_dir": str(tmp_path / "train_case"),
            "time_start": "2000-01-01",
            "time_end": "2000-01-10",
            "relevant_cols": "precipitation:pcp_mm",
            "target_cols": "streamflow:q_cms_obs",
            "constant_cols": "area_km2",
            "input_schema": "precipitation",
            "target_schema": "streamflow",
            "constant_schema": "area_km2",
            "missing_policy": "fill_zero",
            "normalization": "standard",
            "no_cache_stats": False,
            "no_valid_target_filter": False,
            "max_basins": 2,
            "batch_size": 4,
            "hindcast_length": 3,
            "forecast_length": 1,
            "warmup_length": 0,
            "epochs": 1,
            "hidden_size": 8,
            "dropout": 0.0,
            "learning_rate": 1e-3,
            "num_workers": 0,
            "prefetch_factor": 2,
            "persistent_workers": False,
            "pin_memory": False,
            "cuda_prefetch": False,
            "device": 0,
            "chunk_sampler": True,
            "sampling": "weighted_by_sample",
            "time_chunk_size": None,
            "basin_chunk_size": 1,
            "max_batches_per_epoch": 2,
            "save_epoch": 0,
            "config_out": None,
            "dry_run": True,
        },
    )()
    cfg = build_cfg(args)
    input_size, output_size, dataset_length = infer_model_sizes(cfg)

    assert input_size == 2
    assert output_size == 1
    assert dataset_length == 14


def test_train_cloud_zarr_lazy_joint_reads_dataset_manifest(tmp_path):
    cfg = _multi_source_cfg(tmp_path, normalization="none")
    manifest = {
        "input_schema": cfg["data_cfgs"]["lazy_zarr"]["input_schema"],
        "target_schema": cfg["data_cfgs"]["lazy_zarr"]["target_schema"],
        "constant_schema": cfg["data_cfgs"]["lazy_zarr"]["constant_schema"],
        "datasets": cfg["data_cfgs"]["lazy_zarr"]["datasets"],
    }
    manifest_path = tmp_path / "joint_datasets.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    args = SimpleNamespace(
        datasets="camels_us,camels_gb",
        dataset_config=manifest_path,
        source="cloud",
        timeseries_uri=None,
        attributes_uri=None,
        use_hydrodataset_reader=False,
        case_dir=str(tmp_path / "joint_case"),
        time_start="2000-01-01",
        time_end="2000-01-10",
        relevant_cols="precipitation",
        target_cols="streamflow",
        constant_cols="",
        input_schema="",
        target_schema="",
        constant_schema="",
        missing_policy="fill_and_mask",
        normalization="none",
        normalization_scope="global",
        no_cache_stats=False,
        no_valid_target_filter=True,
        max_basins=None,
        batch_size=2,
        hindcast_length=3,
        forecast_length=1,
        warmup_length=0,
        epochs=1,
        hidden_size=8,
        dropout=0.0,
        learning_rate=1e-3,
        num_workers=0,
        prefetch_factor=2,
        persistent_workers=False,
        pin_memory=False,
        cuda_prefetch=False,
        device=0,
        chunk_sampler=True,
        sampling="weighted_by_dataset",
        time_chunk_size=None,
        basin_chunk_size=None,
        max_batches_per_epoch=4,
        save_epoch=0,
        config_out=None,
        dry_run=True,
        worker_cache_size_mb=1,
        dataloader_timeout=0.0,
        dataloader_in_order=False,
    )

    train_cfg = build_cfg(args)
    assert train_cfg["data_cfgs"]["sampler_hyperparam"]["drop_last"] is False
    dataset = CloudZarrLazyDataset(train_cfg, "train")
    sampler = CloudZarrChunkBatchSampler(
        dataset,
        batch_size=2,
        sampling="weighted_by_dataset",
        max_batches_per_epoch=4,
        shuffle=False,
    )
    dataset._batch_level_dataloader_enabled = True
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=0,
        collate_fn=_prebatched_collate_fn,
    )
    model = SimpleLSTM(
        input_size=dataset.x_feature_count,
        output_size=dataset.y_feature_count,
        hidden_size=8,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss, n_iter = torch_single_train(
        model,
        optimizer,
        RMSELoss(),
        loader,
        device=torch.device("cpu"),
        which_first_tensor="sequence",
    )

    assert [source.dataset_id for source in dataset.sources] == [
        "camels_us",
        "camels_gb",
    ]
    assert dataset.x_feature_count == 4
    assert n_iter == 4
    assert np.isfinite(loss)


def test_manifest_defaults_and_full_time_range(tmp_path):
    from experiments.train_cloud_zarr_lazy_joint import dataset_entries

    manifest = {
        "dataset_defaults": {
            "basins": "all",
            "time_range": "all",
            "relevant_cols": ["precipitation"],
            "target_cols": ["streamflow"],
        },
        "datasets": [
            {"dataset_id": "camels_us"},
            {"dataset_id": "camels_gb", "weight": 2.0},
        ],
    }
    args = SimpleNamespace(
        datasets="all",
        time_start="2000-01-01",
        time_end="2000-01-10",
        relevant_cols="unused",
        target_cols="unused",
        constant_cols="",
        timeseries_uri=None,
        attributes_uri=None,
        max_basins=None,
    )

    entries = dataset_entries(args, manifest)

    assert [entry["time_range"] for entry in entries] == ["all", "all"]
    assert [entry["basins"] for entry in entries] == ["all", "all"]
    assert entries[0]["relevant_cols"] == ["precipitation"]
    assert entries[1]["weight"] == 2.0


def test_dataset_full_time_range_uses_entire_store(tmp_path):
    ts_uri, attr_uri = _write_tiny_zarr(tmp_path, "camels_us", ["01013500", "01022500"])
    cfg = _base_cfg(tmp_path, ts_uri, attr_uri)
    cfg["data_cfgs"]["lazy_zarr"]["normalization"] = "none"
    cfg["data_cfgs"]["lazy_zarr"]["datasets"] = [
        {
            "dataset_id": "camels_us",
            "basins": ["01013500", "01022500"],
            "time_range": "all",
            "relevant_cols": {"precipitation": "pcp_mm"},
            "target_cols": {"streamflow": "q_cms_obs"},
            "constant_cols": {"area": "area_km2"},
            "timeseries_uri": str(ts_uri),
            "attributes_uri": str(attr_uri),
        }
    ]

    dataset = CloudZarrLazyDataset(cfg, "train")

    assert dataset.source_states[0].time_length == 10
    assert dataset.source_states[0].times[0] == pd.Timestamp("2000-01-01")
    assert dataset.source_states[0].times[-1] == pd.Timestamp("2000-01-10")


def test_training_performance_monitor_writes_epoch_json(tmp_path):
    monitor = TrainingPerformanceMonitor(tmp_path, torch.device("cpu"))
    monitor.start_epoch(3)
    monitor.record_data_wait(0.2)
    monitor.record_loader_fetch(0.15)
    monitor.record_batch((torch.zeros(4, 3, 1), torch.zeros(4, 1, 1)), 0.1)

    result = monitor.finish_epoch()

    assert result["epoch"] == 3
    assert result["samples"] == 4
    assert result["batch_size_avg"] == 4
    assert result["batch_size_min"] == 4
    assert result["batch_size_max"] == 4
    assert result["avg_dataloader_wait_seconds"] == pytest.approx(0.2)
    assert result["loader_fetch_seconds"] == pytest.approx(0.15)
    output = json.loads((tmp_path / "performance_epoch_0003.json").read_text())
    assert output["samples"] == 4


def test_weighted_sample_sampler_coalesces_chunk_tails(tmp_path):
    dataset = CloudZarrLazyDataset(
        _multi_source_cfg(tmp_path, normalization="none"), "train"
    )
    sampler = CloudZarrChunkBatchSampler(
        dataset,
        batch_size=7,
        drop_last=False,
        shuffle=False,
        sampling="weighted_by_sample",
    )

    batches = list(sampler)
    flattened = [index for batch in batches for index in batch]

    assert all(len(batch) == 7 for batch in batches[:-1])
    assert 1 <= len(batches[-1]) <= 7
    assert len(flattened) == len(dataset)
    assert len(set(flattened)) == len(dataset)
    assert len(batches) == len(sampler)


def test_chunk_sampler_interleaves_worker_affinity_lanes(tmp_path):
    dataset = CloudZarrLazyDataset(
        _multi_source_cfg(tmp_path, normalization="none"), "train"
    )
    sampler = CloudZarrChunkBatchSampler(
        dataset,
        batch_size=2,
        drop_last=False,
        shuffle=False,
        sampling="weighted_by_sample",
        worker_affinity_count=2,
    )

    batches = list(sampler)
    flattened = [index for batch in batches for index in batch]
    lanes_by_chunk: dict[tuple[int, int, int], set[int]] = {}
    for batch_pos, batch in enumerate(batches[:-2]):
        for index in batch:
            lanes_by_chunk.setdefault(sampler.chunk_key(index), set()).add(
                batch_pos % 2
            )

    assert len(flattened) == len(dataset)
    assert len(set(flattened)) == len(dataset)
    assert all(len(lanes) == 1 for lanes in lanes_by_chunk.values())
