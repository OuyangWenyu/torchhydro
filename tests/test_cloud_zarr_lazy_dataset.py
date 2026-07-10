import json
import multiprocessing as mp
import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from torchhydro.datasets.cloud_zarr_lazy_dataset import CloudZarrLazyDataset
from torchhydro.datasets.cloud_zarr_sampler import CloudZarrChunkBatchSampler
from torchhydro.models.crits import RMSELoss
from torchhydro.models.simple_lstm import SimpleLSTM
from torchhydro.trainers.train_utils import torch_single_train
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

    loss, n_iter = torch_single_train(
        model,
        optimizer,
        RMSELoss(),
        loader,
        device=torch.device("cpu"),
        which_first_tensor="sequence",
    )

    assert n_iter == 2
    assert np.isfinite(loss)


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
