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
    basins = np.array(basins)
    times = pd.date_range("2000-01-01", periods=10, freq="D")
    pcp = np.arange(len(basins) * 10, dtype=np.float32).reshape(len(basins), 10) + offset
    flow = (
        np.arange(100, 100 + len(basins) * 10, dtype=np.float32)
        .reshape(len(basins), 10)
        + offset
    )
    data_vars = {
        "pcp_mm": (("basin", "time"), pcp, {"units": "mm/day"}),
        "q_cms_obs": (("basin", "time"), flow, {"units": "m^3/s"}),
    }
    if include_temperature:
        data_vars["temp_mean"] = (
            ("basin", "time"),
            np.ones((len(basins), 10), dtype=np.float32) * (10.0 + offset),
            {"units": "degC"},
        )
    ts = xr.Dataset(
        data_vars,
        coords={"basin": basins, "time": times},
    )
    attrs = xr.Dataset(
        {"area_km2": ("basin", np.arange(len(basins), dtype=np.float32) + 10.0)},
        coords={"basin": basins},
    )
    ts_uri = tmp_path / f"{name}_timeseries.zarr"
    attr_uri = tmp_path / f"{name}_attributes.zarr"
    ts.to_zarr(ts_uri, consolidated=False)
    attrs.to_zarr(attr_uri, consolidated=False)
    return ts_uri, attr_uri


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


def test_cloud_zarr_lazy_dataset_multi_source_global_schema(tmp_path):
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
        "normalization": "standard",
        "stat_cache_dir": str(tmp_path / "stat_cache"),
        "input_schema": ["precipitation", "temperature_mean"],
        "target_schema": ["streamflow"],
        "constant_schema": ["area"],
        "missing_policy": "fill_and_mask",
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
            },
        ],
    }
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
