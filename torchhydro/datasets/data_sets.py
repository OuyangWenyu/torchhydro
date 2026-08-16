"""
Author: Wenyu Ouyang
Date: 2024-04-08 18:16:53
LastEditTime: 2025-11-22 10:46:49
LastEditors: Wenyu Ouyang
Description: A pytorch dataset class; references to https://github.com/neuralhydrology/neuralhydrology
FilePath: \torchhydro\torchhydro\datasets\data_sets.py
Copyright (c) 2024-2024 Wenyu Ouyang. All rights reserved.
"""

import logging
import os
import re
import sys
import time
import torch
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
from torch.utils.data import Dataset
from tqdm import tqdm
from hydroutils.hydro_units import streamflow_unit_conv
# MultiDatasets loader — may fail if hydrodataset version is too new (0.3+)
# Fallback gracefully for CAMELS-US single-source use cases
try:
    from hydrodataset.multi_datasets import MultiDatasets
except ImportError:
    MultiDatasets = None

# DATASET_MAPPING removed from hydrodataset; define locally for MultiSourceDataset
DATASET_MAPPING = {}

from torchhydro.configs.config import DATE_FORMATS
from torchhydro.datasets.data_scalers import ScalerHub
from torchhydro.datasets.data_sources import data_sources_dict
from torch_geometric.data import Data
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from torchhydro.datasets.data_utils import (
    set_unit_to_var,
    warn_if_nan,
    wrap_t_s_dict,
)

LOGGER = logging.getLogger(__name__)


class LookupTableView:
    """Tuple-compatible lookup table backed by compact NumPy arrays.

    This preserves the existing ``lookup_table[item] -> (basin, start, length)``
    interface for Dataset subclasses while allowing the standard BaseDataset
    path to use the underlying arrays directly.
    """

    def __init__(self, basins, starts, lengths):
        self.basins = basins
        self.starts = starts
        self.lengths = lengths

    def __len__(self):
        return len(self.basins)

    def __getitem__(self, item: int):
        return (
            int(self.basins[item]),
            int(self.starts[item]),
            int(self.lengths[item]),
        )


def _is_dask_backed_xarray(obj) -> bool:
    """Return True when an Xarray object contains Dask-backed arrays."""
    if obj is None:
        return False
    if hasattr(getattr(obj, "data", None), "chunks"):
        return True
    if hasattr(obj, "data_vars"):
        return any(
            hasattr(getattr(var, "data", None), "chunks")
            for var in obj.data_vars.values()
        )
    return False


def static_attr_batch_collate_fn(batch):
    """Assemble static attributes once per batch instead of once per sample."""
    if not batch:
        return batch
    first = batch[0]
    if (
        len(first) == 2
        and isinstance(first[0], (tuple, list))
        and len(first[0]) == 2
    ):
        xcs, ys = zip(*batch)
        xs, cs = zip(*xcs)
        return (
            torch.stack(xs, dim=0),
            torch.stack(cs, dim=0),
        ), torch.stack(ys, dim=0)
    if len(first) == 3:
        xs, ys, cs = zip(*batch)
        x_batch = torch.stack(xs, dim=0)
        y_batch = torch.stack(ys, dim=0)
        c_batch = torch.stack(cs, dim=0)
        c_expanded = c_batch.unsqueeze(1).expand(-1, x_batch.shape[1], -1)
        return torch.cat((x_batch, c_expanded), dim=-1), y_batch
    xs, ys = zip(*batch)
    return torch.stack(xs, dim=0), torch.stack(ys, dim=0)


def prebatched_collate_fn(batch):
    """Return a batch that has already been assembled by ``Dataset.__getitems__``.

    PyTorch's map-style DataLoader calls ``dataset.__getitems__(indices)`` when
    available for auto-collated batches. TorchHydro uses this hook for the
    optional batch-level loading path, where the dataset returns an already
    stacked ``(xs, ys)`` tuple. The collate function must therefore pass that
    tuple through unchanged instead of applying the default per-sample stack.
    """
    return batch


def _fill_gaps_da(da: xr.DataArray, fill_nan: Optional[str] = None) -> xr.DataArray:
    """Fill gaps in a DataArray"""
    if fill_nan is None or da is None:
        return da
    assert isinstance(da, xr.DataArray), "Expect da to be DataArray (not dataset)"
    # fill gaps
    if fill_nan == "et_ssm_ignore":
        all_non_nan_idx = []
        for i in range(da.shape[0]):
            non_nan_idx_tmp = np.where(~np.isnan(da[i].values))
            all_non_nan_idx = all_non_nan_idx + non_nan_idx_tmp[0].tolist()
        # some NaN data appear in different dates in different basins
        non_nan_idx = np.unique(all_non_nan_idx).tolist()
        for i in range(da.shape[0]):
            targ_i = da[i][non_nan_idx]
            da[i][non_nan_idx] = targ_i.interpolate_na(
                dim="time", fill_value="extrapolate"
            )
    elif fill_nan == "mean":
        # fill with mean
        for var in da["variable"].values:
            var_data = da.sel(variable=var)  # select the data for the current variable
            mean_val = var_data.mean(
                dim="basin"
            )  # calculate the mean across all basins
            if warn_if_nan(mean_val, nan_mode="all"):
                # when all value are NaN, mean_val will be NaN, we set mean_val to -1
                mean_val = -1
            filled_data = var_data.fillna(
                mean_val
            )  # fill NaN values with the calculated mean
            da.loc[dict(variable=var)] = (
                filled_data  # update the original dataarray with the filled data
            )
    elif fill_nan == "interpolate":
        # Check if this is a station-based DataArray (has 'station' dimension)
        if "station" in da.dims:
            # Create a copy of the DataArray to modify
            result_da = da.copy(deep=True)

            # Handle station data: interpolate along time for each basin and each station
            for i in range(da.shape[0]):  # For each basin
                # Get the number of stations for this basin
                n_stations = da[i].shape[1] if da[i].ndim > 1 else 1

                # For each station, interpolate along time dimension
                for s in range(n_stations):
                    # Get data for this station
                    station_data = da[i, :, s, :]

                    # Check if the entire station has all NaN values
                    if np.isnan(station_data.values).all():
                        # If all values are NaN, fill with 0 or a small value
                        # This prevents the training from crashing
                        result_da[i, :, s, :] = 0.0
                        # print(f"Warning: Station {s} in basin {i} has all NaN values, filled with 0.0")
                    else:
                        # Try to interpolate along time for this station
                        try:
                            filled_station_data = station_data.interpolate_na(
                                dim="time", fill_value="extrapolate"
                            )
                            result_da[i, :, s, :] = filled_station_data
                        except Exception as e:
                            # If interpolation fails, use forward fill + backward fill
                            print(
                                f"Warning: Interpolation failed for station {s} in basin {i}: {e}"
                            )
                            filled_data = station_data.fillna(method="ffill").fillna(
                                method="bfill"
                            )
                            if np.isnan(filled_data.values).any():
                                # If still has NaN after forward/backward fill, use 0
                                filled_data = filled_data.fillna(0.0)
                            result_da[i, :, s, :] = filled_data

            # Return the modified copy
            return result_da
        else:
            # Xarray vectorizes interpolation over every non-core dimension.
            # Processing all basins and variables in one call preserves the
            # per-series result while avoiding one Python/Xarray call per basin.
            return da.interpolate_na(dim="time", fill_value="extrapolate")
    elif fill_nan == "lead_step":
        # for forecast data, we use interpolation to fill NaN values for lead step small than the maximum lead step of each forecast performing
        for i in tqdm(range(da.shape[0]), desc="Processing basins", unit="basin"):
            # first dim must be basin
            # 对于每个时间点(time)，找到最后一个非空的lead_step
            basin_data = da[i]
            for t_idx in range(basin_data.shape[0]):  # 遍历每个时间点
                time_slice = basin_data[t_idx]  # 获取当前时间点的数据
                non_nan_mask = ~np.isnan(time_slice.values.squeeze())
                non_nan_lead_steps = time_slice.lead_step[non_nan_mask]
                if len(non_nan_lead_steps) == 0:
                    # 如果全部都是空值，则跳过不处理
                    continue
                max_lead_step = non_nan_lead_steps.max().values
                # 修改这里：手动实现有限制的插值
                values = time_slice.values.squeeze()
                lead_steps = time_slice.lead_step.values
                valid_mask = (lead_steps <= max_lead_step) & np.isnan(values)
                if np.any(valid_mask):
                    # 只对有效范围内的NaN值进行插值
                    valid_values = values[
                        ~np.isnan(values) & (lead_steps <= max_lead_step)
                    ]
                    valid_lead_steps = lead_steps[
                        ~np.isnan(values) & (lead_steps <= max_lead_step)
                    ]
                    if len(valid_values) > 1:
                        # 使用线性插值
                        interp_values = np.interp(
                            lead_steps[valid_mask], valid_lead_steps, valid_values
                        )
                        values[valid_mask] = interp_values
                        basin_data[t_idx].values = values.reshape(-1, 1)
            da[i] = basin_data
    else:
        raise NotImplementedError(f"fill_nan {fill_nan} not implemented")
    return da


def detect_date_format(date_str):
    """
    检测日期格式，支持单个字符串或字符串列表

    Parameters
    ----------
    date_str : str or list
        日期字符串或日期字符串列表

    Returns
    -------
    str
        检测到的日期格式
    """
    # 如果输入是列表，使用第一个元素
    if isinstance(date_str, (list, tuple)):
        if not date_str:  # 如果列表为空
            raise ValueError("Empty date list")
        date_str = date_str[0]  # 使用第一个日期字符串

    # 确保输入是字符串
    if not isinstance(date_str, str):
        raise ValueError(
            f"Date must be string or list of strings, got {type(date_str)}"
        )

    # 尝试不同的日期格式
    for date_format in DATE_FORMATS:
        try:
            datetime.strptime(date_str, date_format)
            return date_format
        except ValueError:
            continue

    raise ValueError(f"Unknown date format: {date_str}")


class BaseDataset(Dataset):
    """Base data set class to load and preprocess data (batch-first) using PyTorch's Dataset"""

    def __init__(self, cfgs: dict, is_tra_val_te: str):
        """
        Parameters
        ----------
        cfgs
            configs, including data and training + evaluation settings
            which will be used for organizing batch data
        is_tra_val_te
            train, vaild or test
        """
        super(BaseDataset, self).__init__()
        self.data_cfgs = cfgs["data_cfgs"]
        self.training_cfgs = cfgs["training_cfgs"]
        self.evaluation_cfgs = cfgs["evaluation_cfgs"]
        self._pre_load_data(is_tra_val_te)
        # load and preprocess data
        self._load_data()

    def _pre_load_data(self, is_tra_val_te):
        """
        some preprocessing before loading data, such as
        setting the way to organize batch data

        Parameters
        ----------
        is_tra_val_te: bool
            train, valid or test

        Raises
        ------
        ValueError
            _description_
        """
        if is_tra_val_te in {"train", "valid", "test"}:
            self.is_tra_val_te = is_tra_val_te
        else:
            raise ValueError(
                "'is_tra_val_te' must be one of 'train', 'valid' or 'test' "
            )
        self.train_mode = self.is_tra_val_te == "train"
        self.t_s_dict = wrap_t_s_dict(self.data_cfgs, self.is_tra_val_te)
        self.rho = self.training_cfgs["hindcast_length"]
        self.warmup_length = self.training_cfgs["warmup_length"]
        self.horizon = self.training_cfgs["forecast_length"]
        valid_batch_mode = self.training_cfgs["valid_batch_mode"]
        # train + valid with valid_mode is train means we will use the same batch data for train and valid
        self.is_new_batch_way = (
            is_tra_val_te != "valid" or valid_batch_mode != "train"
        ) and is_tra_val_te != "train"
        rolling = self.evaluation_cfgs.get("rolling", 0)
        if self.evaluation_cfgs["hrwin"] is None:
            hrwin = self.rho
        else:
            hrwin = self.evaluation_cfgs["hrwin"]
        if self.evaluation_cfgs["frwin"] is None:
            frwin = self.horizon
        else:
            frwin = self.evaluation_cfgs["frwin"]
        if rolling == 0:
            hrwin = 0 if hrwin is None else hrwin
            frwin = self.nt - hrwin - self.warmup_length
        if self.is_new_batch_way:
            # we will set the batch data for valid and test
            self.rolling = rolling
            self.rho = hrwin
            self.horizon = frwin

    @property
    def data_source(self):
        from torchhydro.configs.data_resolver import open_dataset_from_source_cfgs

        return open_dataset_from_source_cfgs(self.data_cfgs.get("source_cfgs") or {})

    @property
    def streamflow_name(self):
        return self.data_cfgs["target_cols"][0]

    @property
    def precipitation_name(self):
        return self.data_cfgs["relevant_cols"][0]

    @property
    def ngrid(self):
        """How many basins/grids in the dataset

        Returns
        -------
        int
            number of basins/grids
        """
        return len(self.basins)

    @property
    def noutputvar(self):
        """How many output variables in the dataset
        Used in evaluation.

        Returns
        -------
        int
            number of variables
        """
        return len(self.data_cfgs["target_cols"])

    @property
    def nt(self):
        """length of longest time series in all basins

        Returns
        -------
        int
            number of longest time steps
        """
        if isinstance(self.t_s_dict["t_final_range"][0], tuple):
            trange_type_num = len(self.t_s_dict["t_final_range"])
            if trange_type_num not in [self.ngrid, 1]:
                raise ValueError(
                    "The number of time ranges should be equal to the number of basins "
                    "if you choose different time ranges for different basins"
                )
            earliest_date = None
            latest_date = None
            for start_date_str, end_date_str in self.t_s_dict["t_final_range"]:
                date_format = detect_date_format(start_date_str)

                start_date = datetime.strptime(start_date_str, date_format)
                end_date = datetime.strptime(end_date_str, date_format)

                if earliest_date is None or start_date < earliest_date:
                    earliest_date = start_date
                if latest_date is None or end_date > latest_date:
                    latest_date = end_date
            earliest_date = earliest_date.strftime(date_format)
            latest_date = latest_date.strftime(date_format)
        else:
            trange_type_num = 1
            earliest_date = self.t_s_dict["t_final_range"][0]
            latest_date = self.t_s_dict["t_final_range"][1]
        min_time_unit = self.data_cfgs["min_time_unit"]
        min_time_interval = self.data_cfgs["min_time_interval"]

        # 计算时间步长（以小时为单位）
        unit_to_hours = {
            "h": 1,
            "H": 1,
            "d": 24,
            "D": 24,
            "m": 1 / 60,
            "M": 1 / 60,
            "s": 1 / 3600,
            "S": 1 / 3600,
        }
        hours_per_step = min_time_interval * unit_to_hours.get(min_time_unit, 1)

        # 解析时间字符串
        date_format = detect_date_format(
            earliest_date[0]
            if isinstance(earliest_date, (list, tuple))
            else earliest_date
        )

        # 获取开始和结束时间
        if isinstance(earliest_date, (list, tuple)):
            s_date = datetime.strptime(
                earliest_date[0], date_format
            )  # 使用第一个元素作为开始时间
        else:
            s_date = datetime.strptime(earliest_date, date_format)

        if isinstance(latest_date, (list, tuple)):
            e_date = datetime.strptime(
                latest_date[-1], date_format
            )  # 使用最后一个元素作为结束时间
        else:
            e_date = datetime.strptime(latest_date, date_format)

        # 计算总小时数
        total_hours = (e_date - s_date).total_seconds() / 3600

        # 计算时间步数
        return int(total_hours / hours_per_step) + 1

    @property
    def basins(self):
        """Return the basins of the dataset"""
        return self.t_s_dict["sites_id"]

    @property
    def times(self):
        """Return the times of all basins

        TODO: Although we support get different time ranges for different basins,
        we didn't implement the reading function for this case in _read_xyc method.
        Hence, it's better to choose unified time range for all basins
        """
        min_time_unit = self.data_cfgs["min_time_unit"]
        min_time_interval = self.data_cfgs["min_time_interval"]
        time_step = f"{min_time_interval}{min_time_unit}"
        if isinstance(self.t_s_dict["t_final_range"][0], tuple):
            times_ = []
            trange_type_num = len(self.t_s_dict["t_final_range"])
            if trange_type_num not in [self.ngrid, 1]:
                raise ValueError(
                    "The number of time ranges should be equal to the number of basins "
                    "if you choose different time ranges for different basins"
                )
            detect_date_format(self.t_s_dict["t_final_range"][0][0])
            for start_date_str, end_date_str in self.t_s_dict["t_final_range"]:
                s_date = pd.to_datetime(start_date_str)
                e_date = pd.to_datetime(end_date_str)
                time_series = pd.date_range(start=s_date, end=e_date, freq=time_step)
                times_.append(time_series)
        else:
            detect_date_format(self.t_s_dict["t_final_range"][0])
            trange_type_num = 1
            s_date = pd.to_datetime(self.t_s_dict["t_final_range"][0])
            e_date = pd.to_datetime(self.t_s_dict["t_final_range"][1])
            times_ = pd.date_range(start=s_date, end=e_date, freq=time_step)
        return times_

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item: int):
        """Get one sample from the dataset with a unified return format.

        Args:
            item: The index of the sample to retrieve.

        Returns:
            A tuple of (input_data, output_data), where input_data is a tensor
            of input features and output_data is a tensor of target values.
        """
        if hasattr(self, "_lookup_basins"):
            basin = self._lookup_basins[item]
            idx = self._lookup_starts[item]
            actual_length = self._lookup_lengths[item]
        else:
            basin, idx, actual_length = self.lookup_table[item]
        warmup_length = self.warmup_length
        x = self.x[basin, idx - warmup_length : idx + actual_length - warmup_length, :]
        y = self.y[basin, idx : idx + actual_length - warmup_length, :]
        if self.c is None or self.c.shape[-1] == 0:
            return torch.from_numpy(x), torch.from_numpy(y)
        c = self.c[basin, :]
        if getattr(self, "_separate_static_input", False):
            return (torch.from_numpy(x), torch.from_numpy(c)), torch.from_numpy(y)
        if getattr(self, "_return_static_attrs_separately", False):
            return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(c)
        c = np.repeat(c, x.shape[0], axis=0).reshape(c.shape[0], -1).T
        xc = np.concatenate((x, c), axis=1)
        return torch.from_numpy(xc), torch.from_numpy(y)

    def __getitems__(self, items):
        """Get a list of samples or one pre-assembled batch.

        PyTorch DataLoader calls this method with a list of indices when it is
        present on a map-style dataset. The default-compatible path returns the
        same list of samples that repeated ``__getitem__`` calls would produce.

        When ``training_cfgs["batch_level_dataloader"]`` is true, TorchHydro
        assembles the full batch in one dataset call and returns a batched
        ``(xs, ys)`` tuple. This avoids millions of per-sample
        ``__getitem__`` calls in large sliding-window training jobs while
        preserving the model-facing tensor shapes.
        """
        if self._use_batch_level_getitems():
            return self._get_batch_level_items(items)
        return [self.__getitem__(int(item)) for item in items]

    def _use_batch_level_getitems(self) -> bool:
        """Whether this dataset instance should use batch-level assembly."""
        return bool(getattr(self, "_batch_level_dataloader_enabled", False)) and (
            self.__class__ is BaseDataset
        )

    def _get_batch_level_items(self, items):
        """Assemble one DataLoader batch from sample indices.

        This method intentionally supports the standard fixed-length
        ``BaseDataset`` path only. Variable-length, graph, forecast, and flood
        event datasets override ``__getitem__`` with different semantics and
        should continue using their existing collate paths.
        """
        item_indices = np.asarray(items, dtype=np.int64)
        if item_indices.size == 0:
            return torch.empty(0), torch.empty(0)

        if hasattr(self, "_lookup_basins"):
            basins = self._lookup_basins[item_indices].astype(np.intp, copy=False)
            start_indices = self._lookup_starts[item_indices].astype(
                np.intp, copy=False
            )
            actual_lengths = self._lookup_lengths[item_indices].astype(
                np.intp, copy=False
            )
        else:
            lookup_rows = [self.lookup_table[int(item)] for item in item_indices]
            basins = np.asarray([row[0] for row in lookup_rows], dtype=np.int64)
            start_indices = np.asarray([row[1] for row in lookup_rows], dtype=np.int64)
            actual_lengths = np.asarray([row[2] for row in lookup_rows], dtype=np.int64)

        actual_length = int(actual_lengths[0])
        if not np.all(actual_lengths == actual_length):
            raise ValueError(
                "batch_level_dataloader only supports fixed-length batches. "
                "Disable it for variable-length training."
            )

        warmup_length = self.warmup_length
        x_length = actual_length
        y_length = actual_length - warmup_length
        if y_length <= 0:
            raise ValueError(
                "Invalid batch window: actual_length must be larger than "
                f"warmup_length, got {actual_length=} and {warmup_length=}."
            )

        batch_size = len(item_indices)
        x_feature_count = self.x.shape[-1]
        y_feature_count = self.y.shape[-1]
        c_feature_count = (
            0 if self.c is None or self.c.shape[-1] == 0 else self.c.shape[-1]
        )

        x_window_view, y_window_view = self._get_batch_level_window_views(
            x_length, y_length
        )

        # Gather complete sliding windows by (basin, start) in one NumPy
        # operation per tensor. ``sliding_window_view`` itself is a strided view,
        # so this avoids the per-sample Python slicing loop without first
        # materializing a large basin x time-index matrix.
        x_dynamic = x_window_view[basins, start_indices - warmup_length]
        ys = y_window_view[basins, start_indices]

        if getattr(self, "_separate_static_input", False) and c_feature_count:
            return (
                torch.from_numpy(x_dynamic),
                torch.from_numpy(self.c[basins, :]),
            ), torch.from_numpy(ys)

        if c_feature_count:
            xs = np.empty(
                (batch_size, x_length, x_feature_count + c_feature_count),
                dtype=self.x.dtype,
            )
            xs[:, :, :x_feature_count] = x_dynamic
        else:
            xs = x_dynamic
        if c_feature_count:
            xs[:, :, x_feature_count:] = self.c[basins, :][:, None, :]

        return torch.from_numpy(xs), torch.from_numpy(ys)

    def _get_batch_level_window_views(self, x_length, y_length):
        """Return cached sliding-window views for batch-level assembly."""
        cache_key = (x_length, y_length)
        if getattr(self, "_batch_window_view_key", None) != cache_key:
            from numpy.lib.stride_tricks import sliding_window_view

            self._batch_x_window_view = np.moveaxis(
                sliding_window_view(self.x, window_shape=x_length, axis=1),
                -1,
                2,
            )
            self._batch_y_window_view = np.moveaxis(
                sliding_window_view(self.y, window_shape=y_length, axis=1),
                -1,
                2,
            )
            self._batch_window_view_key = cache_key
        return self._batch_x_window_view, self._batch_y_window_view

    @staticmethod
    def _lookup_dtype(max_value, default=np.int32):
        if max_value <= np.iinfo(np.int16).max:
            return np.int16
        return default

    @staticmethod
    def _valid_window_mask(nan_array, starts, window):
        """Return starts whose target window contains at least one observation."""
        if starts.size == 0:
            return np.zeros(0, dtype=bool)
        time_has_value = ~np.all(nan_array, axis=1)
        counts = np.concatenate(([0], np.cumsum(time_has_value, dtype=np.int32)))
        return (counts[starts + window] - counts[starts]) > 0

    def _set_lookup_arrays(self, basins, starts, lengths):
        """Store compact lookup arrays and a tuple-compatible view."""
        max_basin = int(basins.max()) if basins.size else 0
        max_length = int(lengths.max()) if lengths.size else 0
        self._lookup_basins = basins.astype(self._lookup_dtype(max_basin), copy=False)
        self._lookup_starts = starts.astype(np.int32, copy=False)
        self._lookup_lengths = lengths.astype(
            self._lookup_dtype(max_length), copy=False
        )
        self.lookup_table = LookupTableView(
            self._lookup_basins, self._lookup_starts, self._lookup_lengths
        )
        self.num_samples = len(self.lookup_table)

    def _load_data(self):
        total_start = time.perf_counter()
        origin_data = self._profile_dataset_setup_step("_read_xyc", self._read_xyc)
        origin_data = self._profile_dataset_setup_step(
            "origin cloud .load()",
            self._eager_load_cloud_xarray_data,
            origin_data,
        )
        # normalization
        norm_data = self._profile_dataset_setup_step(
            "_normalize", self._normalize, origin_data
        )
        norm_data = self._profile_dataset_setup_step(
            "norm .load()", self._eager_load_cloud_xarray_data, norm_data
        )
        # 启用 NaN 处理以确保数据清洁
        origin_data_wonan, norm_data_wonan = self._profile_dataset_setup_step(
            "_kill_nan", self._kill_nan, origin_data, norm_data
        )
        # origin_data_wonan, norm_data_wonan = origin_data, norm_data  # 备用：跳过 NaN 处理
        self._profile_dataset_setup_step(
            "_trans2nparr", self._trans2nparr, origin_data_wonan, norm_data_wonan
        )
        self._profile_dataset_setup_step("_create_lookup_table", self._create_lookup_table)
        self._profile_dataset_setup_total(total_start)

    def _should_profile_dataset_setup(self) -> bool:
        return bool(self.data_cfgs.get("profile_dataset_setup", False))

    def _profile_dataset_setup_step(self, name, func, *args, **kwargs):
        if not self._should_profile_dataset_setup():
            return func(*args, **kwargs)
        start = time.perf_counter()
        result = func(*args, **kwargs)
        seconds = time.perf_counter() - start
        print(f"[dataset setup profile] {name:24s}: {seconds:10.3f} s")
        return result

    def _profile_dataset_setup_total(self, total_start):
        if self._should_profile_dataset_setup():
            seconds = time.perf_counter() - total_start
            print(f"[dataset setup profile] {'total':24s}: {seconds:10.3f} s")

    def _eager_load_cloud_xarray_data(self, data):
        """Load cloud Dask-backed Xarray data before NaN filling.

        Cloud Zarr reads return dask-backed Xarray objects. Later NaN handling
        calls ``interpolate_na(dim="time")``, which requires the core dimension
        to be contained in one chunk. The current TorchHydro Dataset setup
        converts all Xarray data to NumPy anyway, so loading here keeps local
        behavior unchanged while avoiding Dask rechunk errors for cloud data.
        """
        if not self._should_eager_load_cloud_data() or data is None:
            return data
        loaded_data = dict(data)
        dask_items = [
            (key, value)
            for key, value in data.items()
            if value is not None and _is_dask_backed_xarray(value)
        ]
        if not dask_items:
            return loaded_data

        if self._should_profile_cloud_load_details():
            for key, value in dask_items:
                self._print_cloud_xarray_load_plan(key, value)

        if (
            self.data_cfgs.get("cloud_eager_load_concurrent", True)
            and len(dask_items) > 1
        ):
            return self._eager_load_cloud_xarray_values_concurrently(
                loaded_data, dask_items
            )

        for key, value in dask_items:
            loaded_data[key] = self._eager_load_cloud_xarray_value(value, key)
        return loaded_data

    def _eager_load_cloud_xarray_values_concurrently(self, loaded_data, dask_items):
        """Load multiple Dask-backed Xarray objects in one Dask compute call."""
        try:
            import dask
        except ImportError:
            for key, value in dask_items:
                loaded_data[key] = self._eager_load_cloud_xarray_value(value, key)
            return loaded_data

        compute_kwargs = {}
        scheduler = self.data_cfgs.get("cloud_eager_load_scheduler", "threads")
        if scheduler is not None:
            compute_kwargs["scheduler"] = scheduler
        num_workers = self.data_cfgs.get("cloud_eager_load_num_workers")
        if num_workers is not None:
            compute_kwargs["num_workers"] = num_workers

        if self._should_profile_cloud_load_details():
            keys = ", ".join(key for key, _ in dask_items)
            scheduler_name = compute_kwargs.get("scheduler", "default")
            workers_text = compute_kwargs.get("num_workers", "default")
            print(
                "[dataset setup profile] "
                f"cloud load concurrent: keys=[{keys}], "
                f"scheduler={scheduler_name}, num_workers={workers_text}"
            )

        loaded_values = dask.compute(
            *(value for _, value in dask_items), **compute_kwargs
        )
        for (key, _), loaded_value in zip(dask_items, loaded_values):
            loaded_data[key] = loaded_value
        return loaded_data

    def _eager_load_cloud_xarray_value(self, value, key):
        if value is not None and _is_dask_backed_xarray(value):
            LOGGER.info(
                "Eager loading cloud Xarray data before NaN handling: %s", key
            )
            return value.load()
        return value

    def _should_profile_cloud_load_details(self) -> bool:
        return self._should_profile_dataset_setup() or bool(
            self.data_cfgs.get("profile_cloud_load_details", False)
        )

    def _print_cloud_xarray_load_plan(self, key, value):
        dims = getattr(value, "dims", None)
        shape = getattr(value, "shape", None)
        chunks = getattr(value, "chunks", None)
        dtype = getattr(value, "dtype", None)
        nbytes = getattr(value, "nbytes", None)
        task_count = self._dask_task_count(value)
        nbytes_text = self._format_bytes(nbytes)
        print(
            "[dataset setup profile] "
            f"cloud load plan {key}: dims={dims}, shape={shape}, "
            f"chunks={chunks}, dtype={dtype}, bytes={nbytes_text}, "
            f"dask_tasks={task_count}"
        )

    @staticmethod
    def _dask_task_count(value):
        try:
            graph = value.__dask_graph__()
        except Exception:
            graph = None
        if graph is None:
            return None
        try:
            return len(graph)
        except TypeError:
            try:
                return len(graph.to_dict())
            except Exception:
                return None

    @staticmethod
    def _format_bytes(nbytes):
        if nbytes is None:
            return "unknown"
        try:
            nbytes = float(nbytes)
        except (TypeError, ValueError):
            return str(nbytes)
        units = ["B", "KiB", "MiB", "GiB", "TiB"]
        unit_idx = 0
        while nbytes >= 1024 and unit_idx < len(units) - 1:
            nbytes /= 1024
            unit_idx += 1
        return f"{nbytes:.3f} {units[unit_idx]}"

    def _should_eager_load_cloud_data(self) -> bool:
        return bool(
            self.data_cfgs.get("cloud_eager_load", True)
        ) and self._is_cloud_source()

    def _is_cloud_source(self) -> bool:
        source_cfgs = self.data_cfgs.get("source_cfgs") or {}
        other_settings = source_cfgs.get("other_settings") or {}
        source = other_settings.get("source") or source_cfgs.get("source")
        if source is None:
            try:
                from hydrodataset.configs.settings import (
                    get_default_source,
                    load_settings,
                )

                source = get_default_source(load_settings())
            except Exception:
                source = None
        return source == "cloud"

    def _trans2nparr(self, origin_data, norm_data):
        """To make __getitem__ more efficient,
        we transform x, y, c to numpy array with shape (nsample, nt, nvar)
        """
        for key in origin_data.keys():
            _origin = origin_data[key]
            _norm = norm_data[key]
            if _origin is None or _norm is None:
                norm_arr = None
                origin_arr = None
            else:
                norm_arr = self._to_float32_numpy(_norm.to_numpy())
                origin_arr = self._to_float32_numpy(_origin.to_numpy())
            if key == "relevant_cols":
                self.x_origin = origin_arr
                self.x = norm_arr
            elif key == "target_cols":
                self.y_origin = origin_arr
                self.y = norm_arr
            elif key == "constant_cols":
                self.c_origin = origin_arr
                self.c = norm_arr
            elif key == "forecast_cols":
                self.f_origin = origin_arr
                self.f = norm_arr
            elif key == "global_cols":
                self.g_origin = origin_arr
                self.g = norm_arr
            elif key == "station_cols":
                # GNN特有的站点数据
                self.station_cols_origin = origin_arr
                self.station_cols = norm_arr
            else:
                raise ValueError(
                    f"Unknown data type {key} in origin_data, "
                    "it should be one of relevant_cols, target_cols, constant_cols, forecast_cols, global_cols, station_cols"
                )

    @staticmethod
    def _to_float32_numpy(arr):
        """Convert numeric numpy arrays to float32 once during dataset setup.

        TorchHydro models consume float32 tensors. Converting here avoids doing
        float64 -> float32 copies for every sample in ``__getitem__``.
        """
        if arr is None:
            return None
        if np.issubdtype(arr.dtype, np.number) and arr.dtype != np.float32:
            return arr.astype(np.float32, copy=False)
        return arr

    def _normalize(
        self,
        origin_data,
    ):
        """_summary_

        Parameters
        ----------
        origin_data : dict
            data with key as data type

        Returns
        -------
        _type_
            _description_
        """
        scaler_hub = ScalerHub(
            origin_data,
            data_cfgs=self.data_cfgs,
            is_tra_val_te=self.is_tra_val_te,
            data_source=self.data_source,
        )
        self.target_scaler = scaler_hub.target_scaler
        return scaler_hub.norm_data

    def _selected_time_points_for_denorm(self):
        """get the time points for denormalization

        Returns
        -------
            a list of time points
        """
        # may cause bug when running dpl model, needs more check
        return self.target_scaler.data_target.coords["time"][self.warmup_length :]

    def denormalize(self, norm_data, pace_idx=None):
        """Denormalize the norm_data

        Parameters
        ----------
        norm_data : np.ndarray
            batch-first data
        pace_idx : int, optional
            which pace to show, by default None
            sometimes we may have multiple results for one time period and we flatten them
            so we need a temp time to replace real one

        Returns
        -------
        xr.Dataset
            denormlized data
        """
        target_scaler = self.target_scaler
        target_data = target_scaler.data_target
        # the units are dimensionless for pure DL models
        units = {k: "dimensionless" for k in target_data.attrs["units"].keys()}
        # mainly to get information about the time points of norm_data
        selected_time_points = self._selected_time_points_for_denorm()
        selected_data = target_data.sel(time=selected_time_points)

        # 处理三维数据 (basin, time, variable)
        if norm_data.ndim == 3:
            all_times = selected_data.coords["time"].values
            T = norm_data.shape[1]
            if all_times.shape[0] >= T:
                time_coords = all_times[-T:]
            else:
                if all_times.shape[0] == 0:
                    raise ValueError(
                        "No time coordinates available for denormalization."
                    )
                last_time = all_times[-1]
                pad = np.array(
                    [last_time] * (T - all_times.shape[0]), dtype=all_times.dtype
                )
                time_coords = np.concatenate([all_times, pad])
            norm_data_3d = norm_data
            coords = {
                "basin": selected_data.coords["basin"],
                "time": time_coords,
                "variable": selected_data.coords["variable"],
            }
            dims = ["basin", "time", "variable"]

        # 处理四维数据
        elif norm_data.ndim == 4:
            # Check if the data is organized by basins
            if self.evaluation_cfgs["evaluator"]["recover_mode"] == "bybasins":
                # Shape: (basin_num, i_e_time_length, forecast_length, nf)
                basin_num, i_e_time_length, forecast_length, nf = norm_data.shape

                # If pace_idx is specified, select the specific forecast step
                if (
                    pace_idx is not None
                    and pace_idx != np.nan
                    and pace_idx >= 0
                    and pace_idx < forecast_length
                ):
                    norm_data_3d = norm_data[:, :, pace_idx, :]
                    # 创建新的坐标
                    # 修改这里：确保basin坐标长度与数据维度匹配
                    if basin_num == 1 and len(selected_data.coords["basin"]) > 1:
                        # 当只有一个流域时，选择第一个流域的坐标
                        basin_coord = selected_data.coords["basin"].values[:1]
                    else:
                        basin_coord = selected_data.coords["basin"].values[:basin_num]

                    coords = {
                        "basin": basin_coord,
                        "time": selected_data.coords["time"][:i_e_time_length],
                        "variable": selected_data.coords["variable"],
                    }
                else:
                    # 如果没有指定pace_idx，则创建一个新的维度'horizon'
                    norm_data_3d = norm_data.reshape(
                        basin_num, i_e_time_length * forecast_length, nf
                    )
                    # 创建新的时间坐标：取连续的 time 切片再按 forecast_length 平铺，避免重复索引错误
                    base_times = list(
                        selected_data.coords["time"][:i_e_time_length].values
                    )
                    if not base_times:
                        base_times = list(
                            target_data.coords["time"][:i_e_time_length].values
                        )
                    tiled = base_times * forecast_length
                    new_times = tiled[: i_e_time_length * forecast_length]

                    # 修改这里：确保basin坐标长度与数据维度匹配
                    if basin_num == 1 and len(selected_data.coords["basin"]) > 1:
                        basin_coord = selected_data.coords["basin"].values[:1]
                    else:
                        basin_coord = selected_data.coords["basin"].values[:basin_num]

                    coords = {
                        "basin": basin_coord,
                        "time": new_times,
                        "variable": selected_data.coords["variable"],
                    }
            else:  # byforecast模式
                # 形状为 (forecast_length, basin_num, i_e_time_length, nf)
                forecast_length, basin_num, i_e_time_length, nf = norm_data.shape

                # 如果指定了pace_idx，则选择特定的预测步长
                if (
                    pace_idx is not None
                    and pace_idx != np.nan
                    and pace_idx >= 0
                    and pace_idx < forecast_length
                ):
                    norm_data_3d = norm_data[pace_idx]
                    # 修改这里：确保basin坐标长度与数据维度匹配
                    if basin_num == 1 and len(selected_data.coords["basin"]) > 1:
                        basin_coord = selected_data.coords["basin"].values[:1]
                    else:
                        basin_coord = selected_data.coords["basin"].values[:basin_num]

                    coords = {
                        "basin": basin_coord,
                        "time": selected_data.coords["time"][:i_e_time_length],
                        "variable": selected_data.coords["variable"],
                    }
                else:
                    # If pace_idx is not specified, create a new dimension 'horizon'
                    # Reshape (forecast_length, basin_num, i_e_time_length, nf) -> (basin_num, forecast_length * i_e_time_length, nf)
                    norm_data_3d = np.transpose(norm_data, (1, 0, 2, 3)).reshape(
                        basin_num, forecast_length * i_e_time_length, nf
                    )

                    # 创建新的时间坐标：按 i_e_time_length 切片后平铺，避免重复索引错误
                    base_times = list(
                        selected_data.coords["time"][:i_e_time_length].values
                    )
                    if not base_times:
                        base_times = list(
                            target_data.coords["time"][:i_e_time_length].values
                        )
                    tiled = base_times * forecast_length
                    new_times = tiled[: forecast_length * i_e_time_length]

                    # 修改这里：确保basin坐标长度与数据维度匹配
                    if basin_num == 1 and len(selected_data.coords["basin"]) > 1:
                        basin_coord = selected_data.coords["basin"].values[:1]
                    else:
                        basin_coord = selected_data.coords["basin"].values[:basin_num]

                    coords = {
                        "basin": basin_coord,
                        "time": new_times,
                        "variable": selected_data.coords["variable"],
                    }
            dims = ["basin", "time", "variable"]
        else:
            coords = selected_data.coords
            dims = selected_data.dims
            norm_data_3d = norm_data

        # create DataArray and inverse transform
        denorm_xr_ds = target_scaler.inverse_transform(
            xr.DataArray(
                norm_data_3d,
                dims=dims,
                coords=coords,
                attrs={"units": units},
            )
        )
        return set_unit_to_var(denorm_xr_ds)

    def _to_dataarray_with_unit(self, *args):
        """Convert xarray datasets to xarray data arrays and set units for each variable.

        Parameters
        ----------
        *args : xr.Dataset
            Any number of xarray dataset inputs.

        Returns
        -------
        tuple
            A tuple of converted data arrays, with the same number as the input parameters.
        """
        results = []
        for ds in args:
            if ds is not None:
                # First convert some string-type data to floating-point type
                results.append(self._trans2da_and_setunits(ds))
            else:
                results.append(None)
        return tuple(results)

    def _check_ts_xrds_unit(self, data_forcing_ds, data_output_ds):
        """Check timeseries xarray dataset unit and convert if necessary

        Parameters
        ----------
        data_forcing_ds : xr.Dataset
            the forcing data
        data_output_ds : xr.Dataset
            outputs including streamflow data
        """

        def standardize_unit(unit):
            unit = unit.lower()  # convert to lower case
            unit = re.sub(r"day", "d", unit)
            unit = re.sub(r"hour", "h", unit)
            return unit

        streamflow_unit = data_output_ds[self.streamflow_name].attrs["units"]
        prcp_unit = data_forcing_ds[self.precipitation_name].attrs["units"]

        standardized_streamflow_unit = standardize_unit(streamflow_unit)
        standardized_prcp_unit = standardize_unit(prcp_unit)
        if standardized_streamflow_unit != standardized_prcp_unit:
            streamflow_dataset = data_output_ds[[self.streamflow_name]]
            converted_streamflow_dataset = streamflow_unit_conv(
                streamflow_dataset,
                self.data_source.read_area(self.t_s_dict["sites_id"]),
                target_unit=prcp_unit,
                source_unit=streamflow_unit,
            )
            data_output_ds[self.streamflow_name] = converted_streamflow_dataset[
                self.streamflow_name
            ]
        return data_forcing_ds, data_output_ds

    def _mask_negative_streamflow(self, data_output_ds):
        """Treat negative streamflow values as missing observations.

        CAMELS-US local NetCDF caches store invalid negative streamflow sentinel
        values as NaN. Cloud Zarr data can still contain those negative values,
        which break DapengScaler's sqrt/log normalization for streamflow and can
        make all normalized targets NaN. Mask them before scaling so local and
        cloud data sources follow the same missing-data semantics.
        """
        if data_output_ds is None or self.streamflow_name not in data_output_ds:
            return data_output_ds
        streamflow = data_output_ds[self.streamflow_name]
        attrs = streamflow.attrs.copy()
        data_output_ds = data_output_ds.copy()
        data_output_ds[self.streamflow_name] = streamflow.where(streamflow >= 0)
        data_output_ds[self.streamflow_name].attrs.update(attrs)
        return data_output_ds

    def _read_xyc(self):
        """Read x, y, c data from data source

        Returns
        -------
        dict
            data with key as data type
            the dim must be (basin, time, lead_step, variable) for 4-d xr array;
            the dim must be (basin, time, variable) for 3-d xr array;
            the dim must be (basin, variable) for 2-d xr array;
        """
        # Check if we have multiple time periods (for multi-period training)
        t_range = self.t_s_dict["t_final_range"]

        # Check if first element is a list/tuple (indicating multiple periods)
        if isinstance(t_range[0], (list, tuple)):
            # Validate multi-period format
            self._validate_multi_period_format(t_range)

            # Multiple periods case - can be any number of periods
            all_data = None
            for start_date, end_date in t_range:
                period_data = self._read_xyc_period(start_date, end_date)
                if all_data is None:
                    all_data = period_data
                else:
                    # Concatenate along time dimension
                    for key in period_data:
                        # 确保两个数据集的时间维度都是字符串类型
                        if all_data[key] is not None and period_data[key] is not None:
                            if not isinstance(all_data[key].time.values[0], str):
                                all_data[key]["time"] = all_data[key].time.astype(str)
                            if not isinstance(period_data[key].time.values[0], str):
                                period_data[key]["time"] = period_data[key].time.astype(
                                    str
                                )

                            all_data[key] = xr.concat(
                                [all_data[key], period_data[key]], dim="time"
                            )
            return all_data
        else:
            # Single period case (existing behavior)
            start_date = t_range[0]
            end_date = t_range[1]
            return self._read_xyc_period(start_date, end_date)

    def _read_xyc_period(self, start_date, end_date):
        """Template method for reading x, y, c data for a specific time period

        This method can be overridden by subclasses to customize how data is read
        for each time period while keeping the multi-period handling logic in the parent class.

        Parameters
        ----------
        start_date : str
            start time
        end_date : str
            end time

        Returns
        -------
        dict
            Dictionary containing relevant_cols, target_cols, and constant_cols data
        """
        # Default implementation: delegate to the original method
        return self._read_xyc_specified_time(start_date, end_date)

    def _validate_multi_period_format(self, t_range):
        """Validate format of multi-period time ranges

        Parameters
        ----------
        t_range : list
            List of time periods, where each period should be [start_date, end_date]

        Raises
        ------
        ValueError
            If any period doesn't have exactly 2 elements (start_date, end_date)
        """
        for i, period in enumerate(t_range):
            if not isinstance(period, (list, tuple)) or len(period) != 2:
                raise ValueError(
                    f"Period {i} must be a list/tuple with exactly 2 elements (start_date, end_date), got: {period}"
                )

    def _rm_timeunit_key(self, ds_):
        """this means the data source return a dict with key as time_unit
            in this BaseDataset, we only support unified time range for all basins, so we chose the first key
            TODO: maybe this could be refactored better

        Parameters
        ----------
        ds_ : dict
            the xarray data with time_unit as key

        Returns
        ----------
        ds_ : xr.Dataset
            the output data without time_unit
        """
        if isinstance(ds_, dict):
            ds_ = ds_[list(ds_.keys())[0]]
        return ds_

    def _read_xyc_specified_time(self, start_date, end_date):
        """Read x, y, c data from data source with specified time range
        We set this function as sometimes we need adjust the time range for some specific dataset,
        such as seq2seq dataset (it needs one more period for the end of the time range)

        Parameters
        ----------
        start_date : str
            start time
        end_date : str
            end time
        """
        data_forcing_ds_, data_output_ds_ = self._profile_dataset_setup_step(
            "_read_xyc.read_ts",
            self._read_relevant_and_target_xrdatasets,
            start_date,
            end_date,
        )
        data_forcing_ds_, data_output_ds_ = self._profile_dataset_setup_step(
            "_read_xyc.rm_timeunit",
            self._rm_timeunit_keys,
            data_forcing_ds_,
            data_output_ds_,
        )
        data_forcing_ds, data_output_ds = self._profile_dataset_setup_step(
            "_read_xyc.unit_check",
            self._check_ts_xrds_unit,
            data_forcing_ds_,
            data_output_ds_,
        )
        data_output_ds = self._profile_dataset_setup_step(
            "_read_xyc.mask_negative_streamflow",
            self._mask_negative_streamflow,
            data_output_ds,
        )
        # c
        data_attr_ds = self._profile_dataset_setup_step(
            "_read_xyc.read_attr",
            self.data_source.read_attr_xrdataset,
            self.t_s_dict["sites_id"],
            self.data_cfgs["constant_cols"],
            all_number=True,
        )
        x_origin, y_origin, c_origin = self._profile_dataset_setup_step(
            "_read_xyc.to_dataarray",
            self._to_dataarray_with_unit,
            data_forcing_ds,
            data_output_ds,
            data_attr_ds,
        )
        return {
            "relevant_cols": x_origin.transpose("basin", "time", "variable"),
            "target_cols": y_origin.transpose("basin", "time", "variable"),
            "constant_cols": (
                c_origin.transpose("basin", "variable")
                if c_origin is not None
                else None
            ),
        }

    def _read_relevant_and_target_xrdatasets(self, start_date, end_date):
        relevant_cols = self.data_cfgs["relevant_cols"]
        target_cols = self.data_cfgs["target_cols"]
        if self.data_cfgs.get("combine_relevant_target_read", True):
            try:
                all_cols = list(dict.fromkeys([*relevant_cols, *target_cols]))
                data_ts_ds = self.data_source.read_ts_xrdataset(
                    self.t_s_dict["sites_id"],
                    [start_date, end_date],
                    all_cols,
                )
                data_ts_ds = self._rm_timeunit_key(data_ts_ds)
                return data_ts_ds[relevant_cols], data_ts_ds[target_cols]
            except Exception as exc:
                if self._should_profile_dataset_setup():
                    print(
                        "[dataset setup profile] "
                        "combined relevant/target read failed; fallback to "
                        f"separate reads: {type(exc).__name__}: {exc}"
                    )
        data_forcing_ds_ = self.data_source.read_ts_xrdataset(
            self.t_s_dict["sites_id"],
            [start_date, end_date],
            relevant_cols,
        )
        data_output_ds_ = self.data_source.read_ts_xrdataset(
            self.t_s_dict["sites_id"],
            [start_date, end_date],
            target_cols,
        )
        return data_forcing_ds_, data_output_ds_

    def _rm_timeunit_keys(self, data_forcing_ds_, data_output_ds_):
        return self._rm_timeunit_key(data_forcing_ds_), self._rm_timeunit_key(
            data_output_ds_
        )

    def _trans2da_and_setunits(self, ds):
        """Set units for dataarray transfromed from dataset"""
        result = ds.to_array(dim="variable")
        units_dict = {
            var: ds[var].attrs["units"]
            for var in ds.variables
            if "units" in ds[var].attrs
        }
        result.attrs["units"] = units_dict
        return result

    def _kill_nan(self, origin_data, norm_data):
        """This function is used to remove NaN values in the original data and its normalized data.

        Parameters
        ----------
        origin_data : dict
            the original data
        norm_data : dict
            the normalized data

        Returns
        -------
        dict, dict
            the original data and normalized data after removing NaN values
        """
        data_cfgs = self.data_cfgs
        origins_wonan = {}
        norms_wonan = {}
        for key in origin_data.keys():
            _origin = origin_data[key]
            _norm = norm_data[key]
            if _origin is None or _norm is None:
                origins_wonan[key] = None
                norms_wonan[key] = None
                continue
            kill_way = "interpolate"
            if key == "relevant_cols":
                rm_nan = data_cfgs["relevant_rm_nan"]
            elif key == "target_cols":
                rm_nan = data_cfgs["target_rm_nan"]
            elif key == "constant_cols":
                rm_nan = data_cfgs["constant_rm_nan"]
                kill_way = "mean"
            elif key == "forecast_cols":
                rm_nan = data_cfgs["forecast_rm_nan"]
                kill_way = "lead_step"
            elif key == "global_cols":
                rm_nan = data_cfgs["global_rm_nan"]
            elif key == "station_cols":
                rm_nan = data_cfgs.get("station_rm_nan")
            else:
                raise ValueError(
                    f"Unknown data type {key} in origin_data, "
                    "it should be one of relevant_cols, target_cols, constant_cols, forecast_cols, global_cols and station_cols"
                )

            if rm_nan:
                norm = self._kill_1type_nan(
                    _norm,
                    kill_way,
                    "original data",
                    "nan_filled data",
                )
                origin = self._kill_1type_nan(
                    _origin,
                    kill_way,
                    "original data",
                    "nan_filled data",
                )
            else:
                norm = _norm
                origin = _origin
            # _kill_1type_nan already validates filled arrays. Only validate
            # here when filling was disabled, avoiding a third full scan of
            # every large input array.
            if not rm_nan:
                if key == "target_cols":
                    warn_if_nan(
                        origin, nan_mode="all", data_name="nan_filled target data"
                    )
                    warn_if_nan(
                        norm, nan_mode="all", data_name="nan_filled target data"
                    )
                else:
                    warn_if_nan(
                        origin, nan_mode="any", data_name="nan_filled input data"
                    )
                    warn_if_nan(
                        norm, nan_mode="any", data_name="nan_filled input data"
                    )
            origins_wonan[key] = origin
            norms_wonan[key] = norm
        return origins_wonan, norms_wonan

    def _kill_1type_nan(self, the_data, fill_nan, data_name_before, data_name_after):
        is_any_nan = warn_if_nan(the_data, data_name=data_name_before)
        if not is_any_nan:
            return the_data
        # As input, we cannot have NaN values
        the_filled_data = _fill_gaps_da(the_data, fill_nan=fill_nan)
        warn_if_nan(the_filled_data, data_name=data_name_after)
        return the_filled_data

    def _create_lookup_table(self):
        basin_chunks = []
        start_chunks = []
        length_chunks = []
        # list to collect basins ids of basins without a single training sample
        basin_coordinates = len(self.t_s_dict["sites_id"])
        rho = self.rho
        warmup_length = self.warmup_length
        horizon = self.horizon
        # NOTE: we set seq_len to rho + horizon instead of warmup_length + rho + horizon
        seq_len = rho + horizon
        max_time_length = self.nt
        variable_length_cfgs = self.training_cfgs.get("variable_length_cfgs", {})
        use_variable_length = variable_length_cfgs.get("use_variable_length", False)
        variable_length_type = variable_length_cfgs.get(
            "variable_length_type", "dynamic"
        )  # only used for case when use_variable_length is True
        fixed_lengths = variable_length_cfgs.get("fixed_lengths", [365, 1095, 1825])
        # Use fixed type variable length if enabled and type is fixed
        is_fixed_length_train = use_variable_length and variable_length_type == "fixed"

        def append_lookup_arrays(basin, starts, length):
            if starts.size == 0:
                return
            basin_chunks.append(np.full(starts.shape, basin, dtype=np.int32))
            start_chunks.append(starts.astype(np.int32, copy=False))
            length_chunks.append(np.full(starts.shape, length, dtype=np.int32))

        for basin in tqdm(range(basin_coordinates), file=sys.stdout, disable=False):
            if not self.train_mode:
                # we don't need to ignore those with full nan in target vars for prediction without loss calculation
                # all samples should be included so that we can recover results to specified basins easily
                starts = np.arange(
                    warmup_length,
                    max_time_length - rho - horizon + 1,
                    dtype=np.int32,
                )
                append_lookup_arrays(basin, starts, seq_len)
            else:
                # some dataloader load data with warmup period, so leave some periods for it
                # [warmup_len] -> time_start -> [rho] -> [horizon]
                #                       window: \-----------------/ meaning rho + horizon
                nan_array = np.isnan(self.y[basin, :, :])
                if is_fixed_length_train:
                    for window in fixed_lengths:
                        starts = np.arange(
                            warmup_length,
                            max_time_length - window + 1,
                            dtype=np.int32,
                        )
                        starts = starts[
                            self._valid_window_mask(nan_array, starts, window)
                        ]
                        append_lookup_arrays(basin, starts, window)
                else:
                    starts = np.arange(
                        warmup_length,
                        max_time_length - rho - horizon + 1,
                        dtype=np.int32,
                    )
                    starts = starts[
                        self._valid_window_mask(nan_array, starts, rho + horizon)
                    ]
                    append_lookup_arrays(basin, starts, seq_len)

        if start_chunks:
            basins = np.concatenate(basin_chunks)
            starts = np.concatenate(start_chunks)
            lengths = np.concatenate(length_chunks)
        else:
            basins = np.empty(0, dtype=np.int32)
            starts = np.empty(0, dtype=np.int32)
            lengths = np.empty(0, dtype=np.int32)
        self._set_lookup_arrays(basins, starts, lengths)

    def _create_multi_len_lookup_table(self):
        """
        Create a lookup table for multi-length training
        TODO: not fully tested
        """
        lookup = []
        # list to collect basins ids of basins without a single training sample
        basin_coordinates = len(self.t_s_dict["sites_id"])
        rho = self.rho
        warmup_length = self.warmup_length
        horizon = self.horizon
        seq_len = warmup_length + rho + horizon
        max_time_length = self.nt
        variable_length_cfgs = self.training_cfgs.get("variable_length_cfgs", {})
        use_variable_length = variable_length_cfgs.get("use_variable_length", False)
        variable_length_type = variable_length_cfgs.get(
            "variable_length_type", "dynamic"
        )
        fixed_lengths = variable_length_cfgs.get("fixed_lengths", [365, 1095, 1825])
        # Use fixed type variable length if enabled and type is fixed
        is_fixed_length_train = use_variable_length and variable_length_type == "fixed"

        # 初始化不同长度的lookup表
        self.lookup_tables_by_length = {length: [] for length in fixed_lengths}

        # New: Global lookup table to map a single index to (window_length, index_within_that_window_length_table)
        self.global_lookup_table_indices = []

        for basin in tqdm(range(basin_coordinates), file=sys.stdout, disable=False):
            if not self.train_mode:
                # For prediction, we still use the original rho for simplicity if multi_length_training is enabled
                # or we can extend this logic to support multi-length prediction if needed.
                # For now, let's assume prediction uses a fixed rho or is handled differently.
                # If multi_length_training is active, we might need to decide which window_len to use for prediction.
                # For now, let's stick to the original logic for train_mode=False
                lookup.extend(
                    (basin, f, seq_len)
                    for f in range(warmup_length, max_time_length - rho - horizon + 1)
                )
            else:
                # some dataloader load data with warmup period, so leave some periods for it
                # [warmup_len] -> time_start -> [rho] -> [horizon]
                nan_array = np.isnan(self.y[basin, :, :])
                if is_fixed_length_train:
                    for window in fixed_lengths:
                        for f in range(
                            warmup_length, max_time_length - window - horizon + 1
                        ):
                            # 检查目标区间内是否全为nan
                            if not np.all(nan_array[f + window : f + window + horizon]):
                                # 记录 (basin, 起始位置) 到对应窗口长度的 lookup table
                                self.lookup_tables_by_length[window].append((basin, f))
                                # 记录 (窗口长度, 在该窗口长度 lookup table 中的索引) 到全局索引表
                                self.global_lookup_table_indices.append(
                                    (
                                        window,
                                        len(self.lookup_tables_by_length[window]) - 1,
                                    )
                                )
                else:
                    lookup.extend(
                        (basin, f, seq_len)
                        for f in range(
                            warmup_length, max_time_length - rho - horizon + 1
                        )
                        if not np.all(nan_array[f + rho : f + rho + horizon])
                    )

        if is_fixed_length_train and self.train_mode:
            # If fixed-length training is enabled and in train mode, use the global lookup table
            self.lookup_table = dict(enumerate(self.global_lookup_table_indices))
            self.num_samples = len(self.global_lookup_table_indices)
        else:
            # Otherwise, use the original lookup table (for fixed length training or prediction)
            self.lookup_table = dict(enumerate(lookup))
            self.num_samples = len(self.lookup_table)


class ObsForeDataset(BaseDataset):
    """处理观测和预见期数据的混合数据集

    这个类专门用于处理具有双维度预见期数据格式的数据集，其中 lead_time 和 time 都是独立维度。
    适合表示不同发布时间对不同目标日期的预报。
    """

    def __init__(self, cfgs: dict, is_tra_val_te: str):
        """初始化观测和预见期混合数据集

        Parameters
        ----------
        cfgs : dict
            all configs
        is_tra_val_te : str
            指定是训练集、验证集还是测试集
        """
        # 调用父类初始化方法
        super(ObsForeDataset, self).__init__(cfgs, is_tra_val_te)
        # for each batch, we fix length of hindcast and forecast length.
        # data from different lead time with a number representing the lead time,
        # for example, now is 2020-09-30, our min_time_interval is 1 day, hindcast length is 30 and forecast length is 1,
        # lead_time = 3 means 2020-09-01 to 2020-09-30, and the forecast data is 2020-10-01 from 2020-09-28
        # for forecast data, we have two different configurations:
        # 1st, we can set a same lead time for all forecast time
        # 2020-09-30now, 30hindcast, 2forecast, 3leadtime means 2020-09-01 to 2020-09-30 obs concatenate with 2020-10-01 forecast data from 2020-09-28 and 2020-10-02 forecast data from 2020-09-29
        # 2nd, we can set a increasing lead time for each forecast time
        # 2020-09-30now, 30hindcast, 2forecast, [1, 2]leadtime means 2020-09-01 to 2020-09-30 obs concatenate with 2020-10-01 to 2010-10-02 forecast data from 2020-09-30
        self.lead_time_type = self.training_cfgs["lead_time_type"]
        if self.lead_time_type not in ["fixed", "increasing"]:
            raise ValueError(
                "lead_time_type must be one of 'fixed' or 'increasing', "
                f"but got {self.lead_time_type}"
            )
        self.lead_time_start = self.training_cfgs["lead_time_start"]
        horizon = self.horizon
        offset = np.zeros((horizon,), dtype=int)
        if self.lead_time_type == "fixed":
            offset = offset + self.lead_time_start
        elif self.lead_time_type == "increasing":
            offset = offset + np.arange(
                self.lead_time_start, self.lead_time_start + horizon
            )
        self.horizon_offset = offset
        feature_mapping = self.data_cfgs["feature_mapping"]
        #
        xf_var_indices = {}
        for obs_var, fore_var in feature_mapping.items():
            # 找到x中需要被替换的变量索引
            x_var_indice = [
                i
                for i, var in enumerate(self.data_cfgs["relevant_cols"])
                if var == obs_var
            ][0]
            # 找到f中对应的变量索引
            f_var_indice = [
                i
                for i, var in enumerate(self.data_cfgs["forecast_cols"])
                if var == fore_var
            ][0]
            xf_var_indices[x_var_indice] = f_var_indice
        self.xf_var_indices = xf_var_indices

    def _read_xyc_specified_time(self, start_date, end_date, **kwargs):
        """read f data from data source with specified time range and add it to the whole dict"""
        data_dict = super(ObsForeDataset, self)._read_xyc_specified_time(
            start_date, end_date
        )
        lead_time = kwargs.get("lead_time", None)
        f_origin = self.data_source.read_ts_xrdataset(
            self.t_s_dict["sites_id"],
            [start_date, end_date],
            self.data_cfgs["forecast_cols"],
            forecast_mode=True,
            lead_time=lead_time,
        )
        f_origin_ = self._rm_timeunit_key(f_origin)
        f_data = self._trans2da_and_setunits(f_origin_)
        data_dict["forecast_cols"] = f_data.transpose(
            "basin", "time", "lead_step", "variable"
        )
        return data_dict

    def __getitem__(self, item: int):
        """Get a sample from the dataset

        Parameters
        ----------
        item : int
            index of sample

        Returns
        -------
        tuple
            A pair of (x, y) data, where x contains input features and lead time flags,
            and y contains target values
        """
        # train mode
        basin, idx, _ = self.lookup_table[item]
        warmup_length = self.warmup_length
        # for x, we only chose data before horizon, but we may need forecast data for not all variables
        # hence, to avoid nan values for some variables without forecast in horizon
        # we still get data from the first time to the end of horizon
        x = self.x[basin, idx - warmup_length : idx + self.rho + self.horizon, :]
        # for y, we chose data after warmup_length
        y = self.y[basin, idx : idx + self.rho + self.horizon, :]
        # use offset to get forecast data
        offset = self.horizon_offset
        if self.lead_time_type == "fixed":
            # Fixed lead_time mode - All forecast steps use the same lead_step
            f = self.f[
                basin, idx + self.rho : idx + self.rho + self.horizon, offset[0], :
            ]
        else:
            # Increasing lead_time mode - Each forecast step uses a different lead_step
            f = self.f[basin, idx + self.rho, offset, :]
        xf = self._concat_xf(x, f)
        if self.c is None or self.c.shape[-1] == 0:
            xfc = xf
        else:
            c = self.c[basin, :]
            c = np.repeat(c, xf.shape[0], axis=0).reshape(c.shape[0], -1).T
            xfc = np.concatenate((xf, c), axis=1)

        return torch.from_numpy(xfc).float(), torch.from_numpy(y).float()

    def _concat_xf(self, x, f):
        # Create a copy of x to avoid modifying the original data
        x_combined = x.copy()

        # Iterate through the variable mapping relationship
        for x_idx, f_idx in self.xf_var_indices.items():
            # Replace the variables in the forecast period of x with the forecast variables in f
            # The forecast period of x starts from the rho position
            x_combined[self.warmup_length + self.rho :, x_idx] = f[:, f_idx]

        return x_combined


class BasinSingleFlowDataset(BaseDataset):
    """one time length output for each grid in a batch"""

    def __init__(self, cfgs: dict, is_tra_val_te: str):
        super(BasinSingleFlowDataset, self).__init__(cfgs, is_tra_val_te)

    def __getitem__(self, index):
        xc, ys = super(BasinSingleFlowDataset, self).__getitem__(index)
        y = ys[-1, :]
        return xc, y


class DplDataset(BaseDataset):
    """pytorch dataset for Differential parameter learning"""

    def __init__(self, cfgs: dict, is_tra_val_te: str):
        """
        Parameters
        ----------
        cfgs
            all configs
        is_tra_val_te
            train, vaild or test
        """
        super(DplDataset, self).__init__(cfgs, is_tra_val_te)
        # we don't use y_un_norm as its name because in the main function we will use "y"
        # For physical hydrological models, we need warmup, hence the target values should exclude data in warmup period
        self.warmup_length = self.training_cfgs["warmup_length"]
        self.target_as_input = self.data_cfgs["target_as_input"]
        self.constant_only = self.data_cfgs["constant_only"]
        if self.target_as_input and (not self.train_mode):
            # if the target is used as input and train_mode is False,
            # we need to get the target data in training period to generate pbm params
            self.train_dataset = DplDataset(cfgs, is_tra_val_te="train")

    def __getitem__(self, item):
        """
        Get one mini-batch for dPL (differential parameter learning) model

        TODO: not check target_as_input and constant_only cases yet

        Parameters
        ----------
        item
            index

        Returns
        -------
        tuple
            a mini-batch data;
            x_train (not normalized forcing), z_train (normalized data for DL model), y_train (not normalized output)
        """
        warmup = self.warmup_length
        rho = self.rho
        horizon = self.horizon
        xc_norm, _ = super(DplDataset, self).__getitem__(item)
        basin, time, _ = self.lookup_table[item]
        if self.target_as_input:
            # y_morn and xc_norm are concatenated and used for DL model
            y_norm = torch.from_numpy(
                self.y[basin, time - warmup : time + rho + horizon, :]
            ).float()
            # the order of xc_norm and y_norm matters, please be careful!
            z_train = torch.cat((xc_norm, y_norm), -1)
        elif self.constant_only:
            # only use attributes data for DL model
            z_train = torch.from_numpy(self.c[basin, :]).float()
        else:
            z_train = xc_norm.float()
        x_train = self.x_origin[basin, time - warmup : time + rho + horizon, :]
        y_train = self.y_origin[basin, time : time + rho + horizon, :]
        return (
            torch.from_numpy(x_train).float(),
            z_train,
        ), torch.from_numpy(y_train).float()


class MultiDataSourceWrapper:
    """Wrapper that presents multiple data sources as a unified interface for scalers."""

    def __init__(self, sources, datasets_config=None):
        self.sources = sources
        self.datasets_config = datasets_config
        self._reader = MultiDatasets(datasets=datasets_config)
        # Cache unique IDs mapping
        self._unique_ids_map, _ = self._reader.get_global_unique_ids()

    def _group_ids(self, gage_id_lst):
        ids_by_dataset = {}
        for gage_id in gage_id_lst:
            # Handle duplicates logic if needed, but here we likely want unique source per ID
            # Use cached map
            if gage_id in self._unique_ids_map:
                ds_name = self._unique_ids_map[gage_id]
                if ds_name not in ids_by_dataset:
                    ids_by_dataset[ds_name] = []
                ids_by_dataset[ds_name].append(gage_id)
            else:
                # Fallback/retry or warn
                pass
        return ids_by_dataset

    def read_mean_prcp(self, gage_id_lst: list, unit: str = "mm/d"):
        ids_by_ds = self._group_ids(gage_id_lst)
        results = []
        for ds_name, ids in ids_by_ds.items():
            if ds_name in self.sources:
                ds_obj = self.sources[ds_name]
                if hasattr(ds_obj, "read_mean_prcp"):
                    results.append(ds_obj.read_mean_prcp(ids, unit))

        if not results:
            return None
        return xr.concat(results, dim="basin")

    def read_area(self, gage_id_lst: list):
        ids_by_ds = self._group_ids(gage_id_lst)
        results = []
        for ds_name, ids in ids_by_ds.items():
            if ds_name in self.sources:
                ds_obj = self.sources[ds_name]
                if hasattr(ds_obj, "read_area"):
                    results.append(ds_obj.read_area(ids))

        if not results:
            return None
        return xr.concat(results, dim="basin")

    def read_attr_xrdataset(self, gage_id_lst, var_lst, **kwargs):
        ids_by_ds = self._group_ids(gage_id_lst)
        results = []
        for ds_name, ids in ids_by_ds.items():
            if ds_name in self.sources:
                ds_obj = self.sources[ds_name]
                if hasattr(ds_obj, "read_attr_xrdataset"):
                    results.append(ds_obj.read_attr_xrdataset(ids, var_lst, **kwargs))

        if not results:
            return None
        return xr.concat(results, dim="basin")


class MultiSourceDataset(BaseDataset):
    """A dataset that supports reading from multiple hydrological datasets simultaneously.

    This class uses MultiDatasets to handle IDs and data loading from multiple sources.
    It automatically groups stations by dataset, reads data using the appropriate data source,
    and merges the results.
    """

    def __init__(self, cfgs: dict, is_tra_val_te: str):
        super(MultiSourceDataset, self).__init__(cfgs, is_tra_val_te)

    @property
    def data_source(self):
        """
        For MultiSourceDataset, we returns a wrapper that aggregates all sources.
        """
        from torchhydro.datasets.data_sources import data_sources_dict

        # Initialize all required data sources
        sources = {}
        # Get datasets from config, or default to all available in DATASET_MAPPING
        if "source_cfgs" in self.data_cfgs:
            datasets_config = self.data_cfgs["source_cfgs"].get("datasets", None)
            if isinstance(datasets_config, list):
                req_datasets = datasets_config
            else:
                # Default to all if not specified list
                req_datasets = list(DATASET_MAPPING.keys())
        else:
            req_datasets = list(DATASET_MAPPING.keys())

        for ds_name in req_datasets:
            if ds_name in DATASET_MAPPING:
                _, class_name = DATASET_MAPPING[ds_name]
                if ds_name in data_sources_dict:
                    source_constructor = data_sources_dict[ds_name]
                    # source_path = SETTING["local_data_path"]["datasets-origin"]
                    source_path = self.data_cfgs["source_cfgs"]["source_path"]
                    sources[ds_name] = source_constructor(source_path)

        return MultiDataSourceWrapper(sources, datasets_config)

    def _read_xyc(self):
        """Read x, y, c data from multiple data sources and merge them."""

        # 1. Collect IDs and their source datasets
        # use MultiDatasets logic to identify which dataset each ID belongs to
        # We assume self.t_s_dict["sites_id"] contains the list of all IDs we want to read
        target_ids = self.t_s_dict["sites_id"]

        # Initialize reader just for ID mapping (cache might be used)
        datasets_config = self.data_cfgs.get("source_cfgs", {}).get("datasets", None)
        reader = MultiDatasets(datasets=datasets_config)

        # Use reader to get global unique IDs to find where each ID comes from
        # Note: this might be slow if we have many datasets and didn't cache IDs
        unique_ids_map, duplicates_map = reader.get_global_unique_ids()

        # Group target IDs by dataset
        ids_by_dataset = {}

        for gage_id in target_ids:
            if gage_id in unique_ids_map:
                ds_name = unique_ids_map[gage_id]
                if ds_name not in ids_by_dataset:
                    ids_by_dataset[ds_name] = []
                ids_by_dataset[ds_name].append(gage_id)
            else:
                # Log warning or handle missing ID
                print(f"Warning: ID {gage_id} not found in any configured dataset.")

        # 2. Read data for each group
        x_list, y_list, c_list = [], [], []

        t_range = self.t_s_dict["t_final_range"]

        # Need to instantiate data sources on the fly or use the property
        # Since we need to pass specific paths or args, better to do it here or use a helper

        for ds_name, ids in ids_by_dataset.items():
            if ds_name not in DATASET_MAPPING:
                continue

            # Get the data source instance for this dataset
            # We assume data_sources_dict has entries for these datasets
            # If not, we might need to rely on hydrodataset readers directly?
            # But torchhydro pipeline expects xarray from its own data_sources.
            # Let's assume we have corresponding wrappers in torchhydro.

            # Construct the data source
            # This relies on torchhydro.datasets.data_sources having wrappers for these datasets
            # which usually wrap hydrodataset classes.
            if ds_name in data_sources_dict:
                source_cls = data_sources_dict[ds_name]
                # source_path = SETTING["local_data_path"]["datasets-origin"]
                source_path = self.data_cfgs["source_cfgs"]["source_path"]
                # Some data sources might need extra config
                ds_instance = source_cls(source_path)

                # Read x (forcing)
                if self.data_cfgs["relevant_cols"]:
                    x_ds = ds_instance.read_ts_xrdataset(
                        ids, t_range, self.data_cfgs["relevant_cols"]
                    )
                    x_ds = self._rm_timeunit_key(x_ds)
                    x_list.append(x_ds)

                # Read y (streamflow/target)
                if self.data_cfgs["target_cols"]:
                    y_ds = ds_instance.read_ts_xrdataset(
                        ids, t_range, self.data_cfgs["target_cols"]
                    )
                    y_ds = self._rm_timeunit_key(y_ds)
                    y_list.append(y_ds)

                # Read c (attributes)
                if self.data_cfgs["constant_cols"]:
                    c_ds = ds_instance.read_attr_xrdataset(
                        ids, self.data_cfgs["constant_cols"], all_number=True
                    )
                    c_list.append(c_ds)
            else:
                print(f"Warning: No torchhydro data source found for {ds_name}")

        # 3. Merge datasets
        # We need to concatenate along the 'basin' dimension.
        # xarray's concat or merge should handle this if dimensions match (time, variable)

        if not x_list and not y_list:
            raise ValueError("No data read from any source.")

        # Helper to merge list of datasets
        def safe_concat(ds_list):
            if not ds_list:
                return None
            # Prepare for concatenation: ensure they have the same variables and time index
            # This might be tricky if time ranges slightly differ or if variables are missing
            # Assuming standard read_ts_xrdataset returns aligned time series for the requested t_range
            return xr.concat(ds_list, dim="basin")

        x_merged = safe_concat(x_list)
        y_merged = safe_concat(y_list)
        c_merged = safe_concat(c_list)

        # Ensure units are consistent/handled if needed (e.g., using unit conversion logic from BaseDataset)
        # For now, we assume _check_ts_xrds_unit might need to be called on the merged result
        # OR iteratively before merging.
        # Since _check_ts_xrds_unit takes (forcing, output), let's try running it on merged result.
        # But wait, unit conversion depends on each dataset's metadata.
        # Merging first might lose that per-dataset metadata if attributes aren't preserved per basin.
        # However, xr.concat typically preserves attributes of the first object or drops them.
        # Ideally, we should unify units BEFORE merging.
        # Let's skip complex unit unification for now and assume standard units (mm/d, m3/s) or handle it simply.
        # BaseDataset._check_ts_xrds_unit handles streamflow unit conversion based on attrs.
        # If we concat, we might lose individual attrs.

        # Let's try to convert units PER DATASET before merging
        # But _check_ts_xrds_unit logic is a bit coupled.
        # Let's rely on standard hydrodataset behavior which tries to standardize to mm/d or m3/s.

        if x_merged is not None and y_merged is not None:
            # Just ensures variable names match what's expected
            pass

        # Check if any flow variable exists in y dataset and convert units if needed
        # We try to use streamflow name attribute or fallback to target cols
        flow_var_name = getattr(self, "streamflow_name", None)
        if flow_var_name is None:
            # fallback: check if any target variable is in y
            for target_var in self.data_cfgs["target_cols"]:
                if y_merged is not None and target_var in y_merged:
                    flow_var_name = target_var
                    break

        if flow_var_name and y_merged is not None and flow_var_name in y_merged:
            # We need to read area for ALL basins in y_merged to perform conversion
            # self.data_source is now a MultiDataSourceWrapper which handles this
            area = self.data_source.read_area(y_merged["basin"].values.tolist())

            # Use precipitation unit as target unit if available
            prcp_unit = "mm/d"
            if x_merged is not None and self.data_cfgs.get("relevant_cols"):
                prcp_name = self.precipitation_name
                if prcp_name in x_merged:
                    prcp_unit = x_merged[prcp_name].attrs.get("units", "mm/d")

            if area is not None:
                y_merged.update(
                    streamflow_unit_conv(
                        y_merged[[flow_var_name]], area, target_unit=prcp_unit
                    )
                )
        # Validate and sync basin IDs
        # We must ensure self.t_s_dict["sites_id"] matches the basins actually present in the merged data
        # created by xarray. The order and count must match for BaseDataset lookup table generation.
        loaded_basins = None
        if y_merged is not None:
            loaded_basins = y_merged["basin"].values.astype(str).tolist()
        elif x_merged is not None:
            loaded_basins = x_merged["basin"].values.astype(str).tolist()

        if loaded_basins is not None:
            if len(loaded_basins) != len(self.t_s_dict["sites_id"]):
                print(
                    f"Warning: Requested {len(self.t_s_dict['sites_id'])} stations, but only {len(loaded_basins)} were successfully loaded. Updating sites_id list."
                )
                # Update the list in place
                self.t_s_dict["sites_id"] = loaded_basins
            else:
                # Ensure order matches if count is same (though xarray concat usually preserves order of concat/input,
                # but merged might reorder if not aligned? Let's force update to be safe)
                self.t_s_dict["sites_id"] = loaded_basins

        # Convert to DataArrays with units
        x_origin, y_origin, c_origin = self._to_dataarray_with_unit(
            x_merged, y_merged, c_merged
        )

        return {
            "relevant_cols": (
                x_origin.transpose("basin", "time", "variable")
                if x_origin is not None
                else None
            ),
            "target_cols": (
                y_origin.transpose("basin", "time", "variable")
                if y_origin is not None
                else None
            ),
            "constant_cols": (
                c_origin.transpose("basin", "variable")
                if c_origin is not None
                else None
            ),
        }


class MultiSourceAEFDataset(MultiSourceDataset):
    """
    仿照MultiSourceDataset编写的一个类，
    读取x和y即动态属性和观测值的方式不变（继承自MultiSourceDataset），
    读取c即静态属性改为从指定路径的csv中读取。
    source_cfgs["attributes_paths"] 应该是一个字典: {dataset_name: csv_path}，
    为每个数据集指定对应的静态属性CSV文件。
    CSV文件格式第一列为流域id（hru_id），第一行是列名。
    """

    def _read_xyc(self):
        # 1. 暂存并清空 constant_cols 配置，避免父类 _read_xyc 方法从数据源读取静态属性
        original_constant_cols = self.data_cfgs.get("constant_cols")
        self.data_cfgs["constant_cols"] = None

        try:
            # 2. 调用父类方法读取 X 和 Y
            # 父类会处理 ID 对齐、多源数据读取和合并
            data_dict = super()._read_xyc()
        finally:
            # 3. 恢复 constant_cols 配置
            self.data_cfgs["constant_cols"] = original_constant_cols

        # 4. 获取属性文件配置字典
        attributes_paths = self.data_cfgs.get("source_cfgs", {}).get("attributes_paths")
        if not attributes_paths or not isinstance(attributes_paths, dict):
            raise ValueError(
                "MultiSourceCsvStaticDataset requires 'attributes_paths' dict in source_cfgs mapping dataset names to csv paths"
            )

        # 5. 对此时的 basins 进行分组（确定属于哪个数据集）
        # 此时的 sites_id 是经过父类 _read_xyc 可能更新过的（剔除了无数据的basin）
        target_ids = self.t_s_dict["sites_id"]
        datasets_config = self.data_cfgs.get("source_cfgs", {}).get("datasets", None)
        reader = MultiDatasets(datasets=datasets_config)
        unique_ids_map, _ = reader.get_global_unique_ids()

        ids_by_dataset = {}
        for gage_id in target_ids:
            if gage_id in unique_ids_map:
                ds_name = unique_ids_map[gage_id]
                if ds_name not in ids_by_dataset:
                    ids_by_dataset[ds_name] = []
                ids_by_dataset[ds_name].append(gage_id)
            else:
                # 如果找不到归属的数据集，打印警告
                print(
                    f"Warning: ID {gage_id} not found in any configured dataset for attribute loading."
                )

        c_list = []

        # 6. 遍历每个数据集读取 CSV
        for ds_name, ids in ids_by_dataset.items():
            csv_path = attributes_paths.get(ds_name)
            if not csv_path:
                print(
                    f"Error: No attributes CSV path defined for dataset {ds_name} which contains {len(ids)} basins."
                )
                raise ValueError(
                    f"Missing attributes_paths entry for dataset: {ds_name}"
                )

            # 读取 CSV
            df = pd.read_csv(csv_path)
            # 仿照 AEFDataset 的逻辑，将 ID 转换为去掉前导零的字符串
            # 假设第一列是 ID
            id_col = df.columns[0]
            df[id_col] = df[id_col].astype(int).astype(str).str.lstrip("0")
            df.set_index(id_col, inplace=True)

            # 对请求的 ids 也做同样的处理
            target_ids_stripped = [str(gid).lstrip("0") for gid in ids]

            # 筛选 Basin
            # reindex 会引入 NaN 如果有缺失 ID
            df_subset = df.reindex(target_ids_stripped)

            # 恢复原始 ID 作为索引
            df_subset.index = ids

            # 检查是否有完全缺失的行
            if df_subset.isnull().all(axis=1).any():
                missing_ids = df_subset.index[df_subset.isnull().all(axis=1)].tolist()
                print(
                    f"Warning: {len(missing_ids)} basins in dataset {ds_name} match no static attributes in CSV (first 5: {missing_ids[:5]})"
                )

            # 选择列
            if original_constant_cols:
                # 这种方式会保留所有 original_constant_cols，缺失的列会变成 NaN
                # 如果不需要严格对齐，可以用 intersection
                # df_subset = df_subset[df_subset.columns.intersection(original_constant_cols)]
                # 但一般要求所有属性都存在，故直接 reindex
                df_subset = df_subset.reindex(columns=original_constant_cols)

            # 转换为 Dataset/DataArray
            df_subset.index.name = "basin"
            ds_c = xr.Dataset.from_dataframe(df_subset)

            # 设置默认单位
            for var in ds_c.data_vars:
                ds_c[var].attrs["units"] = "dimensionless"

            c_da_part = self._trans2da_and_setunits(ds_c)
            # 确保维度顺序 (basin, variable)
            if "basin" in c_da_part.dims and "variable" in c_da_part.dims:
                c_da_part = c_da_part.transpose("basin", "variable")

            c_list.append(c_da_part)

        # 7. 合并所有数据集的静态属性
        if not c_list:
            data_dict["constant_cols"] = None
        else:
            # 按 basin 维度合并
            c_merged = xr.concat(c_list, dim="basin")

            # 8. 重新按照 target_ids 排序，确保与 X, Y 顺序一致
            if "basin" in c_merged.coords:
                # 必须确保所有 basin 都在 merged 中（reindexing 会引入 NaN 如果没有）
                c_merged = c_merged.sel(basin=target_ids)

            data_dict["constant_cols"] = c_merged

        return data_dict


class FlexibleDataset(BaseDataset):
    """A dataset whose datasources are from multiple sources according to the configuration"""

    def __init__(self, cfgs: dict, is_tra_val_te: str):
        super(FlexibleDataset, self).__init__(cfgs, is_tra_val_te)

    @property
    def data_source(self):
        source_cfgs = self.data_cfgs["source_cfgs"]
        return {
            name: data_sources_dict[name](path)
            for name, path in zip(
                source_cfgs["source_names"], source_cfgs["source_paths"]
            )
        }

    def _read_xyc(self):
        var_to_source_map = self.data_cfgs["var_to_source_map"]
        x_datasets, y_datasets, c_datasets = [], [], []
        gage_ids = self.t_s_dict["sites_id"]
        t_range = self.t_s_dict["t_final_range"]

        for var_name in var_to_source_map:
            source_name = var_to_source_map[var_name]
            data_source_ = self.data_source[source_name]
            if var_name in self.data_cfgs["relevant_cols"]:
                x_datasets.append(
                    data_source_.read_ts_xrdataset(gage_ids, t_range, [var_name])
                )
            elif var_name in self.data_cfgs["target_cols"]:
                y_datasets.append(
                    data_source_.read_ts_xrdataset(gage_ids, t_range, [var_name])
                )
            elif var_name in self.data_cfgs["constant_cols"]:
                c_datasets.append(
                    data_source_.read_attr_xrdataset(gage_ids, [var_name])
                )

        # 合并所有x, y, c类型的数据集
        x = xr.merge(x_datasets) if x_datasets else xr.Dataset()
        y = xr.merge(y_datasets) if y_datasets else xr.Dataset()
        c = xr.merge(c_datasets) if c_datasets else xr.Dataset()
        # Check if any flow variable exists in y dataset instead of hardcoding "streamflow"
        flow_var_name = (
            self.streamflow_name
            if hasattr(self, "streamflow_name") and self.streamflow_name in y
            else None
        )
        if flow_var_name is None:
            # fallback: check if any target variable is in y
            for target_var in self.data_cfgs["target_cols"]:
                if target_var in y:
                    flow_var_name = target_var
                    break
        if flow_var_name and flow_var_name in y:
            area = data_source_.camels.read_area(self.t_s_dict["sites_id"])
            y.update(streamflow_unit_conv(y[[flow_var_name]], area))
        x_origin, y_origin, c_origin = self._to_dataarray_with_unit(x, y, c)
        return x_origin, y_origin, c_origin

    def _normalize(self):
        var_to_source_map = self.data_cfgs["var_to_source_map"]
        for var_name in var_to_source_map:
            source_name = var_to_source_map[var_name]
            data_source_ = self.data_source[source_name]
            break
        # TODO: only support CAMELS for now
        scaler_hub = ScalerHub(
            self.y_origin,
            self.x_origin,
            self.c_origin,
            data_cfgs=self.data_cfgs,
            is_tra_val_te=self.is_tra_val_te,
            data_source=data_source_.camels,
        )
        self.target_scaler = scaler_hub.target_scaler
        return scaler_hub.x, scaler_hub.y, scaler_hub.c


class Seq2SeqDataset(BaseDataset):
    def __init__(self, cfgs: dict, is_tra_val_te: str):
        super(Seq2SeqDataset, self).__init__(cfgs, is_tra_val_te)

    def _read_xyc_period(self, start_date, end_date):
        """Override template method to adjust start date for Seq2Seq requirements

        NOTE: Seq2SeqDataset needs one more time period at the beginning
        because it includes the concepts of start and end moment of the period.

        Parameters
        ----------
        start_date : str
            start time
        end_date : str
            end time

        Returns
        -------
        dict
            Dictionary containing relevant_cols, target_cols, and constant_cols data
        """
        interval = self.data_cfgs["min_time_interval"]
        time_unit = self.data_cfgs["min_time_unit"]

        # Adjust start date by subtracting one interval
        adjusted_start_date = self._adjust_start_date_by_interval(
            start_date, interval, time_unit
        )

        return self._read_xyc_specified_time(adjusted_start_date, end_date)

    def _adjust_start_date_by_interval(self, start_date, interval, time_unit):
        """Adjust start date by subtracting specified interval

        Parameters
        ----------
        start_date : str
            Original start date string
        interval : int
            Time interval to subtract
        time_unit : str
            Time unit ('h' for hours, 'D' for days)

        Returns
        -------
        str
            Adjusted start date string
        """
        # Determine the date format
        date_format = detect_date_format(start_date)

        # Adjust the start date based on the time unit
        start_date_dt = datetime.strptime(start_date, date_format)
        if time_unit == "h":
            adjusted_start_date = (start_date_dt - timedelta(hours=interval)).strftime(
                date_format
            )
        elif time_unit == "D":
            adjusted_start_date = (start_date_dt - timedelta(days=interval)).strftime(
                date_format
            )
        else:
            raise ValueError(f"Unsupported time unit: {time_unit}")

        return adjusted_start_date

    def _selected_time_points_for_denorm(self):
        # because we have time start and end for each period, similar reason to why we need to override _read_xyc
        return self.target_scaler.data_target.coords["time"][self.warmup_length : -1]

    def __getitem__(self, item: int):
        basin, time, _ = self.lookup_table[item]
        rho = self.rho
        horizon = self.horizon
        hindcast_output_window = self.data_cfgs.get("hindcast_output_window", 0)
        # p cover all encoder-decoder periods; +1 means the period while +0 means start of the current period
        p = self.x[basin, time + 1 : time + rho + horizon + 1, :1]
        # s only cover encoder periods
        s = self.x[basin, time : time + rho, 1:]
        # xe = np.concatenate((p[:rho], s), axis=1)

        if self.c is None or self.c.shape[-1] == 0:
            pc = p
        else:
            c = self.c[basin, :]
            c = np.tile(c, (rho + horizon, 1))
            pc = np.concatenate((p[:rho], c[:rho]), axis=1)
        xe = np.concatenate((pc[:rho], s), axis=1)
        # xh cover decoder periods
        try:
            xd = np.concatenate((p[rho:], c[rho:]), axis=1)
        except ValueError as e:
            print(f"Error in np.concatenate: {e}")
            print(f"p[rho:].shape: {p[rho:].shape}, c[rho:].shape: {c[rho:].shape}")
            raise
        # y cover specified encoder size (hindcast_output_window) and all decoder periods
        y = self.y[
            basin, time + rho - hindcast_output_window + 1 : time + rho + horizon + 1, :
        ]  # qs
        # y_q = y[:, :1]
        # y_s = y[:, 1:]
        # y = np.concatenate((y_s, y_q), axis=1)

        if self.is_tra_val_te == "train":
            return [
                torch.from_numpy(xe).float(),
                torch.from_numpy(xd).float(),
                torch.from_numpy(y).float(),
            ], torch.from_numpy(y).float()
        return [
            torch.from_numpy(xe).float(),
            torch.from_numpy(xd).float(),
        ], torch.from_numpy(y).float()


class SeqForecastDataset(Seq2SeqDataset):
    def __init__(self, data_cfgs: dict, is_tra_val_te: str):
        super(SeqForecastDataset, self).__init__(data_cfgs, is_tra_val_te)

    def __getitem__(self, item: int):
        basin, time, _ = self.lookup_table[item]
        rho = self.rho  # forecast history
        horizon = self.horizon  # forecast length
        hindcast_output_window = self.data_cfgs.get("hindcast_output_window", 0)
        xe = self.x[basin, time : time + rho, :]
        xd = self.x[basin, time + rho : time + rho + horizon, :]
        c = self.c[basin, :]
        # y cover specified all decoder periods
        y = self.y[basin, time + rho - hindcast_output_window : time + rho + horizon, :]

        return [
            torch.from_numpy(xe).float(),
            torch.from_numpy(xd).float(),
            torch.from_numpy(c).float(),
        ], torch.from_numpy(y).float()


class TransformerDataset(Seq2SeqDataset):
    def __init__(self, data_cfgs: dict, is_tra_val_te: str):
        super(TransformerDataset, self).__init__(data_cfgs, is_tra_val_te)

    def __getitem__(self, item: int):
        basin, idx, _ = self.lookup_table[item]
        rho = self.rho
        horizon = self.horizon

        p = self.x[basin, idx + 1 : idx + rho + horizon + 1, 0]
        s = self.x[basin, idx : idx + rho, 1]
        x = np.stack((p[:rho], s), axis=1)

        c = self.c[basin, :]
        c = np.tile(c, (rho + horizon, 1))
        x = np.concatenate((x, c[:rho]), axis=1)

        x_h = np.concatenate((p[rho:].reshape(-1, 1), c[rho:]), axis=1)
        y = self.y[basin, idx + rho + 1 : idx + rho + horizon + 1, :]

        return [
            torch.from_numpy(x).float(),
            torch.from_numpy(x_h).float(),
        ], torch.from_numpy(y).float()


class ForecastDataset(BaseDataset):
    def __init__(self, data_cfgs: dict, is_tra_val_te: str):
        super(ForecastDataset, self).__init__(data_cfgs, is_tra_val_te)

    def __getitem__(self, item):
        basin, idx, _ = self.lookup_table[item]
        warmup_length = self.warmup_length
        x = self.x[basin, idx - warmup_length : idx + self.rho + self.horizon, :]
        y = self.y[basin, idx + self.rho : idx + self.rho + self.horizon, :]
        if self.c is None or self.c.shape[-1] == 0:
            return torch.from_numpy(x).float(), torch.from_numpy(y).float()
        c = self.c[basin, :]
        c = np.repeat(c, x.shape[0], axis=0).reshape(c.shape[0], -1).T
        xc = np.concatenate((x, c), axis=1)
        return torch.from_numpy(xc).float(), torch.from_numpy(y).float()


class HFDataset(BaseDataset):
    def __init__(self, data_cfgs: dict, is_tra_val_te: str):
        super(HFDataset, self).__init__(data_cfgs, is_tra_val_te)

    @property
    def streamflow_input_name(self):
        return self.data_cfgs["relevant_cols"][-1]

    def _read_xyc_period(self, start_date, end_date):
        """Override template method to handle HFDataset's specific time adjustment needs

        HFDataset needs to adjust forcing data time range (extend backward) while keeping
        target data time range unchanged for high-frequency forecasting scenarios.

        Parameters
        ----------
        start_date : str
            start time
        end_date : str
            end time

        Returns
        -------
        dict
            Dictionary containing relevant_cols, target_cols, and constant_cols data
        """
        # Calculate adjusted start date for forcing data
        date_format = detect_date_format(start_date)
        time_unit = self.data_cfgs["min_time_unit"]
        start_date_dt = datetime.strptime(start_date, date_format)

        if time_unit == "h":
            adjusted_start_date = (start_date_dt - timedelta(hours=1)).strftime(
                date_format
            )
        elif time_unit == "D":
            adjusted_start_date = (
                start_date_dt - timedelta(days=self.data_cfgs["min_time_interval"])
            ).strftime(date_format)
        else:
            raise ValueError(f"Unsupported time unit: {time_unit}")

        # Read forcing data with adjusted start date
        data_forcing_ds_ = self.data_source.read_ts_xrdataset(
            self.t_s_dict["sites_id"],
            [adjusted_start_date, end_date],
            self.data_cfgs["relevant_cols"],
        )

        # Read target data with original time range
        data_output_ds_ = self.data_source.read_ts_xrdataset(
            self.t_s_dict["sites_id"],
            [start_date, end_date],
            self.data_cfgs["target_cols"],
        )

        # Handle dict format data sources (time_unit as key)
        data_forcing_ds_ = self._rm_timeunit_key(data_forcing_ds_)
        data_output_ds_ = self._rm_timeunit_key(data_output_ds_)

        # Process unit conversion
        data_forcing_ds, data_output_ds = self._check_ts_xrds_unit(
            data_forcing_ds_, data_output_ds_
        )

        # Read attribute data
        data_attr_ds = self.data_source.read_attr_xrdataset(
            self.t_s_dict["sites_id"],
            self.data_cfgs["constant_cols"],
            all_number=True,
        )

        # Convert to DataArray with units
        x_origin, y_origin, c_origin = self._to_dataarray_with_unit(
            data_forcing_ds, data_output_ds, data_attr_ds
        )

        return {
            "relevant_cols": x_origin.transpose("basin", "time", "variable"),
            "target_cols": y_origin.transpose("basin", "time", "variable"),
            "constant_cols": c_origin.transpose("basin", "variable"),
        }

    def _check_ts_xrds_unit(self, data_forcing_ds, data_output_ds):
        """Check timeseries xarray dataset unit and convert if necessary

        Parameters
        ----------
        data_forcing_ds : xr.Dataset
            the forcing data
        data_output_ds : xr.Dataset
            outputs including streamflow data
        """

        def standardize_unit(unit):
            unit = unit.lower()  # convert to lower case
            unit = re.sub(r"day", "d", unit)
            unit = re.sub(r"hour", "h", unit)
            return unit

        streamflow_unit = data_output_ds[self.streamflow_name].attrs["units"]
        prcp_unit = data_forcing_ds[self.precipitation_name].attrs["units"]

        standardized_streamflow_unit = standardize_unit(streamflow_unit)
        standardized_prcp_unit = standardize_unit(prcp_unit)
        if standardized_streamflow_unit != standardized_prcp_unit:
            streamflow_dataset = data_output_ds[[self.streamflow_name]]
            converted_streamflow_dataset = streamflow_unit_conv(
                streamflow_dataset,
                self.data_source.read_area(self.t_s_dict["sites_id"]),
                target_unit=prcp_unit,
            )
            data_output_ds[self.streamflow_name] = converted_streamflow_dataset[
                self.streamflow_name
            ]
            streamflow_input_dataset = data_forcing_ds[[self.streamflow_input_name]]
            converted_streamflow_input_dataset = streamflow_unit_conv(
                streamflow_input_dataset,
                self.data_source.read_area(self.t_s_dict["sites_id"]),
                target_unit=prcp_unit,
            )
            data_forcing_ds[self.streamflow_input_name] = (
                converted_streamflow_input_dataset[self.streamflow_input_name]
            )

        return data_forcing_ds, data_output_ds

    def __getitem__(self, item: int):
        basin, idx, _ = self.lookup_table[item]
        warmup_length = self.warmup_length
        xf = self.x[
            basin,
            idx - warmup_length + 1 : idx + self.rho + self.horizon + 1,
            :-1,
        ]
        xq = self.x[basin, idx - warmup_length : idx + self.rho + self.horizon, -1]
        xq = xq.reshape(xq.size, 1)
        xf_rho = xf[: self.rho, :]
        xf_hor = xf[self.rho :, :]
        xq_rho = xq[: self.rho, :]
        xq_hor = xq[self.rho :, :]
        y = self.y[basin, idx + self.rho : idx + self.rho + self.horizon, :]
        c = self.c[basin, :]
        c_rho = np.repeat(c, xf_rho.shape[0], axis=0).reshape(c.shape[0], -1).T
        c_hor = np.repeat(c, xf_hor.shape[0], axis=0).reshape(c.shape[0], -1).T
        xfc_rho = np.concatenate((xf_rho, c_rho), axis=1)
        xfc_hor = np.concatenate((xf_hor, c_hor), axis=1)
        return [
            torch.from_numpy(xfc_rho).float(),
            torch.from_numpy(xfc_hor).float(),
            torch.from_numpy(xq_rho).float(),
            torch.from_numpy(xq_hor).float(),
        ], torch.from_numpy(y).float()


class FloodEventDataset(BaseDataset):
    """Dataset class for flood event detection and prediction tasks.

    This dataset is specifically designed to handle flood event data where
    flood_event column contains binary indicators (0 for normal, non-zero for flood).
    It automatically creates a flood_mask from the flood_event data for special
    loss computation purposes.

    The dataset reads data using SelfMadeHydroDataset from hydrodatasource,
    expecting CSV files with columns like: time, rain, inflow, flood_event.
    """

    def __init__(self, cfgs: dict, is_tra_val_te: str):
        """Initialize FloodEventDataset

        Parameters
        ----------
        cfgs : dict
            Configuration dictionary containing data_cfgs, training_cfgs, evaluation_cfgs
        is_tra_val_te : str
            One of 'train', 'valid', or 'test'
        """
        # Find flood_event column index for later processing
        target_cols = cfgs["data_cfgs"]["target_cols"]
        self.flood_event_idx = None
        for i, col in enumerate(target_cols):
            if "flood_event" in col.lower():
                self.flood_event_idx = i
                break

        if self.flood_event_idx is None:
            raise ValueError(
                "flood_event column not found in target_cols. Please ensure flood_event is included in the target columns."
            )
        super(FloodEventDataset, self).__init__(cfgs, is_tra_val_te)

    @property
    def noutputvar(self):
        """How many output variables in the dataset
        Used in evaluation.
        For flood datasets, the number of output variables is 2.
        But we don't need flood_mask in evaluation.

        Returns
        -------
        int
            number of variables
        """
        return len(self.data_cfgs["target_cols"]) - 1

    def _create_flood_mask(self, y):
        """Create flood mask from flood_event column

        Parameters
        ----------
        y : np.ndarray
            Target data with shape [seq_len, n_targets] containing flood_event column

        Returns
        -------
        np.ndarray
            Flood mask with shape [seq_len, 1] where 1 indicates flood event, 0 indicates normal
        """
        if self.flood_event_idx >= y.shape[1]:
            raise ValueError(
                f"flood_event_idx {self.flood_event_idx} exceeds target dimensions {y.shape[1]}"
            )

        # Extract flood_event column
        flood_events = y[:, self.flood_event_idx]

        # Create binary mask: 1 for flood (non-zero), 0 for normal (zero)
        no_flood_data = min(flood_events)
        flood_mask = (flood_events != no_flood_data).astype(np.float32)

        # Reshape to maintain dimension consistency
        flood_mask = flood_mask.reshape(-1, 1)

        return flood_mask

    def _create_lookup_table(self):
        """Create lookup table based on flood events with sliding window

        This method creates samples where:
        1. For each flood event sequence:
           - In training: use sliding window to generate samples with fixed length
           - In testing: use the entire flood event sequence as one sample with its actual length
        2. Each sample covers the full sequence length without internal structure division
        """
        lookup = []

        # Calculate total sample sequence length for training/validation
        sample_seqlen = self.warmup_length + self.rho + self.horizon

        for basin_idx in range(self.ngrid):
            # Get flood events for this basin
            flood_events = self.y_origin[basin_idx, :, self.flood_event_idx]

            # Find flood event sequences (consecutive non-zero values)
            flood_sequences = self._find_flood_sequences(flood_events)

            for seq_start, seq_end in flood_sequences:
                if self.is_new_batch_way:
                    # For test period, use the entire flood event sequence as one sample
                    # But we need to ensure the sample includes enough context (sample_seqlen)
                    flood_length = seq_end - seq_start + 1

                    # Calculate the start index to include enough context before the flood
                    # We want to include some data before the flood event starts
                    context_before = min(sample_seqlen - flood_length, seq_start)
                    context_before = max(context_before, 0)
                    # The actual start index should be early enough to provide context
                    actual_start = seq_start - context_before

                    # The total length should be at least sample_seqlen or the actual flood sequence length
                    total_length = max(sample_seqlen, flood_length + context_before)

                    # Ensure we don't exceed the data bounds
                    if actual_start + total_length > self.nt:
                        total_length = self.nt - actual_start

                    lookup.append((basin_idx, actual_start, total_length))
                else:
                    # For training, use sliding window approach
                    self._create_sliding_window_samples(
                        basin_idx, seq_start, seq_end, sample_seqlen, lookup
                    )

        self.lookup_table = dict(enumerate(lookup))
        self.num_samples = len(self.lookup_table)

    def _find_flood_sequences(self, flood_events):
        """Find sequences of consecutive flood events

        Parameters
        ----------
        flood_events : np.ndarray
            1D array of flood event indicators

        Returns
        -------
        list
            List of tuples (start_idx, end_idx) for each flood sequence
        """
        sequences = []
        in_sequence = False
        start_idx = None

        for i, event in enumerate(flood_events):
            if event > 0 and not in_sequence:
                # Start of a new flood sequence
                in_sequence = True
                start_idx = i
            elif event == 0 and in_sequence:
                # End of current flood sequence
                in_sequence = False
                sequences.append((start_idx, i - 1))
            elif i == len(flood_events) - 1 and in_sequence:
                # End of data while in sequence
                sequences.append((start_idx, i))

        return sequences

    def _create_sliding_window_samples(
        self, basin_idx, seq_start, seq_end, sample_seqlen, lookup
    ):
        """Create samples for a flood sequence using sliding window approach with data validity check

        Parameters
        ----------
        basin_idx : int
            Index of the basin
        seq_start : int
            Start index of flood sequence
        seq_end : int
            End index of flood sequence
        sample_seqlen : int
            Maximum length of each sample (warmup_length + rho + horizon)
        lookup : list
            List to append new samples to (basin_idx, actual_start, actual_length)
        """
        # Generate sliding window samples for this flood sequence
        # Each window should include at least some flood event data

        # Calculate the range where we can place the sliding window
        # The window end should not exceed the flood sequence end
        max_window_start = min(
            seq_end - sample_seqlen + 1, self.nt - sample_seqlen
        )  # Window end should not exceed seq_end or data bounds
        min_window_start = max(
            0, seq_start - sample_seqlen + 1
        )  # Window must include at least the first flood event

        # Ensure we have a valid range
        if max_window_start < min_window_start:
            return  # Skip this flood sequence if no valid window can be created

        # Generate samples with sliding window
        for window_start in range(min_window_start, max_window_start + 1):
            window_end = window_start + sample_seqlen - 1

            # Check if the window is valid (doesn't exceed data bounds and flood sequence)
            if window_end < self.nt and window_end <= seq_end:
                # Check if this window includes at least some flood events
                window_includes_flood = (window_start <= seq_end) and (
                    window_end >= seq_start
                )

                if window_includes_flood:
                    # Find the actual valid data range within this window closest to flood
                    actual_start, actual_length = self._find_valid_data_range(
                        basin_idx, window_start, window_end, seq_start, seq_end
                    )

                    # Only add sample if we have sufficient valid data
                    if (
                        actual_length >= self.rho + self.horizon
                    ):  # At least need rho + horizon
                        lookup.append((basin_idx, actual_start, actual_length))

    def _create_peak_aligned_samples(
        self, basin_idx, seq_start, seq_end, sample_seqlen, lookup
    ):
        """Create peak-aligned training samples for a flood sequence

        Core idea:
        1. Find flood peak t_peak within the sequence
        2. First sample: window right boundary = t_peak
           - Input: [t_peak - hindcast + 1, t_peak] (length: hindcast)
           - Output: [t_peak - hindcast + 1, t_peak + forecast] (length: hindcast + forecast)
        3. Slide window right boundary from t_peak to seq_end
        4. Last sample: window right boundary = seq_end
           - Input: [seq_end - hindcast + 1, seq_end] (length: hindcast)
           - Output: [seq_end - hindcast + 1, seq_end] (length: hindcast, no forecast)

        Parameters
        ----------
        basin_idx : int
            Basin index
        seq_start : int
            Start index of flood sequence
        seq_end : int
            End index of flood sequence
        sample_seqlen : int
            Sample sequence length (warmup_length + rho + horizon)
        lookup : list
            List to append new samples to (basin_idx, start, length)
        """
        # 1. Find flood peak
        peak_idx = self._find_flood_peak_in_sequence(basin_idx, seq_start, seq_end)

        if peak_idx is None:
            # Fallback to original sliding window logic
            print(
                f"Warning: Cannot find peak for basin {basin_idx}, "
                f"flood sequence [{seq_start}, {seq_end}], using fallback"
            )
            self._create_sliding_window_samples(
                basin_idx, seq_start, seq_end, sample_seqlen, lookup
            )
            return

        # 2. Slide window right boundary from peak to flood end
        # window_right slides from peak_idx to seq_end
        total_samples = 0

        for window_right in range(peak_idx, seq_end + 1, self.peak_slide_step):
            # Calculate window range
            # window_end = window_right + forecast
            # window_start = window_end - sample_seqlen + 1
            window_end = window_right + self.horizon
            start_idx = window_end - sample_seqlen + 1

            # Boundary checks
            if start_idx < 0:
                start_idx = 0
                window_end = start_idx + sample_seqlen - 1

            if window_end >= self.nt:
                # Can't extend beyond data bounds
                break

            # Check data validity (avoid NaN)
            actual_start, valid_length = self._find_valid_data_range(
                basin_idx, start_idx, window_end, seq_start, seq_end
            )

            # 只采样完全有效的样本（valid_length >= sample_seqlen）
            # 这样保持固定长度，避免验证时 torch.cat 维度不匹配
            if valid_length >= sample_seqlen:
                lookup.append((basin_idx, actual_start, sample_seqlen))
                total_samples += 1
    def _find_valid_data_range(
        self, basin_idx, window_start, window_end, flood_start, flood_end
    ):
        """Find the continuous valid data range closest to the flood sequence

        Parameters
        ----------
        basin_idx : int
            Basin index
        window_start : int
            Start of the window to check
        window_end : int
            End of the window to check
        flood_start : int
            Start index of the flood sequence
        flood_end : int
            End index of the flood sequence

        Returns
        -------
        tuple
            (actual_start, actual_length) of the valid data range closest to flood sequence
        """
        # Get data for this basin and window
        x_window = self.x[basin_idx, window_start : window_end + 1, :]

        # Check for NaN values in both input and output
        valid_mask = ~np.isnan(x_window).any(axis=1)  # Valid if no NaN in any feature

        # Find the continuous valid sequence closest to the flood sequence
        closest_start, closest_length = self._find_closest_valid_sequence(
            valid_mask, window_start, flood_start, flood_end
        )

        if closest_length <= 0:
            return window_start, 0
        return closest_start, closest_length

    def _find_closest_valid_sequence(
        self, valid_mask, window_start, flood_start, flood_end
    ):
        """Find the continuous valid sequence closest to the flood sequence

        Parameters
        ----------
        valid_mask : np.ndarray
            Boolean array indicating valid positions within the window
        window_start : int
            Start index of the window in the original time series
        flood_start : int
            Start index of the flood sequence in the original time series
        flood_end : int
            End index of the flood sequence in the original time series

        Returns
        -------
        tuple
            (closest_start, closest_length) in original time series coordinates
        """
        if not valid_mask.any():
            return window_start, 0

        # Find all continuous valid sequences within the window
        sequences = []
        current_start = None

        for i, is_valid in enumerate(valid_mask):
            if is_valid and current_start is None:
                current_start = i
            elif not is_valid and current_start is not None:
                sequences.append((current_start, i - current_start))
                current_start = None

        # Handle case where sequence continues to the end
        if current_start is not None:
            sequences.append((current_start, len(valid_mask) - current_start))

        if not sequences:
            return window_start, 0

        # If only one sequence, return it directly
        if len(sequences) == 1:
            seq_start_rel, seq_length = sequences[0]
            seq_start_abs = window_start + seq_start_rel
            return seq_start_abs, seq_length

        # Find the sequence closest to the flood sequence
        flood_center = (flood_start + flood_end) / 2
        closest_sequence = None
        min_distance = float("inf")

        for seq_start_rel, seq_length in sequences:
            seq_start_abs = window_start + seq_start_rel
            seq_end_abs = seq_start_abs + seq_length - 1
            seq_center = (seq_start_abs + seq_end_abs) / 2

            # Calculate distance from sequence center to flood center
            distance = abs(seq_center - flood_center)

            if distance < min_distance:
                min_distance = distance
                closest_sequence = (seq_start_abs, seq_length)

        return closest_sequence or (window_start, 0)

    def __getitem__(self, item: int):
        """Get one sample from the dataset with flood mask

        Returns samples with:
        1. Variable length sequences (no padding)
        2. Flood mask for weighted loss computation
        """
        basin, start_idx, actual_length = self.lookup_table[item]
        warmup_length = self.warmup_length
        end_idx = start_idx + actual_length

        # Get input and target data for the actual valid range
        x = self.x[basin, start_idx:end_idx, :]
        y = self.y[basin, start_idx + warmup_length : end_idx, :]

        # Create flood mask from flood_event column
        flood_mask = self._create_flood_mask(y)

        # Replace the original flood_event column with the new flood_mask
        y_with_flood_mask = y.copy()
        y_with_flood_mask[:, self.flood_event_idx] = flood_mask.squeeze()

        # Handle constant features if available
        if self.c is None or self.c.shape[-1] == 0:
            return (
                torch.from_numpy(x).float(),
                torch.from_numpy(y_with_flood_mask).float(),
            )

        # Add constant features to input
        c = self.c[basin, :]
        c = np.repeat(c, x.shape[0], axis=0).reshape(c.shape[0], -1).T
        xc = np.concatenate((x, c), axis=1)

        return torch.from_numpy(xc).float(), torch.from_numpy(y_with_flood_mask).float()


class FloodEventDplDataset(FloodEventDataset):
    """Dataset class for flood event detection and prediction with differential parameter learning support.

    This dataset combines FloodEventDataset's flood event handling capabilities with
    DplDataset's data format for differential parameter learning (dPL) models.
    It handles flood event sequences and returns data in the format required for
    physical hydrological models with neural network components.
    """

    def __init__(self, cfgs: dict, is_tra_val_te: str):
        """Initialize FloodEventDplDataset

        Parameters
        ----------
        cfgs : dict
            Configuration dictionary containing data_cfgs, training_cfgs, evaluation_cfgs
        is_tra_val_te : str
            One of 'train', 'valid', or 'test'
        """
        super(FloodEventDplDataset, self).__init__(cfgs, is_tra_val_te)

        # Additional attributes for DPL functionality
        self.target_as_input = self.data_cfgs["target_as_input"]
        self.constant_only = self.data_cfgs["constant_only"]

        if self.target_as_input and (not self.train_mode):
            # if the target is used as input and train_mode is False,
            # we need to get the target data in training period to generate pbm params
            self.train_dataset = FloodEventDplDataset(cfgs, is_tra_val_te="train")

    def __getitem__(self, item: int):
        """Get one sample from the dataset in DPL format with flood mask

        Returns data in the format required for differential parameter learning:
        - x_train: not normalized forcing data
        - z_train: normalized data for DL model (with flood mask)
        - y_train: not normalized output data

        Parameters
        ----------
        item : int
            Index of the sample

        Returns
        -------
        tuple
            ((x_train, z_train), y_train) where:
            - x_train: torch.Tensor, not normalized forcing data
            - z_train: torch.Tensor, normalized data for DL model
            - y_train: torch.Tensor, not normalized output data with flood mask
        """
        basin, start_idx, actual_length = self.lookup_table[item]
        end_idx = start_idx + actual_length
        warmup_length = self.warmup_length
        # Get normalized data first (using parent's logic for flood mask)
        xc_norm, y_norm_with_mask = super(FloodEventDplDataset, self).__getitem__(item)

        # Get original (not normalized) data
        x_origin = self.x_origin[basin, start_idx:end_idx, :]
        y_origin = self.y_origin[basin, start_idx + warmup_length : end_idx, :]

        # Create flood mask for original y data
        flood_mask_origin = self._create_flood_mask(y_origin)
        y_origin_with_mask = y_origin.copy()
        y_origin_with_mask[:, self.flood_event_idx] = flood_mask_origin.squeeze()

        # Prepare z_train based on configuration
        if self.target_as_input:
            # y_norm and xc_norm are concatenated and used for DL model
            # the order of xc_norm and y_norm matters, please be careful!
            z_train = torch.cat((xc_norm, y_norm_with_mask), -1)
        elif self.constant_only:
            # only use attributes data for DL model
            if self.c is None or self.c.shape[-1] == 0:
                # If no constant features, use a zero tensor
                z_train = torch.zeros((actual_length, 1)).float()
            else:
                c = self.c[basin, :]
                # Repeat constants for the actual sequence length
                c_repeated = (
                    np.repeat(c, actual_length, axis=0).reshape(c.shape[0], -1).T
                )
                z_train = torch.from_numpy(c_repeated).float()
        else:
            # Use normalized input features with constants
            z_train = xc_norm.float()

        # Prepare x_train (original forcing data with constants if available)
        if self.c is None or self.c.shape[-1] == 0:
            x_train = torch.from_numpy(x_origin).float()
        else:
            c = self.c_origin[basin, :]
            c_repeated = np.repeat(c, actual_length, axis=0).reshape(c.shape[0], -1).T
            x_origin_with_c = np.concatenate((x_origin, c_repeated), axis=1)
            x_train = torch.from_numpy(x_origin_with_c).float()

        # y_train is the original output data with flood mask
        y_train = torch.from_numpy(y_origin_with_mask).float()

        return (x_train, z_train), y_train


class GNNDataset(FloodEventDataset):
    """Optimized GNN Dataset for hydrological Graph Neural Network tasks.

    This dataset extends FloodEventDataset to support Graph Neural Networks by:
    1. Integrating station data via StationHydroDataset
    2. Processing adjacency matrices with flexible edge weight and attribute handling
    3. Merging basin-level features (xc) with station-level features (sxc) per node
    4. Returning GNN-ready format: (sxc, y, edge_index, edge_attr)

    Key Features:
    - Leverages BaseDataset's universal normalization and NaN handling for station data
    - Supports flexible edge weight selection (specify column or default to binary)
    - Always constructs edge_index and edge_attr for each basin
    - Merges basin and station features to create comprehensive node representations

    Configuration keys in data_cfgs.gnn_cfgs:
    - station_cols: List of station variable names to load
    - station_rm_nan: Whether to remove/interpolate NaN values (default: True)
    - station_scaler_type: Scaler type for station data normalization
    - use_adjacency: Whether to load adjacency matrices (default: True)
    - adjacency_src_col: Source node column name (default: "ID")
    - adjacency_dst_col: Destination node column name (default: "NEXTDOWNID")
    - adjacency_edge_attr_cols: Columns for edge attributes (default: ["dist_hdn", "elev_diff", "strm_slope"])
    - adjacency_weight_col: Column to use as edge weights (default: None for binary weights)
    - return_edge_weight: Whether to return edge_weight instead of edge_attr (default: False)

    edge_attr : torch.Tensor
        Edge attributes [num_edges, edge_attr_dim]
    """

    def __init__(self, cfgs: dict, is_tra_val_te: str):
        # Extract and extend configuration for station data
        self._extend_data_cfgs_for_stations(cfgs)

        # Store GNN-specific settings
        self.gnn_cfgs = cfgs["data_cfgs"].get("station_cfgs", {})

        # Initialize parent (this will call BaseDataset._load_data() automatically)
        super(GNNDataset, self).__init__(cfgs, is_tra_val_te)

        # Load adjacency data after main data processing
        self.adjacency_data = self._load_adjacency_data()

    def _extend_data_cfgs_for_stations(self, cfgs):
        """Extend data configuration to include station data as a standard data type

        This allows BaseDataset to handle station data using its universal processing pipeline.
        """
        data_cfgs = cfgs["data_cfgs"]
        gnn_cfgs = data_cfgs.get("station_cfgs", {})

        # Add station_cols to data configuration if specified（这个不见得非得有gnn_cfgs,正常应该是data_cfgs里面继续扩充的）
        if gnn_cfgs.get("station_cols"):
            data_cfgs["station_cols"] = gnn_cfgs["station_cols"]
            # Add station data processing settings to leverage BaseDataset pipeline
            data_cfgs["station_rm_nan"] = gnn_cfgs.get("station_rm_nan", True)

    def _read_xyc(self):
        """Read X, Y, C data including station data using unified approach

        This is the ONLY method we need to override from BaseDataset.
        All other processing (normalization, NaN handling, array conversion)
        is handled automatically by BaseDataset's pipeline.
        """
        # Read standard basin data using parent's logic
        data_dict = super(GNNDataset, self)._read_xyc()

        # Add station data if configured
        if self.data_cfgs.get("station_cols"):
            station_data = self._read_all_station_data()
            data_dict["station_cols"] = station_data

        return data_dict

    def _read_all_station_data(self):
        """Read station data for all basins using StationHydroDataset

        Creates xr.DataArray with the same structure as other data types
        so that BaseDataset can process it using the universal pipeline.
        """
        if not hasattr(self.data_source, "get_stations_by_basin"):
            LOGGER.warning(
                "Data source does not support station data, skipping station data reading"
            )
            return None

        # Convert basin IDs from "songliao_21100150" to "21100150" for StationHydroDataset
        basin_ids_with_prefix = self.t_s_dict["sites_id"]
        basin_ids = self._convert_basin_to_station_ids(basin_ids_with_prefix)
        t_range = self.t_s_dict["t_final_range"]

        # Collect station data for all basins
        all_station_data = []

        for basin_id in basin_ids:
            basin_station_data = self._read_basin_station_data(basin_id, t_range)
            all_station_data.append(basin_station_data)

        # Combine into unified xr.DataArray structure
        if all_station_data and any(data is not None for data in all_station_data):
            combined_station_data = self._combine_station_data_arrays(
                all_station_data, basin_ids
            )
            return combined_station_data
        else:
            return None

    def _read_basin_station_data(self, basin_id, t_range):
        """Read station data for a single basin, supporting multi-period case"""
        try:
            # Get stations for this basin
            station_ids = self.data_source.get_stations_by_basin(basin_id)

            if not station_ids:
                return None

            # Handle multi-period case
            if isinstance(t_range[0], (list, tuple)):
                # Validate that each period has exactly 2 elements (start and end date)
                for i, period in enumerate(t_range):
                    if not isinstance(period, (list, tuple)) or len(period) != 2:
                        raise ValueError(
                            f"Period {i} must be a list/tuple with exactly 2 elements (start_date, end_date), got: {period}"
                        )

                # Multi-period case - read and concatenate data
                all_station_data = None

                for start_date, end_date in t_range:
                    period_station_data = self.data_source.read_station_ts_xrdataset(
                        station_id_lst=station_ids,
                        t_range=[start_date, end_date],
                        var_lst=self.data_cfgs["station_cols"],
                        time_units=self.gnn_cfgs.get("station_time_units", ["1D"]),
                    )

                    if all_station_data is None:
                        all_station_data = period_station_data
                    else:
                        all_station_data = xr.concat(
                            [all_station_data, period_station_data], dim="time"
                        )

                station_data = all_station_data
            else:
                # Single period case (existing behavior)
                station_data = self.data_source.read_station_ts_xrdataset(
                    station_id_lst=station_ids,
                    t_range=t_range,
                    var_lst=self.data_cfgs["station_cols"],
                    time_units=self.gnn_cfgs.get("station_time_units", ["1D"]),
                )

            return self._process_station_xr_data(station_data)

        except Exception as e:
            LOGGER.warning(f"Could not read station data for basin {basin_id}: {e}")
            return None

    def _process_station_xr_data(self, station_data):
        """Process xarray station data into standard format"""
        if not station_data:
            return None

        # Handle multiple time units
        if isinstance(station_data, dict):
            # Use first available time unit
            time_unit = list(station_data.keys())[0]
            station_ds = station_data[time_unit]
        else:
            station_ds = station_data

        if not station_ds or not station_ds.sizes:
            return None

        # Convert to DataArray with standard format
        if isinstance(station_ds, xr.Dataset):
            station_da = station_ds.to_array(dim="variable")
            # Transpose to [time, station, variable]
            station_da = station_da.transpose("time", "station", "variable")
        else:
            station_da = station_ds

        return station_da

    def _combine_station_data_arrays(self, station_data_list, basin_ids):
        """Combine station data from all basins into a unified structure

        Creates an xr.DataArray with dimensions [basin, time, station, variable]
        similar to how other data types are structured in BaseDataset.
        """
        # Find common time dimension and data structure
        valid_data = [data for data in station_data_list if data is not None]
        if not valid_data:
            return None

        # Use time dimension from first valid dataset
        common_time = valid_data[0].coords["time"]

        # Find maximum number of stations and variables across all basins
        max_stations = max(data.sizes.get("station", 0) for data in valid_data)
        max_variables = max(data.sizes.get("variable", 0) for data in valid_data)

        # Create unified data array
        n_basins = len(basin_ids)
        n_time = len(common_time)

        # Initialize with NaN (BaseDataset will handle NaN processing)
        unified_data = np.full(
            (n_basins, n_time, max_stations, max_variables), np.nan, dtype=np.float32
        )

        # Fill with actual data
        for i, (basin_id, station_data) in enumerate(zip(basin_ids, station_data_list)):
            if station_data is not None:
                # Align time dimension
                try:
                    aligned_data = station_data.reindex(
                        time=common_time, method="nearest"
                    )
                    data_array = aligned_data.values

                    # Insert into unified array
                    n_stations_basin = data_array.shape[1]
                    n_vars_basin = data_array.shape[2]
                    unified_data[i, :, :n_stations_basin, :n_vars_basin] = data_array
                except Exception as e:
                    LOGGER.warning(
                        f"Failed to align station data for basin {basin_id}: {e}"
                    )
                    continue

        # Create xr.DataArray with proper coordinates
        station_coords = [f"station_{j}" for j in range(max_stations)]
        variable_coords = self.data_cfgs["station_cols"][:max_variables]

        station_da = xr.DataArray(
            unified_data,
            dims=["basin", "time", "station", "variable"],
            coords={
                "basin": basin_ids,
                "time": common_time,
                "station": station_coords,
                "variable": variable_coords,
            },
        )

        return station_da

    def _load_adjacency_data(self):
        """Load and process adjacency data from .nc files

        Returns
        -------
        dict
            Dictionary containing edge_index, edge_attr for each basin
        """
        if not self.gnn_cfgs.get("use_adjacency", True):
            return None

        if not hasattr(self.data_source, "read_adjacency_xrdataset"):
            LOGGER.warning("Data source does not support adjacency data")
            return None

        adjacency_data = {}
        # basin_ids = self.t_s_dict["sites_id"]
        # Convert basin IDs from "songliao_21100150" to "21100150" for StationHydroDataset
        basin_ids_with_prefix = self.t_s_dict["sites_id"]
        basin_ids = self._convert_basin_to_station_ids(basin_ids_with_prefix)

        for basin_id in basin_ids:
            try:
                # Read adjacency data from .nc file
                adj_df = self.data_source.read_adjacency_xrdataset(basin_id)

                if adj_df is None:
                    LOGGER.warning(
                        f"No adjacency data for basin {basin_id}, using self-loops"
                    )
                    adjacency_data[basin_id] = self._create_self_loop_adjacency(
                        basin_id
                    )
                else:
                    # Let _process_adjacency_dataframe handle the format checking and processing
                    adjacency_data[basin_id] = self._process_adjacency_dataframe(
                        adj_df, basin_id
                    )

            except Exception as e:
                LOGGER.warning(
                    f"Failed to load adjacency data for basin {basin_id}: {e}"
                )
                adjacency_data[basin_id] = self._create_self_loop_adjacency(basin_id)

        return adjacency_data

    def _process_adjacency_dataframe(self, adj_df, basin_id):
        """Process adjacency DataFrame into edge_index and edge_attr tensors

        Standard GNN processing: extract edges and their attributes from DataFrame or xarray Dataset.

        Parameters
        ----------
        adj_df : pd.DataFrame or xr.Dataset
            Adjacency DataFrame/Dataset with columns like ID, NEXTDOWNID, dist_hdn, elev_diff, strm_slope
        basin_id : str
            Basin identifier

        Returns
        -------
        dict
            Dictionary containing edge_index, edge_attr, edge_weight, num_nodes
        """
        import torch
        import pandas as pd
        import xarray as xr
        import numpy as np

        # Convert xarray Dataset to pandas DataFrame if needed
        if isinstance(adj_df, xr.Dataset):
            try:
                # Convert xarray Dataset to pandas DataFrame
                adj_df = adj_df.to_dataframe().reset_index()
                # LOGGER.info(f"Basin {basin_id}: Converted xarray Dataset to DataFrame with shape {adj_df.shape}")
                # LOGGER.info(f"Basin {basin_id}: DataFrame columns = {list(adj_df.columns)}")
            except Exception as e:
                LOGGER.error(
                    f"Basin {basin_id}: Failed to convert xarray Dataset to DataFrame: {e}"
                )
                return self._create_self_loop_adjacency(basin_id)

        # Configuration (simplified)
        src_col = self.gnn_cfgs.get("adjacency_src_col", "ID")
        dst_col = self.gnn_cfgs.get("adjacency_dst_col", "NEXTDOWNID")
        edge_attr_cols = self.gnn_cfgs.get(
            "adjacency_edge_attr_cols", ["dist_hdn", "elev_diff", "strm_slope"]
        )
        weight_col = self.gnn_cfgs.get("adjacency_weight_col", None)  # 新增：指定权重列
        # Check if required columns exist
        if src_col not in adj_df.columns:
            LOGGER.warning(
                f"Basin {basin_id}: Source column '{src_col}' not found in adjacency data. Available columns: {list(adj_df.columns)}"
            )
            return self._create_self_loop_adjacency(basin_id)

        if dst_col not in adj_df.columns:
            LOGGER.warning(
                f"Basin {basin_id}: Destination column '{dst_col}' not found in adjacency data. Available columns: {list(adj_df.columns)}"
            )
            return self._create_self_loop_adjacency(basin_id)

        # Clean and convert numeric columns to proper dtypes in batch
        # Handle string "nan" values that may come from NetCDF files
        numeric_cols = [
            col
            for col in edge_attr_cols + ([weight_col] if weight_col else [])
            if col in adj_df.columns
        ]
        if numeric_cols:
            # Batch replace string "nan" with actual NaN and convert to numeric
            adj_df[numeric_cols] = adj_df[numeric_cols].replace(
                ["nan", "NaN", "NAN"], np.nan
            )
            adj_df[numeric_cols] = adj_df[numeric_cols].apply(
                pd.to_numeric, errors="coerce"
            )
            LOGGER.debug(
                f"Basin {basin_id}: Converted {len(numeric_cols)} numeric columns in batch"
            )

        # Create comprehensive node mapping including all stations in the basin
        # First get all nodes that appear in adjacency matrix (connected nodes)
        connected_nodes = set(adj_df[src_col].dropna()) | set(adj_df[dst_col].dropna())

        # Then get all stations in this basin (including isolated nodes)
        try:
            if hasattr(self.data_source, "get_stations_by_basin"):
                all_basin_stations = self.data_source.get_stations_by_basin(basin_id)
                if all_basin_stations:
                    # Convert station IDs to strings to match adjacency data format
                    all_basin_nodes = set(
                        str(station_id) for station_id in all_basin_stations
                    )
                    # Combine connected nodes with all basin nodes
                    all_nodes = connected_nodes | all_basin_nodes
                    isolated_nodes = all_basin_nodes - connected_nodes
                    if isolated_nodes:
                        LOGGER.info(
                            f"Basin {basin_id}: Found {len(isolated_nodes)} isolated nodes: {isolated_nodes}"
                        )
                else:
                    all_nodes = connected_nodes
            else:
                # Fallback to only connected nodes if station data unavailable
                all_nodes = connected_nodes
        except Exception as e:
            LOGGER.warning(
                f"Basin {basin_id}: Failed to get all basin stations: {e}, using connected nodes only"
            )
            all_nodes = connected_nodes

        if len(all_nodes) == 0:
            LOGGER.warning(f"Basin {basin_id}: No valid nodes found")
            return self._create_self_loop_adjacency(basin_id)

        node_to_idx = {node: idx for idx, node in enumerate(sorted(all_nodes))}
        LOGGER.info(
            f"Basin {basin_id}: Found {len(all_nodes)} total nodes ({len(connected_nodes)} connected, {len(all_nodes) - len(connected_nodes)} isolated)"
        )

        # Extract edges and attributes using vectorized operations
        # First process edges from adjacency matrix
        valid_rows = adj_df.dropna(subset=[src_col, dst_col])
        edges_from_adj = []
        edge_attrs_from_adj = []
        edge_weights_from_adj = []

        if len(valid_rows) > 0:
            # Vectorized edge creation from adjacency matrix
            src_nodes = valid_rows[src_col].map(node_to_idx).values
            dst_nodes = valid_rows[dst_col].map(node_to_idx).values
            edges_from_adj = np.column_stack([src_nodes, dst_nodes])

            # Vectorized edge attributes extraction
            edge_attrs_list = []
            for col in edge_attr_cols:
                if col in valid_rows.columns:
                    attrs = valid_rows[col].fillna(0.0).values
                else:
                    attrs = np.zeros(len(valid_rows))
                edge_attrs_list.append(attrs)
            edge_attrs_from_adj = (
                np.column_stack(edge_attrs_list)
                if edge_attrs_list
                else np.zeros((len(valid_rows), len(edge_attr_cols)))
            )

            # Vectorized edge weights extraction
            if weight_col and weight_col in valid_rows.columns:
                edge_weights_from_adj = valid_rows[weight_col].fillna(1.0).values
            else:
                edge_weights_from_adj = np.ones(len(valid_rows))

        # Add self-loops for isolated nodes (nodes not in adjacency matrix)
        isolated_nodes = all_nodes - connected_nodes
        edges_from_isolated = []
        edge_attrs_from_isolated = []
        edge_weights_from_isolated = []

        if isolated_nodes:
            # Create self-loops for isolated nodes
            isolated_indices = [node_to_idx[node] for node in isolated_nodes]
            edges_from_isolated = np.column_stack([isolated_indices, isolated_indices])
            edge_attrs_from_isolated = np.zeros(
                (len(isolated_nodes), len(edge_attr_cols))
            )
            edge_weights_from_isolated = np.ones(len(isolated_nodes))

        # Combine edges from adjacency matrix and self-loops for isolated nodes
        if len(edges_from_adj) > 0 and len(edges_from_isolated) > 0:
            all_edges = np.vstack([edges_from_adj, edges_from_isolated])
            all_edge_attrs = np.vstack([edge_attrs_from_adj, edge_attrs_from_isolated])
            all_edge_weights = np.concatenate(
                [edge_weights_from_adj, edge_weights_from_isolated]
            )
        elif len(edges_from_adj) > 0:
            all_edges = edges_from_adj
            all_edge_attrs = edge_attrs_from_adj
            all_edge_weights = edge_weights_from_adj
        elif len(edges_from_isolated) > 0:
            all_edges = edges_from_isolated
            all_edge_attrs = edge_attrs_from_isolated
            all_edge_weights = edge_weights_from_isolated
        else:
            # Fallback: create self-loops for all nodes
            # LOGGER.warning(f"Basin {basin_id}: No edges found, creating self-loops for all nodes")
            n_nodes = len(all_nodes)
            node_indices = list(range(n_nodes))
            all_edges = np.column_stack([node_indices, node_indices])
            all_edge_attrs = np.zeros((n_nodes, len(edge_attr_cols)))
            all_edge_weights = np.ones(n_nodes)

        # Convert to tensors
        edge_index = torch.tensor(all_edges.T, dtype=torch.long).contiguous()
        edge_attr = (
            torch.tensor(all_edge_attrs, dtype=torch.float)
            if all_edge_attrs is not None
            else None
        )
        edge_weight = torch.tensor(all_edge_weights, dtype=torch.float)

        return {
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "edge_weight": edge_weight,  # 新增：单独的边权重张量
            "num_nodes": len(all_nodes),
            "node_to_idx": node_to_idx,
            "weight_col": weight_col,  # 记录使用的权重列
        }

    def _create_self_loop_adjacency(self, basin_id):
        """Create self-loop adjacency as fallback"""
        import torch

        try:
            # Try to get station count for this basin
            if hasattr(self.data_source, "get_stations_by_basin"):
                station_ids = self.data_source.get_stations_by_basin(basin_id)
                n_nodes = len(station_ids) if station_ids else 1
            else:
                n_nodes = 1
        except Exception:
            n_nodes = 1

        # Create self-loops: edge_index = [[0,1,2,...], [0,1,2,...]]
        edge_index = torch.arange(n_nodes).repeat(2, 1)

        # Create default edge attributes
        edge_attr_cols = self.gnn_cfgs.get(
            "adjacency_edge_attr_cols", ["dist_hdn", "elev_diff", "strm_slope"]
        )
        if edge_attr_cols:
            edge_attr = torch.zeros((n_nodes, len(edge_attr_cols)), dtype=torch.float)
        else:
            edge_attr = None

        # Create default edge weights (1.0 for self-loops)
        edge_weight = torch.ones(n_nodes, dtype=torch.float)

        return {
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "edge_weight": edge_weight,  # 新增：边权重
            "num_nodes": n_nodes,
            "node_to_idx": {i: i for i in range(n_nodes)},
            "weight_col": None,  # 自环情况下没有指定权重列
        }

    # GNN-specific utility methods
    def get_station_data(self, basin_idx):
        """Get station data for a specific basin

        Since station data is now processed by BaseDataset pipeline,
        it's available as self.station_cols (converted to numpy array).
        """
        if hasattr(self, "station_cols") and self.station_cols is not None:
            return self.station_cols[basin_idx]
        return None

    def get_adjacency_data(self, basin_idx):
        """Get adjacency data for a specific basin

        Returns
        -------
        dict or None
            Dictionary containing edge_index, edge_attr, edge_weight, etc. or None
        """
        if self.adjacency_data is None:
            return None

        # Get the specific basin ID for this basin index
        basin_id_with_prefix = self.t_s_dict["sites_id"][basin_idx]
        # Convert single basin ID to station ID (without prefix)
        basin_id = self._convert_basin_to_station_ids([basin_id_with_prefix])[0]
        return self.adjacency_data.get(basin_id)

    def get_edge_weight(self, basin_idx):
        """Get edge weights for a specific basin

        Parameters
        ----------
        basin_idx : int
            Basin index

        Returns
        -------
        torch.Tensor or None
            Edge weights tensor [num_edges] or None
        """
        adjacency_data = self.get_adjacency_data(basin_idx)
        if adjacency_data is not None:
            return adjacency_data.get("edge_weight")
        return None

    def _convert_basin_to_station_ids(self, basin_ids):
        """Convert basin IDs (with prefix) to station IDs (without prefix) for StationHydroDataset

        Parameters
        ----------
        basin_ids : list
            List of basin IDs with prefix (e.g., ["songliao_21100150"])

        Returns
        -------
        list
            List of station IDs without prefix (e.g., ["21100150"])
        """
        station_ids = []
        for basin_id in basin_ids:
            # Remove common prefixes
            if "_" in basin_id:
                # Extract the part after the last underscore
                station_id = basin_id.split("_")[-1]
            else:
                # If no underscore, use the original ID
                station_id = basin_id
            station_ids.append(station_id)
        return station_ids

    def _convert_station_to_basin_ids(self, station_ids, prefix="songliao"):
        """Convert station IDs (without prefix) to basin IDs (with prefix) for consistency

        Parameters
        ----------
        station_ids : list
            List of station IDs without prefix (e.g., ["21100150"])
        prefix : str
            Prefix to add (default: "songliao")

        Returns
        -------
        list
            List of basin IDs with prefix (e.g., ["songliao_21100150"])
        """
        basin_ids = []
        for station_id in station_ids:
            basin_id = f"{prefix}_{station_id}"
            basin_ids.append(basin_id)
        return basin_ids

    def __getitem__(self, item: int):
        """Get one sample with GNN-specific data format.

        This method merges basin-level features (xc) into each station node's
        features (sxc), so each node's input includes both station and basin
        attributes.

        Args:
            item: The index of the sample to retrieve.

        Returns:
            A tuple of (sxc, y, edge_index, edge_weight) where:
            - sxc: Station features merged with basin features.
                   Shape: [num_stations, seq_length, feature_dim]
            - y: Target values for prediction.
                 Shape: [forecast_length, output_dim]
            - edge_index: Edge connectivity.
                          Shape: [2, num_edges]
            - edge_weight: Edge weights.
                           Shape: [num_edges]
        """
        import torch
        import numpy as np

        # Get basic sample from parent (includes flood mask if FloodEventDataset)
        basic_sample = super(GNNDataset, self).__getitem__(item)

        # Extract x, y from parent's output
        if isinstance(basic_sample, tuple):
            x, y = (
                basic_sample  # x: [seq_length, x_feature_dim], y_full: [full_length, y_feature_dim]
            )
        elif isinstance(basic_sample, dict):
            x = basic_sample.get("x")
            y = basic_sample.get("y")
        else:
            raise ValueError(f"Unexpected basic_sample format: {type(basic_sample)}")

        # Get sample metadata
        basin, time_idx, actual_length = self.lookup_table[item]

        # For GNN prediction, we only need the forecast part of y as target
        # The structure should be: warmup + hindcast (rho) + forecast (horizon)
        # We only predict the forecast (horizon) part

        # Get station data for current basin and time window
        station_data = self.get_station_data(basin)  # [time, station, variable]
        adjacency_data = self.get_adjacency_data(basin)

        # Extract station data for the time window (input sequence)
        if station_data is not None:
            # For station data, we need the input sequence (not just forecast part)
            seq_end = time_idx + actual_length
            sxc_raw = station_data[
                time_idx:seq_end
            ]  # [seq_length, num_stations, station_feature_dim]
        else:
            # If no station data, create dummy station data
            LOGGER.warning(
                f"No station data for basin {basin}, using single dummy station"
            )
            dummy_station_features = 1  # Number of dummy features
            sxc_raw = np.zeros(
                (actual_length, 1, dummy_station_features)
            )  # [seq_length, 1, 1]

        # Get basin-level features (xc) for merging
        # x contains basin-level features, we need to replicate it for each station
        if x is not None and x.ndim >= 2:
            xc = x  # [seq_length, basin_feature_dim]
            basin_feature_dim = xc.shape[-1]
            seq_length, num_stations, station_feature_dim = sxc_raw.shape

            # Replicate basin features for each station and concatenate with station features
            # xc expanded: [seq_length, 1, basin_feature_dim] -> [seq_length, num_stations, basin_feature_dim]
            xc_expanded = np.tile(xc[:, np.newaxis, :], (1, num_stations, 1))

            # Concatenate station features with basin features
            # sxc_temp: [seq_length, num_stations, station_feature_dim + basin_feature_dim]
            sxc_temp = np.concatenate([sxc_raw, xc_expanded], axis=-1)

            # Transpose to get desired shape: [num_stations, seq_length, feature_dim]
            sxc = sxc_temp.transpose(1, 0, 2)
        else:
            # If no basin features, use only station features and transpose
            # sxc: [num_stations, seq_length, station_feature_dim]
            sxc = sxc_raw.transpose(1, 0, 2)

        # Process adjacency data (GNN edge orientation handled here)
        # Edge orientation logic: support 'upstream', 'downstream', 'bidirectional' (default: downstream)
        edge_orientation = self.gnn_cfgs.get("edge_orientation", "downstream")
        if adjacency_data is not None:
            edge_index = adjacency_data["edge_index"]  # [2, num_edges]
            edge_attr = adjacency_data[
                "edge_attr"
            ]  # [num_edges, edge_attr_dim] or None
            edge_weight = adjacency_data.get("edge_weight")  # [num_edges]
            # If edge_weight is None, fill with ones (all edges weight=1)
            if edge_weight is None:
                num_edges = edge_index.shape[1]
                edge_weight = torch.ones(num_edges, dtype=torch.float)

            # Edge orientation handling
            # 原始边: src_col(ID) → dst_col(NEXTDOWNID) = 上游 → 下游
            # downstream(默认): 边指向下游 = 保持原始方向, root=出口节点
            # upstream: 边指向上游 = 反转方向, root=最上游源头节点
            if edge_orientation == 'upstream':
                # Reverse all edges: swap source and target (变为 下游→上游)
                edge_index = edge_index[[1, 0], :]
            elif edge_orientation == "bidirectional":
                # Add reversed edges to make bidirectional
                edge_index_rev = edge_index[[1, 0], :]
                edge_index = torch.cat([edge_index, edge_index_rev], dim=1)
                if edge_attr is not None:
                    edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
                if edge_weight is not None:
                    edge_weight = torch.cat([edge_weight, edge_weight], dim=0)
            # else: downstream (default), keep original upstream→downstream

        else:
            # Default: self-loops for each station
            num_stations = sxc.shape[
                0
            ]  # Now sxc is [num_stations, seq_length, feature_dim]
            edge_index = torch.arange(num_stations).repeat(2, 1)
            edge_attr = None
            edge_weight = torch.ones(num_stations, dtype=torch.float)  # 默认权重为1

        # Ensure edge_attr has proper shape
        if edge_attr is None:
            num_edges = edge_index.shape[1]
            edge_attr_dim = len(
                self.gnn_cfgs.get(
                    "adjacency_edge_attr_cols", ["dist_hdn", "elev_diff", "strm_slope"]
                )
            )
            edge_attr = torch.zeros((num_edges, edge_attr_dim), dtype=torch.float)

        # Convert to tensors if needed
        if not isinstance(sxc, torch.Tensor):
            sxc = torch.tensor(sxc, dtype=torch.float)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float)

        return sxc, y, edge_index, edge_weight  # edge_attr


class AugmentedFloodEventDataset(FloodEventDataset):
    """Dataset class for augmented flood event data with discontinuous time ranges.

    This dataset is designed to handle flood event data that includes augmented
    (generated) future data alongside historical data, where time ranges may be
    discontinuous (e.g., historical data 1990-2010, then augmented data 2026+).

    It connects to hydrodatasource.reader.floodevent.FloodEventDatasource
    and uses the read_ts_xrdataset_augmented method to read augmented data.
    """

    def __init__(self, cfgs: dict, is_tra_val_te: str):
        """Initialize AugmentedFloodEventDataset

        Parameters
        ----------
        cfgs : dict
            Configuration dictionary containing data_cfgs, training_cfgs, evaluation_cfgs
        is_tra_val_te : str
            One of 'train', 'valid', or 'test'
        """
        super(AugmentedFloodEventDataset, self).__init__(cfgs, is_tra_val_te)

        if not hasattr(self.data_source, "read_ts_xrdataset_augmented"):
            raise ValueError(
                "Data source must support read_ts_xrdataset_augmented method"
            )

    def _read_xyc_period(self, start_date, end_date):
        """Override template method to read augmented flood event data for a specific period

        This method leverages the parent class's multi-period handling while using
        augmented data reading methods for generated future data.

        Parameters
        ----------
        start_date : str
            start time
        end_date : str
            end time

        Returns
        -------
        dict
            Dictionary containing relevant_cols, target_cols, and constant_cols data
        """
        return self._read_xyc_specified_time_augmented(start_date, end_date)

    def _read_xyc_specified_time_augmented(self, start_date, end_date):
        """Read x, y, c data from both historical and augmented data sources

        This method reads both historical observed data (using read_ts_xrdataset)
        and augmented future data (using read_ts_xrdataset_augmented), then
        concatenates them along the time dimension to provide a complete dataset.

        Parameters
        ----------
        start_date : str
            start time
        end_date : str
            end time

        Returns
        -------
        dict
            Dictionary containing relevant_cols, target_cols, and constant_cols data
        """

        # Read historical observed data using standard method
        relevant_cols = self.data_cfgs.get("relevant_cols", ["rain"])
        target_cols = self.data_cfgs.get("target_cols", ["inflow", "flood_event"])

        try:
            data_forcing_hist_ = self.data_source.read_ts_xrdataset(
                self.t_s_dict["sites_id"],
                [start_date, end_date],
                relevant_cols,
            )
            data_output_hist_ = self.data_source.read_ts_xrdataset(
                self.t_s_dict["sites_id"],
                [start_date, end_date],
                target_cols,
            )

            # Process historical data
            data_forcing_hist_ = self._rm_timeunit_key(data_forcing_hist_)
            data_output_hist_ = self._rm_timeunit_key(data_output_hist_)

        except Exception as e:
            LOGGER.info(f"无法读取历史数据，可能是时间范围不在历史数据中: {e}")
            data_forcing_hist_ = None
            data_output_hist_ = None

        # Read augmented data using augmented method
        try:
            data_forcing_aug_ = self.data_source.read_ts_xrdataset_augmented(
                self.t_s_dict["sites_id"],
                [start_date, end_date],
                relevant_cols,
            )
            data_output_aug_ = self.data_source.read_ts_xrdataset_augmented(
                self.t_s_dict["sites_id"],
                [start_date, end_date],
                target_cols,
            )

            # Process augmented data
            data_forcing_aug_ = self._rm_timeunit_key(data_forcing_aug_)
            data_output_aug_ = self._rm_timeunit_key(data_output_aug_)

        except Exception as e:
            LOGGER.info(f"无法读取增强数据，可能是时间范围不在增强数据中: {e}")
            data_forcing_aug_ = None
            data_output_aug_ = None

        # Combine historical and augmented data
        data_forcing_ds = self._combine_historical_and_augmented_data(
            data_forcing_hist_, data_forcing_aug_, "forcing"
        )
        data_output_ds = self._combine_historical_and_augmented_data(
            data_output_hist_, data_output_aug_, "target"
        )

        # Check and process combined data
        data_forcing_ds, data_output_ds = self._check_ts_xrds_unit(
            data_forcing_ds, data_output_ds
        )

        # Read constant/attribute data (same as parent class)
        data_attr_ds = self.data_source.read_attr_xrdataset(
            self.t_s_dict["sites_id"],
            self.data_cfgs["constant_cols"],
            all_number=True,
        )

        # Convert to DataArray with units
        x_origin, y_origin, c_origin = self._to_dataarray_with_unit(
            data_forcing_ds, data_output_ds, data_attr_ds
        )

        return {
            "relevant_cols": x_origin.transpose("basin", "time", "variable"),
            "target_cols": y_origin.transpose("basin", "time", "variable"),
            "constant_cols": (
                c_origin.transpose("basin", "variable")
                if c_origin is not None
                else None
            ),
        }

    def _combine_historical_and_augmented_data(self, hist_data, aug_data, data_type):
        """Combine historical observed data and augmented generated data

        This method concatenates historical and augmented data along the time dimension,
        handling cases where data may be discontinuous or overlapping.

        Parameters
        ----------
        hist_data : xr.Dataset or None
            Historical observed data
        aug_data : xr.Dataset or None
            Augmented generated data
        data_type : str
            Type of data ("forcing" or "target") for logging purposes

        Returns
        -------
        xr.Dataset
            Combined dataset with historical and augmented data concatenated
        """
        import xarray as xr

        # Handle cases where one or both data sources are None
        if hist_data is None and aug_data is None:
            raise ValueError(f"Both historical and augmented {data_type} data are None")
        elif hist_data is None:
            LOGGER.info(
                f"No historical {data_type} data found, using only augmented data"
            )
            return aug_data
        elif aug_data is None:
            LOGGER.info(
                f"No augmented {data_type} data found, using only historical data"
            )
            return hist_data

        # Both datasets exist - need to combine them
        try:
            # Check if there's time overlap between datasets
            hist_times = hist_data.time.values if "time" in hist_data.dims else []
            aug_times = aug_data.time.values if "time" in aug_data.dims else []

            if len(hist_times) == 0:
                LOGGER.info(
                    f"Historical {data_type} data has no time dimension, using only augmented data"
                )
                return aug_data
            elif len(aug_times) == 0:
                LOGGER.info(
                    f"Augmented {data_type} data has no time dimension, using only historical data"
                )
                return hist_data

            # Find overlap period
            hist_start, hist_end = hist_times[0], hist_times[-1]
            aug_start, aug_end = aug_times[0], aug_times[-1]

            # Check for overlap
            if hist_end < aug_start:
                # No overlap - historical data ends before augmented data starts
                LOGGER.info(
                    f"No temporal overlap for {data_type} data, concatenating sequentially"
                )
                combined_data = xr.concat([hist_data, aug_data], dim="time")
            elif aug_end < hist_start:
                # No overlap - augmented data ends before historical data starts
                LOGGER.info(
                    f"Augmented {data_type} data precedes historical data, concatenating"
                )
                combined_data = xr.concat([aug_data, hist_data], dim="time")
            else:
                # There is overlap - need to handle carefully
                LOGGER.info(
                    f"Temporal overlap detected for {data_type} data, "
                    f"merging with priority to historical data"
                )

                # Create time index for the full range
                all_times = sorted(set(list(hist_times) + list(aug_times)))

                # Reindex both datasets to the full time range
                hist_reindexed = hist_data.reindex(time=all_times, method=None)
                aug_reindexed = aug_data.reindex(time=all_times, method=None)

                # Combine: use historical data where available, fill with augmented data
                combined_data = hist_reindexed.where(
                    ~hist_reindexed.isnull(), aug_reindexed
                )

            # Sort by time to ensure proper ordering
            combined_data = combined_data.sortby("time")

            LOGGER.info(
                f"Successfully combined {data_type} data: "
                f"historical shape {hist_data.dims if hasattr(hist_data, 'dims') else 'N/A'}, "
                f"augmented shape {aug_data.dims if hasattr(aug_data, 'dims') else 'N/A'}, "
                f"combined shape {combined_data.dims}"
            )

            return combined_data

        except Exception as e:
            LOGGER.error(f"Failed to combine {data_type} data: {e}")
            # Fallback: prefer historical data if combination fails
            LOGGER.warning(f"Falling back to historical {data_type} data only")
            return hist_data

    def _handle_discontinuous_time_ranges(self, data_dict, start_date, end_date):
        """Handle discontinuous time ranges by filling gaps with NaN values

        This method creates a continuous time index and fills missing periods
        with NaN values, handling cases such as training data covers 1990-2010,
        augmented data starts from 2026+, and test data covers 2011-2025.

        Parameters
        ----------
        data_dict : dict
            Dictionary containing xarray data with keys 'relevant_cols',
            'target_cols', 'constant_cols'
        start_date : str
            Overall start date for the continuous timeline
        end_date : str
            Overall end date for the continuous timeline

        Returns
        -------
        dict
            Dictionary with continuous time index and NaN-filled gaps
        """
        # Create continuous daily time index from start_date to end_date
        try:
            continuous_time = pd.date_range(start=start_date, end=end_date, freq="D")
        except Exception as e:
            LOGGER.warning(f"Failed to create continuous time index: {e}")
            return data_dict

        # Process each data type (relevant_cols, target_cols)
        processed_dict = {}

        for data_key in ["relevant_cols", "target_cols"]:
            if data_key in data_dict and data_dict[data_key] is not None:
                original_data = data_dict[data_key]

                try:
                    # Check if data has time dimension
                    if "time" not in original_data.dims:
                        LOGGER.warning(
                            f"{data_key} has no time dimension, skipping time alignment"
                        )
                        processed_dict[data_key] = original_data
                        continue

                    # Reindex to continuous time, filling gaps with NaN
                    aligned_data = original_data.reindex(
                        time=continuous_time,
                        method=None,  # No interpolation, fill with NaN
                        fill_value=float("nan"),
                    )

                    processed_dict[data_key] = aligned_data

                    # Log information about the alignment
                    original_time_points = len(original_data.time)
                    aligned_time_points = len(aligned_data.time)
                    nan_points = aligned_data.isnull().sum().sum().values

                    LOGGER.info(
                        f"{data_key}: aligned from {original_time_points} to "
                        f"{aligned_time_points} time points, with {nan_points} "
                        f"NaN values for discontinuous periods"
                    )

                except Exception as e:
                    LOGGER.error(
                        f"Failed to align {data_key} to continuous timeline: {e}"
                    )
                    processed_dict[data_key] = original_data
            else:
                processed_dict[data_key] = data_dict.get(data_key)

        # Constant cols don't need time alignment
        processed_dict["constant_cols"] = data_dict.get("constant_cols")

        return processed_dict


class FloodEventSeq2SeqDataset(AugmentedFloodEventDataset):
    """Dataset class for Seq2Seq models with flood event detection and peak-aligned sampling.

    This dataset combines:
    1. Flood event data organization (from FloodEventDataset)
    2. Augmented data support (from AugmentedFloodEventDataset)
    3. Peak-aligned sampling (from FloodEventDataset)
    4. Seq2Seq data format (encoder input, decoder input, targets)

    Returns data in the format required by GeneralSeq2Seq model:
    - Training: [encoder_input, decoder_input, targets], targets
    - Inference: [encoder_input, decoder_input], targets
    """

    def __init__(self, cfgs: dict, is_tra_val_te: str):
        super(FloodEventSeq2SeqDataset, self).__init__(cfgs, is_tra_val_te)
        # Get hindcast_output_window from data_cfgs (default to rho if not specified)
        self.hindcast_output_window = self.data_cfgs.get(
            "hindcast_output_window", self.rho
        )

    def __getitem__(self, item: int):
        """Get one sample from the dataset in Seq2Seq format

        Returns
        -------
        Training mode:
            [encoder_input, decoder_input, targets], targets
        Inference mode:
            [encoder_input, decoder_input], targets

        Where:
        - encoder_input: (warmup_length + rho steps, feature_size) - warmup + hindcast period forcing data
        - decoder_input: (horizon steps, feature_size) - forecast period forcing data
        - targets: (hindcast_output_window + horizon steps, output_size) - full output sequence
        """
        basin, start_idx, actual_length = self.lookup_table[item]
        warmup_length = self.warmup_length
        rho = self.rho
        horizon = self.horizon
        end_idx = start_idx + actual_length

        # Get input data (forcing variables like rainfall)
        x = self.x[basin, start_idx:end_idx, :]

        # Get target data (streamflow, flood_event)
        y = self.y[basin, start_idx + warmup_length : end_idx, :]

        # Create flood mask from original data
        y_origin = self.y_origin[basin, start_idx + warmup_length : end_idx, :]
        flood_mask = self._create_flood_mask(y_origin)

        # Replace flood_event column with flood_mask
        y_with_flood_mask = y.copy()
        y_with_flood_mask[:, self.flood_event_idx] = flood_mask.squeeze()

        # Split data for Seq2Seq format
        # Encoder input: 包含 warmup + rho 数据，让 encoder LSTM 预热
        # 当 warmup_length=0 时，长度为 rho；当 warmup_length>0 时，长度为 warmup+rho
        x_encoder = x[: warmup_length + rho, :]

        # Decoder input: horizon period forcing data
        x_decoder = x[warmup_length + rho : warmup_length + rho + horizon, :]

        # Targets: hindcast_output_window + horizon period targets
        # Calculate start index for targets to include specified hindcast window
        target_start = max(0, rho - self.hindcast_output_window)
        y_targets = y_with_flood_mask[target_start : rho + horizon, :]

        # Handle constant features if available
        if self.c is not None and self.c.shape[-1] > 0:
            c = self.c[basin, :]

            # Add constants to encoder input
            c_encoder = (
                np.repeat(c, x_encoder.shape[0], axis=0).reshape(c.shape[0], -1).T
            )
            x_encoder = np.concatenate((x_encoder, c_encoder), axis=1)

            # Add constants to decoder input
            c_decoder = (
                np.repeat(c, x_decoder.shape[0], axis=0).reshape(c.shape[0], -1).T
            )
            x_decoder = np.concatenate((x_decoder, c_decoder), axis=1)

        # Convert to tensors
        encoder_input = torch.from_numpy(x_encoder).float()
        decoder_input = torch.from_numpy(x_decoder).float()

        # For Seq2Seq model:
        # - targets_for_model: only inflow (for teacher forcing), shape: [seq, 1]
        # - targets_for_loss: inflow + flood_event (for loss computation), shape: [seq, 2]
        targets_for_model = torch.from_numpy(
            y_targets[:, :1]
        ).float()  # Only inflow column
        targets_for_loss = torch.from_numpy(
            y_targets
        ).float()  # All columns (inflow + flood_event)

        # Return format depends on training/inference mode
        if self.is_tra_val_te == "train":
            # Training mode: include targets for teacher forcing
            # Model receives targets_for_model (1 column)
            # Loss receives targets_for_loss (2 columns)
            return [encoder_input, decoder_input, targets_for_model], targets_for_loss
        else:
            # Inference mode: no teacher forcing targets
            return [encoder_input, decoder_input], targets_for_loss
class AEFDataset(BaseDataset):
    def _read_xyc_specified_time(self, start_date, end_date):
        """Read x, y, c data from data source with specified time range
        We set this function as sometimes we need adjust the time range for some specific dataset,
        such as seq2seq dataset (it needs one more period for the end of the time range)

        Parameters
        ----------
        start_date : str
            start time
        end_date : str
            end time
        """
        data_forcing_ds_ = self.data_source.read_ts_xrdataset(
            self.t_s_dict["sites_id"],
            [start_date, end_date],
            self.data_cfgs["relevant_cols"],
        )
        # y
        data_output_ds_ = self.data_source.read_ts_xrdataset(
            self.t_s_dict["sites_id"],
            [start_date, end_date],
            self.data_cfgs["target_cols"],
        )
        data_forcing_ds_ = self._rm_timeunit_key(data_forcing_ds_)
        data_output_ds_ = self._rm_timeunit_key(data_output_ds_)
        data_forcing_ds, data_output_ds = self._check_ts_xrds_unit(
            data_forcing_ds_, data_output_ds_
        )
        # c
        csv_path = os.environ.get("TORCHHYDRO_EMBEDDING_CSV") or r"/Users/cylenlc/AlphaEarthViz/data/向量csv/BasinEmbeddings_2017_2024_avg_method_2.csv"
        df_embeddings = pd.read_csv(csv_path)
        df_embeddings["hru_id"] = (
            df_embeddings["hru_id"].astype(int).astype(str).str.lstrip("0")
        )
        target_ids = [site_id.lstrip("0") for site_id in self.t_s_dict["sites_id"]]
        df_filtered = df_embeddings[df_embeddings["hru_id"].isin(target_ids)]
        df_filtered = df_filtered.set_index("hru_id").reindex(target_ids).reset_index()
        if len(df_filtered) != len(target_ids):
            print(
                "Warning: Not all basin IDs from self.t_s_dict['sites_id'] were found in the CSV."
            )
        df_with_index = df_filtered.set_index("hru_id")
        embeddings_data = df_with_index.values
        c_data_vars = {
            col: (("basin"), embeddings_data[:, i])
            for i, col in enumerate([f"A{j:02d}" for j in range(64)])
        }
        c_origin = xr.Dataset(c_data_vars, coords={"basin": self.t_s_dict["sites_id"]})

        x_origin, y_origin, c_origin_with_unit = self._to_dataarray_with_unit(
            data_forcing_ds, data_output_ds, c_origin
        )
        return {
            "relevant_cols": x_origin.transpose("basin", "time", "variable"),
            "target_cols": y_origin.transpose("basin", "time", "variable"),
            "constant_cols": (
                c_origin_with_unit.transpose("basin", "variable")
                if c_origin_with_unit is not None
                else None
            ),
        }


class TgHydroDataset(BaseDataset):
    def __init__(self, cfgs: dict, is_tra_val_te: str, upstream_indices: dict = None):
        super().__init__(cfgs, is_tra_val_te)
        self.upstream_indices = upstream_indices or {}
        self.hindcast_length = self.rho
        self.forecast_length = self.horizon
        tg_cfgs = self.data_cfgs.get("tg_cfgs", {})
        self.node_attributes = tg_cfgs.get("node_attributes", None)
        self.basin_names = tg_cfgs.get("basin_names", [])
        self.basin_id = tg_cfgs.get("basin_id", None)
        self.use_confluence_nodes_only = tg_cfgs.get("use_confluence_nodes_only", True)
        self.node_mask = None
        self.edge_index = None
        self.num_nodes = len(self.basins)
        rels = self.data_cfgs.get("relevant_cols", [])

        def _find_first_index(names):
            for n in names:
                if n in rels:
                    return rels.index(n)
            return None

        self._var_idx_precip = _find_first_index(
            [
                "precipitation",
                "total_precipitation",
                "total_precipitation_hourly",
                "total_precipitation_surface",
                "prcp",
                "pr",
            ]
        )
        self._var_idx_lstm = _find_first_index(
            [
                "lstm_streamflow",
                "lstm_pred",
                "lstm_prediction",
            ]
        )
        self._var_idx_obs = _find_first_index(
            [
                "observed_streamflow",
                "streamflow",
                "qobs",
            ]
        )
        self._var_idx_discharge = _find_first_index(["discharge"])
        self._load_graph_from_datasource()
        if self.node_mask is None:
            self._init_node_mask()
        if self.edge_index is None:
            self._init_edge_index()

    def _create_lookup_table(self):
        """
        Create lookup table for Graph Dataset.
        For GNNs, each sample represents the state of ALL nodes at a specific time step.
        So we only iterate over time, not basins.
        """
        lookup = []
        rho = self.rho
        warmup_length = self.warmup_length
        horizon = self.horizon
        seq_len = rho + horizon
        max_time_length = self.nt

        # We only iterate over time, not basins
        # Validation/Test: maintain full sequence
        # Training: Currently we also maintain full sequence to keep graph structure fixed.
        # Handling of NaN targets should be done via loss masking, not by skipping samples here,
        # unless all nodes are NaN at that time step.

        # Check for time steps where ALL basins are NaN (optional optimization)
        # nan_array_all = np.isnan(self.y[:, :, 0]).all(axis=0) # [time]

        for f in range(warmup_length, max_time_length - rho - horizon):
            # We use basin=None to indicate "all basins/nodes"
            lookup.append((None, f, seq_len))

        self.lookup_table = dict(enumerate(lookup))
        self.num_samples = len(self.lookup_table)

    def __getitem__(self, item: int):
        """Get one sample from the dataset as a PyG Data object.

        Parameters
        ----------
        item : int
            Sample index

        Returns
        -------
        torch_geometric.data.Data
            PyG Data object containing:
            - x: node features [num_nodes, hindcast_length, num_features]
            - y: target values [num_nodes, forecast_length]
            - edge_index: graph connectivity [2, num_edges]
            - node_mask: (optional) mask for confluence nodes [num_nodes]
        """

        basin, idx, actual_length = self.lookup_table[item]
        warmup_length = self.warmup_length
        M = self.rho
        F = self.horizon
        # 若数据源提供了 node_ids，并且与 basins 顺序不一致，则按 node_ids 重排 x/y
        node_ids = getattr(self, "node_ids", None)
        if node_ids is not None and isinstance(node_ids, (list, np.ndarray)):
            # 构建 basin->index 的映射并重排
            basin_to_idx = {basin: i for i, basin in enumerate(self.basins)}
            reorder_idx = [basin_to_idx.get(b, None) for b in node_ids]
            if all(i is not None for i in reorder_idx):
                x_all = self.x[
                    reorder_idx, idx - warmup_length : idx + actual_length + 1, :
                ]
                y_all = self.y[reorder_idx, idx : idx + actual_length, :]
            else:
                x_all = self.x[:, idx - warmup_length : idx + actual_length + 1, :]
                y_all = self.y[:, idx : idx + actual_length, :]
        else:
            x_all = self.x[:, idx - warmup_length : idx + actual_length + 1, :]
            y_all = self.y[:, idx : idx + actual_length, :]
        start_obs_rel = warmup_length
        end_obs_rel = warmup_length + M
        end_future_rel = warmup_length + M + F
        if self._var_idx_precip is None or self._var_idx_lstm is None:
            raise ValueError(
                "relevant_cols 需包含 'precipitation' 与 'lstm_streamflow'"
            )
        # Extend to include horizon F
        past_rain = x_all[
            :, start_obs_rel + 1 : end_future_rel + 1, self._var_idx_precip
        ]
        past_lstm = x_all[:, start_obs_rel + 1 : end_future_rel + 1, self._var_idx_lstm]

        # Obs only up to M, then NaN for F
        if self._var_idx_obs is not None:
            real_obs = x_all[:, start_obs_rel:end_obs_rel, self._var_idx_obs]
        else:
            real_obs = y_all[:, :M, 0]

        if self._var_idx_discharge is not None:
            discharge_vals = x_all[
                :, start_obs_rel:end_obs_rel, self._var_idx_discharge
            ]
            mask = ~np.isnan(discharge_vals)
            real_obs = np.where(mask, discharge_vals, real_obs)

        # Create placeholder for future obs (filled with NaN or 0)
        # Using 0 or NaN? NaN is safer to detect leakage, but model needs to handle it if not overwritten immediately.
        # But we will overwrite it in the model before use.
        future_obs_placeholder = np.full(
            (real_obs.shape[0], F), np.nan, dtype=real_obs.dtype
        )
        past_obs = np.concatenate([real_obs, future_obs_placeholder], axis=1)

        time_series = np.stack(
            [
                past_rain.transpose(0, 1),
                past_lstm.transpose(0, 1),
                past_obs.transpose(0, 1),
            ],
            axis=-1,
        )
        if self.c is None or self.c.shape[-1] == 0:
            input_features = time_series
        else:
            if node_ids is not None and isinstance(node_ids, (list, np.ndarray)):
                basin_to_idx = {basin: i for i, basin in enumerate(self.basins)}
                reorder_idx = [basin_to_idx.get(b, None) for b in node_ids]
                if all(i is not None for i in reorder_idx):
                    c = self.c[reorder_idx, :]
                else:
                    c = self.c
            else:
                c = self.c
            # Extend c repetition to M + F
            c_exp = np.repeat(c[:, np.newaxis, :], M + F, axis=1)
            input_features = np.concatenate((time_series, c_exp), axis=-1)
        # For PyG, we want [num_nodes, M+F, num_features]
        future_targets = self.y[:, idx + M : idx + M + F, 0]
        # output_data = future_targets.transpose(1, 0)

        # Create PyG Data object
        data = Data(
            x=torch.from_numpy(
                input_features
            ).float(),  # [num_nodes, M+F, num_features]
            y=torch.from_numpy(future_targets).float(),  # [num_nodes, F]
            edge_index=self.edge_index,  # [2, num_edges]
        )

        # Add node mask if using confluence nodes only
        if self.use_confluence_nodes_only and self.node_mask is not None:
            data.node_mask = torch.from_numpy(self.node_mask.astype(np.bool_))

        return data

    def _read_xyc_specified_time(self, start_date, end_date):
        tg_cfgs = self.data_cfgs.get("tg_cfgs", {})
        var_source_map = tg_cfgs.get(
            "var_source_map", tg_cfgs.get("source_selection", {})
        )
        default_choice = tg_cfgs.get("default_source", None)
        source_kwargs = {}
        combine = tg_cfgs.get("combine", True)
        if combine:
            source_kwargs["combine"] = True

        def _split_source(ds):
            base, inter = None, None
            if isinstance(ds, dict):
                keys = {k.lower(): k for k in ds.keys()}
                if "base" in keys:
                    base = ds[keys["base"]]
                if "intermediate" in keys:
                    inter = ds[keys["intermediate"]]
            elif isinstance(ds, xr.Dataset):
                if "source" in ds.data_vars:
                    src_da = ds["source"]
                    if "basin" in src_da.dims:
                        basins = (
                            ds["basin"].values
                            if "basin" in ds.coords
                            else src_da["basin"].values
                        )
                        src_vals = src_da.values.astype(str)
                        base_mask = np.isin(src_vals, ["base", "Base", "BASE"])
                        inter_mask = np.isin(
                            src_vals,
                            ["intermediate", "Intermediate", "INTERMEDIATE", "inter"],
                        )
                        base_idx = np.where(base_mask)[0]
                        inter_idx = np.where(inter_mask)[0]
                        if base_idx.size > 0:
                            base = ds.isel(basin=base_idx)
                        if inter_idx.size > 0:
                            inter = ds.isel(basin=inter_idx)
                elif "source" in ds.coords or "source" in ds.dims:
                    try:
                        base = ds.sel(source="base")
                    except Exception:
                        base = None
                    try:
                        inter = ds.sel(source="intermediate")
                    except Exception:
                        inter = None
            if base is None and inter is None:
                base = ds
            return base, inter

        def _select_timeseries(ds_raw, var_list):
            ds_base, ds_inter = _split_source(ds_raw)
            if ds_base is not None:
                ds_base = self._rm_timeunit_key(ds_base)
            if ds_inter is not None:
                ds_inter = self._rm_timeunit_key(ds_inter)
            if ds_base is None and ds_inter is None:
                return None
            if ds_inter is None:
                return ds_base[var_list]
            if ds_base is None:
                return ds_inter[var_list]
            base_basins = (
                ds_base["basin"].values if "basin" in ds_base.coords else np.array([])
            )
            inter_basins = (
                ds_inter["basin"].values if "basin" in ds_inter.coords else np.array([])
            )
            basins_union = np.union1d(base_basins, inter_basins)
            union_time = None
            if "time" in ds_base.dims and "time" in ds_inter.dims:
                union_time = np.union1d(ds_base.time.values, ds_inter.time.values)
                ds_base = ds_base.reindex(time=union_time)
                ds_inter = ds_inter.reindex(time=union_time)
            if basins_union.size > 0:
                if "basin" in ds_base.dims:
                    ds_base = ds_base.reindex(basin=basins_union)
                if "basin" in ds_inter.dims:
                    ds_inter = ds_inter.reindex(basin=basins_union)
            data_vars = {}
            suffix_map = tg_cfgs.get(
                "source_suffix", {"base": "_base", "intermediate": "_inter"}
            )
            for v in var_list:
                choice = var_source_map.get(v, default_choice)
                if isinstance(choice, (list, tuple)) or choice == "both":
                    if ds_inter is not None and v in ds_inter:
                        data_vars[f"{v}{suffix_map.get('intermediate', '_inter')}"] = (
                            ds_inter[v]
                        )
                    if ds_base is not None and v in ds_base:
                        data_vars[f"{v}{suffix_map.get('base', '_base')}"] = ds_base[v]
                else:
                    if choice == "intermediate" and v in ds_inter:
                        data_vars[v] = ds_inter[v]
                    elif v in ds_base:
                        data_vars[v] = ds_base[v]
                    elif v in ds_inter:
                        data_vars[v] = ds_inter[v]
            coords = {}
            if basins_union.size > 0:
                coords["basin"] = basins_union
            if union_time is not None:
                coords["time"] = union_time
            return xr.Dataset(data_vars, coords=coords)

        def _select_attributes(ds_raw, var_list):
            ds_base, ds_inter = _split_source(ds_raw)
            if ds_base is not None:
                ds_base = self._rm_timeunit_key(ds_base)
            if ds_inter is not None:
                ds_inter = self._rm_timeunit_key(ds_inter)
            if ds_base is None and ds_inter is None:
                return None
            if ds_inter is None:
                return ds_base[var_list]
            if ds_base is None:
                return ds_inter[var_list]
            base_basins = (
                ds_base["basin"].values if "basin" in ds_base.coords else np.array([])
            )
            inter_basins = (
                ds_inter["basin"].values if "basin" in ds_inter.coords else np.array([])
            )
            basins_union = np.union1d(base_basins, inter_basins)
            if basins_union.size > 0:
                if "basin" in ds_base.dims:
                    ds_base = ds_base.reindex(basin=basins_union)
                if "basin" in ds_inter.dims:
                    ds_inter = ds_inter.reindex(basin=basins_union)
            data_vars = {}
            suffix_map = tg_cfgs.get(
                "source_suffix", {"base": "_base", "intermediate": "_inter"}
            )
            for v in var_list:
                choice = var_source_map.get(v, default_choice)
                if isinstance(choice, (list, tuple)) or choice == "both":
                    if ds_inter is not None and v in ds_inter:
                        data_vars[f"{v}{suffix_map.get('intermediate', '_inter')}"] = (
                            ds_inter[v]
                        )
                    if ds_base is not None and v in ds_base:
                        data_vars[f"{v}{suffix_map.get('base', '_base')}"] = ds_base[v]
                else:
                    if choice == "intermediate" and v in ds_inter:
                        data_vars[v] = ds_inter[v]
                    elif v in ds_base:
                        data_vars[v] = ds_base[v]
                    elif v in ds_inter:
                        data_vars[v] = ds_inter[v]
            coords = {"basin": basins_union} if basins_union.size > 0 else {}
            return xr.Dataset(data_vars, coords=coords)

        data_forcing_ds_raw = self.data_source.read_ts_xrdataset(
            self.t_s_dict["sites_id"],
            [start_date, end_date],
            self.data_cfgs["relevant_cols"],
            **source_kwargs,
        )
        data_output_ds_raw = self.data_source.read_ts_xrdataset(
            self.t_s_dict["sites_id"],
            [start_date, end_date],
            self.data_cfgs["target_cols"],
            **source_kwargs,
        )
        data_attr_ds_raw = self.data_source.read_attr_xrdataset(
            self.t_s_dict["sites_id"],
            self.data_cfgs["constant_cols"],
            all_number=True,
            **source_kwargs,
        )

        data_forcing_ds_raw = self._rm_timeunit_key(data_forcing_ds_raw)
        data_output_ds_raw = self._rm_timeunit_key(data_output_ds_raw)
        data_forcing_ds, data_output_ds = self._check_ts_xrds_unit(
            data_forcing_ds_raw, data_output_ds_raw
        )

        data_forcing_ds_sel = _select_timeseries(
            data_forcing_ds, self.data_cfgs["relevant_cols"]
        )
        data_output_ds_sel = _select_timeseries(
            data_output_ds, self.data_cfgs["target_cols"]
        )
        data_attr_ds_sel = _select_attributes(
            data_attr_ds_raw, self.data_cfgs["constant_cols"]
        )

        x_origin, y_origin, c_origin = self._to_dataarray_with_unit(
            data_forcing_ds_sel, data_output_ds_sel, data_attr_ds_sel
        )
        return {
            "relevant_cols": x_origin.transpose("basin", "time", "variable"),
            "target_cols": y_origin.transpose("basin", "time", "variable"),
            "constant_cols": (
                c_origin.transpose("basin", "variable")
                if c_origin is not None
                else None
            ),
        }

    def _init_node_mask(self):
        """Initialize node mask for identifying confluence nodes.

        A confluence node is defined as having 2 or more upstream nodes.
        """
        mask = None
        upstream = None
        if self.upstream_indices:
            upstream = self.upstream_indices
        else:
            if hasattr(self.data_source, "read_upstream_indices"):
                try:
                    upstream = self.data_source.read_upstream_indices(
                        self.basin_id, self.basins
                    )
                except Exception:
                    upstream = None
            elif hasattr(self.data_source, "get_upstream_indices"):
                try:
                    upstream = self.data_source.get_upstream_indices(
                        self.basin_id, self.basins
                    )
                except Exception:
                    upstream = None
        if upstream is not None:
            mask_list = []
            for i, b in enumerate(self.basins):
                u = upstream.get(b)
                if u is None:
                    u = upstream.get(i)
                if u is None:
                    u = upstream.get(str(b))
                mask_list.append(bool(u) and len(u) >= 2)
            mask = np.array(mask_list, dtype=bool)
        self.node_mask = mask

    def _init_edge_index(self):
        """Initialize edge_index from upstream_indices for PyG graph structure.

        Converts the upstream_indices dict to PyG edge_index format [2, num_edges].
        Edge direction: from upstream to downstream.
        """
        if not self.upstream_indices:
            # No edges if upstream_indices is empty
            self.edge_index = torch.empty((2, 0), dtype=torch.long)
            return

        # Create basin to index mapping
        basin_to_idx = {basin: idx for idx, basin in enumerate(self.basins)}

        # Build edge list: (source, target) pairs
        edges = []
        for downstream_basin, upstream_basins in self.upstream_indices.items():
            # Get downstream node index
            if downstream_basin in basin_to_idx:
                downstream_idx = basin_to_idx[downstream_basin]
            elif isinstance(downstream_basin, int):
                downstream_idx = downstream_basin
            else:
                continue  # Skip if basin not found

            # Add edges from each upstream basin to downstream
            if isinstance(upstream_basins, list):
                for upstream_basin in upstream_basins:
                    if upstream_basin in basin_to_idx:
                        upstream_idx = basin_to_idx[upstream_basin]
                    elif isinstance(upstream_basin, int):
                        upstream_idx = upstream_basin
                    else:
                        continue
                    edges.append([upstream_idx, downstream_idx])

        if edges:
            # Convert to PyG edge_index format: [2, num_edges]
            self.edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            # No valid edges found
            self.edge_index = torch.empty((2, 0), dtype=torch.long)

    def _load_graph_from_datasource(self):
        graph = None
        if hasattr(self.data_source, "read_graph_data"):
            try:
                graph = self.data_source.read_graph_data(self.basin_id, self.basins)
            except Exception:
                graph = None
        elif hasattr(self.data_source, "get_graph_data"):
            try:
                graph = self.data_source.get_graph_data(self.basin_id, self.basins)
            except Exception:
                graph = None
        if graph is None:
            if hasattr(self.data_source, "read_upstream_indices"):
                try:
                    self.upstream_indices = self.data_source.read_upstream_indices(
                        self.basin_id, self.basins
                    )
                except Exception:
                    self.upstream_indices = {}
            elif hasattr(self.data_source, "get_upstream_indices"):
                try:
                    self.upstream_indices = self.data_source.get_upstream_indices(
                        self.basin_id, self.basins
                    )
                except Exception:
                    self.upstream_indices = {}
            return
        if isinstance(graph, dict):
            ei = graph.get("edge_index")
            nm = graph.get("node_mask")
            node_ids = graph.get("node_ids")
            if ei is not None:
                if isinstance(ei, torch.Tensor):
                    self.edge_index = ei
                else:
                    try:
                        self.edge_index = torch.tensor(ei, dtype=torch.long)
                        if self.edge_index.dim() == 2 and self.edge_index.shape[0] != 2:
                            self.edge_index = self.edge_index.t().contiguous()
                    except Exception:
                        self.edge_index = None
            if nm is not None:
                try:
                    self.node_mask = np.array(nm, dtype=bool)
                except Exception:
                    self.node_mask = None
            if node_ids is not None:
                try:
                    self.node_ids = list(node_ids)
                except Exception:
                    self.node_ids = None
            if not self.upstream_indices and "upstream_indices" in graph:
                self.upstream_indices = graph.get("upstream_indices")


class Camels_merge_AEFDataset(BaseDataset):
    def _read_xyc_specified_time(self, start_date, end_date):
        """Read x, y, c data from data source with specified time range
        We set this function as sometimes we need adjust the time range for some specific dataset,
        such as seq2seq dataset (it needs one more period for the end of the time range)

        Parameters
        ----------
        start_date : str
            start time
        end_date : str
            end time
        """
        data_forcing_ds_ = self.data_source.read_ts_xrdataset(
            self.t_s_dict["sites_id"],
            [start_date, end_date],
            self.data_cfgs["relevant_cols"],
        )
        # y
        data_output_ds_ = self.data_source.read_ts_xrdataset(
            self.t_s_dict["sites_id"],
            [start_date, end_date],
            self.data_cfgs["target_cols"],
        )
        data_forcing_ds_ = self._rm_timeunit_key(data_forcing_ds_)
        data_output_ds_ = self._rm_timeunit_key(data_output_ds_)
        data_forcing_ds, data_output_ds = self._check_ts_xrds_unit(
            data_forcing_ds_, data_output_ds_
        )
        # c
        data_attr_ds = self.data_source.read_attr_xrdataset(
            self.t_s_dict["sites_id"],
            self.data_cfgs["constant_cols"],
            all_number=True,
        )

        csv_path = os.environ.get("TORCHHYDRO_EMBEDDING_CSV") or r"/Users/cylenlc/AlphaEarthViz/data/向量csv/BasinEmbeddings_2017_2024_avg_method_2.csv"
        df_embeddings = pd.read_csv(csv_path)
        df_embeddings["hru_id"] = (
            df_embeddings["hru_id"].astype(int).astype(str).str.lstrip("0")
        )
        target_ids = [site_id.lstrip("0") for site_id in self.t_s_dict["sites_id"]]
        df_filtered = df_embeddings[df_embeddings["hru_id"].isin(target_ids)]
        df_filtered = df_filtered.set_index("hru_id").reindex(target_ids).reset_index()

        if len(df_filtered) != len(target_ids):
            print("Warning: Some basin IDs not found in the embeddings CSV.")

        df_indexed = df_filtered.set_index("hru_id")
        embedding_values = df_indexed.values
        embedding_vars = {
            f"A{j:02d}": (("basin"), embedding_values[:, j]) for j in range(64)
        }

        c_origin = xr.Dataset(
            embedding_vars, coords={"basin": self.t_s_dict["sites_id"]}
        )
        c_origin = c_origin.reindex(basin=data_attr_ds.basin.values)
        combined_ds = xr.merge([data_attr_ds, c_origin])

        x_origin, y_origin, c_origin_with_unit = self._to_dataarray_with_unit(
            data_forcing_ds, data_output_ds, combined_ds
        )
        return {
            "relevant_cols": x_origin.transpose("basin", "time", "variable"),
            "target_cols": y_origin.transpose("basin", "time", "variable"),
            "constant_cols": (
                c_origin_with_unit.transpose("basin", "variable")
                if c_origin_with_unit is not None
                else None
            ),
        }


class FNODataset(BaseDataset):
    def _load_data(self):
        origin_data = self._read_xyc()
        # normalization
        norm_data = self._normalize(origin_data)
        # 启用 NaN 处理以确保数据清洁
        origin_data_wonan, norm_data_wonan = self._kill_nan(origin_data, norm_data)
        origin_data_wonan, norm_data_wonan = self._add_absolute_time_axis(
            origin_data_wonan,
            norm_data_wonan,
        )
        self._trans2nparr(origin_data_wonan, norm_data_wonan)
        self._create_lookup_table()

    def _add_absolute_time_axis(self, origin_data, norm_data):
        """
        Add absolute time coordinate normalized to [0, 1] as an extra dynamic feature.

        Parameters
        ----------
        origin_data : dict
            output of _kill_nan, contains xarray DataArrays
        norm_data : dict
            normalized version of origin_data

        Returns
        -------
        origin_data_new, norm_data_new : dict
            same structure as input, but relevant_cols has +1 variable (time)
        """

        # -------- 1. Extract time coordinate --------
        time_coord = origin_data["relevant_cols"].coords["time"]
        # datetime64[ns] → int64
        time_ns = time_coord.values.astype("datetime64[ns]").astype("int64")

        t_min = time_ns.min()
        t_max = time_ns.max()

        time_norm = (time_ns - t_min) / (t_max - t_min)
        time_norm = time_norm.astype("float32")  # (T,)

        # -------- 2. Expand to (basin, time, 1) --------
        n_basins = origin_data["relevant_cols"].sizes["basin"]
        n_time = origin_data["relevant_cols"].sizes["time"]

        time_feat = np.broadcast_to(time_norm[None, :, None], (n_basins, n_time, 1))

        # -------- 3. Wrap as xarray.DataArray --------
        time_da = xr.DataArray(
            time_feat,
            dims=("basin", "time", "variable"),
            coords={
                "basin": origin_data["relevant_cols"].coords["basin"],
                "time": origin_data["relevant_cols"].coords["time"],
                "variable": ["absolute_time"],
            },
            attrs={"description": "absolute time normalized to [0,1]"},
        )

        # -------- 4. Concatenate to relevant_cols --------
        origin_data = origin_data.copy()
        norm_data = norm_data.copy()

        origin_data["relevant_cols"] = xr.concat(
            [origin_data["relevant_cols"], time_da], dim="variable"
        )

        norm_data["relevant_cols"] = xr.concat(
            [norm_data["relevant_cols"], time_da], dim="variable"
        )

        return origin_data, norm_data
