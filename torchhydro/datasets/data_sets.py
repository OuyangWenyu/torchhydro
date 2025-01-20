"""
Author: Wenyu Ouyang
Date: 2024-04-08 18:16:53
LastEditTime: 2024-08-10 15:10:27
LastEditors: Wenyu Ouyang
Description: A pytorch dataset class; references to https://github.com/neuralhydrology/neuralhydrology
FilePath: \torchhydro\torchhydro\datasets\data_sets.py
Copyright (c) 2024-2024 Wenyu Ouyang. All rights reserved.
"""

import logging
import os.path
import re
import sys
from datetime import datetime
from datetime import timedelta
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
import torch
import xarray as xr
import polars as pl
from hydrodatasource.reader.data_source import SelfMadeHydroDataset
from hydrodatasource.utils.utils import streamflow_unit_conv
from torch.utils.data import Dataset
from tqdm import tqdm

from torchhydro.configs.config import DATE_FORMATS
from torchhydro.datasets.data_scalers import ScalerHub
from torchhydro.datasets.data_sources import data_sources_dict
from torchhydro.datasets.data_utils import (
    warn_if_nan,
    wrap_t_s_dict, warn_if_nan_pq,
)
from torchhydro.trainers.train_utils import total_fab

LOGGER = logging.getLogger(__name__)


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
            if warn_if_nan(mean_val):
                # when all value are NaN, mean_val will be NaN, we set mean_val to -1
                mean_val = -1
            filled_data = var_data.fillna(
                mean_val
            )  # fill NaN values with the calculated mean
            da.loc[dict(variable=var)] = (
                filled_data  # update the original dataarray with the filled data
            )
    elif fill_nan == "interpolate":
        # fill interpolation
        for i in range(da.shape[0]):
            da[i] = da[i].interpolate_na(dim="time", fill_value="extrapolate")
    else:
        raise NotImplementedError(f"fill_nan {fill_nan} not implemented")
    return da

def _fill_gaps_pq(df: pl.DataFrame, fill_nan: Optional[str] = None) -> pl.DataFrame:
    """Fill gaps in a DataArray"""
    if fill_nan is None or df is None:
        return df
    # fill gaps
    if fill_nan == "et_ssm_ignore":
        all_non_nan_idx = []
        for col in df.columns[:-2]:
            non_nan_idx_tmp = np.where(~np.isnan(np.float32(df[col].to_numpy())))
            all_non_nan_idx = all_non_nan_idx + non_nan_idx_tmp[0].tolist()
        # some NaN data appear in different dates in different basins
        non_nan_idx = np.unique(all_non_nan_idx).tolist()
        df = df[non_nan_idx]
    elif fill_nan == "mean":
        # fill with mean
        for var in df.columns:
            var_data = df[var]  # select the data for the current variable
            mean_val = var_data.mean()  # calculate the mean across all basins
            if warn_if_nan(mean_val):
                # when all value are NaN, mean_val will be NaN, we set mean_val to -1
                mean_val = -1
            filled_data = var_data.fill_nan(mean_val)  # fill NaN values with the calculated mean
            df = df.with_columns(filled_data.alias(var)) # update the original dataarray with the filled dat
    elif fill_nan == "interpolate":
        # fill interpolation
        for col in df.columns:
            inter_col = df[col].interpolate()
            df = df.with_columns(inter_col.alias(col))
    else:
        raise NotImplementedError(f"fill_nan {fill_nan} not implemented")
    return df


def detect_date_format(date_str):
    for date_format in DATE_FORMATS:
        try:
            datetime.strptime(date_str, date_format)
            return date_format
        except ValueError:
            continue
    raise ValueError(f"Unknown date format: {date_str}")


class BaseDataset(Dataset):
    """Base data set class to load and preprocess data (batch-first) using PyTorch's Dataset"""

    def __init__(self, data_cfgs: dict, is_tra_val_te: str):
        """
        Parameters
        ----------
        data_cfgs
            parameters for reading source data
        is_tra_val_te
            train, vaild or test
        """
        super(BaseDataset, self).__init__()
        self.data_cfgs = data_cfgs
        if is_tra_val_te in {"train", "valid", "test"}:
            self.is_tra_val_te = is_tra_val_te
        else:
            raise ValueError(
                "'is_tra_val_te' must be one of 'train', 'valid' or 'test' "
            )
        # load and preprocess data
        self._load_data()

    @property
    def data_source(self):
        source_name = self.data_cfgs["source_cfgs"]["source_name"]
        source_path = self.data_cfgs["source_cfgs"]["source_path"]
        other_settings = self.data_cfgs["source_cfgs"].get("other_settings", {})
        return data_sources_dict[source_name](source_path, **other_settings)

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
            # trange_type_num = 1
            earliest_date = self.t_s_dict["t_final_range"][0]
            latest_date = self.t_s_dict["t_final_range"][1]
        min_time_unit = self.data_cfgs["min_time_unit"]
        min_time_interval = self.data_cfgs["min_time_interval"]
        time_step = f"{min_time_interval}{min_time_unit}"
        s_date = pd.to_datetime(earliest_date)
        e_date = pd.to_datetime(latest_date)
        time_series = pd.date_range(start=s_date, end=e_date, freq=time_step)
        return len(time_series)

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
        return self.num_samples if self.train_mode else self.ngrid

    def __getitem__(self, item: int):
        if not self.train_mode:
            x = self.x[item, :, :]
            y = self.y[item, :, :]
            if self.c is None or self.c.shape[-1] == 0:
                return torch.from_numpy(x).float(), torch.from_numpy(y).float()
            c = self.c[item, :]
            c = np.repeat(c, x.shape[0], axis=0).reshape(c.shape[0], -1).T
            xc = np.concatenate((x, c), axis=1)
            return torch.from_numpy(xc).float(), torch.from_numpy(y).float()
        basin, idx = self.lookup_table[item]
        warmup_length = self.warmup_length
        x = self.x[basin, idx - warmup_length: idx + self.rho + self.horizon, :]
        y = self.y[basin, idx: idx + self.rho + self.horizon, :]
        if self.c is None or self.c.shape[-1] == 0:
            return torch.from_numpy(x).float(), torch.from_numpy(y).float()
        c = self.c[basin, :]
        c = np.repeat(c, x.shape[0], axis=0).reshape(c.shape[0], -1).T
        xc = np.concatenate((x, c), axis=1)
        return torch.from_numpy(xc).float(), torch.from_numpy(y).float()

    def _pre_load_data(self):
        self.train_mode = self.is_tra_val_te == "train"
        self.t_s_dict = wrap_t_s_dict(self.data_cfgs, self.is_tra_val_te)
        self.rho = self.data_cfgs["forecast_history"]
        self.warmup_length = self.data_cfgs["warmup_length"]
        self.horizon = self.data_cfgs["forecast_length"]

    def _load_data(self):
        self._pre_load_data()
        self._read_xyc()
        # normalization
        norm_x, norm_y, norm_c = self._normalize()
        self.x, self.y, self.c = self._kill_nan(norm_x, norm_y, norm_c)
        self._trans2nparr()
        self._create_lookup_table()

    def _trans2nparr(self):
        """To make __getitem__ more efficient,
        we transform x, y, c to numpy array with shape (nsample, nt, nvar)
        """
        self.x = self.x.transpose("basin", "time", "variable").to_numpy()
        self.y = self.y.transpose("basin", "time", "variable").to_numpy()
        if self.c is not None and self.c.shape[-1] > 0:
            self.c = self.c.transpose("basin", "variable").to_numpy()
            self.c_origin = self.c_origin.transpose("basin", "variable").to_numpy()
        self.x_origin = self.x_origin.transpose("basin", "time", "variable").to_numpy()
        self.y_origin = self.y_origin.transpose("basin", "time", "variable").to_numpy()

    def _normalize(self):
        scaler_hub = ScalerHub(
            self.y_origin,
            self.x_origin,
            self.c_origin,
            data_cfgs=self.data_cfgs,
            is_tra_val_te=self.is_tra_val_te,
            data_source=self.data_source,
        )
        self.target_scaler = scaler_hub.target_scaler
        return scaler_hub.x, scaler_hub.y, scaler_hub.c

    def _to_dataarray_with_unit(self, data_forcing_ds, data_output_ds, data_attr_ds):
        # trans to dataarray to better use xbatch
        if data_output_ds is not None:
            data_output = self._trans2da_and_setunits(data_output_ds)
        else:
            data_output = None
        if data_forcing_ds is not None:
            data_forcing = self._trans2da_and_setunits(data_forcing_ds)
        else:
            data_forcing = None
        if data_attr_ds is not None:
            # firstly, we should transform some str type data to float type
            data_attr = self._trans2da_and_setunits(data_attr_ds)
        else:
            data_attr = None
        return data_forcing, data_output, data_attr

    def _check_ts_xrds_unit(self, data_forcing_ds, data_output_ds):
        """Check timeseries xarray dataset unit and convert if necessary

        Parameters
        ----------
        data_forcing_ds : _type_
            _description_
        data_output_ds : _type_
            _description_
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
            data_output_ds = streamflow_unit_conv(
                data_output_ds,
                self.data_source.read_area(self.t_s_dict["sites_id"]),
                target_unit=prcp_unit,
            )
        return data_forcing_ds, data_output_ds

    def _read_xyc(self):
        """Read x, y, c data from data source

        Returns
        -------
        tuple[xr.Dataset, xr.Dataset, xr.Dataset]
            x, y, c data
        """
        # x
        data_forcing_ds_ = self.data_source.read_ts_xrdataset(
            self.t_s_dict["sites_id"],
            self.t_s_dict["t_final_range"],
            self.data_cfgs["relevant_cols"],
        )
        # y
        data_output_ds_ = self.data_source.read_ts_xrdataset(
            self.t_s_dict["sites_id"],
            self.t_s_dict["t_final_range"],
            self.data_cfgs["target_cols"],
        )
        if isinstance(data_output_ds_, dict) or isinstance(data_forcing_ds_, dict):
            # this means the data source return a dict with key as time_unit
            # in this BaseDataset, we only support unified time range for all basins, so we chose the first key
            # TODO: maybe this could be refactored better
            data_forcing_ds_ = data_forcing_ds_[list(data_forcing_ds_.keys())[0]]
            data_output_ds_ = data_output_ds_[list(data_output_ds_.keys())[0]]
        data_forcing_ds, data_output_ds = self._check_ts_xrds_unit(
            data_forcing_ds_, data_output_ds_
        )
        # c
        data_attr_ds = self.data_source.read_attr_xrdataset(
            self.t_s_dict["sites_id"],
            self.data_cfgs["constant_cols"],
            all_number=True,
        )
        self.x_origin, self.y_origin, self.c_origin = self._to_dataarray_with_unit(
            data_forcing_ds, data_output_ds, data_attr_ds
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

    def _kill_nan(self, x, y, c):
        data_cfgs = self.data_cfgs
        y_rm_nan = data_cfgs["target_rm_nan"]
        x_rm_nan = data_cfgs["relevant_rm_nan"]
        c_rm_nan = data_cfgs["constant_rm_nan"]
        if x_rm_nan:
            # As input, we cannot have NaN values
            _fill_gaps_da(x, fill_nan="interpolate")
            warn_if_nan(x)
        if y_rm_nan:
            _fill_gaps_da(y, fill_nan="interpolate")
            warn_if_nan(y)
        if c_rm_nan:
            _fill_gaps_da(c, fill_nan="mean")
            warn_if_nan(c)
        warn_if_nan(x, nan_mode="all")
        warn_if_nan(y, nan_mode="all")
        warn_if_nan(c, nan_mode="all")
        return x, y, c

    def _create_lookup_table(self):
        lookup = []
        # list to collect basins ids of basins without a single training sample
        basin_coordinates = len(self.t_s_dict["sites_id"])
        rho = self.rho
        warmup_length = self.warmup_length
        horizon = self.horizon
        max_time_length = self.nt
        for basin in tqdm(range(basin_coordinates), file=sys.stdout, disable=False):
            if self.is_tra_val_te != "train":
                lookup.extend(
                    (basin, f)
                    for f in range(warmup_length, max_time_length - rho - horizon + 1)
                )
            else:
                # some dataloader load data with warmup period, so leave some periods for it
                # [warmup_len] -> time_start -> [rho] -> [horizon]
                nan_array = torch.isnan(self.y[basin])
                # TODO: 这里的流程有待优化，速度太慢
                lookup.extend(
                    (basin, f)
                    for f in range(warmup_length, max_time_length - rho - horizon + 1)
                    if not torch.all(nan_array[f + rho: f + rho + horizon])
                )
        self.lookup_table = dict(enumerate(lookup))
        self.num_samples = len(self.lookup_table)


class BasinSingleFlowDataset(BaseDataset):
    """one time length output for each grid in a batch"""

    def __init__(self, data_cfgs: dict, is_tra_val_te: str):
        super(BasinSingleFlowDataset, self).__init__(data_cfgs, is_tra_val_te)

    def __getitem__(self, index):
        xc, ys = super(BasinSingleFlowDataset, self).__getitem__(index)
        y = ys[-1, :]
        return xc, y

    def __len__(self):
        return self.num_samples


class DplDataset(BaseDataset):
    """pytorch dataset for Differential parameter learning"""

    def __init__(self, data_cfgs: dict, is_tra_val_te: str):
        """
        Parameters
        ----------
        data_cfgs
            configs for reading source data
        is_tra_val_te
            train, vaild or test
        """
        super(DplDataset, self).__init__(data_cfgs, is_tra_val_te)
        # we don't use y_un_norm as its name because in the main function we will use "y"
        # For physical hydrological models, we need warmup, hence the target values should exclude data in warmup period
        self.warmup_length = data_cfgs["warmup_length"]
        self.target_as_input = data_cfgs["target_as_input"]
        self.constant_only = data_cfgs["constant_only"]
        if self.target_as_input and (not self.train_mode):
            # if the target is used as input and train_mode is False,
            # we need to get the target data in training period to generate pbm params
            self.train_dataset = DplDataset(data_cfgs, is_tra_val_te="train")

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
        if self.train_mode:
            xc_norm, _ = super(DplDataset, self).__getitem__(item)
            basin, time = self.lookup_table[item]
            if self.target_as_input:
                # y_morn and xc_norm are concatenated and used for DL model
                y_norm = torch.from_numpy(
                    self.y[basin, time - warmup: time + rho + horizon, :]
                ).float()
                # the order of xc_norm and y_norm matters, please be careful!
                z_train = torch.cat((xc_norm, y_norm), -1)
            elif self.constant_only:
                # only use attributes data for DL model
                z_train = torch.from_numpy(self.c[basin, :]).float()
            else:
                z_train = xc_norm.float()
            x_train = self.x_origin[basin, time - warmup: time + rho + horizon, :]
            y_train = self.y_origin[basin, time: time + rho + horizon, :]
        else:
            x_norm = self.x[item, :, :]
            if self.target_as_input:
                # when target_as_input is True,
                # we need to use training data to generate pbm params
                x_norm = self.train_dataset.x[item, :, :]
            if self.c is None or self.c.shape[-1] == 0:
                xc_norm = torch.from_numpy(x_norm).float()
            else:
                c_norm = self.c[item, :]
                c_norm = (
                    np.repeat(c_norm, x_norm.shape[0], axis=0)
                    .reshape(c_norm.shape[0], -1)
                    .T
                )
                xc_norm = torch.from_numpy(
                    np.concatenate((x_norm, c_norm), axis=1)
                ).float()
            if self.target_as_input:
                # when target_as_input is True,
                # we need to use training data to generate pbm params
                # when used as input, warmup_length not included for y
                y_norm = torch.from_numpy(self.train_dataset.y[item, :, :]).float()
                # the order of xc_norm and y_norm matters, please be careful!
                z_train = torch.cat((xc_norm, y_norm), -1)
            elif self.constant_only:
                # only use attributes data for DL model
                z_train = torch.from_numpy(self.c[item, :]).float()
            else:
                z_train = xc_norm
            x_train = self.x_origin[item, :, :]
            y_train = self.y_origin[item, warmup:, :]
        return (
            torch.from_numpy(x_train).float(),
            z_train,
        ), torch.from_numpy(y_train).float()

    def __len__(self):
        return self.num_samples if self.train_mode else len(self.t_s_dict["sites_id"])


class FlexibleDataset(BaseDataset):
    """A dataset whose datasources are from multiple sources according to the configuration"""

    def __init__(self, data_cfgs: dict, is_tra_val_te: str):
        super(FlexibleDataset, self).__init__(data_cfgs, is_tra_val_te)

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
        if "streamflow" in y:
            area = data_source_.camels.read_area(self.t_s_dict["sites_id"])
            y.update(streamflow_unit_conv(y[["streamflow"]], area))
        self.x_origin, self.y_origin, self.c_origin = self._to_dataarray_with_unit(
            x, y, c
        )

    def _normalize(self):
        var_to_source_map = self.data_cfgs["var_to_source_map"]
        for var_name in var_to_source_map:
            source_name = var_to_source_map[var_name]
            data_source_ = self.data_source[source_name]
            break
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


class HydroMeanDataset(BaseDataset):
    def __init__(self, data_cfgs: dict, is_tra_val_te: str):
        super(HydroMeanDataset, self).__init__(data_cfgs, is_tra_val_te)

    @property
    def data_source(self):
        time_unit = (
            str(self.data_cfgs["min_time_interval"]) + self.data_cfgs["min_time_unit"]
        )
        return SelfMadeHydroDataset(
            self.data_cfgs["source_cfgs"]["source_path"],
            time_unit=[time_unit],
        )

    def _normalize(self):
        x, y, c = super()._normalize()
        return x.compute(), y.compute(), c.compute()

    def _read_xyc(self):
        data_target_ds = self._prepare_target()
        if data_target_ds is not None:
            y_origin = self._trans2da_and_setunits(data_target_ds)
        else:
            y_origin = None

        data_forcing_ds = self._prepare_forcing()
        if data_forcing_ds is not None:
            x_origin = self._trans2da_and_setunits(data_forcing_ds)
        else:
            x_origin = None

        if self.data_cfgs["constant_cols"]:
            data_attr_ds = self.data_source.read_BA_xrdataset(
                self.t_s_dict["sites_id"],
                self.data_cfgs["constant_cols"],
                self.data_cfgs["source_cfgs"]["source_path"]["attributes"],
            )
            c_orgin = self._trans2da_and_setunits(data_attr_ds)
        else:
            c_orgin = None
        self.x_origin, self.y_origin, self.c_origin = x_origin, y_origin, c_orgin

    def __len__(self):
        return self.num_samples

    def _prepare_forcing(self):
        return self._read_from_minio(self.data_cfgs["relevant_cols"])

    def _prepare_target(self):
        return self._read_from_minio(self.data_cfgs["target_cols"])

    def _read_from_minio(self, var_lst):
        gage_id_lst = self.t_s_dict["sites_id"]
        t_range = self.t_s_dict["t_final_range"]
        interval = self.data_cfgs["min_time_interval"]
        time_unit = (
            str(self.data_cfgs["min_time_interval"]) + self.data_cfgs["min_time_unit"]
        )

        subset_list = []
        for start_date, end_date in t_range:
            adjusted_end_date = (
                datetime.strptime(end_date, "%Y-%m-%d-%H") + timedelta(hours=interval)
            ).strftime("%Y-%m-%d-%H")
            subset = self.data_source.read_ts_xrdataset(
                gage_id_lst,
                t_range=[start_date, adjusted_end_date],
                var_lst=var_lst,
                time_units=[time_unit],
            )
            subset_list.append(subset[time_unit])
        return xr.concat(subset_list, dim="time")


class Seq2SeqDataset(HydroMeanDataset):
    def __init__(self, data_cfgs: dict, is_tra_val_te: str):
        super(Seq2SeqDataset, self).__init__(data_cfgs, is_tra_val_te)

    def _read_xyc(self):
        data_target_ds = self._prepare_target()
        if data_target_ds is not None:
            y_origin = self._trans2da_and_setunits(data_target_ds)
        else:
            y_origin = None

        data_forcing_ds = self._prepare_forcing()
        if data_forcing_ds is not None:
            x_origin = self._trans2da_and_setunits(data_forcing_ds)
            x_origin = xr.where(x_origin < 0, float("nan"), x_origin)
        else:
            x_origin = None

        if self.data_cfgs["constant_cols"]:
            data_attr_ds = self.data_source.read_attr_xrdataset(
                self.t_s_dict["sites_id"],
                self.data_cfgs["constant_cols"],
            )
            c_orgin = self._trans2da_and_setunits(data_attr_ds)
        else:
            c_orgin = None
        self.x_origin, self.y_origin, self.c_origin = x_origin, y_origin, c_orgin

    def __getitem__(self, item: int):
        basin, idx = self.lookup_table[item]
        rho = self.rho
        horizon = self.horizon
        prec = self.data_cfgs["prec_window"]

        p = self.x[basin, idx + 1: idx + rho + horizon + 1, 0].reshape(-1, 1)
        s = self.x[basin, idx: idx + rho, 1:]
        x = np.concatenate((p[:rho], s), axis=1)

        c = self.c[basin, :]
        c = np.tile(c, (rho + horizon, 1))
        x = np.concatenate((x, c[:rho]), axis=1)

        x_h = np.concatenate((p[rho:], c[rho:]), axis=1)
        y = self.y[basin, idx + rho - prec + 1: idx + rho + horizon + 1, :]

        if self.is_tra_val_te == "train":
            return [
                torch.from_numpy(x).float(),
                torch.from_numpy(x_h).float(),
                torch.from_numpy(y).float(),
            ], torch.from_numpy(y).float()
        return [
            torch.from_numpy(x).float(),
            torch.from_numpy(x_h).float(),
        ], torch.from_numpy(y).float()


class TransformerDataset(Seq2SeqDataset):
    def __init__(self, data_cfgs: dict, is_tra_val_te: str):
        super(TransformerDataset, self).__init__(data_cfgs, is_tra_val_te)

    def __getitem__(self, item: int):
        basin, idx = self.lookup_table[item]
        rho = self.rho
        horizon = self.horizon

        p = self.x[basin, idx + 1: idx + rho + horizon + 1, 0]
        s = self.x[basin, idx: idx + rho, 1]
        x = np.stack((p[:rho], s), axis=1)

        c = self.c[basin, :]
        c = np.tile(c, (rho + horizon, 1))
        x = np.concatenate((x, c[:rho]), axis=1)

        x_h = np.concatenate((p[rho:].reshape(-1, 1), c[rho:]), axis=1)
        y = self.y[basin, idx + rho + 1: idx + rho + horizon + 1, :]

        return [
            torch.from_numpy(x).float(),
            torch.from_numpy(x_h).float(),
        ], torch.from_numpy(y).float()


class GNNDataset(Seq2SeqDataset):
    def __init__(self, data_cfgs: dict, is_tra_val_te: str):
        self.data_cfgs = data_cfgs
        self.is_tra_val_te = is_tra_val_te
        upstream_ds = self.get_upstream_ds()
        # upstream_ds['basin_id'] = upstream_ds['basin_id'].astype(str).str.zfill(8)
        self.x_up = upstream_ds
        super(GNNDataset, self).__init__(data_cfgs, is_tra_val_te)

    def __getitem__(self, item: int):
        basin, idx = self.lookup_table[item]
        rho, horizon = self.rho, self.horizon
        prec = self.data_cfgs["prec_window"]
        p = self.x[basin, idx + 1: idx + rho + horizon + 1, 0]
        s = self.x[basin, idx: idx + rho, 1:]
        x = torch.concatenate([p[:rho].unsqueeze(1), s], dim=1)
        stream_up = self.x_up[basin, idx: idx + rho, :]
        c = self.c[basin, :]
        c = torch.tensor(np.tile(c, (rho + horizon, 1)))
        x = torch.concatenate((x, c[:rho], stream_up), dim=1)
        p_rho = p[rho:].unsqueeze(1)
        pad_p = torch.nn.functional.pad(p_rho, (0, 0, 0, c[rho:].shape[0] - p_rho.shape[0]), 'constant', 0) \
                if p_rho.shape[0] < c[rho:].shape[0] else p_rho
        x_h = torch.concatenate((pad_p, c[rho:]), dim=1)
        y = self.y[basin, idx + rho - prec + 1: idx + rho + horizon + 1, :]
        if y.shape[0] < horizon + prec:
           y = torch.nn.functional.pad(y, (0, 0, 0, horizon + prec - y.shape[0]), 'constant', 0)
        result = (([x.float(), x_h.float(), y.float()], y.float()) if self.is_tra_val_te == "train"
                  else ([x.float(), x_h.float()], y.float()))
        return result

    def __len__(self):
        # 15118/train, 2626/test
        return self.num_samples

    @property
    def data_source(self):
        source_name = self.data_cfgs["source_cfgs"]["source_name"]
        source_path = self.data_cfgs["source_cfgs"]["source_path"]
        other_settings = self.data_cfgs["source_cfgs"].get("other_settings", {})
        return data_sources_dict[source_name](source_path, **other_settings)

    def _read_xyc(self):
        y_origin = self._prepare_target()
        x_origin = self._prepare_forcing()
        x_origin = x_origin.with_columns(pl.when(pl.col(pl.Float64)<0).then(np.nan).otherwise(pl.col(pl.Float64)).name.keep())
        basin_sites = [site_id for site_id in self.t_s_dict["sites_id"] if ((len(site_id.split('_'))==2) & ('HML' not in site_id))]
        station_sites = [site_id for site_id in self.t_s_dict["sites_id"] if ((len(site_id.split('_'))==3) | ('HML' in site_id))]
        station_basins_df = self.data_cfgs['basins_stations_df']
        name_dict = {site: station_basins_df[station_basins_df['station_id']==site]['basin_id'].values[0] for site in station_sites}
        if self.data_cfgs["constant_cols"]:
            data_attr_ds = self.data_source.read_attr_xrdataset(basin_sites, self.data_cfgs["constant_cols"])
            c_origin = self._trans2da_and_setunits(data_attr_ds)
            zc_origin = xr.DataArray().assign_coords({'basin': None})
            for name in name_dict.keys():
                zone_attr_ds = self.data_source.read_attr_xrdataset(name_dict[name], self.data_cfgs["constant_cols"])
                z_origin = self._trans2da_and_setunits(zone_attr_ds).assign_coords(basin=name)
                zc_origin = xr.concat([zc_origin, z_origin], dim='basin')
            c_origin = xr.concat([c_origin, zc_origin.dropna(dim="basin")], dim="basin")
        else:
            c_origin = None
        self.x_origin, self.y_origin, self.c_origin = x_origin, y_origin, c_origin

    def _read_from_minio(self, var_lst):
        gage_id_lst = self.t_s_dict["sites_id"]
        basin_sites = [site_id for site_id in gage_id_lst if ((len(site_id.split('_'))==2) & ('HML' not in site_id))]
        station_sites = [site_id for site_id in gage_id_lst if ((len(site_id.split('_'))==3) | ('HML' in site_id))]
        t_range = self.t_s_dict["t_final_range"]
        interval = self.data_cfgs["min_time_interval"]
        time_unit = (
            str(self.data_cfgs["min_time_interval"]) + self.data_cfgs["min_time_unit"]
        )
        subset_list = []
        for start_date, end_date in t_range:
            adjusted_end_date = (
                datetime.strptime(end_date, "%Y-%m-%d-%H") + timedelta(hours=interval)
            ).strftime("%Y-%m-%d-%H")
            subset = self.data_source.read_ts_xrdataset(
                basin_sites,
                t_range=[start_date, adjusted_end_date],
                var_lst=var_lst,
                time_units=[time_unit],
            )
            zt_origin_path = os.path.join(self.data_cfgs['validation_path'], f'stations_weather_{len(var_lst)}.parquet')
            if not os.path.exists(zt_origin_path):
                station_basins_df = self.data_cfgs['basins_stations_df']
                name_dict = {site: station_basins_df[station_basins_df['station_id']==site]['basin_id'].values[0] for site in station_sites}
                zt_origin = pl.DataFrame()
                # 291个站点，约45分钟生成zt_origin
                for name in name_dict.keys():
                    zone_ts_ds = self.data_source.read_ts_xrdataset([name_dict[name]], t_range=[start_date, adjusted_end_date],
                                                                    var_lst=var_lst, time_units=[time_unit])[time_unit]
                    zone_ts_ds = zone_ts_ds.with_columns(pl.Series('basin_id', np.repeat(name, len(zone_ts_ds))))
                    zt_origin = zt_origin.vstack(zone_ts_ds)
                zt_origin.write_parquet(zt_origin_path)
            else:
                zt_origin = pl.read_parquet(zt_origin_path)
                tm_range = pd.date_range(start_date, adjusted_end_date, freq=time_unit)
                zt_part = (zt_origin.group_by('basin_id', maintain_order=True).agg(pl.all().slice(0, len(tm_range))).
                            explode(pl.exclude('basin_id')))
                zt_origin = pl.concat([zt_part[:, 1:-1], zt_part[['basin_id', 'time']]], how='horizontal')
            zt_origin = zt_origin.with_columns(pl.col(pl.Float64).cast(pl.Float32))
            subsets = pl.concat([subset[time_unit], zt_origin])
            subset_list.append(subsets)
        return pl.concat(subset_list)

    def _normalize(self):
        if self.data_cfgs["pre_norm"]:
            scaler_hub = ScalerHub(
                self.y_origin,
                self.x_origin,
                self.c_origin,
                data_cfgs=self.data_cfgs,
                is_tra_val_te=self.is_tra_val_te,
                data_source=self.data_source)
            self.target_scaler = scaler_hub.target_scaler
            return scaler_hub.x, scaler_hub.y, scaler_hub.c.compute()
        else:
            return self.x_origin, self.y_origin, self.c_origin.compute()


    def _kill_nan(self, x, y, c):
        data_cfgs = self.data_cfgs
        y_rm_nan = data_cfgs["target_rm_nan"]
        x_rm_nan = data_cfgs["relevant_rm_nan"]
        c_rm_nan = data_cfgs["constant_rm_nan"]
        if x_rm_nan:
            # As input, we cannot have NaN values
            _fill_gaps_pq(x, fill_nan="interpolate")
            warn_if_nan_pq(x)
        if y_rm_nan:
            _fill_gaps_pq(y, fill_nan="interpolate")
            warn_if_nan_pq(y)
        if c_rm_nan:
            _fill_gaps_da(c, fill_nan="mean")
            warn_if_nan(c)
        warn_if_nan_pq(x, nan_mode="all")
        warn_if_nan_pq(y, nan_mode="all")
        warn_if_nan(c, nan_mode="all")
        return x, y, c


    def _trans2nparr(self):
        """To make __getitem__ more efficient,
        we transform x, y, c to numpy array with shape (nsample, nt, nvar)
        """
        station_len = len(self.x['basin_id'].unique())
        x_truncated = (self.x.group_by('basin_id', maintain_order=True).agg(pl.all().slice(0, len(self.x) // station_len)).
                       explode(pl.exclude("basin_id")))
        y_trucnated = (self.y.group_by('basin_id', maintain_order=True).agg(pl.all().slice(0, len(self.x) // station_len)).
                       explode(pl.exclude("basin_id")))
        self.x = x_truncated[:, 1:-1].to_torch().reshape(station_len, len(self.x) // station_len, self.x.width-2)
        self.y = y_trucnated[:, 1:-1].to_torch().reshape(station_len, len(self.y) // station_len, self.y.width-2)
        basins_len = len(self.x_up['basin_id'].unique())
        self.x_up = self.x_up[:, 2:].to_torch().reshape(basins_len, len(self.x_up) // basins_len, self.x_up.width-2)
        if self.c is not None and self.c.shape[-1] > 0:
            self.c = self.c.transpose("basin", "variable").to_numpy()
            self.c_origin = self.c_origin.transpose("basin", "variable").to_numpy()


    def get_upstream_ds(self):
        res_dir = self.data_cfgs["test_path"]
        min_time_interval = self.data_cfgs['min_time_interval']
        obj_len = len(self.data_cfgs['object_ids'])
        cut_len = self.data_cfgs['upstream_cut']
        tra_val_te_ds = os.path.join(res_dir, f'upstream_ds_{self.is_tra_val_te}_{obj_len}_{min_time_interval}h.parquet')
        if os.path.exists(tra_val_te_ds):
            total_df = pl.read_parquet(tra_val_te_ds)
            save_cols = min(cut_len+2, len(total_df.columns)-2)
        else:
            nodes_df = self.data_cfgs['basins_stations_df']
            nodes_df['basin_id'] = nodes_df['basin_id'].astype(str).str.zfill(8)
            nodes_df['node_id'] = nodes_df['node_id'].astype(str)
            max_app_cols = int(nodes_df['upstream_len'].max()) - 1
            save_cols = min(cut_len+2, max_app_cols)
            total_df = pl.DataFrame()
            freq = f'{min_time_interval}h'
            for node_name in np.unique(nodes_df['station_id'].to_numpy()):
                station_df = pl.DataFrame()
                # basin_id = nodes_df[nodes_df['station_id'] == node_name]['basin_id'].to_list()[0]
                node_id = nodes_df['node_id'][nodes_df['station_id'] == node_name].to_list()[0]
                up_set = nx.ancestors(self.data_cfgs['graph'], node_id)
                if len(up_set) > 0:
                    up_node_names = nodes_df['station_id'][nodes_df['node_id'].isin(up_set)]
                    # read_data_with_id放在if下方减少读盘次数
                    # 上游站点最多的10701300站，在成图阶段上游有30个站，但是basin_stations里以其为下游的站却有34个，原因暂不明确
                    # 与流域相交的站点，和上游能找到的站点数量不等，后者比前者少得多，可能是因为USGS的站点也有“三岔口问题”
                    streamflow_dfs = [self.read_data_with_id(up_node_name) for up_node_name in up_node_names]
                    if len(streamflow_dfs) > max_app_cols:
                        streamflow_dfs = streamflow_dfs[:max_app_cols]
                    for date_tuple in self.data_cfgs[f"t_range_{self.is_tra_val_te}"]:
                        date_times = pl.datetime_range(pd.to_datetime(date_tuple[0]), pd.to_datetime(date_tuple[1]), interval=freq, eager=True)
                        up_col_dict = {f'streamflow_up_{i}': np.zeros(len(date_times), dtype=np.float32) for i in range(max_app_cols)}
                        up_str_df = pl.DataFrame(up_col_dict)
                        for i in range(len(streamflow_dfs)):
                            data_table = streamflow_dfs[i]
                            # 有的time列是object，和datetime64[μs]不等，所以这里先转成datetime再比较
                            if len(data_table)>0:
                                pl_times = pl.from_pandas(pd.to_datetime(data_table['time'])).cast(pl.Datetime)
                                data_table = data_table.with_columns(pl.Series(pl_times).alias('time'))
                                # 存在这样的情况：2013年直接跳到2018年，导致时间轴上没有数据
                                up_str_col = (data_table.filter(data_table['time'].is_in(date_times)).
                                              unique('time', keep='first', maintain_order=True))['streamflow']
                                up_str_arr = np.float32(up_str_col.fill_nan(0).to_numpy())
                                if len(up_str_arr) < len(date_times):
                                    if len(up_str_arr) > 0:
                                        up_str_arr = np.pad(up_str_arr, (0, len(date_times) - len(up_str_arr)), 'edge')
                                    else:
                                        up_str_arr = np.zeros(len(date_times), dtype=np.float32)
                            else:
                                up_str_arr = np.zeros(len(date_times), dtype=np.float32)
                            up_str_df = up_str_df.with_columns(pl.Series(f'streamflow_up_{i}', up_str_arr))
                        station_df = station_df.with_columns([pl.Series('basin_id', np.repeat(node_name, len(date_times))),
                                                              pl.Series('time', date_times)])
                        station_df = pl.concat([station_df, up_str_df], how='horizontal')
                        total_df = pl.concat([total_df, station_df])
                else:
                    for date_tuple in self.data_cfgs[f"t_range_{self.is_tra_val_te}"]:
                        date_times = pl.datetime_range(pd.to_datetime(date_tuple[0]), pd.to_datetime(date_tuple[1]), interval=freq, eager=True)
                        up_str_df = pl.DataFrame({f'streamflow_up_{i}': np.zeros(len(date_times), dtype=np.float32) for i in range(max_app_cols)})
                        station_df = station_df.with_columns([pl.Series('basin_id', np.repeat(node_name, len(date_times))),
                                                              pl.Series('time', date_times)])
                        station_df = pl.concat([station_df, up_str_df],  how='horizontal')
                        total_df = pl.concat([total_df, station_df])
            if total_fab.global_rank == 0:
                total_df.write_parquet(tra_val_te_ds)
                total_fab.barrier()
        total_df = total_df[total_df.columns[:save_cols]]
        return total_df

    def read_data_with_id(self, node_name: str):
        import geopandas as gpd
        # node_name: IOWA, WY_DCP_XXXXX
        # iowa流量站有653个，但是数据足够多的只有222个被整编到nc文件中
        min_time_interval = self.data_cfgs['min_time_interval']
        if ('_' in node_name) & (len(node_name.split('_'))==3):
            iowa_stream_ds = pl.read_parquet("/ftproot/iowa_streamflow_stas.parquet")
            if node_name in iowa_stream_ds['station']:
                node_df = iowa_stream_ds.filter(iowa_stream_ds['station']==node_name)
                node_df = node_df.rename({'utc_valid': 'time'})
                node_df = node_df[['time', 'streamflow']]
                sta_basin_df = self.data_cfgs['basins_stations_df']
                sta_basin_df['basin_id'] = sta_basin_df['basin_id'].astype(str).str.zfill(8)
                basin_id = sta_basin_df[sta_basin_df['station_id'] == node_name]['basin_id'].to_list()[0]
                area_gdf = gpd.read_file(self.data_cfgs['basins_shp'])
                area = area_gdf[area_gdf['BASIN_ID'].str.contains(basin_id)]['AREA'].to_list()[0]
                # GNNMultiTaskHydro.get_upstream_graph方法，如果遇到iowa的站必须除以流域面积做平均，否则误差会极大
                # iowa流量站单位是KCFS(1000 ft3/s)，这里除以流域面积，并变成mm/h
                node_df = node_df.with_columns((pl.col('streamflow') / (35.31 * area) * 3600 * min_time_interval).alias('streamflow'))
            else:
                node_df = pl.DataFrame()
        # node_name: songliao_21401550 or HML_XXXXX
        elif ('_' in node_name) & (len(node_name.split('_'))==2):
            if 'HML' not in node_name:
                node_df = pl.read_csv(f'/ftproot/basins-interim/timeseries/1h/{node_name}.csv')[['time', 'streamflow']]
            else:
                hml_stream_ds = pl.read_parquet(f'/ftproot/hml_camels_stations.parquet')
                if node_name.split('_')[-1] in hml_stream_ds['station']:
                    node_df = hml_stream_ds.filter(hml_stream_ds['station']==node_name.split('_')[-1])
                    node_df = node_df.rename({'valid[UTC]': 'time', 'Flow[kcfs]': 'streamflow'})
                    node_df = node_df[['time', 'streamflow']]
                    sta_basin_df = self.data_cfgs['basins_stations_df']
                    sta_basin_df['basin_id'] = sta_basin_df['basin_id'].astype(str).str.zfill(8)
                    basin_id = sta_basin_df[sta_basin_df['station_id'] == node_name]['basin_id'].to_list()[0]
                    area_gdf = gpd.read_file(self.data_cfgs['basins_shp'])
                    area = area_gdf[area_gdf['BASIN_ID'].str.contains(basin_id)]['AREA'].to_list()[0]
                    # GNNMultiTaskHydro.get_upstream_graph方法，如果遇到HML的站必须除以流域面积做平均，否则误差会极大
                    # HML流量站单位是KCFS(1000 ft3/s)，这里除以流域面积，并变成mm/h
                    node_df = node_df.with_columns((pl.col('streamflow') / (35.31 * area) * 3600 * min_time_interval).alias('streamflow'))
                else:
                    node_df = pl.DataFrame()
        # node_name: str(21401550)
        elif '_' not in node_name:
            csv_path = f'/ftproot/basins-interim/timeseries/1h/camels_{node_name}.csv'
            node_csv_path = csv_path if os.path.exists(csv_path) else csv_path.replace('camels', 'songliao')
            node_df = pl.read_csv(node_csv_path)[['time', 'streamflow']]
        else:
            node_df = pl.DataFrame()
        if 'streamflow' in node_df.columns:
            node_df = node_df.with_columns([pl.col('streamflow').cast(pl.Float32)])
        return node_df
