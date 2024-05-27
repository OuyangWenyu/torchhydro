"""
Author: Wenyu Ouyang
Date: 2024-04-08 18:16:53
LastEditTime: 2024-05-27 17:48:17
LastEditors: Wenyu Ouyang
Description: A pytorch dataset class; references to https://github.com/neuralhydrology/neuralhydrology
FilePath: \torchhydro\torchhydro\datasets\data_sets.py
Copyright (c) 2024-2024 Wenyu Ouyang. All rights reserved.
"""

import logging
import re
import sys
import torch
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
from torch.utils.data import Dataset
from tqdm import tqdm
from hydrodatasource.utils.utils import streamflow_unit_conv

from torchhydro.configs.config import DATE_FORMATS
from torchhydro.datasets.data_scalers import (
    ScalerHub,
    MutiBasinScaler,
)
from torchhydro.datasets.data_sources import data_sources_dict

from torchhydro.datasets.data_utils import (
    warn_if_nan,
    wrap_t_s_dict,
)
from hydrodatasource.reader.data_source import HydroBasins

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
        return data_sources_dict[source_name](source_path)

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
            trange_type_num = 1
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
        x = self.x[basin, idx - warmup_length : idx + self.rho + self.horizon, :]
        y = self.y[basin, idx : idx + self.rho + self.horizon, :]
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

        streamflow_unit = data_output_ds["streamflow"].attrs["units"]
        prcp_unit = data_forcing_ds["prcp"].attrs["units"]

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
                lookup.append((basin, warmup_length))
            else:
                # some dataloader load data with warmup period, so leave some periods for it
                # [warmup_len] -> time_start -> [rho] -> [horizon]
                nan_array = np.isnan(self.y[basin, :, :])
                lookup.extend(
                    (basin, f)
                    for f in range(warmup_length, max_time_length - rho - horizon + 1)
                    if not np.all(nan_array[f + rho : f + rho + horizon])
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
    # TODO: USE NUMPY ARRAY INSTEAD OF DATAARRAY FOR GET_ITEM
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
        if self.train_mode:
            xc_norm, _ = super(DplDataset, self).__getitem__(item)
            basin, time = self.lookup_table[item]
            warmup_length = self.warmup_length
            if self.target_as_input:
                # y_morn and xc_norm are concatenated and used for DL model
                y_norm = torch.from_numpy(
                    self.y[basin, time - warmup_length : time + self.rho, :]
                ).float()
                # the order of xc_norm and y_norm matters, please be careful!
                z_train = torch.cat((xc_norm, y_norm), -1)
            elif self.constant_only:
                # only use attributes data for DL model
                z_train = torch.from_numpy(self.c[basin, :]).float()
            else:
                z_train = xc_norm.float()
            x_train = (
                self.x_origin.sel(
                    basin=basin,
                    time=slice(
                        time - np.timedelta64(warmup_length, "D"),
                        time + np.timedelta64(self.rho - 1, "D"),
                    ),
                )
                .to_array()
                .to_numpy()
                .T
            )
            y_train = (
                self.y_origin.sel(
                    basin=basin,
                    time=slice(
                        time,
                        time + np.timedelta64(self.rho - 1, "D"),
                    ),
                )
                .to_array()
                .to_numpy()
                .T
            )
        else:
            basin = self.t_s_dict["sites_id"][item]
            x_norm = self.x.sel(basin=basin).to_numpy().T
            if self.target_as_input:
                # when target_as_input is True,
                # we need to use training data to generate pbm params
                x_norm = self.train_dataset.sel(basin=basin).to_numpy().T
            if self.c is None or self.c.shape[-1] == 0:
                xc_norm = torch.from_numpy(x_norm).float()
            else:
                c_norm = self.c.sel(basin=basin).values
                c_norm = (
                    np.repeat(c_norm, x_norm.shape[0], axis=0)
                    .reshape(c_norm.shape[0], -1)
                    .T
                )
                xc_norm = np.concatenate((x_norm, c_norm), axis=1)
            warmup_length = self.warmup_length
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
                z_train = torch.from_numpy(xc_norm).float()
            x_train = self.x_origin.sel(basin=basin).to_array().to_numpy().T
            y_train = (
                self.y_origin.sel(basin=basin)
                .isel(time=slice(warmup_length, None))
                .to_array()
                .to_numpy()
                .T
            )
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
        self.forecast_length = data_cfgs["forecast_length"]
        super(HydroMeanDataset, self).__init__(data_cfgs, is_tra_val_te)

    @property
    def data_source(self):
        return HydroBasins(self.data_cfgs["source_cfgs"]["source_path"])

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

    def __getitem__(self, item: int):
        basin, time = self.lookup_table[item]
        forecast_history = self.rho
        horizon = self.forecast_length
        warmup_length = self.warmup_length
        gpm_tp = (
            self.x.sel(
                variable="gpm_tp",
                basin=basin,
                time=slice(
                    time - np.timedelta64(warmup_length + forecast_history - 1, "h"),
                    time,
                ),
            )
            .to_numpy()
            .T
        ).reshape(-1, 1)
        gfs_tp = (
            self.x.sel(
                variable="gfs_tp",
                basin=basin,
                time=slice(
                    time + np.timedelta64(1, "h"),
                    time + np.timedelta64(horizon, "h"),
                ),
            )
            .to_numpy()
            .T
        ).reshape(-1, 1)
        x = np.concatenate((gpm_tp, gfs_tp), axis=0)
        if self.c is not None and self.c.shape[-1] > 0:
            c = self.c.sel(basin=basin).values
            c = np.tile(c, (warmup_length + forecast_history + horizon, 1))
            x = np.concatenate((x, c), axis=1)
        y = (
            self.y.sel(
                basin=basin,
                time=slice(
                    time + np.timedelta64(1, "h"),
                    time + np.timedelta64(horizon, "h"),
                ),
            )
            .to_numpy()
            .T
        )
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()

    def __len__(self):
        return self.num_samples

    def _prepare_target(self):
        gage_id_lst = self.t_s_dict["sites_id"]
        # t_range加一小时或三小时
        t_range = [
            (
                datetime.fromisoformat(date_tuple[0]) + timedelta(hours=3),
                datetime.fromisoformat(date_tuple[1]) + timedelta(hours=3),
            )
            for date_tuple in self.t_s_dict["t_final_range"]
        ]
        var_lst = self.data_cfgs["target_cols"]
        path = self.data_cfgs["source_cfgs"]["source_path"]["target"]

        if var_lst is None or not var_lst:
            return None

        data = self.data_source.merge_nc_minio_datasets(path, gage_id_lst, var_lst)

        all_vars = data.data_vars
        if any(var not in data.variables for var in var_lst):
            raise ValueError(f"var_lst must all be in {all_vars}")
        subset_list = []
        for start_date, end_date in t_range:
            adjusted_end_date = (
                end_date + timedelta(hours=self.forecast_length)
            ).strftime("%Y-%m-%d")
            subset = data.sel(time=slice(start_date, adjusted_end_date))
            subset_list.append(subset)
        return xr.concat(subset_list, dim="time")

    def _prepare_forcing(self):
        gage_id_lst = self.t_s_dict["sites_id"]
        t_range = self.t_s_dict["t_final_range"]
        var_lst = self.data_cfgs["relevant_cols"]
        path = self.data_cfgs["source_cfgs"]["source_path"]["forcing"]

        if var_lst is None:
            return None

        data = self.data_source.merge_nc_minio_datasets(path, gage_id_lst, var_lst)

        var_subset_list = []
        for start_date, end_date in t_range:
            adjusted_start_date = (
                datetime.strptime(start_date, "%Y-%m-%d") - timedelta(hours=self.rho)
            ).strftime("%Y-%m-%d")
            adjusted_end_date = (
                datetime.strptime(end_date, "%Y-%m-%d")
                + timedelta(hours=self.forecast_length)
            ).strftime("%Y-%m-%d")
            subset = data.sel(time=slice(adjusted_start_date, adjusted_end_date))
            var_subset_list.append(subset)

        return xr.concat(var_subset_list, dim="time")


class HydroGridDataset(HydroMeanDataset):
    def __init__(self, data_cfgs: dict, is_tra_val_te: str):
        super(HydroGridDataset, self).__init__(data_cfgs, is_tra_val_te)

    def _load_data(self):
        self.data_source = HydroBasins(self.data_cfgs["data_path"])
        self.forecast_length = self.data_cfgs["forecast_length"]
        self._pre_load_data()

        data_target_ds = self._prepare_target()
        if data_target_ds is not None:
            y_origin = self._trans2da_and_setunits(data_target_ds)
        else:
            y_origin = None

        data_gpm = self._prepare_forcing(0)

        if self.data_cfgs["relevant_cols"][1] != ["None"]:
            data_gfs = self._prepare_forcing(1)
        else:
            data_gfs = None

        if self.data_cfgs["relevant_cols"][2] != ["None"]:
            data_smap = self._prepare_forcing(2)
        else:
            data_smap = None

        if self.data_cfgs["constant_cols"]:
            data_attr_ds = self.data_source.read_BA_xrdataset(
                self.t_s_dict["sites_id"],
                self.data_cfgs["constant_cols"],
                self.data_cfgs["data_path"]["attributes"],
            )
            data_attr = self._trans2da_and_setunits(data_attr_ds)
        else:
            data_attr = None

        scaler_hub = MutiBasinScaler(
            y_origin,
            data_gpm,
            data_attr,
            data_gfs,
            data_smap,
            self.data_cfgs,
            self.is_tra_val_te,
            self.data_source,
        )

        self.x, self.y, self.c, self.g, self.s = self.kill_nan(
            scaler_hub.x, scaler_hub.y, scaler_hub.c, scaler_hub.g, scaler_hub.s
        )

        self.target_scaler = scaler_hub.target_scaler

        self._create_lookup_table()

    def kill_nan(self, x, y, c, g, s):
        data_cfgs = self.data_cfgs
        y_rm_nan = data_cfgs["target_rm_nan"]
        x_rm_nan = data_cfgs["relevant_rm_nan"]
        c_rm_nan = data_cfgs["constant_rm_nan"]

        if x_rm_nan:
            for xx in x.values():
                for i in range(xx.shape[0]):
                    xx[i] = xx[i].interpolate_na(
                        dim="time_now", fill_value="extrapolate"
                    )
                warn_if_nan(xx)

        if y_rm_nan:
            _fill_gaps_da(y, fill_nan="interpolate")
            warn_if_nan(y)

        if c_rm_nan and c is not None:
            _fill_gaps_da(c, fill_nan="mean")
            warn_if_nan(c)

        if x_rm_nan and g is not None:
            for gg in g.values():
                for i in range(gg.shape[0]):
                    gg[i] = gg[i].interpolate_na(dim="time", fill_value="extrapolate")
                warn_if_nan(gg)
        if x_rm_nan and s is not None:
            for ss in s.values():
                for i in range(ss.shape[0]):
                    ss[i] = ss[i].interpolate_na(dim="time", fill_value="extrapolate")
                warn_if_nan(ss)

        return x, y, c, g, s

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item: int):
        basin, time = self.lookup_table[item]
        forecast_history = self.rho
        horizon = self.forecast_length
        warmup_length = self.warmup_length
        gpm_tp = (
            self.x[basin]
            .sel(
                time=slice(
                    time - np.timedelta64(warmup_length + forecast_history - 1, "h"),
                    time,
                )
            )
            .values
        )
        x = np.transpose(gpm_tp, (1, 0, 2, 3))

        y = (
            self.y.sel(basin=basin)
            .sel(
                time=slice(
                    time + np.timedelta64(1, "h"),
                    time + np.timedelta64(horizon, "h"),
                )
            )
            .values
        ).T

        if self.c is not None and self.g is None and self.s is None:
            c = self.get_c(basin, x.shape[0])
            return (
                [torch.from_numpy(x).float(), torch.from_numpy(c).float()],
                torch.from_numpy(y).float(),
            )

        elif self.g is not None and self.c is None and self.s is None:
            g = self.get_g(basin, time)
            if x.shape[2] != g.shape[2]:
                x = np.transpose(x, (0, 1, 3, 2))
            x_g = np.concatenate((x, g), axis=1)
            return (torch.from_numpy(x_g).float(), torch.from_numpy(y).float())

        elif self.s is not None and self.c is None and self.g is None:
            s = self.get_s(basin, time)
            return (
                [torch.from_numpy(x).float(), torch.from_numpy(s).float()],
                torch.from_numpy(y).float(),
            )

        elif self.c is not None and self.g is not None and self.s is None:
            c = self.get_c(basin, x.shape[0])
            g = self.get_g(basin, time)
            if x.shape[2] != g.shape[2]:
                x = np.transpose(x, (0, 1, 3, 2))
            x_g = np.concatenate((x, g), axis=1)
            return (
                [torch.from_numpy(x_g).float(), torch.from_numpy(c).float()],
                torch.from_numpy(y).float(),
            )

        elif self.c is not None and self.s is not None and self.g is None:
            c = self.get_c(basin, x.shape[0])
            s = self.get_s(basin, time)
            return (
                [
                    torch.from_numpy(x).float(),
                    torch.from_numpy(c).float(),
                    torch.from_numpy(s).float(),
                ],
                torch.from_numpy(y).float(),
            )

        elif self.s is not None and self.g is not None and self.c is None:
            s = self.get_s(basin, time)
            g = self.get_g(basin, time)
            if x.shape[2] != g.shape[2]:
                x = np.transpose(x, (0, 1, 3, 2))
            x_g = np.concatenate((x, g), axis=1)
            return (
                [torch.from_numpy(x_g).float(), torch.from_numpy(s).float()],
                torch.from_numpy(y).float(),
            )

        elif self.s is not None and self.g is not None and self.c is not None:
            c = self.get_c(basin, x.shape[0])
            g = self.get_g(basin, time)
            s = self.get_s(basin, time)
            if x.shape[2] != g.shape[2]:
                x = np.transpose(x, (0, 1, 3, 2))
            x_g = np.concatenate((x, g), axis=1)
            return (
                [
                    torch.from_numpy(x_g).float(),
                    torch.from_numpy(c).float(),
                    torch.from_numpy(s).float(),
                ],
                torch.from_numpy(y).float(),
            )
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()

    def get_c(self, basin, shape):
        c = self.c.sel(basin=basin).values
        c = np.tile(c, (shape, 1))
        return c

    def get_g(self, basin, time):
        g = (
            self.g[basin]
            .sel(
                time=slice(
                    time - np.timedelta64(self.warmup_length + self.rho - 1, "h"),
                    time + np.timedelta64(self.forecast_length, "h"),
                )
            )
            .values
        )
        return np.transpose(g, (1, 0, 2, 3))

    def get_s(self, basin, time):
        length = int(self.forecast_length * 2.5)
        s = (
            self.s[basin]
            .sel(
                time=slice(
                    time - np.timedelta64(self.warmup_length + self.rho - 1, "h"),
                    time - np.timedelta64(length, "h"),
                )
            )
            .values
        )
        return np.transpose(s, (1, 0, 2, 3))

    def _prepare_forcing(self, data_type):
        gage_id_lst = self.t_s_dict["sites_id"]
        t_range = self.t_s_dict["t_final_range"]
        var_lst = self.data_cfgs["relevant_cols"][data_type]
        if var_lst is None:
            return None

        if data_type == 0:
            path = self.data_cfgs["data_path"]["gpm"]
        elif data_type == 1:
            path = self.data_cfgs["data_path"]["gfs"]
        elif data_type == 2:
            path = self.data_cfgs["data_path"]["smap"]

        file_lst = self.data_source.read_file_lst(path)
        data_dict = {}
        for basin in gage_id_lst:
            data = self.data_source.read_grid_data(file_lst, basin)
            subset_list = []
            for start_date, end_date in t_range:
                adjusted_start_date = (
                    datetime.strptime(start_date, "%Y-%m-%d")
                    - timedelta(hours=self.rho)
                ).strftime("%Y-%m-%d")
                adjusted_end_date = (
                    datetime.strptime(end_date, "%Y-%m-%d")
                    + timedelta(hours=self.forecast_length)
                ).strftime("%Y-%m-%d")
                subset = data.sel(time=slice(adjusted_start_date, adjusted_end_date))
                subset_list.append(subset)
            merged_dataset = xr.concat(subset_list, dim="time")
            data_dict[basin] = merged_dataset.to_array(dim="variable")

        return data_dict


class HydroMultiSourceDataset(HydroMeanDataset):
    def __init__(self, data_cfgs: dict, is_tra_val_te: str):
        super(HydroMultiSourceDataset, self).__init__(data_cfgs, is_tra_val_te)

    def get_x(self, variable, basin, time, forecast_history, offset1=0, offset2=0):
        return (
            self.x.sel(
                variable=variable,
                basin=basin,
                time=slice(
                    time
                    - np.timedelta64(forecast_history - 1, "h")
                    + np.timedelta64(offset1, "h"),
                    (time if offset2 == 0 else time + np.timedelta64(offset2, "h")),
                ),
            )
            .to_numpy()
            .T.reshape(-1, len(variable))
        )

    def get_y(self, basin, time, forecast_length, prec_window):
        return (
            self.y.sel(
                basin=basin,
                time=slice(
                    time - np.timedelta64(prec_window, "h"),
                    time + np.timedelta64((forecast_length - 1), "1h"),
                ),
            )
            .to_numpy()
            .T
        )

    def __getitem__(self, item: int):
        basin, time = self.lookup_table[item]
        forecast_history = self.rho
        sm_length = forecast_history - self.data_cfgs["cnn_size"]
        x, y, s = None, None, None
        var_lst = self.data_cfgs["relevant_cols"]
        station_tp_present = "sta_tp" in var_lst

        if station_tp_present:
            delay = 6
            gpm_tp = self.get_x("gpm_tp", basin, time, forecast_history, 0, -delay)
            station_tp = self.get_x("sta_tp", basin, time, forecast_history)
            expanded_gpm_tp = np.vstack([gpm_tp, station_tp[-delay:, :]])
            x = np.hstack([expanded_gpm_tp, station_tp])
            if "streamflow" in var_lst:
                yy = self.get_x("streamflow", basin, time, forecast_history)
                x = np.hstack([yy, x])
        else:
            x = self.get_x("gpm_tp", basin, time, forecast_history)
        if self.c is not None and self.c.shape[-1] > 0:
            c = self.c.sel(basin=basin).values
            # TODO: WARNING: length of c has been divided by 3
            c = np.tile(c, (int(forecast_history / 3), 1))
            x = np.concatenate((x, c), axis=1)

        mode = self.data_cfgs["model_mode"]
        if mode == "dual":
            s = self.get_x(
                ["sm_surface", "sm_rootzone"],
                basin,
                time,
                forecast_history,
                sm_length,
                0,
            )
        else:
            all_features = [
                self.get_x(
                    ["sm_surface", "sm_rootzone"],
                    basin,
                    time,
                    forecast_history,
                    sm_length - offset,
                    -offset,
                )
                for offset in range(forecast_history)
            ]
            s = np.array(
                [
                    feat.squeeze()
                    for feat in all_features
                    if feat.shape[0] == forecast_history - sm_length
                ]
            )
        y = self.get_y(basin, time, self.forecast_length, self.data_cfgs["prec_window"])
        return [
            torch.from_numpy(x).float(),
            torch.from_numpy(s).float(),
            torch.from_numpy(y).float(),
        ], torch.from_numpy(y).float()

    def _prepare_forcing(self):
        gage_id_lst = self.t_s_dict["sites_id"]
        t_range = self.t_s_dict["t_final_range"]
        var_lst = self.data_cfgs["relevant_cols"]
        path = self.data_cfgs["source_cfgs"]["source_path"]["forcing"]

        if var_lst is None:
            return None

        data = self.data_source.merge_nc_minio_datasets(path, gage_id_lst, var_lst)

        var_subset_list = []
        for start_date, end_date in t_range:
            adjusted_start_date = (
                datetime.strptime(start_date, "%Y-%m-%d-%H")
                - timedelta(hours=self.rho * self.data_cfgs["min_time_interval"])
            ).strftime("%Y-%m-%d-%H")
            adjusted_end_date = (
                datetime.strptime(end_date, "%Y-%m-%d-%H")
                + timedelta(hours=self.data_cfgs["min_time_interval"])
            ).strftime("%Y-%m-%d-%H")
            subset = data.sel(time=slice(adjusted_start_date, adjusted_end_date))
            var_subset_list.append(subset)

        return xr.concat(var_subset_list, dim="time")

    def _prepare_target(self):
        gage_id_lst = self.t_s_dict["sites_id"]
        t_range = self.t_s_dict["t_final_range"]
        var_lst = self.data_cfgs["target_cols"]
        path = self.data_cfgs["source_cfgs"]["source_path"]["target"]

        if var_lst is None or not var_lst:
            return None

        data = self.data_source.merge_nc_minio_datasets(path, gage_id_lst, var_lst)

        all_vars = data.data_vars
        if any(var not in data.variables for var in var_lst):
            raise ValueError(f"var_lst must all be in {all_vars}")
        subset_list = []

        for start_date, end_date in t_range:
            adjusted_start_date = (
                datetime.strptime(start_date, "%Y-%m-%d-%H")
                - timedelta(
                    hours=(
                        self.data_cfgs["prec_window"]
                        * self.data_cfgs["min_time_interval"]
                    )
                )
            ).strftime("%Y-%m-%d-%H")

            adjusted_end_date = (
                datetime.strptime(end_date, "%Y-%m-%d-%H")
                + timedelta(
                    hours=self.forecast_length * self.data_cfgs["min_time_interval"]
                )
            ).strftime("%Y-%m-%d-%H")
            subset = data.sel(time=slice(adjusted_start_date, adjusted_end_date))
            subset_list.append(subset)
        return xr.concat(subset_list, dim="time")


class Seq2SeqDataset(HydroMultiSourceDataset):
    def __init__(self, data_cfgs: dict, is_tra_val_te: str):
        super(Seq2SeqDataset, self).__init__(data_cfgs, is_tra_val_te)
        self.input_features = self.data_cfgs["relevant_cols"]

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
            data_attr_ds = self.data_source.read_BA_xrdataset(
                self.t_s_dict["sites_id"],
                self.data_cfgs["constant_cols"],
                self.data_cfgs["source_cfgs"]["source_path"]["attributes"],
            )
            c_orgin = self._trans2da_and_setunits(data_attr_ds)
        else:
            c_orgin = None
        self.x_origin, self.y_origin, self.c_origin = x_origin, y_origin, c_orgin

    def __getitem__(self, item: int):
        basin, time = self.lookup_table[item]
        forecast_history = self.rho
        x = self.get_x(self.input_features, basin, time, forecast_history)

        if self.c is not None and self.c.shape[-1] > 0:
            c = self.c.sel(basin=basin).values
            c = np.tile(c, (forecast_history // self.data_cfgs["min_time_interval"], 1))
            x = np.concatenate((x, c), axis=1)

        prec_window = self.data_cfgs["prec_window"]
        y = self.get_y(basin, time, self.forecast_length, prec_window)

        if self.is_tra_val_te == "train":
            return [
                torch.from_numpy(x).float(),
                torch.from_numpy(y).float(),
            ], torch.from_numpy(y).float()
        return [torch.from_numpy(x).float()], torch.from_numpy(y).float()
