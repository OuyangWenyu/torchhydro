"""
Author: Wenyu Ouyang
Date: 2024-04-08 18:16:53
LastEditTime: 2025-04-17 08:55:40
LastEditors: Wenyu Ouyang
Description: A pytorch dataset class; references to https://github.com/neuralhydrology/neuralhydrology
FilePath: /HydroForecastEval/mnt/disk1/owen/code/torchhydro/torchhydro/datasets/data_sets.py
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
from torchhydro.datasets.data_scalers import ScalerHub
from torchhydro.datasets.data_sources import data_sources_dict

from torchhydro.datasets.data_utils import (
    set_unit_to_var,
    warn_if_nan,
    wrap_t_s_dict,
)

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
        self.rho = self.data_cfgs["hindcast_length"]
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

    def denormalize(self, norm_data, rolling=0):
        """Denormalize the norm_data

        Parameters
        ----------
        norm_data : np.ndarray
            batch-first data
        rolling: int
            default 0, if rolling is used, perform forecasting using rolling window size

        Returns
        -------
        xr.Dataset
            denormlized data
        """
        target_scaler = self.target_scaler
        target_data = target_scaler.data_target
        # the units are dimensionless for pure DL models
        units = {k: "dimensionless" for k in target_data.attrs["units"].keys()}
        if target_scaler.pbm_norm:
            units = {**units, **target_data.attrs["units"]}
        if rolling > 0:
            hindcast_output_window = target_scaler.data_cfgs["hindcast_output_window"]
            rho = target_scaler.data_cfgs["hindcast_length"]
            # TODO: -1 because seq2seqdataset has one more time, hence we need to cut it, as rolling will be refactored, we will modify it later
            selected_time_points = target_data.coords["time"][
                rho - hindcast_output_window : -1
            ]
        else:
            warmup_length = self.warmup_length
            selected_time_points = target_data.coords["time"][warmup_length:]

        selected_data = target_data.sel(time=selected_time_points)
        denorm_xr_ds = target_scaler.inverse_transform(
            xr.DataArray(
                norm_data,
                dims=selected_data.dims,
                coords=selected_data.coords,
                attrs={"units": units},
            )
        )
        return set_unit_to_var(denorm_xr_ds)

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
        return data_forcing_ds, data_output_ds

    def _read_xyc(self):
        """Read x, y, c data from data source

        Returns
        -------
        tuple[xr.Dataset, xr.Dataset, xr.Dataset]
            x, y, c data
        """
        # x
        start_date = self.t_s_dict["t_final_range"][0]
        end_date = self.t_s_dict["t_final_range"][1]
        self._read_xyc_specified_time(start_date, end_date)

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
            x = self._kill_1type_nan(
                x,
                "interpolate",
                "original forcing data",
                "nan_filled forcing data",
            )
        if y_rm_nan:
            y = self._kill_1type_nan(
                y, "interpolate", "original output data", "nan_filled output data"
            )
        if c_rm_nan:
            c = self._kill_1type_nan(
                c, "mean", "original attribute data", "nan_filled attribute data"
            )
        warn_if_nan(x, nan_mode="any", data_name="nan_filled forcing data")
        warn_if_nan(y, nan_mode="all", data_name="output data")
        warn_if_nan(c, nan_mode="any", data_name="nan_filled attribute data")
        return x, y, c

    def _kill_1type_nan(self, the_data, fill_nan, data_name_before, data_name_after):
        is_any_nan = warn_if_nan(the_data, data_name=data_name_before)
        if not is_any_nan:
            return the_data
        # As input, we cannot have NaN values
        the_filled_data = _fill_gaps_da(the_data, fill_nan=fill_nan)
        warn_if_nan(the_filled_data, data_name=data_name_after)
        return the_filled_data

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
                nan_array = np.isnan(self.y[basin, :, :])
                lookup.extend(
                    (basin, f)
                    for f in range(warmup_length, max_time_length - rho - horizon + 1)
                    if not np.all(nan_array[f + rho : f + rho + horizon])
                )
        self.lookup_table = dict(enumerate(lookup))
        self.num_samples = len(self.lookup_table)


<<<<<<< HEAD
class HoDataset(BaseDataset):
    def __init__(self, data_cfgs: dict, is_tra_val_te: str):
        super(HoDataset, self).__init__(data_cfgs, is_tra_val_te)

    @property
    def streamflow_input_name(self):
        return self.data_cfgs["relevant_cols"][-1]

    def __getitem__(self, item: int):
        if not self.train_mode:
            xf = self.x[item, 1:, :-1]
            # xq = self.x[item, :-1, -1]
            # xq = xq.reshape(xq.size, 1)
            # x = np.concatenate((xf, xq), axis=1)
            x = xf
            y = self.y[item, :, :]
            if self.c is None or self.c.shape[-1] == 0:
                return torch.from_numpy(x).float(), torch.from_numpy(y).float()
            c = self.c[item, :]
            c = np.repeat(c, x.shape[0], axis=0).reshape(c.shape[0], -1).T
            xc = np.concatenate((x, c), axis=1)
            return torch.from_numpy(xc).float(), torch.from_numpy(y).float()
        basin, idx = self.lookup_table[item]
        warmup_length = self.warmup_length
        xf = self.x[
            basin,
            idx - warmup_length + 1 : idx + self.rho + self.horizon + 1,
            :-1,
        ]
        xq = self.x[basin, idx - warmup_length : idx + self.rho + self.horizon, -1]
        xq = xq.reshape(xq.size, 1)
        # x = np.concatenate((xf, xq), axis=1)
        x = xf
        y = self.y[basin, idx : idx + self.rho + self.horizon, :]
        if self.c is None or self.c.shape[-1] == 0:
            return torch.from_numpy(x).float(), torch.from_numpy(y).float()
        c = self.c[basin, :]
        c = np.repeat(c, x.shape[0], axis=0).reshape(c.shape[0], -1).T
        xc = np.concatenate((x, c), axis=1)
        return [
            torch.from_numpy(xc).float(),
            torch.from_numpy(xq).float(),
        ], torch.from_numpy(y).float()

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
        date_format = detect_date_format(start_date)
        time_unit = self.data_cfgs["min_time_unit"]
        horizon = self.horizon
        start_date_dt = datetime.strptime(start_date, date_format)
        if time_unit == "h":
            adjusted_start_date = (start_date_dt - timedelta(hours=1)).strftime(
                date_format
            )
        elif time_unit == "D":
            adjusted_start_date = (start_date_dt - timedelta(days=1)).strftime(
                date_format
            )
        else:
            raise ValueError(f"Unsupported time unit: {time_unit}")
        data_forcing_ds_ = self.data_source.read_ts_xrdataset(
            self.t_s_dict["sites_id"],
            [adjusted_start_date, end_date],
            self.data_cfgs["relevant_cols"],
        )
        # y
        data_output_ds_ = self.data_source.read_ts_xrdataset(
            self.t_s_dict["sites_id"],
            [start_date, end_date],
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


class HoSameDataset(HoDataset):
    def __init__(self, data_cfgs: dict, is_tra_val_te: str):
        super(HoSameDataset, self).__init__(data_cfgs, is_tra_val_te)

    def __getitem__(self, item: int):
        if not self.train_mode:
            xf = self.x[item, 1:, :-1]
            xq = self.x[item, :-1, -1]
            xq = xq.reshape(xq.size, 1)
            x = np.concatenate((xf, xq), axis=1)
            # x = xf
            y = self.y[item, :, :]
            if self.c is None or self.c.shape[-1] == 0:
                return torch.from_numpy(x).float(), torch.from_numpy(y).float()
            c = self.c[item, :]
            c = np.repeat(c, x.shape[0], axis=0).reshape(c.shape[0], -1).T
            xc = np.concatenate((x, c), axis=1)
            return torch.from_numpy(xc).float(), torch.from_numpy(y).float()
        basin, idx = self.lookup_table[item]
        warmup_length = self.warmup_length
        xf = self.x[
            basin,
            idx - warmup_length + 1 : idx + self.rho + self.horizon + 1,
            :-1,
        ]
        xq = self.x[basin, idx - warmup_length : idx + self.rho + self.horizon, -1]
        xq = xq.reshape(xq.size, 1)
        x = np.concatenate((xf, xq), axis=1)
        # x = xf
        y = self.y[basin, idx : idx + self.rho + self.horizon, :]
        if self.c is None or self.c.shape[-1] == 0:
            return torch.from_numpy(x).float(), torch.from_numpy(y).float()
        c = self.c[basin, :]
        c = np.repeat(c, x.shape[0], axis=0).reshape(c.shape[0], -1).T
        xc = np.concatenate((x, c), axis=1)
        return torch.from_numpy(xc).float(), torch.from_numpy(y).float()


class FoDataset(HoDataset):
    def __init__(self, data_cfgs: dict, is_tra_val_te: str):
        super(FoDataset, self).__init__(data_cfgs, is_tra_val_te)

    def __getitem__(self, item: int):
        if not self.train_mode:
            x = self.x[item, 1:, :-1]
            y = self.y[item, :, :]
            if self.c is None or self.c.shape[-1] == 0:
                return torch.from_numpy(x).float(), torch.from_numpy(y).float()
            c = self.c[item, :]
            c = np.repeat(c, x.shape[0], axis=0).reshape(c.shape[0], -1).T
            xc = np.concatenate((x, c), axis=1)
            return torch.from_numpy(xc).float(), torch.from_numpy(y).float()
        basin, idx = self.lookup_table[item]
        warmup_length = self.warmup_length
        x = self.x[
            basin,
            idx - warmup_length + 1 : idx + self.rho + self.horizon + 1,
            :-1,
        ]
        xy = self.x[basin, idx - warmup_length : idx + self.rho + self.horizon, -1]
        xy = xy.reshape(xy.size, 1)
        y = self.y[basin, idx : idx + self.rho + self.horizon, :]
        if self.c is None or self.c.shape[-1] == 0:
            return torch.from_numpy(x).float(), torch.from_numpy(y).float()
        c = self.c[basin, :]
        c = np.repeat(c, x.shape[0], axis=0).reshape(c.shape[0], -1).T
        xc = np.concatenate((x, c), axis=1)
        return [
            torch.from_numpy(xc).float(),
            torch.from_numpy(xy).float(),
        ], torch.from_numpy(y).float()


class HFDataset(HoDataset):
    def __init__(self, data_cfgs: dict, is_tra_val_te: str):
        super(HFDataset, self).__init__(data_cfgs, is_tra_val_te)

    def __getitem__(self, item: int):
        if not self.train_mode:
            # 先把forcing都取出来
            xf = self.x[item, 1:, :-1]
            xf = xf.reshape(1, xf.shape[0], xf.shape[1])
            # 再把hindcast输入的streamflow取出来
            xq = self.x[item, : self.rho, -1]
            xq = xq.reshape(xq.size, 1)
            # 取hindcast部分的forcing
            xf_hind = xf[:, : self.rho, :]
            # 取forecast部分的forcing
            xf_fore = xf[:, self.rho :, :]
            # 取y
            y = self.y[item, :, :]
            # 取c
            c = self.c[item, :]
            # 转到二维
            xf_hind = xf_hind.squeeze(0)
            xf_fore = xf_fore.squeeze(0)
            # hindcast部分和c拼接
            hind_c = np.repeat(c, xf_hind.shape[0], axis=0).reshape(c.shape[0], -1).T
            xf_hind_c = np.concatenate((xf_hind, hind_c), axis=1)
            x_hind_c = np.concatenate((xf_hind_c, xq), axis=-1)
            # forecast部分和c拼接
            fore_c = np.repeat(c, xf_fore.shape[0], axis=0).reshape(c.shape[0], -1).T
            xf_fore_c = np.concatenate((xf_fore, fore_c), axis=1)
            return [
                torch.from_numpy(x_hind_c).float(),
                torch.from_numpy(xf_fore_c).float(),
            ], torch.from_numpy(y).float()
        basin, idx = self.lookup_table[item]
        warmup_length = self.warmup_length
        # 先把hindcast和forecast的forcing取出来
        xf = self.x[
            basin,
            idx - warmup_length + 1 : idx + self.rho + self.horizon + 1,
            :-1,
        ]
        xf = xf.reshape(1, xf.shape[0], xf.shape[1])
        xf_hind = xf[:, : self.rho, :]
        xf_fore = xf[:, self.rho :, :]
        # 再把hindcast和forecast输入的streamflow取出来
        xq = self.x[basin, idx - warmup_length : idx + self.rho + self.horizon, -1]
        xq = xq.reshape(1, xq.size, 1)
        # 取hindcast部分的流量
        xq_hind = xq[:, : self.rho, :]
        # 取forecast部分的流量
        xq_fore = xq[:, : self.rho, :]
        # 取c
        c = self.c[basin, :]
        # 取y
        y = self.y[basin, idx : idx + self.rho + self.horizon, :]
        # 转到二维
        xf_hind = xf_hind.squeeze(0)
        xf_fore = xf_fore.squeeze(0)
        xq_hind = xq_hind.squeeze(0)
        xq_fore = xq_fore.squeeze(0)
        # hindcast部分和c拼接
        hind_c = np.repeat(c, xf_hind.shape[0], axis=0).reshape(c.shape[0], -1).T
        xf_hind_c = np.concatenate((xf_hind, hind_c), axis=1)
        x_hind_c = np.concatenate((xf_hind_c, xq_hind), axis=-1)
        # forecast部分和c拼接
        fore_c = np.repeat(c, xf_fore.shape[0], axis=0).reshape(c.shape[0], -1).T
        xf_fore_c = np.concatenate((xf_fore, fore_c), axis=1)

        return [
            torch.from_numpy(x_hind_c).float(),
            torch.from_numpy(xf_fore_c).float(),
            torch.from_numpy(xq_fore).float(),
        ], torch.from_numpy(y).float()


class ForecasetDataset(BaseDataset):
    """Dataset for eval lstm specific horizon forecast"""

    def __init__(self, data_cfgs: dict, is_tra_val_te: str):
        super(ForecasetDataset, self).__init__(data_cfgs, is_tra_val_te)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item: int):
        basin, idx = self.lookup_table[item]
        warmup_length = self.warmup_length
        x = self.x[basin, idx - warmup_length : idx + self.rho + self.horizon, :]
        y = self.y[basin, idx + self.rho : idx + self.rho + self.horizon, :]
        if self.c is None or self.c.shape[-1] == 0:
            return torch.from_numpy(x).float(), torch.from_numpy(y).float()
        c = self.c[basin, :]
        c = np.repeat(c, x.shape[0], axis=0).reshape(c.shape[0], -1).T
        xc = np.concatenate((x, c), axis=1)
        return torch.from_numpy(xc).float(), torch.from_numpy(y).float()

    def denormalize(self, norm_data, rolling=0):
        """Denormalize the norm_data

        Parameters
        ----------
        norm_data : np.ndarray
            batch-first data
        rolling: int
            default 0, if rolling is used, perform forecasting using rolling window size

        Returns
        -------
        xr.Dataset
            denormlized data
        """
        target_scaler = self.target_scaler
        target_data = target_scaler.data_target
        # the units are dimensionless for pure DL models
        units = {k: "dimensionless" for k in target_data.attrs["units"].keys()}
        if target_scaler.pbm_norm:
            units = {**units, **target_data.attrs["units"]}
        if rolling > 0:
            hindcast_output_window = target_scaler.data_cfgs["hindcast_output_window"]
            rho = target_scaler.data_cfgs["hindcast_length"]
            # TODO: -1 because seq2seqdataset has one more time, hence we need to cut it, as rolling will be refactored, we will modify it later
            selected_time_points = target_data.coords["time"][
                rho - hindcast_output_window :
            ]
        else:
            warmup_length = self.warmup_length
            selected_time_points = target_data.coords["time"][warmup_length:]

        selected_data = target_data.sel(time=selected_time_points)
        denorm_xr_ds = target_scaler.inverse_transform(
            xr.DataArray(
                norm_data.transpose(2, 0, 1),
                dims=selected_data.dims,
                coords=selected_data.coords,
                attrs={"units": units},
            )
        )
        return set_unit_to_var(denorm_xr_ds)


class OffsetForecasetDataset(ForecasetDataset):
    """
    Dataset for eval lstm specific horizon forecast with offset
    """

    def __init__(self, data_cfgs: dict, is_tra_val_te: str):
        super(OffsetForecasetDataset, self).__init__(data_cfgs, is_tra_val_te)

    def __getitem__(self, item: int):
        basin, idx = self.lookup_table[item]
        warmup_length = self.warmup_length
        x = self.x[
            basin,
            idx - warmup_length + self.horizon : idx + self.rho + self.horizon * 2,
            :-1,
        ]  # forcing without streamflow
        xy = self.x[
            basin,
            idx - warmup_length : idx + self.rho + self.horizon,
            -1,
        ].reshape(
            -1, 1
        )  # streamflow
        x = np.concatenate((x, xy), axis=1)
        y = self.y[basin, idx + self.rho : idx + self.rho + self.horizon, :]
        if self.c is None or self.c.shape[-1] == 0:
            return torch.from_numpy(x).float(), torch.from_numpy(y).float()
        c = self.c[basin, :]
        c = np.repeat(c, x.shape[0], axis=0).reshape(c.shape[0], -1).T
        xc = np.concatenate((x, c), axis=1)
        return torch.from_numpy(xc).float(), torch.from_numpy(y).float()

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
        time_unit = self.data_cfgs["min_time_unit"]
        horizon = self.horizon
        # Determine the date format
        date_format = detect_date_format(start_date)

        start_date_dt = datetime.strptime(start_date, date_format)
        if time_unit == "h":
            adjusted_start_date = (start_date_dt - timedelta(hours=horizon)).strftime(
                date_format
            )
        elif time_unit == "D":
            adjusted_start_date = (start_date_dt - timedelta(days=horizon)).strftime(
                date_format
            )
        else:
            raise ValueError(f"Unsupported time unit: {time_unit}")
        data_forcing_ds_ = self.data_source.read_ts_xrdataset(
            self.t_s_dict["sites_id"],
            [adjusted_start_date, end_date],
            self.data_cfgs["relevant_cols"],
        )
        # y
        data_output_ds_ = self.data_source.read_ts_xrdataset(
            self.t_s_dict["sites_id"],
            [start_date, end_date],
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

    def denormalize(self, norm_data, rolling=0):
        """Denormalize the norm_data

        Parameters
        ----------
        norm_data : np.ndarray
            batch-first data
        rolling: int
            default 0, if rolling is used, perform forecasting using rolling window size

        Returns
        -------
        xr.Dataset
            denormlized data
        """
        target_scaler = self.target_scaler
        target_data = target_scaler.data_target
        # the units are dimensionless for pure DL models
        units = {k: "dimensionless" for k in target_data.attrs["units"].keys()}
        if target_scaler.pbm_norm:
            units = {**units, **target_data.attrs["units"]}
        if rolling > 0:
            hindcast_output_window = target_scaler.data_cfgs["hindcast_output_window"]
            rho = target_scaler.data_cfgs["hindcast_length"]
            # TODO: -1 because seq2seqdataset has one more time, hence we need to cut it, as rolling will be refactored, we will modify it later
            selected_time_points = target_data.coords["time"][
                rho - hindcast_output_window :
            ]  # add -1 when seq2seq dataset, need refactor
        else:
            warmup_length = self.warmup_length
            selected_time_points = target_data.coords["time"][warmup_length:]

        selected_data = target_data.sel(time=selected_time_points)
        denorm_xr_ds = target_scaler.inverse_transform(
            xr.DataArray(
                norm_data.transpose(2, 0, 1),
                dims=selected_data.dims,
                coords=selected_data.coords,
                attrs={"units": units},
            )
        )
        return set_unit_to_var(denorm_xr_ds)


class MultiInputOffsetForecasetDataset(OffsetForecasetDataset):
    def __init__(self, data_cfgs: dict, is_tra_val_te: str):
        super(MultiInputOffsetForecasetDataset, self).__init__(data_cfgs, is_tra_val_te)
        self.offset_length = self.data_cfgs["offset_length"]

    def __getitem__(self, item: int):
        basin, idx = self.lookup_table[item]
        warmup_length = self.warmup_length
        x = self.x[basin, idx - warmup_length : idx + self.rho + self.horizon, :-1]
        xy = self.x[
            basin,
            idx - warmup_length + self.rho - self.offset_length : idx + self.rho,
            -1,
        ].reshape(
            -1, 1
        )  # streamflow
        y = self.y[basin, idx + self.rho : idx + self.rho + self.horizon, :]
        if self.c is None or self.c.shape[-1] == 0:
            return torch.from_numpy(x).float(), torch.from_numpy(y).float()
        c = self.c[basin, :]
        c = np.repeat(c, x.shape[0], axis=0).reshape(c.shape[0], -1).T
        xc = np.concatenate((x, c), axis=1)
        return [
            torch.from_numpy(xc).float(),
            torch.from_numpy(xy).float(),
        ], torch.from_numpy(y).float()

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


class OffsetForecasetDataset2(ForecasetDataset):
    """
    Dataset for eval lstm specific horizon forecast with offset
    """

    def __init__(self, data_cfgs: dict, is_tra_val_te: str):
        super(OffsetForecasetDataset2, self).__init__(data_cfgs, is_tra_val_te)

    def __getitem__(self, item: int):
        basin, idx = self.lookup_table[item]
        warmup_length = self.warmup_length
        x = self.x[
            basin,
            idx - warmup_length + self.horizon : idx + self.rho + self.horizon * 2,
            :,
        ]  # forcing without streamflow
        xy = self.y[
            basin,
            idx - warmup_length : idx + self.rho + self.horizon,
            :,
        ]
        x = np.concatenate((x, xy), axis=1)
        y = self.y[
            basin, idx + self.rho + self.horizon : idx + self.rho + self.horizon * 2, :
        ]
        if self.c is None or self.c.shape[-1] == 0:
            return torch.from_numpy(x).float(), torch.from_numpy(y).float()
        c = self.c[basin, :]
        c = np.repeat(c, x.shape[0], axis=0).reshape(c.shape[0], -1).T
        xc = np.concatenate((x, c), axis=1)
        return torch.from_numpy(xc).float(), torch.from_numpy(y).float()

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
        time_unit = self.data_cfgs["min_time_unit"]
        horizon = self.horizon
        # Determine the date format
        date_format = detect_date_format(start_date)

        start_date_dt = datetime.strptime(start_date, date_format)
        if time_unit == "h":
            adjusted_start_date = (start_date_dt - timedelta(hours=horizon)).strftime(
                date_format
            )
        elif time_unit == "D":
            adjusted_start_date = (start_date_dt - timedelta(days=horizon)).strftime(
                date_format
            )
        else:
            raise ValueError(f"Unsupported time unit: {time_unit}")
        data_forcing_ds_ = self.data_source.read_ts_xrdataset(
            self.t_s_dict["sites_id"],
            [adjusted_start_date, end_date],
            self.data_cfgs["relevant_cols"],
        )
        # y
        data_output_ds_ = self.data_source.read_ts_xrdataset(
            self.t_s_dict["sites_id"],
            [adjusted_start_date, end_date],
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

    def denormalize(self, norm_data, rolling=0):
        """Denormalize the norm_data

        Parameters
        ----------
        norm_data : np.ndarray
            batch-first data
        rolling: int
            default 0, if rolling is used, perform forecasting using rolling window size

        Returns
        -------
        xr.Dataset
            denormlized data
        """
        target_scaler = self.target_scaler
        target_data = target_scaler.data_target
        # the units are dimensionless for pure DL models
        units = {k: "dimensionless" for k in target_data.attrs["units"].keys()}
        if target_scaler.pbm_norm:
            units = {**units, **target_data.attrs["units"]}
        if rolling > 0:
            hindcast_output_window = target_scaler.data_cfgs["hindcast_output_window"]
            rho = target_scaler.data_cfgs["hindcast_length"]
            horizon = self.horizon
            # TODO: -1 because seq2seqdataset has one more time, hence we need to cut it, as rolling will be refactored, we will modify it later
            selected_time_points = target_data.coords["time"][
                rho - hindcast_output_window + horizon :
            ]  # add -1 when seq2seq dataset, need refactor
        else:
            warmup_length = self.warmup_length
            selected_time_points = target_data.coords["time"][warmup_length:]

        selected_data = target_data.sel(time=selected_time_points)
        denorm_xr_ds = target_scaler.inverse_transform(
            xr.DataArray(
                norm_data.transpose(2, 0, 1),
                dims=selected_data.dims,
                coords=selected_data.coords,
                attrs={"units": units},
            )
        )
        return set_unit_to_var(denorm_xr_ds)


class MultiInputOffsetForecasetDataset2(OffsetForecasetDataset2):
    def __init__(self, data_cfgs: dict, is_tra_val_te: str):
        super(MultiInputOffsetForecasetDataset2, self).__init__(
            data_cfgs, is_tra_val_te
        )
        self.offset_length = self.data_cfgs["offset_length"]

    def __getitem__(self, item: int):
        basin, idx = self.lookup_table[item]
        warmup_length = self.warmup_length
        x = self.x[basin, idx - warmup_length : idx + self.rho + self.horizon, :]
        xy = self.y[
            basin,
            idx - warmup_length + self.rho - self.offset_length : idx + self.rho,
            :,
        ]
        y = self.y[basin, idx + self.rho : idx + self.rho + self.horizon, :]
        if self.c is None or self.c.shape[-1] == 0:
            return torch.from_numpy(x).float(), torch.from_numpy(y).float()
        c = self.c[basin, :]
        c = np.repeat(c, x.shape[0], axis=0).reshape(c.shape[0], -1).T
        xc = np.concatenate((x, c), axis=1)
        return [
            torch.from_numpy(xc).float(),
            torch.from_numpy(xy).float(),
        ], torch.from_numpy(y).float()
=======
class ObsForeDataset(BaseDataset):
    """处理观测和预见期数据的混合数据集

    这个类专门用于处理具有双维度预见期数据格式的数据集，其中 lead_time 和 time 都是独立维度。
    适合表示不同发布时间对不同目标日期的预报。
    """

    def __init__(self, data_cfgs: dict, is_tra_val_te: str):
        """初始化观测和预见期混合数据集

        Parameters
        ----------
        data_cfgs : dict
            数据配置字典
        is_tra_val_te : str
            指定是训练集、验证集还是测试集
        """
        # 调用父类初始化方法
        super(ObsForeDataset, self).__init__(data_cfgs, is_tra_val_te)

        # 记录预见期数据的起始索引
        self.forecast_start_idx = None

    def _load_data(self):
        """重写加载数据方法，添加预见期数据处理"""
        self._pre_load_data()
        self._read_xyc()
        # 检查是否需要添加预见期数据
        if self.data_cfgs.get("use_forecast_data", False):
            self._append_forecast_data()
        # normalization
        norm_x, norm_y, norm_c = self._normalize()
        self.x, self.y, self.c = self._kill_nan(norm_x, norm_y, norm_c)
        self._trans2nparr()
        self._create_lookup_table()

    def _append_forecast_data(self):
        """添加预见期数据到现有数据集

        从预见期数据源中读取数据，并添加到时间序列末端
        """
        forecast_cfg = self.data_cfgs.get("forecast_cfg", {})
        if not forecast_cfg:
            LOGGER.warning("forecast_cfg 未找到，跳过预见期数据添加")
            return

        # 获取预见期数据配置
        forecast_source_path = forecast_cfg.get("source_path")
        forecast_source_name = forecast_cfg.get(
            "source_name", self.data_cfgs["source_cfgs"]["source_name"]
        )
        num_samples = forecast_cfg.get("num_samples", 5)  # 默认添加5个样本
        lead_time_selector = forecast_cfg.get(
            "lead_time_selector"
        )  # 可以是函数或固定值列表

        if not forecast_source_path:
            LOGGER.warning("forecast_source_path 未指定，跳过预见期数据添加")
            return

        # 读取预见期数据
        self._read_forecast_data(
            forecast_source_path, forecast_source_name, num_samples, lead_time_selector
        )

    def _read_forecast_data(
        self,
        forecast_source_path,
        forecast_source_name,
        num_samples,
        lead_time_selector,
    ):
        """读取预见期数据并合并到现有数据中

        Parameters
        ----------
        forecast_source_path : str
            预见期数据源路径
        forecast_source_name : str
            预见期数据源名称
        num_samples : int
            需要添加的样本数量
        lead_time_selector : callable or list
            选择lead_time的函数或固定值列表
        """
        # 创建预见期数据源
        other_settings = self.data_cfgs["source_cfgs"].get("other_settings", {})
        forecast_data_source = data_sources_dict[forecast_source_name](
            forecast_source_path, **other_settings
        )

        # 获取最后一个时间点作为预见期数据的起始点
        end_date = self.t_s_dict["t_final_range"][1]
        date_format = detect_date_format(end_date)
        end_date_dt = datetime.strptime(end_date, date_format)

        # 获取预见期数据加载模式
        forecast_cfg = self.data_cfgs.get("forecast_cfg", {})
        forecast_mode = forecast_cfg.get("forecast_mode", "forecast_matrix")

        # 读取预见期数据
        forecast_forcing_ds = forecast_data_source.read_forecast_xrdataset(
            self.t_s_dict["sites_id"],
            end_date_dt,
            self.data_cfgs["relevant_cols"],
            lead_time_selector=lead_time_selector,
            num_samples=num_samples,
            forecast_mode=forecast_mode,
        )

        # 读取预见期目标数据（如果有）
        forecast_output_ds = None
        if self.data_cfgs.get("forecast_target_available", False):
            forecast_output_ds = forecast_data_source.read_forecast_xrdataset(
                self.t_s_dict["sites_id"],
                end_date_dt,
                self.data_cfgs["target_cols"],
                lead_time_selector=lead_time_selector,
                num_samples=num_samples,
                forecast_mode=forecast_mode,
            )

        # 处理预报矩阵模式
        if forecast_mode == "forecast_matrix":
            # 选择特定的预报路径
            lead_time_idx = forecast_cfg.get("lead_time_idx", -1)  # 默认选择最新的预报

            # 从预报矩阵中选择特定的预报路径
            selected_forecast = forecast_forcing_ds.isel(lead_time=lead_time_idx)

            # 将选定的预报数据转换为时间序列格式
            forecast_x = selected_forecast.to_array(dim="variable")

            # 如果有预见期目标数据，也进行相同的处理
            forecast_y = None
            if forecast_output_ds is not None:
                selected_output = forecast_output_ds.isel(lead_time=lead_time_idx)
                forecast_y = selected_output.to_array(dim="variable")
        else:
            # 将预见期数据转换为与现有数据相同的格式
            forecast_x, forecast_y, _ = self._to_dataarray_with_unit(
                forecast_forcing_ds, forecast_output_ds, None
            )

        # 记录预见期数据的起始索引
        self.forecast_start_idx = self.x_origin.sizes["time"]

        # 将预见期数据添加到现有数据中
        self.x_origin = xr.concat([self.x_origin, forecast_x], dim="time")

        # 如果有预见期目标数据，也添加到现有数据中
        if forecast_y is not None:
            self.y_origin = xr.concat([self.y_origin, forecast_y], dim="time")
        else:
            # 如果没有预见期目标数据，用NaN填充
            dummy_y = xr.full_like(
                self.y_origin.isel(time=slice(-1, None))
                .expand_dims("time", axis=1)
                .repeat(forecast_x.sizes["time"], dim="time"),
                np.nan,
            )
            self.y_origin = xr.concat([self.y_origin, dummy_y], dim="time")

        # 标记哪些数据是预见期数据
        self.is_forecast = np.zeros(self.x_origin.sizes["time"], dtype=bool)
        self.is_forecast[self.forecast_start_idx :] = True

    def __getitem__(self, item: int):
        """获取数据集中的一个样本

        Parameters
        ----------
        item : int
            样本索引

        Returns
        -------
        tuple
            (x, y) 数据对，其中 x 包含输入特征和预见期标志，y 包含目标值
        """
        # 获取基础数据
        if not self.train_mode:
            x = self.x[item, :, :]
            y = self.y[item, :, :]

            # 创建预见期标志
            forecast_flag = np.zeros((x.shape[0], 1))

            # 如果存在forecast_start_idx属性，使用它标记预见期数据
            if (
                hasattr(self, "forecast_start_idx")
                and self.forecast_start_idx is not None
            ):
                forecast_flag[self.forecast_start_idx :] = 1.0

            # 添加预见期标志到输入特征
            if self.c is None or self.c.shape[-1] == 0:
                xc = np.concatenate((x, forecast_flag), axis=1)
            else:
                c = self.c[item, :]
                c = np.repeat(c, x.shape[0], axis=0).reshape(x.shape[0], -1).T
                xc = np.concatenate((x, c, forecast_flag), axis=1)

            return torch.from_numpy(xc).float(), torch.from_numpy(y).float()

        # 训练模式
        basin, idx = self.lookup_table[item]
        warmup_length = self.warmup_length
        x = self.x[basin, idx - warmup_length : idx + self.rho + self.horizon, :]
        y = self.y[basin, idx : idx + self.rho + self.horizon, :]

        # 创建预见期标志
        forecast_flag = np.zeros((x.shape[0], 1))

        # 如果有预见期数据，标记它们
        if hasattr(self, "forecast_start_idx") and self.forecast_start_idx is not None:
            forecast_start_pos = max(0, self.forecast_start_idx - (idx - warmup_length))
            if forecast_start_pos < x.shape[0]:
                forecast_flag[forecast_start_pos:] = 1.0

        # 添加预见期标志到输入特征
        if self.c is None or self.c.shape[-1] == 0:
            xc = np.concatenate((x, forecast_flag), axis=1)
        else:
            c = self.c[basin, :]
            c = np.repeat(c, x.shape[0], axis=0).reshape(c.shape[0], -1).T
            xc = np.concatenate((x, c, forecast_flag), axis=1)

        return torch.from_numpy(xc).float(), torch.from_numpy(y).float()
>>>>>>> 59b9bf759fd89ee57423e10bd9541605034b6274


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
    def __init__(self, data_cfgs: dict, is_tra_val_te: str):
        super(Seq2SeqDataset, self).__init__(data_cfgs, is_tra_val_te)

    def _read_xyc(self):
        """
        NOTE: the lookup table is same as BaseDataset,
        but the data retrieved from datasource should has one more period,
        because we include the concepts of start and end moment of the period

        Returns
        -------
        tuple[xr.Dataset, xr.Dataset, xr.Dataset]
            x, y, c data
        """
        start_date = self.t_s_dict["t_final_range"][0]
        end_date = self.t_s_dict["t_final_range"][1]
        interval = self.data_cfgs["min_time_interval"]
        time_unit = self.data_cfgs["min_time_unit"]

        # Determine the date format
        date_format = detect_date_format(end_date)

        # Adjust the end date based on the time unit
        end_date_dt = datetime.strptime(end_date, date_format)
        if time_unit == "h":
            adjusted_end_date = (end_date_dt + timedelta(hours=interval)).strftime(
                date_format
            )
        elif time_unit == "D":
            adjusted_end_date = (end_date_dt + timedelta(days=interval)).strftime(
                date_format
            )
        else:
            raise ValueError(f"Unsupported time unit: {time_unit}")
        self._read_xyc_specified_time(start_date, adjusted_end_date)

    def _normalize(self):
        x, y, c = super()._normalize()
        # TODO: this work for minio? maybe better to move to basedataset
        return x.compute(), y.compute(), c.compute()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item: int):
        basin, time = self.lookup_table[item]
        rho = self.rho
        horizon = self.horizon
        hindcast_output_window = self.data_cfgs.get("hindcast_output_window", 0)
        # p cover all encoder-decoder periods; +1 means the period while +0 means start of the current period
        p = self.x[basin, time + 1 : time + rho + horizon + 1, 0].reshape(-1, 1)
        # s only cover encoder periods
        s = self.x[basin, time : time + rho, 1:]
        x = np.concatenate((p[:rho], s), axis=1)

        if self.c is None or self.c.shape[-1] == 0:
            xc = x
        else:
            c = self.c[basin, :]
            c = np.tile(c, (rho + horizon, 1))
            xc = np.concatenate((x, c[:rho]), axis=1)
        # xh cover decoder periods
        try:
            xh = np.concatenate((p[rho:], c[rho:]), axis=1)
        except ValueError as e:
            print(f"Error in np.concatenate: {e}")
            print(f"p[rho:].shape: {p[rho:].shape}, c[rho:].shape: {c[rho:].shape}")
            raise
        # y cover specified encoder size (hindcast_output_window) and all decoder periods
        y = self.y[
            basin, time + rho - hindcast_output_window + 1 : time + rho + horizon + 1, :
        ]

        if self.is_tra_val_te == "train":
            return [
                torch.from_numpy(xc).float(),
                torch.from_numpy(xh).float(),
                torch.from_numpy(y).float(),
            ], torch.from_numpy(y).float()
        return [
            torch.from_numpy(xc).float(),
            torch.from_numpy(xh).float(),
        ], torch.from_numpy(y).float()


class Seq2SeqDataset2(Seq2SeqDataset):
    def __init__(self, data_cfgs, is_tra_val_te):
        super().__init__(data_cfgs, is_tra_val_te)

    def __getitem__(self, item: int):
        basin, time = self.lookup_table[item]
        rho = self.rho
        horizon = self.horizon
        hindcast_output_window = self.data_cfgs.get("hindcast_output_window", 0)

        q = self.x[basin, time : time + rho, -1].reshape(-1, 1)
        others = self.x[basin, time + 1 : time + rho + horizon + 1, :-1]
        xe_ = np.concatenate((q, others[:rho]), axis=1)

        if self.c is None or self.c.shape[-1] == 0:
            xe = xe_
        else:
            c = self.c[basin, :]
            c = np.tile(c, (rho + horizon, 1))
            xe = np.concatenate((xe_, c[:rho]), axis=1)
        xd = np.concatenate((others[rho:], c[rho:]), axis=1)
        # y cover specified encoder size (hindcast_output_window) and all decoder periods
        y = self.y[
            basin, time + rho - hindcast_output_window + 1 : time + rho + horizon + 1, :
        ]

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
        basin, time = self.lookup_table[item]
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
        basin, idx = self.lookup_table[item]
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
