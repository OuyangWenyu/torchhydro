"""
Author: Wenyu Ouyang
Date: 2024-04-08 18:16:53
LastEditTime: 2025-07-13 17:59:58
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
from torchhydro.datasets.data_scalers import ScalerHub
from torchhydro.datasets.data_sources import data_sources_dict
from tqdm import tqdm

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
        return self.num_samples

    def __getitem__(self, item: int):
        """Get one sample from the dataset with unified return format

        Returns:
        --------
        tuple[torch.Tensor, torch.Tensor]
            (input_data, output_data)
        """
        basin, idx, actual_length = self.lookup_table[item]
        warmup_length = self.warmup_length
        x = self.x[basin, idx - warmup_length : idx + actual_length, :]
        y = self.y[basin, idx : idx + actual_length, :]
        if self.c is None or self.c.shape[-1] == 0:
            return torch.from_numpy(x).float(), torch.from_numpy(y).float()
        c = self.c[basin, :]
        c = np.repeat(c, x.shape[0], axis=0).reshape(c.shape[0], -1).T
        xc = np.concatenate((x, c), axis=1)
        return torch.from_numpy(xc).float(), torch.from_numpy(y).float()

    def _load_data(self):
        origin_data = self._read_xyc()
        # normalization
        norm_data = self._normalize(origin_data)
        origin_data_wonan, norm_data_wonan = self._kill_nan(origin_data, norm_data)
        self._trans2nparr(origin_data_wonan, norm_data_wonan)
        self._create_lookup_table()

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
                norm_arr = _norm.to_numpy()
                origin_arr = _origin.to_numpy()
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
            else:
                raise ValueError(
                    f"Unknown data type {key} in origin_data, "
                    "it should be one of relevant_cols, target_cols, constant_cols"
                )

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
        if target_scaler.pbm_norm:
            units = {**units, **target_data.attrs["units"]}
        # mainly to get information about the time points of norm_data
        selected_time_points = self._selected_time_points_for_denorm()
        selected_data = target_data.sel(time=selected_time_points)

        # 处理四维数据
        if norm_data.ndim == 4:
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
                    # 创建新的时间坐标，重复i_e_time_length次
                    new_times = []
                    for i in range(forecast_length):
                        if i < len(selected_data.coords["time"]):
                            new_times.extend(
                                selected_data.coords["time"][:i_e_time_length]
                            )

                    # 确保时间坐标长度与数据匹配
                    if len(new_times) > i_e_time_length * forecast_length:
                        new_times = new_times[: i_e_time_length * forecast_length]
                    elif len(new_times) < i_e_time_length * forecast_length:
                        # 如果时间坐标不足，使用最后一个时间点填充
                        last_time = (
                            new_times[-1]
                            if new_times
                            else selected_data.coords["time"][0]
                        )
                        while len(new_times) < i_e_time_length * forecast_length:
                            new_times.append(last_time)

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

                    # 创建新的时间坐标
                    new_times = []
                    for i in range(forecast_length):
                        if i < len(selected_data.coords["time"]):
                            new_times.extend(
                                selected_data.coords["time"][:i_e_time_length]
                            )

                    # 确保时间坐标长度与数据匹配
                    if len(new_times) > forecast_length * i_e_time_length:
                        new_times = new_times[: forecast_length * i_e_time_length]
                    elif len(new_times) < forecast_length * i_e_time_length:
                        # 如果时间坐标不足，使用最后一个时间点填充
                        last_time = (
                            new_times[-1]
                            if new_times
                            else selected_data.coords["time"][0]
                        )
                        while len(new_times) < forecast_length * i_e_time_length:
                            new_times.append(last_time)

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
            )
            data_output_ds[self.streamflow_name] = converted_streamflow_dataset[
                self.streamflow_name
            ]
        return data_forcing_ds, data_output_ds

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
        # x
        start_date = self.t_s_dict["t_final_range"][0]
        end_date = self.t_s_dict["t_final_range"][1]
        return self._read_xyc_specified_time(start_date, end_date)

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
            else:
                raise ValueError(
                    f"Unknown data type {key} in origin_data, "
                    "it should be one of relevant_cols, target_cols, constant_cols, forecast_cols and global_cols"
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
            if key == "target_cols" or not rm_nan:
                warn_if_nan(origin, nan_mode="all", data_name="nan_filled target data")
                warn_if_nan(norm, nan_mode="all", data_name="nan_filled target data")
            else:
                warn_if_nan(origin, nan_mode="any", data_name="nan_filled input data")
                warn_if_nan(norm, nan_mode="any", data_name="nan_filled input data")
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
        for basin in tqdm(range(basin_coordinates), file=sys.stdout, disable=False):
            if not self.train_mode:
                # we don't need to ignore those with full nan in target vars for prediction without loss calculation
                # all samples should be included so that we can recover results to specified basins easily
                lookup.extend(
                    (basin, f, seq_len)
                    for f in range(warmup_length, max_time_length - rho - horizon + 1)
                )
            else:
                # some dataloader load data with warmup period, so leave some periods for it
                # [warmup_len] -> time_start -> [rho] -> [horizon]
                #                       window: \-----------------/ meaning rho + horizon
                nan_array = np.isnan(self.y[basin, :, :])
                if is_fixed_length_train:
                    for window in fixed_lengths:
                        lookup.extend(
                            (basin, f, window)
                            for f in range(
                                warmup_length,
                                max_time_length - window + 1,
                            )
                            # if all nan in window, we skip this sample
                            if not np.all(nan_array[f : f + window])
                        )
                else:
                    lookup.extend(
                        (basin, f, seq_len)
                        for f in range(
                            warmup_length, max_time_length - rho - horizon + 1
                        )
                        # if all nan in rho + horizon window, we skip this sample
                        if not np.all(nan_array[f : f + rho + horizon])
                    )
        self.lookup_table = dict(enumerate(lookup))
        self.num_samples = len(self.lookup_table)

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
        super(BasinSingleFlowDataset, self).__init__(cfgs, is_tra_val_te, **kwargs)

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
        date_format = detect_date_format(start_date)

        # Adjust the end date based on the time unit
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
        return self._read_xyc_specified_time(adjusted_start_date, end_date)

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
        start_date_dt = datetime.strptime(start_date, date_format)
        if time_unit == "h":
            adjusted_start_date = (start_date_dt - timedelta(hours=1)).strftime(
                date_format
            )
            adjusted_start_date_y = (
                start_date_dt
                + timedelta(hours=self.horizon * self.data_cfgs["min_time_interval"])
            ).strftime(date_format)
        elif time_unit == "D":
            adjusted_start_date = (
                start_date_dt - timedelta(days=self.data_cfgs["min_time_interval"])
            ).strftime(date_format)
            adjusted_start_date_y = (
                start_date_dt
                + timedelta(days=self.horizon * self.data_cfgs["min_time_interval"])
            ).strftime(date_format)
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
    """Dataset for Graph Neural Networks with station data support.

    This dataset extends FloodEventDataset to support station-based data
    including station timeseries and adjacency matrix information.
    """

    def __init__(self, cfgs: dict, is_tra_val_te: str):
        # Initialize station-related attributes
        self.station_data = None
        self.station_data_norm = None
        self.station_data_origin = None
        self.adjacency_matrices = None
        self.station_ids_by_basin = None
        self.station_scaler = None

        # Station-specific configuration
        self.station_cfgs = cfgs["data_cfgs"].get("station_cfgs", {})
        self.station_cols = self.station_cfgs.get("station_cols", [])
        self.station_rm_nan = self.station_cfgs.get("station_rm_nan", True)
        self.use_adjacency = self.station_cfgs.get("use_adjacency", True)

        super(GNNDataset, self).__init__(cfgs, is_tra_val_te)

    def _read_xyc_specified_time(self, start_date, end_date):
        """Read x, y, c data including station data from data source"""
        # Get basic data from parent class
        data_dict = super(GNNDataset, self)._read_xyc_specified_time(
            start_date, end_date
        )

        # Read station data if station_cols is specified
        if self.station_cols:
            station_data = self._read_station_data(start_date, end_date)
            data_dict["station_cols"] = station_data

            # Read adjacency matrices if enabled
            if self.use_adjacency:
                adjacency_data = self._read_adjacency_data()
                data_dict["adjacency_cols"] = adjacency_data

        return data_dict

    def _read_station_data(self, start_date, end_date):
        """Read station timeseries data for all basins"""
        station_data_dict = {}

        # Get basin IDs
        basin_ids = self.t_s_dict["sites_id"]

        # Get time units for station data
        time_units = self.station_cfgs.get("station_time_units", ["1D"])

        for basin_id in basin_ids:
            try:
                # Get station IDs for this basin
                station_ids = self.data_source.get_stations_by_basin(basin_id)

                if not station_ids:
                    # If no stations, create empty data structure
                    station_data_dict[basin_id] = None
                    continue

                # Read station data from cache
                station_data = self.data_source.read_station_ts_xrdataset(
                    station_id_lst=station_ids,
                    t_range=[start_date, end_date],
                    var_lst=self.station_cols,
                    time_units=time_units,
                )

                # Convert to DataArray and handle multiple time units
                if station_data:
                    # Use first time unit if multiple are available
                    time_unit = list(station_data.keys())[0]
                    station_ds = station_data[time_unit]

                    if not station_ds.sizes:
                        station_data_dict[basin_id] = None
                    else:
                        # Convert to DataArray
                        station_da = station_ds.to_array(dim="variable")
                        # Set units attribute
                        units_dict = {
                            var: station_ds[var].attrs.get("units", "unknown")
                            for var in station_ds.variables
                            if var in self.station_cols
                        }
                        station_da.attrs["units"] = units_dict
                        station_data_dict[basin_id] = station_da
                else:
                    station_data_dict[basin_id] = None

            except Exception as e:
                LOGGER.warning(f"Failed to read station data for basin {basin_id}: {e}")
                station_data_dict[basin_id] = None

        return station_data_dict

    def _read_adjacency_data(self):
        """Read adjacency matrices for all basins"""
        adjacency_dict = {}
        basin_ids = self.t_s_dict["sites_id"]

        for basin_id in basin_ids:
            try:
                # Read adjacency matrix for this basin
                adjacency_df = self.data_source.read_basin_adjacency(basin_id)

                if not adjacency_df.empty:
                    # Convert to numpy array
                    adjacency_matrix = adjacency_df.values
                    adjacency_dict[basin_id] = adjacency_matrix
                else:
                    adjacency_dict[basin_id] = None

            except Exception as e:
                LOGGER.warning(
                    f"Failed to read adjacency data for basin {basin_id}: {e}"
                )
                adjacency_dict[basin_id] = None

        return adjacency_dict

    def _normalize(self, origin_data):
        """Normalize data including station data"""
        # Get basic normalization from parent class
        scaler_hub = ScalerHub(
            origin_data,
            data_cfgs=self.data_cfgs,
            is_tra_val_te=self.is_tra_val_te,
            data_source=self.data_source,
        )
        self.target_scaler = scaler_hub.target_scaler
        norm_data = scaler_hub.norm_data

        # Normalize station data separately
        if "station_cols" in origin_data and origin_data["station_cols"]:
            norm_data["station_cols"] = self._normalize_station_data(
                origin_data["station_cols"]
            )

        # Adjacency matrices don't need normalization
        if "adjacency_cols" in origin_data:
            norm_data["adjacency_cols"] = origin_data["adjacency_cols"]

        return norm_data

    def _normalize_station_data(self, station_data_dict):
        """Normalize station data using specified scaler"""
        if not station_data_dict:
            return station_data_dict

        # Get scaler type from config
        scaler_type = self.station_cfgs.get("station_scaler_type", "DapengScaler")

        # Collect all non-None station data for fitting scaler
        all_station_data = []
        basin_to_data_map = {}

        for basin_id, station_data in station_data_dict.items():
            if station_data is not None:
                basin_to_data_map[basin_id] = station_data
                # Reshape to (n_samples, n_features) for scaler
                data_2d = station_data.values.reshape(-1, station_data.shape[-1])
                all_station_data.append(data_2d)

        if not all_station_data:
            return station_data_dict

        # Concatenate all station data
        combined_data = np.concatenate(all_station_data, axis=0)

        # Remove NaN values for scaler fitting
        valid_mask = ~np.isnan(combined_data).any(axis=1)
        if not np.any(valid_mask):
            LOGGER.warning("All station data contains NaN, skipping normalization")
            return station_data_dict

        clean_data = combined_data[valid_mask]

        # Initialize scaler based on type
        if scaler_type == "DapengScaler":
            from torchhydro.datasets.data_scalers import DapengScaler

            self.station_scaler = DapengScaler(
                feature_range=self.data_cfgs.get("station_scaler_params", {}).get(
                    "feature_range", [0, 1]
                )
            )
        elif scaler_type == "SklearnScaler":
            from torchhydro.datasets.data_scalers import SklearnScaler

            scaler_name = self.data_cfgs.get("station_scaler_params", {}).get(
                "scaler_name", "StandardScaler"
            )
            self.station_scaler = SklearnScaler(scaler_name=scaler_name)
        else:
            raise ValueError(f"Unknown station scaler type: {scaler_type}")

        # Fit scaler only on training data
        if self.is_tra_val_te == "train":
            self.station_scaler.fit(clean_data)

        # Apply normalization to each basin's station data
        normalized_station_data = {}
        for basin_id, station_data in basin_to_data_map.items():
            original_shape = station_data.shape
            data_2d = station_data.values.reshape(-1, original_shape[-1])

            # Apply normalization
            try:
                normalized_2d = self.station_scaler.transform(data_2d)
                normalized_data = normalized_2d.reshape(original_shape)

                # Create new DataArray with normalized data
                normalized_da = xr.DataArray(
                    normalized_data,
                    dims=station_data.dims,
                    coords=station_data.coords,
                    attrs=station_data.attrs,
                )
                normalized_station_data[basin_id] = normalized_da
            except Exception as e:
                LOGGER.warning(
                    f"Failed to normalize station data for basin {basin_id}: {e}"
                )
                normalized_station_data[basin_id] = station_data

        # Add None entries for basins without station data
        for basin_id in station_data_dict:
            if basin_id not in normalized_station_data:
                normalized_station_data[basin_id] = None

        return normalized_station_data

    def _kill_nan(self, origin_data, norm_data):
        """Handle NaN values including station data"""
        # Get basic NaN handling from parent class
        origins_wonan, norms_wonan = super(GNNDataset, self)._kill_nan(
            origin_data, norm_data
        )

        # Handle station data NaN values
        if "station_cols" in origin_data and origin_data["station_cols"]:
            origin_station_wonan = self._kill_station_nan(origin_data["station_cols"])
            norm_station_wonan = self._kill_station_nan(norm_data["station_cols"])

            origins_wonan["station_cols"] = origin_station_wonan
            norms_wonan["station_cols"] = norm_station_wonan

        # Adjacency matrices don't need NaN handling
        if "adjacency_cols" in origin_data:
            origins_wonan["adjacency_cols"] = origin_data["adjacency_cols"]
            norms_wonan["adjacency_cols"] = norm_data["adjacency_cols"]

        return origins_wonan, norms_wonan

    def _kill_station_nan(self, station_data_dict):
        """Remove NaN values from station data"""
        if not station_data_dict:
            return station_data_dict

        processed_station_data = {}

        for basin_id, station_data in station_data_dict.items():
            if station_data is None:
                processed_station_data[basin_id] = None
                continue

            if self.station_rm_nan:
                # Apply interpolation to fill NaN values
                filled_data = _fill_gaps_da(station_data, fill_nan="interpolate")
                processed_station_data[basin_id] = filled_data
            else:
                processed_station_data[basin_id] = station_data

        return processed_station_data

    def _trans2nparr(self, origin_data, norm_data):
        """Transform data to numpy arrays including station data"""
        # Call parent method for basic data
        super(GNNDataset, self)._trans2nparr(origin_data, norm_data)

        # Handle station data
        if "station_cols" in origin_data and origin_data["station_cols"]:
            self.station_data_origin = self._station_dict_to_array(
                origin_data["station_cols"]
            )
            self.station_data = self._station_dict_to_array(norm_data["station_cols"])

        # Handle adjacency matrices
        if "adjacency_cols" in origin_data and origin_data["adjacency_cols"]:
            self.adjacency_matrices = origin_data["adjacency_cols"]

    def _station_dict_to_array(self, station_data_dict):
        """Convert station data dict to numpy array for efficient access"""
        if not station_data_dict:
            return None

        # Create a mapping from basin_id to basin_index
        basin_ids = self.t_s_dict["sites_id"]
        basin_to_idx = {basin_id: idx for idx, basin_id in enumerate(basin_ids)}

        # Initialize array structure
        # We'll create a list of arrays, one for each basin
        station_arrays = []

        for basin_id in basin_ids:
            station_data = station_data_dict.get(basin_id)
            if station_data is None:
                # Create empty array for basins without station data
                station_arrays.append(None)
            else:
                # Convert DataArray to numpy array
                # Expected shape: (n_stations, n_time, n_variables)
                array_data = station_data.values.transpose(
                    1, 0, 2
                )  # station, time, variable
                station_arrays.append(array_data)

        return station_arrays

    def get_station_data(self, basin_idx, time_idx):
        """Get station data for a specific basin and time range"""
        if self.station_data is None or basin_idx >= len(self.station_data):
            return None

        station_data = self.station_data[basin_idx]
        if station_data is None:
            return None

        # Return station data for the specified time range
        return station_data[:, time_idx, :]

    def get_adjacency_matrix(self, basin_idx):
        """Get adjacency matrix for a specific basin"""
        if self.adjacency_matrices is None:
            return None

        basin_id = self.t_s_dict["sites_id"][basin_idx]
        return self.adjacency_matrices.get(basin_id)

    def __getitem__(self, item: int):
        """Get one sample from the dataset

        Returns
        -------
        tuple
            Input and output data for GNN model
        """
        # TODO: Implement based on specific GNN model requirements
        # This is a placeholder that can be customized based on your GNN architecture

        # Get basic data from parent class
        x, y = super(GNNDataset, self).__getitem__(item)

        # Get station data and adjacency matrix if available
        basin, idx, _ = self.lookup_table[item]

        # Get station data for this time window
        station_data = self.get_station_data(basin, slice(idx, idx + self.rho))

        # Get adjacency matrix
        adj_matrix = self.get_adjacency_matrix(basin)

        # Return data structure suitable for GNN
        # You can customize this based on your specific GNN model requirements
        return {
            "x": x,  # Basic input features
            "y": y,  # Target output
            "station_data": station_data,  # Station time series data
            "adj_matrix": adj_matrix,  # Adjacency matrix
            "basin_idx": basin,  # Basin index
            "time_idx": idx,  # Time index
        }

    def __len__(self):
        return self.num_samples
