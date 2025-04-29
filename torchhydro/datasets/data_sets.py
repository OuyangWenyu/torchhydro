"""
Author: Wenyu Ouyang
Date: 2024-04-08 18:16:53
LastEditTime: 2025-04-19 17:35:29
LastEditors: Wenyu Ouyang
Description: A pytorch dataset class; references to https://github.com/neuralhydrology/neuralhydrology
FilePath: /torchhydro/torchhydro/datasets/data_sets.py
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
        current_idx = self.evaluation_cfgs["current_idx"]
        if rolling == 0:
            hrwin = current_idx
            frwin = self.nt - current_idx
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

    def denormalize(self, norm_data, is_real_time=True):
        """Denormalize the norm_data

        Parameters
        ----------
        norm_data : np.ndarray
            batch-first data
        is_real_time : bool, optional
            whether the data is real time data, by default True
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
            "constant_cols": c_origin.transpose("basin", "variable"),
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
            if key == "target_cols":
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
        max_time_length = self.nt
        for basin in tqdm(range(basin_coordinates), file=sys.stdout, disable=False):
            if not self.train_mode:
                # we don't need to ignore those with full nan in target vars for prediction without loss calculation
                # all samples should be included so that we can recover results to specified basins easily
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
        """获取数据集中的一个样本

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
        basin, idx = self.lookup_table[item]
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
            x_combined[self.rho :, x_idx] = f[:, f_idx]

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
        if "streamflow" in y:
            area = data_source_.camels.read_area(self.t_s_dict["sites_id"])
            y.update(streamflow_unit_conv(y[["streamflow"]], area))
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

    def denormalize(self, norm_data, is_real_time=True):
        """Denormalize the norm_data

        Parameters
        ----------
        norm_data : np.ndarray
            batch-first data
        is_real_time : bool, optional
            whether the data is real time data, by default True
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
        warmup_length = self.warmup_length
        selected_time_points = target_data.coords["time"][warmup_length:-1]
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

    def __getitem__(self, item: int):
        basin, time = self.lookup_table[item]
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


class ForecastDataset(BaseDataset):
    def __init__(self, data_cfgs: dict, is_tra_val_te: str):
        super(ForecastDataset, self).__init__(data_cfgs, is_tra_val_te)

    def __getitem__(self, item):
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
        basin, idx = self.lookup_table[item]
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
