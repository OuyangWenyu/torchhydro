"""
Author: Wenyu Ouyang
Date: 2022-02-13 21:20:18
LastEditTime: 2023-01-11 14:44:00
LastEditors: Xinzhuo Wu
Description: A pytorch dataset class; references to https://github.com/neuralhydrology/neuralhydrology
FilePath: \torchhydro\torchhydro\datasets\data_sets.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
import io
import logging
import sys
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Optional

import boto3
import numpy as np
import pint_xarray  # noqa: F401
import requests
import torch
import xarray as xr
from botocore.config import Config
from hydrodataset import HydroDataset
from torch.utils.data import Dataset
from tqdm import tqdm

from torchhydro.datasets.data_scalers import (
    ScalerHub,
    Muti_Basin_GPM_GFS_SCALER,
)
from torchhydro.datasets.data_source_gpm_gfs import GPM_GFS
from torchhydro.datasets.data_utils import (
    warn_if_nan,
    wrap_t_s_dict,
    unify_streamflow_unit,
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
            if warn_if_nan(mean_val):
                # when all value are NaN, mean_val will be NaN, we set mean_val to -1
                mean_val = -1
            filled_data = var_data.fillna(
                mean_val
            )  # fill NaN values with the calculated mean
            da.loc[
                dict(variable=var)
            ] = filled_data  # update the original dataarray with the filled data
    elif fill_nan == "interpolate":
        # fill interpolation
        for i in range(da.shape[0]):
            da[i] = da[i].interpolate_na(dim="time", fill_value="extrapolate")
    else:
        raise NotImplementedError(f"fill_nan {fill_nan} not implemented")
    return da


class BaseDataset(Dataset):
    """Base data set class to load and preprocess data (batch-first) using PyTorch's Dataset"""

    def __init__(self, data_source: HydroDataset, data_cfgs: dict, is_tra_val_te: str):
        """
        Parameters
        ----------
        data_source
            object for reading source data
        data_cfgs
            parameters for reading source data
        is_tra_val_te
            train, vaild or test
        """
        super(BaseDataset, self).__init__()
        self.data_source = data_source
        self.data_cfgs = data_cfgs
        if is_tra_val_te in {"train", "valid", "test"}:
            self.is_tra_val_te = is_tra_val_te
        else:
            raise ValueError(
                "'is_tra_val_te' must be one of 'train', 'valid' or 'test' "
            )
        # load and preprocess data
        self._load_data()

    def __len__(self):
        return self.num_samples if self.train_mode else len(self.t_s_dict["sites_id"])

    def __getitem__(self, item: int):
        if not self.train_mode:
            basin = self.t_s_dict["sites_id"][item]
            # we don't need warmup_length for models yet
            x = self.x.sel(basin=basin).to_numpy().T
            y = self.y.sel(basin=basin).to_numpy().T
            if self.c is None or self.c.shape[-1] == 0:
                return torch.from_numpy(x).float(), torch.from_numpy(y).float()
            # TODO: not CHECK attributes reading
            c = self.c.sel(basin=basin).values
            c = np.repeat(c, x.shape[0], axis=0).reshape(c.shape[0], -1).T
            xc = np.concatenate((x, c), axis=1)
            return torch.from_numpy(xc).float(), torch.from_numpy(y).float()
        basin, time = self.lookup_table[item]
        seq_length = self.rho
        warmup_length = self.warmup_length
        x = (
            self.x.sel(
                basin=basin,
                time=slice(
                    time - np.timedelta64(warmup_length, "D"),
                    time + np.timedelta64(seq_length - 1, "D"),
                ),
            ).to_numpy()
        ).T
        if self.c is not None and self.c.shape[-1] > 0:
            c = self.c.sel(basin=basin).values
            c = np.tile(c, (warmup_length + seq_length, 1))
            x = np.concatenate((x, c), axis=1)
        # for y, we don't need warmup as warmup are only used for get initial value for some state variables
        y = (
            self.y.sel(
                basin=basin,
                time=slice(
                    time,
                    time + np.timedelta64(seq_length - 1, "D"),
                ),
            )
            .to_numpy()
            .T
        )
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()

    def _load_data(self):
        train_mode = self.is_tra_val_te == "train"
        self.t_s_dict = wrap_t_s_dict(
            self.data_source, self.data_cfgs, self.is_tra_val_te
        )
        # y
        data_flow_ds = self.data_source.read_ts_xrdataset(
            self.t_s_dict["sites_id"],
            self.t_s_dict["t_final_range"],
            self.data_cfgs["target_cols"],
        )
        # x
        data_forcing_ds = self.data_source.read_ts_xrdataset(
            self.t_s_dict["sites_id"],
            self.t_s_dict["t_final_range"],
            # 6 comes from here
            self.data_cfgs["relevant_cols"],
        )
        # c
        data_attr_ds = self.data_source.read_attr_xrdataset(
            self.t_s_dict["sites_id"],
            self.data_cfgs["constant_cols"],
            all_number=True,
        )
        # trans to dataarray to better use xbatch
        if self.data_source.streamflow_unit != "mm/d":
            data_flow_ds = unify_streamflow_unit(
                data_flow_ds, self.data_source.read_area(self.t_s_dict["sites_id"])
            )
        data_flow = self._trans2da_and_setunits(data_flow_ds)
        if data_forcing_ds is not None:
            data_forcing = self._trans2da_and_setunits(data_forcing_ds)
        else:
            data_forcing = None
        if data_attr_ds is not None:
            # firstly, we should transform some str type data to float type
            data_attr = self._trans2da_and_setunits(data_attr_ds)
        else:
            data_attr = None
        # save unnormalized data to use in physics-based modeling, we will use streamflow with unit of mm/day
        self.x_origin = data_forcing_ds
        self.y_origin = data_flow_ds
        self.c_origin = data_attr_ds
        # normalization
        scaler_hub = ScalerHub(
            data_flow,
            data_forcing,
            data_attr,
            data_cfgs=self.data_cfgs,
            is_tra_val_te=self.is_tra_val_te,
            data_source=self.data_source,
        )

        self.x, self.y, self.c = self.kill_nan(scaler_hub.x, scaler_hub.c, scaler_hub.y)
        self.train_mode = train_mode
        self.rho = self.data_cfgs["forecast_history"]
        self.target_scaler = scaler_hub.target_scaler
        self.warmup_length = self.data_cfgs["warmup_length"]
        self._create_lookup_table()

    @property
    def basins(self):
        """Return the basins of the dataset"""
        return self.t_s_dict["sites_id"]

    @property
    def times(self):
        """Return the time range of the dataset"""
        return self.t_s_dict["t_final_range"]

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

    def kill_nan(self, x, c, y):
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
        return x, y, c

    def _create_lookup_table(self):
        lookup = []
        # list to collect basins ids of basins without a single training sample
        basins = self.t_s_dict["sites_id"]
        rho = self.rho
        warmup_length = self.warmup_length
        dates = self.y["time"].to_numpy()
        time_length = len(dates)
        is_tra_val_te = self.is_tra_val_te
        for basin in tqdm(
            basins,
            file=sys.stdout,
            disable=False,
            desc=f"Creating {is_tra_val_te} lookup table",
        ):
            # some dataloader load data with warmup period, so leave some periods for it
            # [warmup_len] -> time_start -> [rho]
            lookup.extend(
                (basin, dates[f])
                for f in range(warmup_length, time_length)
                if f < time_length - rho + 1
            )
        self.lookup_table = dict(enumerate(lookup))
        self.num_samples = len(self.lookup_table)


class BasinSingleFlowDataset(BaseDataset):
    """one time length output for each grid in a batch"""

    def __init__(self, data_source: HydroDataset, data_cfgs: dict, is_tra_val_te: str):
        super(BasinSingleFlowDataset, self).__init__(
            data_source, data_cfgs, is_tra_val_te
        )

    def __getitem__(self, index):
        xc, ys = super(BasinSingleFlowDataset, self).__getitem__(index)
        y = ys[-1, :]
        return xc, y

    def __len__(self):
        return self.num_samples


class DplDataset(BaseDataset):
    """pytorch dataset for Differential parameter learning"""

    def __init__(self, data_source: HydroDataset, data_cfgs: dict, is_tra_val_te: str):
        """
        Parameters
        ----------
        data_source
            object for reading source data
        data_cfgs
            configs for reading source data
        is_tra_val_te
            train, vaild or test
        """
        super(DplDataset, self).__init__(data_source, data_cfgs, is_tra_val_te)
        # we don't use y_un_norm as its name because in the main function we will use "y"
        # For physical hydrological models, we need warmup, hence the target values should exclude data in warmup period
        self.warmup_length = data_cfgs["warmup_length"]
        self.target_as_input = data_cfgs["target_as_input"]
        self.constant_only = data_cfgs["constant_only"]
        if self.target_as_input and (not self.train_mode):
            # if the target is used as input and train_mode is False,
            # we need to get the target data in training period to generate pbm params
            self.train_dataset = DplDataset(
                data_source, data_cfgs, is_tra_val_te="train"
            )

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


class GPM_GFS_Dataset(Dataset):
    def __init__(self, data_source: GPM_GFS, data_cfgs: dict, is_tra_val_te: str):
        super(GPM_GFS_Dataset, self).__init__()
        self.data_source = data_source
        self.data_cfgs = data_cfgs
        if is_tra_val_te in {"train", "valid", "test"}:
            self.is_tra_val_te = is_tra_val_te
        else:
            raise ValueError(
                "'is_tra_val_te' must be one of 'train', 'valid' or 'test' "
            )
        # load and preprocess data
        self._load_data()

    def _load_data(self):
        """
        Loads and prepares data for hydrological modeling, accommodating different types of data sources and configurations.

        This internal method orchestrates the loading and preprocessing of various datasets, including water level, streamflow, rainfall, GFS data, and constant attributes. It applies necessary transformations and scaling to these datasets, aligning them with the configuration settings and preparing them for use in model training and evaluation.

        Process:
        1. Sets up basic configuration parameters such as training mode, historical data length, forecast length, and warmup length.
        2. Creates a dictionary of site and time information based on the data source and configuration.
        3. Depending on the target column configuration, loads either water level or streamflow data, and applies transformations.
        4. Loads rainfall data and, if configured, GFS data, soil attributes data.
        5. Loads constant attribute data if specified in the configuration.
        6. Initializes the `Muti_Basin_GPM_GFS_SCALER` for scaling and normalization of all datasets.
        7. Removes NaN values from the datasets and stores the processed data in class attributes.
        8. Creates a lookup table for data indexing and retrieval.

        The method ensures that all necessary data is loaded, processed, and made ready for the subsequent stages of hydrological modeling. It accounts for various configurations and data types, providing a flexible and robust approach to data preparation.
        """
        train_mode = self.is_tra_val_te == "train"
        self.train_mode = train_mode
        self.rho = self.data_cfgs["forecast_history"]
        self.forecast_length = self.data_cfgs["forecast_length"]
        self.warmup_length = self.data_cfgs["warmup_length"]
        self.t_s_dict = wrap_t_s_dict(
            self.data_source, self.data_cfgs, self.is_tra_val_te
        )
        if self.data_cfgs["target_cols"] == ["waterlevel"]:
            data_waterlevel_ds = self.data_source.read_waterlevel_xrdataset(
                self.t_s_dict["sites_id"],
                self.t_s_dict["t_final_range"],
                self.data_cfgs["target_cols"],
                self.forecast_length,
                self.data_cfgs["water_level_source_path"],
                self.data_cfgs["user"],
            )
            if data_waterlevel_ds is not None:
                y_origin = self._trans2da_and_setunits(data_waterlevel_ds)
            else:
                data_streamflow_ds = None

        elif self.data_cfgs["target_cols"] == ["streamflow"]:
            data_streamflow_ds = self.data_source.read_streamflow_xrdataset(
                self.t_s_dict["sites_id"],
                self.t_s_dict["t_final_range"],
                self.data_cfgs["target_cols"],
                self.forecast_length,
                self.data_cfgs["user"],
                self.data_cfgs["streamflow_source_path"],
            )
            if data_streamflow_ds is not None:
                y_origin = self._trans2da_and_setunits(data_streamflow_ds)
            else:
                y_origin = self._trans2da_and_setunits(data_streamflow_ds)

        data_rainfall = self.data_source.read_rainfall_xrdataset(
            self.t_s_dict["sites_id"],
            self.t_s_dict["t_final_range"],
            self.data_cfgs["relevant_cols"][0],
            self.data_cfgs["rainfall_source_path"],
            self.data_cfgs["user"],
        )

        if self.data_cfgs["relevant_cols"][1] != ["None"]:
            data_gfs = self.data_source.read_gfs_xrdataset(
                self.t_s_dict["sites_id"],
                self.t_s_dict["t_final_range"],
                self.data_cfgs["relevant_cols"][1],
                self.forecast_length,
                self.rho,
                self.data_cfgs["gfs_source_path"],
                self.data_cfgs["user"],
            )
        else:
            data_gfs = None

        if self.data_cfgs["relevant_cols"][2] != ["None"]:
            data_soil = self.data_source.read_soil_xrdataset(
                self.t_s_dict["sites_id"],
                self.t_s_dict["t_final_range"],
                self.data_cfgs["relevant_cols"][1],
                self.forecast_length,
                self.rho,
                self.data_cfgs["soil_source_path"],
                self.data_cfgs["user"],
            )
        else:
            data_soil = None

        if self.data_cfgs["constant_cols"]:
            data_attr_ds = self.data_source.read_attr_xrdataset(
                self.t_s_dict["sites_id"],
                self.data_cfgs["constant_cols"],
                self.data_cfgs["attributes_path"],
                self.data_cfgs["user"],
            )
            data_attr = self._trans2da_and_setunits(data_attr_ds)
        else:
            data_attr = None

        scaler_hub = Muti_Basin_GPM_GFS_SCALER(
            y_origin,
            data_rainfall,
            data_attr,
            data_gfs,
            data_soil,
            data_cfgs=self.data_cfgs,
            is_tra_val_te=self.is_tra_val_te,
            data_source=self.data_source,
        )

        self.x, self.y, self.c, self.g, self.s = self.kill_nan(
            scaler_hub.x, scaler_hub.y, scaler_hub.c, scaler_hub.g, scaler_hub.s
        )

        self.target_scaler = scaler_hub.target_scaler

        self._create_lookup_table()

    def kill_nan(self, x, y, c, g, s):
        """
        Removes or interpolates NaN values in the provided datasets.

        This method is responsible for handling NaN (Not a Number) values in various datasets including time series data (x),
        observation data (y), constant attributes (c), and GFS data (g). It applies specified strategies to either remove or
        interpolate NaN values based on the data configurations.

        Parameters:
        - x: Dictionary of time series data for each basin.
        - y: xarray Dataset or NumPy array of observation data.
        - c: xarray Dataset or NumPy array of constant attribute data.
        - g: Dictionary of GFS data for each basin.
        - s: Dictionary of soil attributes data for each basin.

        Process:
        1. For time series data (x), interpolates NaN values if specified in the configuration.
        2. For observation data (y), fills gaps using interpolation or another specified method.
        3. For constant attributes (c), fills gaps using the mean or another specified method, if applicable.
        4. For GFS data (g), interpolates NaN values if specified.
        5. For soil attributes data, interpolates NaN values if specified.

        Each dataset is checked for NaN values, and a warning is issued if NaNs are still present after processing.

        Returns:
        The processed datasets (x, y, c, g, s) with NaN values handled according to the specified configurations.

        This method plays a crucial role in ensuring data quality and consistency, particularly in hydrological modeling where NaN values can significantly impact model performance and results.
        """
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

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item: int):
        """
        Retrieves a dataset sample based on the specified index, including input features (x, c, g, s) and target values (y).

        Parameters:
        - item (int): An index for the dataset sample to retrieve.

        Process:
        1. Determines the basin and time corresponding to the given index from a lookup table.
        2. Selects a sequence of data based on configured sequence length (`rho`) and output sequence length (`forecast_length`).
        3. Transforms time series data (`x`) for the specified basin and time into a numpy array, and transposes it for model input.
        4. Transforms observation data (`y`) for the specified basin and time into a numpy array.
        5. Conditionally adds constant attributes (`c`), GFS data (`g`), and additional data (`s`) based on their availability:
            - If `c` is not None and `g`, `s` are None, includes `c`.
            - If `g` is not None and `c`, `s` are None, concatenates `g` with `x`.
            - If `s` is not None and `c`, `g` are None, includes `s`.
            - If both `c` and `g` are not None and `s` is None, concatenates `g` with `x` and includes `c`.
            - If both `c` and `s` are not None and `g` is None, includes both `c` and `s`.
            - If both `s` and `g` are not None and `c` is None, concatenates `g` with `x` and includes `s`.
            - If `c`, `g`, and `s` are all not None, concatenates `g` with `x` and includes both `c` and `s`.
        6. If none of `c`, `g`, `s` are available, only `x` and `y` are included.
        7. Converts the selected data into PyTorch tensors suitable for model input.

        Returns:
        - A tuple containing a list of PyTorch tensors for input features (including time series data, constant attributes, GFS data, and additional data as available), and a PyTorch tensor for the target values.

        This method allows for flexible data retrieval from the dataset, accommodating varying availability of additional data sources (`c`, `g`, `s`) and ensuring proper alignment and formatting for model training and evaluation.
        """
        # here time is time_now in gpm_gfs_data
        basin, time = self.lookup_table[item]
        seq_length = self.rho
        output_seq_len = self.forecast_length

        x = (
            self.x[basin]
            .sel(time_now=time)
            .sel(step=slice(0, seq_length + output_seq_len - 1))
            .values
        )
        x = np.transpose(x, (1, 0, 2, 3))

        y = (
            self.y.sel(basin=basin)
            .sel(
                time=slice(
                    time,
                    time + np.timedelta64(output_seq_len - 1, "h"),
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
                    time - np.timedelta64(self.rho, "h"),
                    time + np.timedelta64(self.forecast_length - 1, "h"),
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
                    time - np.timedelta64(self.rho, "h"),
                    time - np.timedelta64(length + 1, "h"),
                )
            )
            .values
        )
        return np.transpose(s, (1, 0, 2, 3))

    def basins(self):
        """Return the basins of the dataset"""
        return self.t_s_dict["sites_id"]

    def times(self):
        """Return the time range of the dataset"""
        return self.t_s_dict["t_final_range"]

    def _create_lookup_table(self):
        """
        Creates a lookup table mapping each sample index to a corresponding basin and time step.

        This method is crucial for efficiently accessing data samples during the training and evaluation of models. It generates a dictionary where each key is a sample index, and the value is a tuple of basin ID and time step. The method ensures that the dataset is properly indexed for random access in machine learning workflows.

        Process:
        1. Initializes an empty list for the lookup table.
        2. Retrieves the list of basin IDs and the time steps from the time series data.
        3. Calculates the total number of time steps and divides it by the number of time ranges to find the length of each time range.
        4. Iterates through each basin and time range, appending tuples of basin and time step to the lookup list.
        5. Converts the list into a dictionary for faster access, where each key-value pair corresponds to a sample index and its associated basin and time information.

        Post-Process:
        - Stores the lookup table as a class attribute.
        - Sets the total number of samples in the dataset for reference.

        This method ensures that each sample in the dataset can be quickly and accurately retrieved by its index, which is essential for batch processing in machine learning models, especially in time series forecasting and hydrological applications.
        """
        lookup = []
        basins = self.t_s_dict["sites_id"]
        dates = self.x[basins[0]]["time_now"].to_numpy()  # 取其中一个流域的时间作为标尺
        time_total_length = len(dates)
        time_num = len(self.t_s_dict["t_final_range"])
        time_single_length = int(time_total_length / time_num)
        is_tra_val_te = self.is_tra_val_te
        for basin in tqdm(
            basins,
            file=sys.stdout,
            disable=False,
            desc=f"Creating {is_tra_val_te} lookup table",
        ):
            for num in range(time_num):
                lookup.extend(
                    (basin, dates[f + num * time_single_length])
                    for f in range(0, time_total_length)
                    if f < time_single_length
                )
        self.lookup_table = dict(enumerate(lookup))
        self.num_samples = len(self.lookup_table)


# Todo
class GPM_GFS_batch_loading_Dataset(Dataset):
    def __init__(self, data_cfgs: dict, minio_cfgs: dict, is_tra_val_te: str):
        super(GPM_GFS_batch_loading_Dataset, self).__init__()
        self.data_cfgs = data_cfgs
        self.minio_cfgs = minio_cfgs
        if is_tra_val_te in {"train", "valid", "test"}:
            self.is_tra_val_te = is_tra_val_te
        else:
            raise ValueError(
                "'is_tra_val_te' must be one of 'train', 'valid' or 'test' "
            )
        self.rho = self.data_cfgs["forecast_history"]
        self.forecast_length = self.data_cfgs["forecast_length"]
        self.loading_batch = self.data_cfgs["loading_batch"]
        self.t_s_dict = self._wrap_t_s_dict()
        self._calculate_mean_std()

    def _wrap_t_s_dict(self):
        basins_id = self.data_cfgs["object_ids"]

        if f"t_range_{self.is_tra_val_te}" in self.data_cfgs:
            t_range_list = self.data_cfgs[f"t_range_{self.is_tra_val_te}"]
        else:
            raise Exception(
                f"Error! The mode {self.is_tra_val_te} was not found. Please add it."
            )
        return OrderedDict(sites_id=basins_id, t_final_range=t_range_list)

    def _calculate_mean_std(self):
        basin_list = self.t_s_dict["sites_id"]
        basin_num = len(basin_list)
        loading_batch = self.loading_batch
        global_mean = 0
        global_M2 = 0
        global_count = 0

        for i in range(0, basin_num, loading_batch):
            batch_basin_list = basin_list[i : i + loading_batch]
            precip_dict, Q_xr = self.read_precip_streamflow_xrdataset(
                batch_basin_list,
                self.t_s_dict["t_final_range"],
                self.minio_cfgs,
                self.forecast_length,
            )

    def read_precip_streamflow_xrdataset(
        self,
        gage_id_lst: list = None,
        t_range: list = None,
        minio_cfgs: dict = None,
        forecast_length: int = None,
    ):
        """
        Reads precipitation and streamflow data from MinIO for specified basin IDs.

        This function connects to MinIO using provided configuration parameters and retrieves
        data for each basin ID in `gage_id_lst`. It processes precipitation and streamflow data
        within the specified `t_range` and extends streamflow data by `forecast_length`.
        The data is read using xarray, and subsets are merged and returned as dictionaries.

        Parameters:
        - gage_id_lst (list): List of basin IDs.
        - t_range (list): Time range for data selection.
        - minio_cfgs (dict): Configuration parameters for MinIO connection.
        - forecast_length (int): Length to extend the streamflow forecast.

        Returns:
        tuple: A tuple of two dictionaries containing precipitation data (`precip_dict`)
        and streamflow data (`Q_dict`), keyed by basin ID.

        Raises:
        NotImplementedError: If MinIO connection parameters or basin IDs are not provided,
        or if there's an error in loading the data.
        """

        if minio_cfgs is None:
            raise NotImplementedError(
                "You have not configured the parameters for connecting to MinIO"
            )
        if gage_id_lst is None:
            raise NotImplementedError("The basin ID has not been specified")

        endpoint_url = minio_cfgs["endpoint_url"]
        access_key = minio_cfgs["access_key"]
        secret_key = minio_cfgs["secret_key"]
        precip_bucket_name = minio_cfgs["bucket_name"]["precip"]
        precip_folder_prefix = minio_cfgs["folder_prefix"]["precip"]
        Q_bucket_name = minio_cfgs["bucket_name"]["streamflow"]
        Q_folder_prefix = minio_cfgs["folder_prefix"]["streamflow"]

        s3_client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=Config(signature_version="s3v4"),
        )

        precip_dict = {}
        Q_xr = xr.Dataset()

        for basin in gage_id_lst:
            precip_object_key = f"{precip_folder_prefix}{basin}.nc"
            precip_url = s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": precip_bucket_name, "Key": precip_object_key},
                ExpiresIn=3600,
            )
            precip_response = requests.get(precip_url)

            if precip_response.status_code == 200:
                with io.BytesIO(precip_response.content) as f:
                    precip = xr.open_dataset(f)
                    subset_list = []
                    for period in t_range:
                        start_date = period["start"]
                        end_date = period["end"]

                        subset = precip.sel(time_now=slice(start_date, end_date))
                        subset_list.append(subset)

                    merged_dataset_tp = xr.concat(subset_list, dim="time_now")
                    precip_dict[basin] = merged_dataset_tp.to_array(dim="variable")
            else:
                raise NotImplementedError(
                    "Error loading file:",
                    precip_object_key,
                    "; Status Code:",
                    precip_response.status_code,
                )

            Q_object_key = f"{Q_folder_prefix}{basin}.nc"
            Q_url = s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": Q_bucket_name, "Key": Q_object_key},
                ExpiresIn=3600,
            )
            Q_response = requests.get(Q_url)

            if Q_response.status_code == 200:
                with io.BytesIO(Q_response.content) as f:
                    Q = xr.open_dataset(f)
                    subset_list = []
                    for period in t_range:
                        start_date = period["start"]
                        end_date = period["end"]

                        end_date = datetime.strptime(period["end"], "%Y-%m-%d")
                        new_end_date = end_date + timedelta(hours=forecast_length)
                        end_date_str = new_end_date.strftime("%Y-%m-%d")

                        subset = Q.sel(time=slice(start_date, end_date_str))
                        subset_list.append(subset)
                    merged_dataset_Q = xr.concat(subset_list, dim="time")
                    Q_xr = xr.concat([merged_dataset_Q, Q_xr], dim="basin").to_array(
                        dim="variable"
                    )
            else:
                raise NotImplementedError(
                    "Error loading file:",
                    Q_object_key,
                    "; Status Code:",
                    Q_response.status_code,
                )
        return precip_dict, Q_xr

    def _load_data(self):
        train_mode = self.is_tra_val_te == "train"
        self.train_mode = train_mode
        self.rho = self.data_cfgs["forecast_history"]
        self.forecast_length = self.data_cfgs["forecast_length"]
        self.warmup_length = self.data_cfgs["warmup_length"]
        self.t_s_dict = wrap_t_s_dict(
            self.data_source, self.data_cfgs, self.is_tra_val_te
        )
        if self.data_cfgs["target_cols"] == ["waterlevel"]:
            data_waterlevel_ds = self.data_source.read_waterlevel_xrdataset(
                self.t_s_dict["sites_id"],
                self.t_s_dict["t_final_range"],
                self.data_cfgs["target_cols"],
                self.forecast_length,
            )

            if data_waterlevel_ds is not None:
                data_waterlevel = self._trans2da_and_setunits(data_waterlevel_ds)
            else:
                data_waterlevel = None

            self.y_origin = data_waterlevel

        elif self.data_cfgs["target_cols"] == ["streamflow"]:
            data_streamflow_ds = self.data_source.read_streamflow_xrdataset(
                self.t_s_dict["sites_id"],
                self.t_s_dict["t_final_range"],
                self.data_cfgs["target_cols"],
                self.forecast_length,
            )

            if data_streamflow_ds is not None:
                data_streamflow = self._trans2da_and_setunits(data_streamflow_ds)
            else:
                data_streamflow = None

            self.y_origin = data_streamflow

        data_forcing_ds = self.data_source.read_gpm_xrdataset(
            self.t_s_dict["sites_id"],
            self.t_s_dict["t_final_range"],
            # 1 comes from here
            self.data_cfgs["relevant_cols"],
        )

        data_forcing = {}
        if data_forcing_ds is not None:
            for basin, data in data_forcing_ds.items():
                result = data.to_array(dim="variable")
                data_forcing[basin] = result
        else:
            data_forcing = None

        self.x_origin = data_forcing

        scaler_hub = Muti_Basin_GPM_GFS_SCALER(
            self.y_origin,
            data_forcing,
            data_attr=None,
            data_cfgs=self.data_cfgs,
            is_tra_val_te=self.is_tra_val_te,
            data_source=self.data_source,
        )

        self.x, self.y = self.kill_nan(scaler_hub.x, scaler_hub.y)

        self.target_scaler = scaler_hub.target_scaler

        self._create_lookup_table()

    def kill_nan(self, x, y):
        data_cfgs = self.data_cfgs
        y_rm_nan = data_cfgs["target_rm_nan"]
        x_rm_nan = data_cfgs["relevant_rm_nan"]
        if x_rm_nan:
            # As input, we cannot have NaN values
            for xx in x.values():
                for i in range(xx.shape[0]):
                    xx[i] = xx[i].interpolate_na(
                        dim="time_now", fill_value="extrapolate"
                    )
                warn_if_nan(xx)
        if y_rm_nan:
            _fill_gaps_da(y, fill_nan="interpolate")
            warn_if_nan(y)

        return x, y

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

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item: int):
        # here time is time_now in gpm_gfs_data
        basin, time = self.lookup_table[item]
        seq_length = self.rho
        output_seq_len = self.forecast_length

        xx = (
            self.x[basin]
            .sel(time_now=time)
            .sel(step=slice(0, seq_length + output_seq_len - 1))
            .values
        )

        x = xx.reshape(xx.shape[0], xx.shape[1], 1, xx.shape[2], xx.shape[3])
        y = (
            self.y.sel(basin=basin)
            .sel(
                time=slice(
                    time,
                    time + np.timedelta64(output_seq_len - 1, "h"),
                )
            )
            .values
        ).T
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()

    def basins(self):
        """Return the basins of the dataset"""
        return self.t_s_dict["sites_id"]

    def times(self):
        """Return the time range of the dataset"""
        return self.t_s_dict["t_final_range"]

    def _create_lookup_table(self):
        lookup = []
        basins = self.t_s_dict["sites_id"]
        dates = self.x[basins[0]]["time_now"].to_numpy()  # 取其中一个流域的时间作为标尺
        time_total_length = len(dates)
        time_num = len(self.t_s_dict["t_final_range"])
        time_single_length = int(time_total_length / time_num)
        is_tra_val_te = self.is_tra_val_te
        for basin in tqdm(
            basins,
            file=sys.stdout,
            disable=False,
            desc=f"Creating {is_tra_val_te} lookup table",
        ):
            for num in range(time_num):
                lookup.extend(
                    (basin, dates[f + num * time_single_length])
                    for f in range(0, time_total_length)
                    if f < time_single_length
                )
        self.lookup_table = dict(enumerate(lookup))
        self.num_samples = len(self.lookup_table)
