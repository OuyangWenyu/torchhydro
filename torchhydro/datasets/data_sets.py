"""
Author: Wenyu Ouyang
Date: 2022-02-13 21:20:18
LastEditTime: 2024-02-12 19:12:52
LastEditors: Wenyu Ouyang
Description: A pytorch dataset class; references to https://github.com/neuralhydrology/neuralhydrology
FilePath: \torchhydro\torchhydro\datasets\data_sets.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""

import os
import io
import logging
import sys
import boto3
import pint_xarray  # noqa: F401
import requests
import torch
import xarray as xr
import numpy as np

from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Optional
from botocore.config import Config
from torch.utils.data import Dataset
from tqdm import tqdm

from torchhydro.datasets.data_scalers import (
    ScalerHub,
    Muti_Basin_GPM_GFS_SCALER,
)

from torchhydro.datasets.data_utils import (
    warn_if_nan,
    wrap_t_s_dict,
    unify_streamflow_unit,
)
from hydrodata.reader.data_source import HydroGrids, HydroBasins
from hydrodataset import Camels

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
        self.data_source = Camels(self.data_cfgs["data_path"])
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
        self.t_s_dict = wrap_t_s_dict(self.data_cfgs, self.is_tra_val_te)
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
    def __init__(self, data_cfgs: dict, is_tra_val_te: str):
        super(GPM_GFS_Dataset, self).__init__()
        self.data_cfgs = data_cfgs
        self.grids_data = HydroGrids(self.data_cfgs["data_path"])
        self.basin_data = HydroBasins(self.data_cfgs["data_path"])
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
        self.t_s_dict = wrap_t_s_dict(self.data_cfgs, self.is_tra_val_te)

        data_target_ds = self.prepare_Y()
        if data_target_ds is not None:
            y_origin = self._trans2da_and_setunits(data_target_ds)
        else:
            y_origin = None

        data_rainfall = self.prepare_PPT()

        if self.data_cfgs["relevant_cols"][1] != ["None"]:
            data_gfs = self.prepare_GFS()
        else:
            data_gfs = None

        if self.data_cfgs["relevant_cols"][2] != ["None"]:
            data_soil = self.prepare_SP()
        else:
            data_soil = None

        if self.data_cfgs["constant_cols"]:
            data_attr_ds = self.basin_data.read_BA_xrdataset(
                self.t_s_dict["sites_id"],
                self.data_cfgs["constant_cols"],
                self.data_cfgs["user"],
                self.data_cfgs["attributes_path"],
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
            basin_data=self.basin_data,
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
                    time + np.timedelta64(1, "h"),
                    time + np.timedelta64(output_seq_len, "h"),
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

    def prepare_PPT(self):
        gage_id_lst = self.t_s_dict["sites_id"]
        t_range = self.t_s_dict["t_final_range"]
        var_lst = self.data_cfgs["relevant_cols"][0]
        path = self.data_cfgs["rainfall_source_path"]
        user = self.data_cfgs["user"]

        if var_lst is None:
            return None

        PPT_dict = {}
        for basin in gage_id_lst:
            data = self.grids_data.read_PPT_xrdataset(gage_id_lst, path, user, basin)
            subset_list = []
            for period in t_range:
                start_date = period["start"]
                end_date = period["end"]
                subset = data.sel(time_now=slice(start_date, end_date))
                subset_list.append(subset)
            merged_dataset = xr.concat(subset_list, dim="time_now")
            PPT_dict[basin] = merged_dataset.to_array(dim="variable")
        return PPT_dict

    def prepare_GFS(self):
        gage_id_lst = self.t_s_dict["sites_id"]
        t_range = self.t_s_dict["t_final_range"]
        var_lst = self.data_cfgs["relevant_cols"][1]
        path = self.data_cfgs["gfs_source_path"]
        user = self.data_cfgs["user"]

        if var_lst is None:
            return None

        GFS_dict = {}
        for basin in gage_id_lst:
            data = self.grids_data.read_GFS_xrdataset(gage_id_lst, path, user, basin)
            subset_list = []
            for period in t_range:
                start_date = datetime.strptime(period["start"], "%Y-%m-%d")
                new_start_date = start_date - timedelta(hours=self.rho)
                start_date_str = new_start_date.strftime("%Y-%m-%d")

                end_date = datetime.strptime(period["end"], "%Y-%m-%d")
                new_end_date = end_date + timedelta(hours=self.forecast_length)
                end_date_str = new_end_date.strftime("%Y-%m-%d")

                subset = data[var_lst].sel(time=slice(start_date_str, end_date_str))
                subset_list.append(subset)
            merged_dataset = xr.concat(subset_list, dim="time")
            GFS_dict[basin] = merged_dataset.to_array(dim="variable")

        return GFS_dict

    def prepare_SP(self):
        gage_id_lst = self.t_s_dict["sites_id"]
        t_range = self.t_s_dict["t_final_range"]
        var_lst = self.data_cfgs["relevant_cols"][2]
        path = self.data_cfgs["soil_source_path"]
        user = self.data_cfgs["user"]

        if var_lst is None:
            return None

        SP_dict = {}
        for basin in gage_id_lst:
            data = self.grids_data.read_GFS_xrdataset(gage_id_lst, path, user, basin)
            subset_list = []
            for period in t_range:
                start_date = datetime.strptime(period["start"], "%Y-%m-%d")
                new_start_date = start_date - timedelta(hours=self.rho)
                start_date_str = new_start_date.strftime("%Y-%m-%d")

                end_date = datetime.strptime(period["end"], "%Y-%m-%d")
                new_end_date = end_date + timedelta(hours=self.forecast_length)
                end_date_str = new_end_date.strftime("%Y-%m-%d")

                subset = data[var_lst].sel(time=slice(start_date_str, end_date_str))
                subset_list.append(subset)
            merged_dataset = xr.concat(subset_list, dim="time")
            SP_dict[basin] = merged_dataset.to_array(dim="variable")

        return SP_dict

    def prepare_Y(self):
        gage_id_lst = self.t_s_dict["sites_id"]
        t_range = self.t_s_dict["t_final_range"]
        var_lst = self.data_cfgs["target_cols"]
        path = self.data_cfgs["streamflow_source_path"]
        user = self.data_cfgs["user"]

        if var_lst is None or not var_lst:
            return None

        data = self.basin_data.read_Y_xrdataset(gage_id_lst, path, user)

        all_vars = data.data_vars
        if any(var not in data.variables for var in var_lst):
            raise ValueError(f"var_lst must all be in {all_vars}")
        subset_list = []
        for period in t_range:
            start_date = period["start"]
            end_date = datetime.strptime(period["end"], "%Y-%m-%d")
            new_end_date = end_date + timedelta(hours=self.forecast_length)
            end_date_str = new_end_date.strftime("%Y-%m-%d")
            subset = data.sel(time=slice(start_date, end_date_str))
            subset_list.append(subset)
        return xr.concat(subset_list, dim="time")


class GPM_GFS_Mean_Dataset(Dataset):
    def __init__(self, data_cfgs: dict, is_tra_val_te: str):
        super(GPM_GFS_Mean_Dataset, self).__init__()
        self.data_cfgs = data_cfgs
        self.basin_data = HydroBasins(self.data_cfgs["data_path"])
        if is_tra_val_te in {"train", "valid", "test"}:
            self.is_tra_val_te = is_tra_val_te
        else:
            raise ValueError(
                "'is_tra_val_te' must be one of 'train', 'valid' or 'test' "
            )

        self._load_data()

    def _load_data(self):
        train_mode = self.is_tra_val_te == "train"
        self.train_mode = train_mode
        self.rho = self.data_cfgs["forecast_history"]
        self.forecast_length = self.data_cfgs["forecast_length"]
        self.warmup_length = self.data_cfgs["warmup_length"]
        self.t_s_dict = wrap_t_s_dict(self.data_cfgs, self.is_tra_val_te)

        data_target_ds = self.prepare_Y()
        if data_target_ds is not None:
            y_origin = self._trans2da_and_setunits(data_target_ds)
        else:
            y_origin = None

        data_rainfall_ds = self.prepare_MPPT()

        if data_rainfall_ds is not None:
            x_origin = self._trans2da_and_setunits(data_rainfall_ds)
        else:
            x_origin = None

        if self.data_cfgs["constant_cols"]:
            data_attr_ds = self.basin_data.read_BA_xrdataset(
                self.t_s_dict["sites_id"],
                self.data_cfgs["constant_cols"],
                self.data_cfgs["attributes_path"],
            )
            c_orgin = self._trans2da_and_setunits(data_attr_ds)
        else:
            c_orgin = None

        scaler_hub = ScalerHub(
            y_origin,
            x_origin,
            constant_vars=c_orgin,
            data_cfgs=self.data_cfgs,
            is_tra_val_te=self.is_tra_val_te,
            data_source=self.basin_data,
        )
        self.target_scaler = scaler_hub.target_scaler
        self.x, self.y, self.c = self.kill_nan(scaler_hub.x, scaler_hub.y, scaler_hub.c)
        self._create_lookup_table()

    def _create_lookup_table(self):
        lookup = []
        basins = self.t_s_dict["sites_id"]
        rho = self.rho
        forecast_length = self.forecast_length
        warmup_length = self.warmup_length
        dates = self.y["time"].to_numpy()
        time_num = len(self.t_s_dict["t_final_range"])
        time_total_length = len(dates)
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
                    for f in range(
                        warmup_length + rho, time_single_length - forecast_length * 2
                    )
                    if f < time_single_length
                )
        self.lookup_table = dict(enumerate(lookup))
        self.num_samples = len(self.lookup_table)

    def __getitem__(self, item: int):
        basin, time = self.lookup_table[item]
        seq_length = self.rho
        output_seq_len = self.forecast_length
        warmup_length = self.warmup_length
        gpm_tp = (
            self.x.sel(
                variable="gpm_tp",
                basin=basin,
                time=slice(
                    time - np.timedelta64(warmup_length + seq_length - 1, "h"),
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
                    time + np.timedelta64(output_seq_len, "h"),
                ),
            )
            .to_numpy()
            .T
        ).reshape(-1, 1)
        x = np.concatenate((gpm_tp, gfs_tp), axis=0)
        if self.c is not None and self.c.shape[-1] > 0:
            c = self.c.sel(basin=basin).values
            c = np.tile(c, (warmup_length + seq_length + output_seq_len, 1))
            x = np.concatenate((x, c), axis=1)
        y = (
            self.y.sel(
                basin=basin,
                time=slice(
                    time + np.timedelta64(1, "h"),
                    time + np.timedelta64(output_seq_len, "h"),
                ),
            )
            .to_numpy()
            .T
        )
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()

    @property
    def basins(self):
        """Return the basins of the dataset"""
        return self.t_s_dict["sites_id"]

    @property
    def times(self):
        """Return the time range of the dataset"""
        return self.t_s_dict["t_final_range"]

    def __len__(self):
        return self.num_samples

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

    def kill_nan(self, x, y, c):
        data_cfgs = self.data_cfgs
        y_rm_nan = data_cfgs["target_rm_nan"]
        x_rm_nan = data_cfgs["relevant_rm_nan"]
        c_rm_nan = data_cfgs["constant_rm_nan"]
        if x_rm_nan:
            # As input, we cannot have NaN values
            x = x.compute()
            _fill_gaps_da(x, fill_nan="interpolate")
            warn_if_nan(x)
        if y_rm_nan:
            y = y.compute()
            _fill_gaps_da(y, fill_nan="interpolate")
            warn_if_nan(y)
        if c_rm_nan:
            c = c.compute()
            _fill_gaps_da(c, fill_nan="mean")
            warn_if_nan(c)
        return x, y, c

    def prepare_Y(self):
        gage_id_lst = self.t_s_dict["sites_id"]
        t_range = self.t_s_dict["t_final_range"]
        var_lst = self.data_cfgs["target_cols"]
        path = self.data_cfgs["streamflow_source_path"]

        if var_lst is None or not var_lst:
            return None

        data = self.basin_data.merge_nc_minio_datasets(path, gage_id_lst, var_lst)

        all_vars = data.data_vars
        if any(var not in data.variables for var in var_lst):
            raise ValueError(f"var_lst must all be in {all_vars}")
        subset_list = []
        for period in t_range:
            start_date = period["start"]
            end_date = datetime.strptime(period["end"], "%Y-%m-%d")
            new_end_date = end_date + timedelta(hours=self.forecast_length)
            end_date_str = new_end_date.strftime("%Y-%m-%d")
            subset = data.sel(time=slice(start_date, end_date_str))
            subset_list.append(subset)
        return xr.concat(subset_list, dim="time")

    def prepare_MPPT(self):
        gage_id_lst = self.t_s_dict["sites_id"]
        t_range = self.t_s_dict["t_final_range"]
        var_lst = self.data_cfgs["relevant_cols"]
        path = self.data_cfgs["rainfall_source_path"]

        if var_lst is None:
            return None

        data = self.basin_data.merge_nc_minio_datasets(path, gage_id_lst, var_lst)

        var_subset_list = []
        for period in t_range:
            start_date = period["start"]
            end_date = period["end"]
            subset = data.sel(time=slice(start_date, end_date))
            var_subset_list.append(subset)

        return xr.concat(var_subset_list, dim="time")

    def prepare_MGFS(self):
        gage_id_lst = self.t_s_dict["sites_id"]
        t_range = self.t_s_dict["t_final_range"]
        var_lst = self.data_cfgs["relevant_cols"]
        path = self.data_cfgs["gfs_source_path"]

        if var_lst is None:
            return None

        data = self.basin_data.merge_nc_minio_datasets(path, gage_id_lst, var_lst)
        subset_list = []
        for period in t_range:
            start_date = period["start"]
            new_start_date = start_date - timedelta(hours=self.rho)
            start_date_str = new_start_date.strftime("%Y-%m-%d")

            end_date = period["end"]
            new_end_date = end_date + timedelta(hours=self.forecast_length)
            end_date_str = new_end_date.strftime("%Y-%m-%d")

            subset = data.sel(time=slice(start_date_str, end_date_str))
            subset_list.append(subset)

        return xr.concat(subset_list, dim="time")

    def prepare_MSP(self):
        gage_id_lst = self.t_s_dict["sites_id"]
        t_range = self.t_s_dict["t_final_range"]
        var_lst = self.data_cfgs["relevant_cols"][2]
        path = self.data_cfgs["soil_source_path"]

        if var_lst is None:
            return None

        data = self.basin_data.merge_nc_minio_datasets(path, gage_id_lst, var_lst)
        subset_list = []
        for period in t_range:
            start_date = period["start"]
            new_start_date = start_date - timedelta(hours=self.rho)
            start_date_str = new_start_date.strftime("%Y-%m-%d")

            end_date = period["end"]
            new_end_date = end_date + timedelta(hours=self.forecast_length)
            end_date_str = new_end_date.strftime("%Y-%m-%d")

            subset = data.sel(time=slice(start_date_str, end_date_str))
            subset_list.append(subset)

        return xr.concat(subset_list, dim="time")
