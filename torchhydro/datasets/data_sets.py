"""
Author: Wenyu Ouyang
Date: 2022-02-13 21:20:18
LastEditTime: 2023-12-29 11:05:57
LastEditors: Xinzhuo Wu
Description: A pytorch dataset class; references to https://github.com/neuralhydrology/neuralhydrology
FilePath: \torchhydro\torchhydro\datasets\data_sets.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
import time
import logging
import sys
from typing import Optional
import numpy as np
import pint_xarray  # noqa: F401
import torch
import xarray as xr
from hydrodataset import HydroDataset
from torchhydro.datasets.data_source_gpm_gfs import GPM_GFS
from torch.utils.data import Dataset
from tqdm import tqdm
from torchhydro.datasets.data_scalers import ScalerHub, Muti_Basin_GPM_GFS_SCALER
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
