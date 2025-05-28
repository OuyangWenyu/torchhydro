"""
Author: Lili Yu
Date: 2025-05-10 18:00:00
LastEditTime: 2025-05-10 18:00:00
LastEditors: Lili Yu
Description: narx model dataset
"""

import sys
import re
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta

from hydrodatasource.utils.utils import streamflow_unit_conv
from torchhydro.configs.config import DATE_FORMATS
from torchhydro.datasets.data_sets import BaseDataset
from torchhydro.datasets.data_utils import (
    wrap_t_s_dict,
)
from torchhydro.models.basintree import BasinTree
from torchhydro.datasets.data_scalers import ScalerHub

def detect_date_format(date_str):
    for date_format in DATE_FORMATS:
        try:
            datetime.strptime(date_str, date_format)
            return date_format
        except ValueError:
            continue
    raise ValueError(f"Unknown date format: {date_str}")

class NarxDataset(BaseDataset):
    """
    a dataset for Narx model.
    """
    def __init__(self, data_cfgs: dict, is_tra_val_te: str):
        """
        Initialize the Narx dataset.
        narx model is more suitable for nested catchment flood prediction,
        while only fr have the nestedness information in camels, so choose fr to make dataset.
        Parameters
        ----------
        data_cfgs: data configures, setting via console.
        is_tra_val_te: three mode, train, validate and test.

        batch_size may need redressal.
        """
        super(NarxDataset, self).__init__(data_cfgs, is_tra_val_te)
        self.data_cfgs = data_cfgs
        self.b_nestedness = self.data_cfgs["b_nestedness"]
        # self.data_educed_model = None  # only nested_model now
        self.basin_list = None
        self._pre_load_data()
        self._generate_data_educed_model()
        self.data_cfgs["batch_size"] = len(self.basin_list)
        if is_tra_val_te in {"train", "valid", "test"}:
            self.is_tra_val_te = is_tra_val_te
        else:
            raise ValueError(
                "'is_tra_val_te' must be one of 'train', 'valid' or 'test' "
            )
        # load and preprocess data
        self._load_data()

    def __getitem__(self, item: int):
        """
        DataLoader load data from here and deliver into model.

        Parameters
        ----------
        item: the locate in lookup_table of basin need to read.

        Returns
        -------

        """
        if not self.train_mode:  # not train mode
            x = self.x[item, :, :]  # forcing data   [batch(basin), sequence, features]
            y = self.y[item, :, :]  # var_out, streamflow
            if self.c is None or self.c.shape[-1] == 0:  # attributions
                return torch.from_numpy(x).float(), torch.from_numpy(y).float()
            c = self.c[item, :]
            c = np.repeat(c, x.shape[0], axis=0).reshape(c.shape[0], -1).T
            xc = np.concatenate((x, c), axis=1)
            return torch.from_numpy(xc).float(), torch.from_numpy(y).float()
        basin, idx = self.lookup_table[item]
        warmup_length = self.warmup_length  # warmup_length = 0
        x = self.x[basin, idx - warmup_length: idx + self.rho + self.horizon, :]  # [batch(basin), time, features]  rho = 0, horizon = 30
        y = self.y[basin, idx: idx + self.rho + self.horizon, :]  # warmup period only used to warmup model, no need to compare the y calculated with the y observed. no need to subtract warmup_length here. see dpl4sac.
        if self.c is None or self.c.shape[-1] == 0:
            return torch.from_numpy(x).float(), torch.from_numpy(y).float()
        c = self.c[basin, :]
        c = np.repeat(c, x.shape[0], axis=0).reshape(c.shape[0], -1).T  # repeat the attributes for each time-step.
        xc = np.concatenate((x, c), axis=1)  # incorporate, as the input of model.

        return torch.from_numpy(xc).float(), torch.from_numpy(y).float()  # deliver into model prcp, pet, attributes and streamflow etc.  DataLoader

    def __len__(self):
        """
            expected to return the size of the dataset by many:class:`torch.utils.data.Sampler` implementations and
            the default options of :class:`torch.utils.data.DataLoader`.
        """
        return self.num_samples if self.train_mode else self.ngrid  # ngrid means nbasin

    @property
    def ngrid(self):
        """How many basins/grids in the dataset

        Returns
        -------
        int
            number of basins/grids
        """
        return len(self.basin_list)

    def _generate_data_educed_model(self):
        if not self.b_nestedness:
            raise ValueError("Error: naxrdataset needs nestedness information.")
        else:
            nestedness_info = self.data_source.read_nestedness_csv()
            basin_tree_ = BasinTree(nestedness_info, self.basins)
            self.data_educed_model = basin_tree_.get_basin_trees()
            self.basin_list = self.data_educed_model["basin_list"]

    def _pre_load_data(self):
        """preload data.
        some arguments setting.
        """
        self.train_mode = self.is_tra_val_te == "train"
        self.t_s_dict = wrap_t_s_dict(self.data_cfgs, self.is_tra_val_te)
        self.rho = self.data_cfgs["forecast_history"]  # the length of the history data for forecasting, the rho in LSTM.
        self.warmup_length = self.data_cfgs["warmup_length"]  # For physics-based models, we need warmup; default is 0 as DL models generally don't need it
        self.horizon = self.data_cfgs["forecast_length"]
        self.b_nestedness = self.data_cfgs["b_nestedness"]

    def _load_data(self):
        """load data to make dataset.

        """
        # self._pre_load_data()
        self._read_xyc()
        # normalization
        norm_x, norm_y, norm_c = self._normalize()
        self.x, self.y, self.c = self._kill_nan(norm_x, norm_y, norm_c)  # deal with nan value
        self._trans2nparr()
        self.x = np.concatenate((self.x, self.y), axis=2)  # notice: for narx model, the input features need to contain the history output(target) features data in train and test period, so concatenate y into x here.
        self._create_lookup_table()

    def _trans2nparr(self):
        """To make __getitem__ more efficient,
        we transform x, y, c to numpy array with shape (nsample, nt, nvar).    means [batch(basin), time, features]
        """
        self.x = self.x.transpose("basin", "time", "variable").to_numpy()  # tanspose, means T.  batch first here
        self.y = self.y.transpose("basin", "time", "variable").to_numpy()
        if self.c is not None and self.c.shape[-1] > 0:
            self.c = self.c.transpose("basin", "variable").to_numpy()
            self.c_origin = self.c_origin.transpose("basin", "variable").to_numpy()
        self.x_origin = self.x_origin.transpose("basin", "time", "variable").to_numpy()
        self.y_origin = self.y_origin.transpose("basin", "time", "variable").to_numpy()

    def _normalize(self):
        """normalize
            target_vars, streamflow.
            relevant_vars, forcing, e.g. prcp, pet, srad, etc.
            constant_vars, attributes, e.g. area, slope, elev, etc.
        """
        scaler_hub = ScalerHub(
            self.y_origin,  # streamflow
            self.x_origin,  # prcp, pet
            self.c_origin,  # attrs
            data_cfgs=self.data_cfgs,
            is_tra_val_te=self.is_tra_val_te,
            data_source=self.data_source,
        )
        self.target_scaler = scaler_hub.target_scaler
        return scaler_hub.x, scaler_hub.y, scaler_hub.c

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
            streamflow_dataset = data_output_ds[[self.streamflow_name]]
            converted_streamflow_dataset = streamflow_unit_conv(
                streamflow_dataset,
                self.data_source.read_area(self.basin_list),
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
            x, y, c data. forcing, target(streamflow), attributions.
        """

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
        # x
        data_forcing_ds_ = self.data_source.read_ts_xrdataset(
            self.basin_list,
            [start_date, end_date],
            self.data_cfgs["relevant_cols"],  # forcing data
        )
        # y
        data_output_ds_ = self.data_source.read_ts_xrdataset(
            self.basin_list,
            [start_date, end_date],
            self.data_cfgs["target_cols"],  # target data, streamflow.
        )
        # turn dict into list
        if isinstance(data_output_ds_, dict) or isinstance(data_forcing_ds_, dict):
            data_forcing_ds_ = data_forcing_ds_[list(data_forcing_ds_.keys())[0]]
            data_output_ds_ = data_output_ds_[list(data_output_ds_.keys())[0]]
        data_forcing_ds, data_output_ds_ = self._check_ts_xrds_unit(
            data_forcing_ds_, data_output_ds_
        )
        data_attr_ds = None
        self.x_origin, self.y_origin, self.c_origin = self._to_dataarray_with_unit(
            data_forcing_ds, data_output_ds_, data_attr_ds
        )

    def _create_lookup_table(self):
        """
        create lookup table.

        Returns
        -------

        """
        lookup = []
        # list to collect basins ids of basins without a single training sample
        basin_coordinates = len(self.basin_list)
        rho = self.rho  # forcast_history
        warmup_length = self.warmup_length
        horizon = self.horizon  # forcast_length
        max_time_length = self.nt  # length of longest time series in all basins
        for basin in tqdm(range(basin_coordinates), file=sys.stdout, disable=False):
            if self.is_tra_val_te != "train":  # validate and test period
                lookup.extend(
                    (basin, f)
                    for f in range(warmup_length, max_time_length - rho - horizon + 1)
                )
            else:  # train period
                # some dataloader load data with warmup period, so leave some periods for it
                # [warmup_len] -> time_start -> [rho] -> [horizon]
                nan_array = np.isnan(self.y[basin, :, :])  # nan value
                lookup.extend(
                    (basin, f)
                    for f in range(warmup_length, max_time_length - rho - horizon + 1)
                    if not np.all(nan_array[f + rho : f + rho + horizon])
                )
        self.lookup_table = dict(enumerate(lookup))
        self.num_samples = len(self.lookup_table)


class StlDataset(BaseDataset):
    """
    a dataset for stl model.
    decomposition -> normalization
    """
    def __init__(
            self,
            data_cfgs: dict,
            is_tra_val_te: str,
            data_decomposed,
    ):
        """
        Initialize the Stl dataset.

        Parameters
        ----------
        data_cfgs: data configures, setting via console.
        is_tra_val_te: three mode, train, validate and test.
        data_decomposed:
        """
        super(StlDataset, self).__init__(data_cfgs, is_tra_val_te)
        self.data_cfgs = data_cfgs
        self._pre_load_data()
        if is_tra_val_te in {"train", "valid", "test"}:
            self.is_tra_val_te = is_tra_val_te
        else:
            raise ValueError(
                "'is_tra_val_te' must be one of 'train', 'valid' or 'test' "
            )
        self.data_decomposed = data_decomposed
        self.y_decomposed = None
        self.y_trend = None
        self.y_season = None
        self.y_residuals = None
        # load and preprocess data
        self._load_data()

    def __len__(self):
        """
            expected to return the size of the dataset by many:class:`torch.utils.data.Sampler` implementations and
            the default options of :class:`torch.utils.data.DataLoader`.
        """
        return self.num_samples if self.train_mode else self.ngrid  # ngrid means nbasin

    def __getitem__(self, item: int):
        """
        class _SingleProcessDataLoaderIter(_BaseDataLoaderIter):
        """
        if not self.train_mode:
            x = self.d[item, :, :]
            y = self.d[item, :, :]
            if self.c is None or self.c.shape[-1] == 0:
                return torch.from_numpy(x).float(), torch.from_numpy(y).float()
            c = self.c[item, :]
            c = np.repeat(c, x.shape[0], axis=0).reshape(c.shape[0], -1).T
            xc = np.concatenate((x, c), axis=1)
            return torch.from_numpy(xc).float(),  torch.from_numpy(y).float()
        basin, idx = self.lookup_table[item]
        warmup_length = self.warmup_length
        x = self.d[basin, idx - warmup_length: idx + self.rho + self.horizon, :]
        y = self.d[basin, idx: idx + self.rho + self.horizon, :]
        if self.c is None or self.c.shape[-1] == 0:
            return torch.from_numpy(x).float(), torch.from_numpy(y).float()
        c = self.c[basin, :]
        c = np.repeat(c, x.shape[0], axis=0).reshape(c.shape[0], -1).T
        xc = np.concatenate((x, c), axis=1)
        return torch.from_numpy(xc).float(), torch.from_numpy(y).float()

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
        time_series = time_series[~((time_series.month == 2) & (time_series.day == 29))]
        return len(time_series)

    def _pre_load_data(self):
        """preload data.
        some arguments setting.
        """
        self.train_mode = self.is_tra_val_te == "train"
        self.t_s_dict = wrap_t_s_dict(self.data_cfgs, self.is_tra_val_te)
        self.rho = self.data_cfgs["forecast_history"]  # the length of the history data for forecasting, the rho in LSTM.
        self.warmup_length = self.data_cfgs["warmup_length"]  # For physics-based models, we need warmup; default is 0 as DL models generally don't need it
        self.horizon = self.data_cfgs["forecast_length"]

    def _load_data(self):
        """load data to make dataset.

        """
        # self._pre_load_data()
        self._read_xyc()
        # normalization
        norm_x, norm_y, norm_c, norm_d = self._normalize()  #
        self.x, self.y, self.c, self.d = self._kill_nan(norm_x, norm_y, norm_c, norm_d)  # deal with nan value
        self._trans2nparr()
        self._create_lookup_table()

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
        # x
        data_forcing_ds_ = self.data_decomposed[0]
        # y
        data_output_ds_ = self.data_decomposed[1]
        if isinstance(data_output_ds_, dict) or isinstance(data_forcing_ds_, dict):
            # this means the data source return a dict with key as time_unit
            # in this BaseDataset, we only support unified time range for all basins, so we chose the first key
            # TODO: maybe this could be refactored better
            data_forcing_ds_ = data_forcing_ds_[list(data_forcing_ds_.keys())[0]]
            data_output_ds_ = data_output_ds_[list(data_output_ds_.keys())[0]]
        data_forcing_ds_, data_output_ds_ = self._check_ts_xrds_unit(
            data_forcing_ds_, data_output_ds_
        )
        # c
        data_attr_ds = self.data_decomposed[2]
        # y_decomposed  output  streamflow
        y_decomposed_ds = self.data_decomposed[3]
        self.x_origin, self.y_origin, self.c_origin = self._to_dataarray_with_unit(
            data_forcing_ds_, data_output_ds_, data_attr_ds
        )
        self.y_decomposed, _, _ = self._to_dataarray_with_unit(y_decomposed_ds)

    def _normalize(self):
        scaler_hub = ScalerHub(
            self.y_origin,
            self.x_origin,
            self.c_origin,
            self.y_decomposed,  #
            data_cfgs=self.data_cfgs,
            is_tra_val_te=self.is_tra_val_te,
            data_source=self.data_source,
        )
        self.target_scaler = scaler_hub.target_scaler
        return scaler_hub.x, scaler_hub.y, scaler_hub.c, scaler_hub.d

    def _trans2nparr(self):
        """To make __getitem__ more efficient,
        we transform x, y, c to numpy array with shape (nsample, nt, nvar)
        """
        self.x = self.x.transpose("basin", "time", "variable").to_numpy()
        self.y = self.y.transpose("basin", "time", "variable").to_numpy()
        self.d = self.d.transpose("basin", "time", "variable").to_numpy()
        if self.c is not None and self.c.shape[-1] > 0:
            self.c = self.c.transpose("basin", "variable").to_numpy()
            self.c_origin = self.c_origin.transpose("basin", "variable").to_numpy()
        self.x_origin = self.x_origin.transpose("basin", "time", "variable").to_numpy()
        self.y_origin = self.y_origin.transpose("basin", "time", "variable").to_numpy()
