"""narx model dataset"""

import sys
import re
import numpy as np
import torch
from tqdm import tqdm
from hydrodatasource.utils.utils import streamflow_unit_conv

from torchhydro.datasets.data_sets import BaseDataset
from torchhydro.datasets.data_utils import (
    wrap_t_s_dict,
)
from torchhydro.models.basintree import BasinTree
from torchhydro.datasets.data_scalers import ScalerHub

class NarxDataset(BaseDataset):
    """
    a dataset for Narx model.
    """
    def __init__(self, data_cfgs: dict, is_tra_val_te: str):
        """
        Initialize the Narx dataset.  for fr in camels.
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
        self.data_educed_model = None  # only nested_model now
        self.basin_list = None
        self._pre_load_data()
        self._generate_data_educed_model()
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
        fetch and deliver into model
        /home/yulili/.conda/envs/torchhydro/lib/python3.13/site-packages/torch/utils/data/_utils/fetch.py  class _MapDatasetFetcher(_BaseDatasetFetcher) call this method.
        /home/yulili/.conda/envs/torchhydro/lib/python3.13/site-packages/torch/utils/data/dataloader.py  _next_data(self)
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
        basin, idx = self.lookup_table[item]  # [1972, 3569, 2907, 894, 442, 3341, 2420, 3443, 4291, 1303, 2368, 2954, 4125, 2773, 2832, 3836, 1948, 120]
        warmup_length = self.warmup_length  # warmup_length = 0
        x = self.x[basin, idx - warmup_length: idx + self.rho + self.horizon, :]  # [batch(basin), time, features]  rho = 0, horizon = 30
        y = self.y[basin, idx: idx + self.rho + self.horizon, :]  # idx - warmup_length ?  why do not subtraction warmup_length here?
        if self.c is None or self.c.shape[-1] == 0:
            return torch.from_numpy(x).float(), torch.from_numpy(y).float()
        c = self.c[basin, :]
        c = np.repeat(c, x.shape[0], axis=0).reshape(c.shape[0], -1).T  # repeat the attributes for each tim-step.
        xc = np.concatenate((x, c), axis=1)  # incorporate, as the input of model.

        return torch.from_numpy(xc).float(), torch.from_numpy(y).float()  # deliver into model prcp, pet, attributes and streamflow etc.  DataLoader

    def __len__(self):
        """
            expected to return the size of the dataset by many:class:`torch.utils.data.Sampler` implementations and 
            the default options of :class:`torch.utils.data.DataLoader`.
        """
        return self.num_samples if self.train_mode else self.ngrid  # ngrid means nbasin

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
        self._create_lookup_table()  # todo：

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
        self.x_origin, self.y_origin, self.c_origin = self._to_dataarray_with_unit(   # origin data
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
