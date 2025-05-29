
import copy
import json
import os
import shutil
from typing import Optional
import xarray as xr
import numpy as np
import pandas as pd
from shutil import SameFileError

from hydroutils.hydro_stat import (
    cal_stat_prcp_norm,
    cal_stat_gamma,
    cal_stat,
)

from torchhydro.datasets.data_utils import (
    _trans_norm,
    _prcp_norm,
    wrap_t_s_dict,
)


class DapengScaler(object):
    def __init__(
        self,
        target_vars: np.ndarray,
        relevant_vars: np.ndarray,
        constant_vars: np.ndarray,
        data_cfgs: dict,
        is_tra_val_te: str,
        other_vars: Optional[dict] = None,
        data_source: object = None,
    ):
        """
        The normalization and denormalization methods from Dapeng's 1st WRR paper.
        Some use StandardScaler, and some use special norm methods

        Parameters
        ----------
        target_vars
            output variables
        relevant_vars
            input dynamic variables
        constant_vars
            input static variables
        data_cfgs
            data parameter config in data source
        is_tra_val_te
            train/valid/test
        other_vars
            if more input are needed, list them in other_vars

        prcp_norm_cols
            data items which use _prcp_norm method to normalize
        gamma_norm_cols
            data items which use log(\sqrt(x)+.1) method to normalize
        pbm_norm
            if true, use pbm_norm method to normalize; the output of pbms is not normalized data, so its inverse is different.
        """
        self.data_target = target_vars
        self.data_forcing = relevant_vars
        self.data_attr = constant_vars
        self.data_cfgs = data_cfgs
        self.t_s_dict = wrap_t_s_dict(data_cfgs, is_tra_val_te)
        self.data_other = other_vars
        self.gamma_norm_cols = self.data_cfgs["scaler_params"]["gamma_norm_cols"]
        self.prcp_norm_cols = self.data_cfgs["scaler_params"]["prcp_norm_cols"]
        self.pbm_norm = self.data_cfgs["scaler_params"]["pbm_norm"]
        if self.prcp_norm_cols is None:
            self.prcp_norm_cols = [
                "streamflow",
            ]
        if self.gamma_norm_cols is None:
            self.gamma_norm_cols = [
                "gpm_tp",
                "sta_tp",
                "total_precipitation_hourly",
                "temperature_2m",
                "dewpoint_temperature_2m",
                "surface_net_solar_radiation",
                "sm_surface",
                "sm_rootzone",
            ]
        # both prcp_norm_cols and gamma_norm_cols use log(\sqrt(x)+.1) method to normalize
        self.log_norm_cols = self.gamma_norm_cols + self.prcp_norm_cols
        self.data_source = data_source

        # save stat_dict of training period in test_path for valid/test
        stat_file = os.path.join(data_cfgs["test_path"], "dapengscaler_stat.json")
        # for testing sometimes such as pub cases, we need stat_dict_file from trained dataset
        if is_tra_val_te == "train" and data_cfgs["stat_dict_file"] is None:
            self.stat_dict = self.cal_stat_all()
            with open(stat_file, "w") as fp:
                json.dump(self.stat_dict, fp)
        else:
            # for valid/test, we need to load stat_dict from train
            if data_cfgs["stat_dict_file"] is not None:
                # we used a assigned stat file, typically for PUB exps
                # shutil.copy(data_cfgs["stat_dict_file"], stat_file)
                try:
                    shutil.copy(data_cfgs["stat_dict_file"], stat_file)
                except SameFileError:
                    print(
                        f"The source file and the target file are the same: {data_cfgs['stat_dict_file']}, skipping the copy operation."
                    )
                except Exception as e:
                    print(f"Error: {e}")
            assert os.path.isfile(stat_file)
            with open(stat_file, "r") as fp:
                self.stat_dict = json.load(fp)

    @property
    def mean_prcp(self):
        """This property is used to be divided by streamflow to normalize streamflow,
        hence, its unit is same as streamflow

        Returns
        -------
        np.ndarray
            mean_prcp with the same unit as streamflow
        """
        final_unit = self.data_target.attrs["units"]["streamflow"]
        # mean_prcp = self.data_source.read_mean_prcp(
        #     self.t_s_dict["sites_id"], unit=final_unit
        # )
        mean_prcp = self.data_source.read_mean_prcp(
            self.data_target.basin.data.tolist(), unit=final_unit
        )
        return mean_prcp.to_array().transpose("basin", "variable").to_numpy()

    def inverse_transform(self, target_values):
        """
        Denormalization for output variables

        Parameters
        ----------
        target_values
            output variables

        Returns
        -------
        np.array
            denormalized predictions
        """
        stat_dict = self.stat_dict
        if self.data_cfgs["b_decompose"]:   # todo:
            target_cols = self.data_cfgs["decomposed_item"]
            attrs = self.data_other.attrs
        else:
            target_cols = self.data_cfgs["target_cols"]
            attrs = self.data_target.attrs
        if self.pbm_norm:
            # for pbm's output, its unit is mm/day, so we don't need to recover its unit
            pred = target_values
        else:
            pred = _trans_norm(
                target_values,
                target_cols,
                stat_dict,
                log_norm_cols=self.log_norm_cols,
                to_norm=False,
            )
            for i in range(len(target_cols)):
                var = target_cols[i]
                if var in self.prcp_norm_cols:
                    pred.loc[dict(variable=var)] = _prcp_norm(
                        pred.sel(variable=var).to_numpy(),
                        self.mean_prcp,
                        to_norm=False,
                    )
                else:
                    pred.loc[dict(variable=var)] = pred.sel(variable=var)
        # add attrs for units
        pred.attrs.update(attrs)
        return pred.to_dataset(dim="variable")

    def cal_stat_all(self):
        """
        Calculate statistics of outputs(streamflow etc), inputs(forcing and attributes) and other data(decomposed from
        streamflow, trend, season and residuals now)(optional)

        Returns
        -------
        dict
            a dict with statistic values
        """
        # streamflow, et, ssm, etc
        target_cols = self.data_cfgs["target_cols"]
        stat_dict = {}
        for i in range(len(target_cols)):
            var = target_cols[i]
            if var in self.prcp_norm_cols:
                stat_dict[var] = cal_stat_prcp_norm(
                    self.data_target.sel(variable=var).to_numpy(),
                    self.mean_prcp,
                )
            elif var in self.gamma_norm_cols:
                stat_dict[var] = cal_stat_gamma(
                    self.data_target.sel(variable=var).to_numpy()
                )
            else:
                stat_dict[var] = cal_stat(self.data_target.sel(variable=var).to_numpy())

        # forcing
        forcing_lst = self.data_cfgs["relevant_cols"]
        x = self.data_forcing
        for k in range(len(forcing_lst)):
            var = forcing_lst[k]
            if var in self.gamma_norm_cols:
                stat_dict[var] = cal_stat_gamma(x.sel(variable=var).to_numpy())
            else:
                stat_dict[var] = cal_stat(x.sel(variable=var).to_numpy())

        # const attribute
        attr_data = self.data_attr
        attr_lst = self.data_cfgs["constant_cols"]
        for k in range(len(attr_lst)):
            var = attr_lst[k]
            stat_dict[var] = cal_stat(attr_data.sel(variable=var).to_numpy())

        # other data, only decomposed data by STL now.  trend, season and residuals decomposed from streamflow.
        if self.data_other is not None:
            decomposed_item = self.data_cfgs["decomposed_item"]
            decomposed_data = self.data_other
            for i in range(len(decomposed_item)):
                var = decomposed_item[i]
                stat_dict[var] = cal_stat(decomposed_data.sel(variable=var).to_numpy())

        return stat_dict

    def get_data_obs(self, to_norm: bool = True) -> np.array:
        """
        Get observation values

        Parameters
        ----------
        to_norm
            if true, perform normalization

        Returns
        -------
        np.array
            the output value for modeling
        """
        stat_dict = self.stat_dict
        data = self.data_target
        out = xr.full_like(data, np.nan)
        # if we don't set a copy() here, the attrs of data will be changed, which is not our wish
        out.attrs = copy.deepcopy(data.attrs)
        target_cols = self.data_cfgs["target_cols"]
        if "units" not in out.attrs:
            Warning("The attrs of output data does not contain units")
            out.attrs["units"] = {}
        for i in range(len(target_cols)):
            var = target_cols[i]
            if var in self.prcp_norm_cols:
                out.loc[dict(variable=var)] = _prcp_norm(
                    data.sel(variable=var).to_numpy(),
                    self.mean_prcp,
                    to_norm=True,
                )
            else:
                out.loc[dict(variable=var)] = data.sel(variable=var).to_numpy()
            out.attrs["units"][var] = "dimensionless"
        out = _trans_norm(
            out,
            target_cols,
            stat_dict,
            log_norm_cols=self.log_norm_cols,
            to_norm=to_norm,
        )
        return out

    def get_data_ts(self, to_norm=True) -> np.array:
        """
        Get dynamic input data

        Parameters
        ----------
        to_norm
            if true, perform normalization

        Returns
        -------
        np.array
            the dynamic inputs for modeling
        """
        stat_dict = self.stat_dict
        var_lst = self.data_cfgs["relevant_cols"]
        data = self.data_forcing
        data = _trans_norm(
            data, var_lst, stat_dict, log_norm_cols=self.log_norm_cols, to_norm=to_norm
        )
        return data

    def get_data_const(self, to_norm=True) -> np.array:
        """
        Attr data and normalization

        Parameters
        ----------
        to_norm
            if true, perform normalization

        Returns
        -------
        np.array
            the static inputs for modeling
        """
        stat_dict = self.stat_dict
        var_lst = self.data_cfgs["constant_cols"]
        data = self.data_attr
        data = _trans_norm(data, var_lst, stat_dict, to_norm=to_norm)
        return data

    def get_data_other(self, to_norm: bool = True) -> np.array:
        """
        Get observation values

        Parameters
        ----------
        to_norm
            if true, perform normalization

        Returns
        -------
        np.array
            the output value for modeling
        """
        stat_dict = self.stat_dict
        data = self.data_other
        out = xr.full_like(data, np.nan)
        # if we don't set a copy() here, the attrs of data will be changed, which is not our wish
        out.attrs = copy.deepcopy(data.attrs)
        decomposed_item = self.data_cfgs["decomposed_item"]
        if "units" not in out.attrs:
            Warning("The attrs of output data does not contain units")
            out.attrs["units"] = {}
        for i in range(len(decomposed_item)):
            var = decomposed_item[i]
            out.loc[dict(variable=var)] = data.sel(variable=var).to_numpy()
            out.attrs["units"][var] = "dimensionless"
        out = _trans_norm(
            out,
            decomposed_item,
            stat_dict,
            log_norm_cols=self.log_norm_cols,
            to_norm=to_norm,
        )
        return out


    def transform(self):
        """
        Read data and perform normalization for DL models

        Returns
        -------
        tuple
            x: 3-d  gages_num*time_num*var_num
            y: 3-d  gages_num*time_num*1
            c: 2-d  gages_num*var_num
            d: 3-d  gages_num*time_num*3
        """
        x = self.get_data_ts()
        y = self.get_data_obs()
        c = self.get_data_const()
        if self.data_other is not None:
            d = self.get_data_other()
        else:
            d = None
        return x, y, c, d


b_save_result = True

class SlidingWindowScaler(object):
    """sliding window scaler"""
    def __init__(
        self,
        # sw_stride: int,
        target_vars: np.ndarray,
        relevant_vars: np.ndarray,
        constant_vars: np.ndarray,
        data_cfgs: dict,
        is_tra_val_te: str,
        other_vars: Optional[dict] = None,
        data_source: object = None,
    ):
        """
        The normalization and denormalization methods of sliding window scaler.

        Parameters
        ----------
        target_vars
            output variables
        relevant_vars
            input dynamic variables
        constant_vars
            input static variables
        data_cfgs
            data parameter config in data source
        is_tra_val_te
            train/valid/test
        other_vars
            if more input are needed, list them in other_vars
        """
        self.data_target = target_vars
        self.data_forcing = relevant_vars
        self.data_attr = constant_vars
        self.data_cfgs = data_cfgs
        self.t_s_dict = wrap_t_s_dict(data_cfgs, is_tra_val_te)
        self.is_tra_val_te = is_tra_val_te
        self.data_other = other_vars
        self.data_source = data_source
        self.pbm_norm = self.data_cfgs["scaler_params"]["pbm_norm"]  # physical based model
        self.sw_width = self.data_cfgs["scaler_params"]["sw_width"]
        self.series_length = self.data_target.shape[2]
        self.series_nbasin = self.data_target.shape[1]
        self.n_windows = int(self.series_length / self.sw_width)
        self.n_residual = self.series_length % self.sw_width
        self.statistic_dict = self.cal_stat_all()


    def cal_statistics(
        self,
        x: np.ndarray,
    ):
        """
        calculate two statistics indices of a series: min and max for all windows.
        Parameters
         x

        Returns
        -------

        """
        min = [0]*self.n_windows
        max = [0]*self.n_windows
        for i in range(self.n_windows):
            start_i = i*self.sw_width
            end_i = (i + 1) * self.sw_width -1
            x_i = x[:, start_i:end_i]
            min[i] = np.min(x_i, axis=1)
            max[i] = np.max(x_i, axis=1)
        if self.n_residual > 0:
            x_i = x[:, -self.n_residual:]
            min_ = np.min(x_i, axis=1)
            max_ = np.max(x_i, axis=1)
            min.append(min_)
            max.append(max_)
        return [min, max]

    def cal_statistics_attr(
            self,
            x,
    ):
        """calculate the statistics value of attributions."""
        min = [0]*self.series_nbasin
        max = [0]*self.series_nbasin
        for i in range(self.series_nbasin):
            x_i = x[i]
            x_i = x_i[~np.isnan(x_i)]
            min[i] = np.min(x_i)
            max[i] = np.max(x_i)
        return [min, max]

    def cal_stat_all(self):
        """calculate the statistics values of series
        calculate statistics of outputs(streamflow etc), inputs(forcing and attributes) and other data(decomposed from
        streamflow, trend, season and residuals now)(optional)

        Returns
        -------
        dict
            a dict with statistic values
        """
        # streamflow, et, ssm, etc
        target_cols = self.data_cfgs["target_cols"]
        stat_dict = {}
        for i in range(len(target_cols)):
            var = target_cols[i]
            stat_dict[var] = self.cal_statistics(self.data_target.sel(variable=var).to_numpy())

        # forcing
        if self.data_forcing is not None:
            forcing_lst = self.data_cfgs["relevant_cols"]
            x = self.data_forcing
            for k in range(len(forcing_lst)):
                var = forcing_lst[k]
                stat_dict[var] = self.cal_statistics(x.sel(variable=var).to_numpy())

        # other data, only decomposed data by STL now.  trend, season and residuals decomposed from streamflow.
        if self.data_other is not None:
            decomposed_item = self.data_cfgs["decomposed_item"]
            decomposed_data = self.data_other
            for i in range(len(decomposed_item)):
                var = decomposed_item[i]
                stat_dict[var] = self.cal_statistics(decomposed_data.sel(variable=var).to_numpy())

        # const attribute
        if self.data_attr is not None:
            attr_data = self.data_attr
            attr_data = attr_data.to_numpy()
            attr_data = np.transpose(attr_data)
            var = "attributions"
            stat_dict[var] = self.cal_statistics_attr(attr_data)

        return stat_dict

    def norm_singlewindow(
        self,
        x,
        min,
        max,
        b_norm: bool=True,
    ):
        """
        normalize or denormalize for a single window
         data format
        Parameters
        ----------
        x: data within window
        min: min value of data
        max: max value of data
        b_norm: normalize or denormalize.

        Returns
        -------
        normalized_x
        denormalized_x
        """
        _, n_t = np.shape(x)
        min = np.expand_dims(min,1).repeat(n_t,axis=1)
        max = np.expand_dims(max,1).repeat(n_t,axis=1)
        range = max - min
        if b_norm:
            normalized_x = (x - min) / range
            return normalized_x
        else:
            denormalized_x = x * range + min
            return denormalized_x

    def norm_wholeseries(
        self,
        x,
        statistics,
        b_norm: bool=True,
    ):
        """
        normalize or denormalize a whole series.

        Parameters
        ----------
        x: data need to normalization.
        statistics: the statistics values of the series.

        Returns
        -------
        X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Transformed array.
        """
        min = statistics[0]
        max = statistics[1]
        out = np.zeros((self.series_nbasin, self.series_length))
        try:
            for i in range(self.n_windows):
                start_i = i * self.sw_width
                end_i = (i + 1) * self.sw_width -1
                x_i = x[:, start_i:end_i]
                min_i = min[i]
                max_i = max[i]
                normalized_x_i = self.norm_singlewindow(x_i, min_i, max_i, b_norm)
                out[:, start_i:end_i] = normalized_x_i
        except IndexError:
            raise IndexError("too many indices for array: array is 1-dimensional, but 2 were indexed")
        if self.n_residual > 0:
            x_i = x[:, -self.n_residual:]
            min_i = min[-1]
            max_i = max[-1]
            normalized_x_i = self.norm_singlewindow(x_i, min_i, max_i, b_norm)
            out[:, -self.n_residual:] = normalized_x_i

        return out

    def _trans_norm(
        self,
        x: xr.DataArray,
        var_lst: list,
        stat_dict: dict,
        to_norm: bool = True,
        **kwargs,
    ) -> np.array:
        """
        norm a DataArray.

        Parameters
        ----------
        X : data need to normalization.

        Returns
        -------
        X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Transformed array.
        """
        if x is None:
            return None
        if type(var_lst) is str:
            var_lst = [var_lst]
        out = xr.full_like(x, np.nan)
        for item in var_lst:
            stat = stat_dict[item]
            out.loc[dict(variable=item)] = self.norm_wholeseries(x.sel(variable=item).values, stat, to_norm)
        if to_norm:
            # after normalization, all units are dimensionless
            out.attrs = {}
        # after denormalization, recover units
        else:
            if "recover_units" in kwargs.keys() and kwargs["recover_units"] is not None:
                recover_units = kwargs["recover_units"]
                for item in var_lst:
                    out.attrs["units"][item] = recover_units[item]
        return out

    def _trans_norm_attr(
        self,
        x: xr.DataArray,
        var_lst: list,
        stat_dict: dict,
        to_norm: bool = True,
        **kwargs,
    ) -> np.array:
        """norm attr"""
        if x is None:
            return None
        out = xr.full_like(x, np.nan)
        var = "attributions"
        stat = stat_dict[var]
        x_ = np.transpose(x.values)
        x_norm = self.norm_singlewindow(x_, stat[0], stat[1], to_norm)
        x_norm = np.transpose(x_norm)
        for i in range(len(var_lst)):
            item = var_lst[i]
            out.loc[dict(variable=item)] = x_norm[i]
        if to_norm:
            # after normalization, all units are dimensionless
            out.attrs = {}
        # after denormalization, recover units
        else:
            if "recover_units" in kwargs.keys() and kwargs["recover_units"] is not None:
                recover_units = kwargs["recover_units"]
                for item in var_lst:
                    out.attrs["units"][item] = recover_units[item]
        return out


    def get_data_ts(self, to_norm=True) -> np.array:
        """
        normalize forcing data.

        Parameters
        ----------
        to_norm
            if true, perform normalization

        Returns
        -------
        np.array
            the dynamic inputs for modeling
        """
        stat_dict = self.statistic_dict
        var_lst = self.data_cfgs["relevant_cols"]
        data = self.data_forcing
        data = self._trans_norm(
            data, var_lst, stat_dict, to_norm=to_norm
        )
        if b_save_result:
            pd_series = data.to_pandas()
            pd_series.index.name = "time"
            # file_name = r"D:\torchhydro\tests\results\test_camels\slidingwindowscaler_camelsus"
            file_name = r"/mnt/d/torchhydro/tests/results/test_camels/slidingwindowscaler_camelsus/forcing_norm.csv"
            pd_series.to_csv(file_name, sep=" ")

        return data

    def get_data_obs(self, to_norm: bool = True) -> np.array:
        """
        normalize target data.

        Parameters
        ----------
        to_norm
            if true, perform normalization

        Returns
        -------
        np.array
            the output value for modeling
        """
        stat_dict = self.statistic_dict
        data = self.data_target
        out = xr.full_like(data, np.nan)
        # if we don't set a copy() here, the attrs of data will be changed, which is not our wish
        out.attrs = copy.deepcopy(data.attrs)
        target_cols = self.data_cfgs["target_cols"]
        if "units" not in out.attrs:
            Warning("The attrs of output data does not contain units")
            out.attrs["units"] = {}
        for i in range(len(target_cols)):
            var = target_cols[i]
            out.loc[dict(variable=var)] = data.sel(variable=var).to_numpy()
            out.attrs["units"][var] = "dimensionless"
        out = self._trans_norm(
            out,
            target_cols,
            stat_dict,
            to_norm=to_norm,
        )

        if b_save_result:
            pd_series = data.to_pandas()
            pd_series.index.name = "time"
            # file_name = r"D:\torchhydro\tests\results\test_camels\slidingwindowscaler_camelsus"
            file_name = r"/mnt/d/torchhydro/tests/results/test_camels/slidingwindowscaler_camelsus/target_norm.csv"
            pd_series.to_csv(file_name, sep=" ")

        return out

    def get_data_const(self, to_norm=True) -> np.array:
        """
       normalize attribution data.

        Parameters
        ----------
        to_norm
            if true, perform normalization

        Returns
        -------
        np.array
            the static inputs for modeling
        """
        stat_dict = self.statistic_dict
        var_lst = self.data_cfgs["constant_cols"]
        data = self.data_attr
        data = self._trans_norm_attr(data, var_lst, stat_dict, to_norm=to_norm)

        if b_save_result:
            pd_series = data.to_pandas()
            pd_series.index.name = "time"
            # file_name = r"D:\torchhydro\tests\results\test_camels\slidingwindowscaler_camelsus"
            file_name = r"/mnt/d/torchhydro/tests/results/test_camels/slidingwindowscaler_camelsus/attribution_norm.csv"
            pd_series.to_csv(file_name, sep=" ")

        return data

    def get_data_other(self, to_norm: bool = True) -> np.array:
        """
        normalize other data.

        Parameters
        ----------
        to_norm
            if true, perform normalization

        Returns
        -------
        np.array
            the output value for modeling
        """
        stat_dict = self.statistic_dict
        data = self.data_other
        out = xr.full_like(data, np.nan)
        # if we don't set a copy() here, the attrs of data will be changed, which is not our wish
        out.attrs = copy.deepcopy(data.attrs)
        decomposed_item = self.data_cfgs["decomposed_item"]
        if "units" not in out.attrs:
            Warning("The attrs of output data does not contain units")
            out.attrs["units"] = {}
        for i in range(len(decomposed_item)):
            var = decomposed_item[i]
            out.loc[dict(variable=var)] = data.sel(variable=var).to_numpy()
            out.attrs["units"][var] = "dimensionless"
        out = self._trans_norm(
            out,
            decomposed_item,
            stat_dict,
            to_norm=to_norm,
        )

        if b_save_result:
            pd_series = data.to_pandas()
            pd_series.index.name = "time"
            # file_name = r"D:\torchhydro\tests\results\test_camels\slidingwindowscaler_camelsus"
            file_name = r"/mnt/d/torchhydro/tests/results/test_camels/slidingwindowscaler_camelsus/other_norm.csv"
            pd_series.to_csv(file_name, sep=" ")

        return out

    def transform(self):
        """
        normalization

        Returns
        -------
        tuple
            x: 3-d  gages_num*time_num*var_num
            y: 3-d  gages_num*time_num*1
            c: 2-d  gages_num*var_num
            d: 3-d  gages_num*time_num*3
        """
        if self.data_forcing is not None:
            x = self.get_data_ts()
        else:
            x = None
        y = self.get_data_obs()
        if self.data_other is not None:
            d = self.get_data_other()
        else:
            d = None
        if self.data_attr is not None:
            c = self.get_data_const()
        else:
            c = None

        return x, y, c, d

    def inverse_transform(self, target_values):
        """
        Denormalization for output variables

        Parameters
        ----------
        target_values
            output variables

        Returns
        -------
        np.array
            denormalized predictions
        """
        stat_dict = self.statistic_dict
        if self.data_cfgs["b_decompose"]:   # todo:
            target_cols = self.data_cfgs["decomposed_item"]
            attrs = self.data_other.attrs
        else:
            target_cols = self.data_cfgs["target_cols"]
            attrs = self.data_target.attrs
        if self.pbm_norm:
            # for pbm's output, its unit is mm/day, so we don't need to recover its unit
            pred = target_values
        else:
            pred = self._trans_norm(
                target_values,
                target_cols,
                stat_dict,
                to_norm=False,
            )
        for i in range(len(target_cols)):
            var = target_cols[i]
            pred.loc[dict(variable=var)] = pred.sel(variable=var)
        # add attrs for units
        pred.attrs.update(attrs)
        return pred.to_dataset(dim="variable")
