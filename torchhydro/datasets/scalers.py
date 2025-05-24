
import copy
import json
import os
import shutil
from typing import Optional
import xarray as xr
import numpy as np
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
        prcp_norm_cols=None,
        gamma_norm_cols=None,
        pbm_norm=False,
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
        if prcp_norm_cols is None:
            prcp_norm_cols = [
                "streamflow",
            ]
        if gamma_norm_cols is None:
            gamma_norm_cols = [
                "gpm_tp",
                "sta_tp",
                "total_precipitation_hourly",
                "temperature_2m",
                "dewpoint_temperature_2m",
                "surface_net_solar_radiation",
                "sm_surface",
                "sm_rootzone",
            ]
        self.data_target = target_vars
        self.data_forcing = relevant_vars
        self.data_attr = constant_vars
        self.data_cfgs = data_cfgs
        self.t_s_dict = wrap_t_s_dict(data_cfgs, is_tra_val_te)
        self.data_other = other_vars
        self.prcp_norm_cols = prcp_norm_cols
        self.gamma_norm_cols = gamma_norm_cols
        # both prcp_norm_cols and gamma_norm_cols use log(\sqrt(x)+.1) method to normalize
        self.log_norm_cols = gamma_norm_cols + prcp_norm_cols
        self.pbm_norm = pbm_norm
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
            decomposed_item = ["trend", "season", "residuals"]
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
        rm_nan
            if true, fill NaN value with 0
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
        rm_nan
            if true, fill NaN value with 0
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
        rm_nan
            if true, fill NaN value with 0
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
        rm_nan
            if true, fill NaN value with 0
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
        decomposed_item = ["trend", "season", "residuals"]
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


    def load_data(self):
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