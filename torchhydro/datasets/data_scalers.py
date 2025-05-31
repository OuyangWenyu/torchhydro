"""
Author: Wenyu Ouyang
Date: 2024-04-08 18:17:44
LastEditTime: 2025-04-19 14:05:54
LastEditors: Wenyu Ouyang
Description: normalize the data
FilePath: /torchhydro/torchhydro/datasets/data_scalers.py
Copyright (c) 2024-2024 Wenyu Ouyang. All rights reserved.
"""

import copy
import json
import os
import pickle as pkl
import shutil
from typing import Optional
import pint_xarray  # noqa: F401
import xarray as xr
import numpy as np
from shutil import SameFileError
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
    MaxAbsScaler,
)

from hydroutils.hydro_stat import (
    cal_stat_prcp_norm,
    cal_stat_gamma,
    cal_stat,
    cal_4_stat_inds,
)

from torchhydro.datasets.data_utils import (
    _trans_norm,
    _prcp_norm,
    wrap_t_s_dict,
    unify_streamflow_unit,
)

SCALER_DICT = {
    "StandardScaler": StandardScaler,
    "RobustScaler": RobustScaler,
    "MinMaxScaler": MinMaxScaler,
    "MaxAbsScaler": MaxAbsScaler,
}


class ScalerHub(object):
    """
    A class for Scaler
    """

    def __init__(
        self,
        vars_data,
        data_cfgs=None,
        is_tra_val_te=None,
        data_source=None,
        **kwargs,
    ):
        """
        Perform normalization

        Parameters
        ----------
        vars_data
            data for all variables used.
            the dim must be (basin, time, lead_step, var) for 4-d array;
            the dim must be (basin, time, var) for 3-d array;
            the dim must be (basin, time) for 2-d array;
        data_cfgs
            configs for reading data
        is_tra_val_te
            train, valid or test
        data_source
            data source to get original data info
        kwargs
            other optional parameters for ScalerHub
        """
        self.data_cfgs = data_cfgs
        scaler_type = data_cfgs["scaler"]
        pbm_norm = data_cfgs["scaler_params"]["pbm_norm"]
        if scaler_type == "DapengScaler":
            gamma_norm_cols = data_cfgs["scaler_params"]["gamma_norm_cols"]
            prcp_norm_cols = data_cfgs["scaler_params"]["prcp_norm_cols"]
            scaler = DapengScaler(
                vars_data,
                data_cfgs,
                is_tra_val_te,
                prcp_norm_cols=prcp_norm_cols,
                gamma_norm_cols=gamma_norm_cols,
                pbm_norm=pbm_norm,
                data_source=data_source,
            )
        elif scaler_type in SCALER_DICT.keys():
            scaler = SklearnScaler(
                vars_data,
                data_cfgs,
                is_tra_val_te,
                pbm_norm=pbm_norm,
            )
        else:
            raise NotImplementedError(
                "We don't provide this Scaler now!!! Please choose another one: DapengScaler or key in SCALER_DICT"
            )
        self.norm_data = scaler.load_norm_data(vars_data)
        # we will use target_scaler during denormalization
        self.target_scaler = scaler
        print("Finish Normalization\n")


class SklearnScaler(object):
    def __init__(
        self,
        vars_data,
        data_cfgs,
        is_tra_val_te,
        pbm_norm=False,
    ):
        """_summary_

        Parameters
        ----------
        vars_data : dict
            vars data map
        data_cfgs : _type_
            _description_
        is_tra_val_te : bool
            _description_
        pbm_norm : bool, optional
            _description_, by default False
        """
        # we will use data_target and target_scaler for denormalization
        self.data_target = vars_data["target_cols"]
        self.target_scaler = None
        self.data_cfgs = data_cfgs
        self.is_tra_val_te = is_tra_val_te
        self.pbm_norm = pbm_norm

    def load_norm_data(self, vars_data):
        # TODO: not fully tested for differentiable models
        norm_dict = {}
        scaler_type = self.data_cfgs["scaler"]
        for k, v in vars_data.items():
            scaler = SCALER_DICT[scaler_type]()
            if v.ndim == 3:
                # for forcings and outputs
                num_instances, num_time_steps, num_features = v.shape
                v_np = v.to_numpy().reshape(-1, num_features)
                scaler, data_norm = self._sklearn_scale(
                    self.data_cfgs, self.is_tra_val_te, scaler, k, v_np
                )
                data_norm = data_norm.reshape(
                    num_instances, num_time_steps, num_features
                )
                norm_xrarray = xr.DataArray(
                    data_norm,
                    coords={
                        "basin": v.coords["basin"],
                        "time": v.coords["time"],
                        "variable": v.coords["variable"],
                    },
                    dims=["basin", "time", "variable"],
                )
            elif v.ndim == 2:
                num_instances, num_features = v.shape
                v_np = v.to_numpy().reshape(-1, num_features)
                scaler, data_norm = self._sklearn_scale(
                    self.data_cfgs, self.is_tra_val_te, scaler, k, v_np
                )
                # don't need to reshape data_norm again as it is 2-d
                norm_xrarray = xr.DataArray(
                    data_norm,
                    coords={
                        "basin": v.coords["basin"],
                        "variable": v.coords["variable"],
                    },
                    dims=["basin", "variable"],
                )
            elif v.ndim == 4:
                # for forecast data
                num_instances, num_time_steps, num_lead_steps, num_features = v.shape
                v_np = v.to_numpy().reshape(-1, num_features)
                scaler, data_norm = self._sklearn_scale(
                    self.data_cfgs, self.is_tra_val_te, scaler, k, v_np
                )
                data_norm = data_norm.reshape(
                    num_instances, num_time_steps, num_lead_steps, num_features
                )
                norm_xrarray = xr.DataArray(
                    data_norm,
                    coords={
                        "basin": v.coords["basin"],
                        "time": v.coords["time"],
                        "lead_step": v.coords["lead_step"],
                        "variable": v.coords["variable"],
                    },
                    dims=["basin", "time", "lead_step", "variable"],
                )
            else:
                raise NotImplementedError(
                    "Please check your data, the dim of data must be 2, 3 or 4"
                )

            norm_dict[k] = norm_xrarray
            if k == "target_cols":
                # we need target cols scaler for denormalization
                self.target_scaler = scaler
        return norm_dict

    def _sklearn_scale(self, data_cfgs, is_tra_val_te, scaler, norm_key, data):
        save_file = os.path.join(data_cfgs["case_dir"], f"{norm_key}_scaler.pkl")
        if is_tra_val_te == "train" and data_cfgs["stat_dict_file"] is None:
            data_norm = scaler.fit_transform(data)
            # Save scaler in case_dir for valid/test
            with open(save_file, "wb") as outfile:
                pkl.dump(scaler, outfile)
        else:
            if data_cfgs["stat_dict_file"] is not None:
                # NOTE: you need to set data_cfgs["stat_dict_file"] as a str with ";" as its seperator
                # the sequence of the stat_dict_file must be same as the sequence of norm_keys
                # for example: "stat_dict_file": "target_stat_dict_file;relevant_stat_dict_file;constant_stat_dict_file"
                shutil.copy(data_cfgs["stat_dict_file"][norm_key], save_file)
            if not os.path.isfile(save_file):
                raise FileNotFoundError("Please genereate xx_scaler.pkl file")
            with open(save_file, "rb") as infile:
                scaler = pkl.load(infile)
                data_norm = scaler.transform(data)
        return scaler, data_norm

    def inverse_transform(self, target_values):
        """
        Denormalization for output variables

        Parameters
        ----------
        target_values
            output variables (xr.DataArray or np.ndarray)

        Returns
        -------
        xr.Dataset
            denormalized predictions or observations
        """
        coords = self.data_target.coords
        attrs = self.data_target.attrs
        # input must be xr.DataArray
        if not isinstance(target_values, xr.DataArray):
            # the shape of target_values must be (basin, time, variable)
            target_values = xr.DataArray(
                target_values,
                coords={
                    "basin": coords["basin"],
                    "time": coords["time"],
                    "variable": coords["variable"],
                },
                dims=["basin", "time", "variable"],
            )
        # transform to numpy array for sklearn inverse_transform
        shape = target_values.shape
        arr = target_values.to_numpy().reshape(-1, shape[-1])
        # sklearn inverse_transform
        arr_inv = self.target_scaler.inverse_transform(arr)
        # reshape to original shape
        arr_inv = arr_inv.reshape(shape)
        result = xr.DataArray(
            arr_inv,
            coords=target_values.coords,
            dims=target_values.dims,
            attrs=attrs,
        )
        # add attrs for units
        result.attrs.update(self.data_target.attrs)
        return result.to_dataset(dim="variable")


class DapengScaler(object):
    def __init__(
        self,
        vars_data,
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
        vars_data: dict
            data for all variables used
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
        self.data_target = vars_data["target_cols"]
        self.data_cfgs = data_cfgs
        self.t_s_dict = wrap_t_s_dict(data_cfgs, is_tra_val_te)
        self.data_other = other_vars
        self.prcp_norm_cols = prcp_norm_cols
        self.gamma_norm_cols = gamma_norm_cols
        # both prcp_norm_cols and gamma_norm_cols use log(\sqrt(x)+.1) method to normalize
        self.log_norm_cols = gamma_norm_cols + prcp_norm_cols
        self.pbm_norm = pbm_norm
        self.data_source = data_source
        # save stat_dict of training period in case_dir for valid/test
        stat_file = os.path.join(data_cfgs["case_dir"], "dapengscaler_stat.json")
        # for testing sometimes such as pub cases, we need stat_dict_file from trained dataset
        if is_tra_val_te == "train" and data_cfgs["stat_dict_file"] is None:
            self.stat_dict = self.cal_stat_all(vars_data)
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
        mean_prcp = self.data_source.read_mean_prcp(
            self.t_s_dict["sites_id"], unit=final_unit
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
        target_vars = self.data_cfgs["target_cols"]
        if self.pbm_norm:
            # for (differentiable models) pbm's output, its unit is mm/day, so we don't need to recover its unit
            pred = target_values
        else:
            pred = _trans_norm(
                target_values,
                target_vars,
                stat_dict,
                log_norm_cols=self.log_norm_cols,
                to_norm=False,
            )
            for i in range(len(self.data_cfgs["target_cols"])):
                var = self.data_cfgs["target_cols"][i]
                if var in self.prcp_norm_cols:
                    pred.loc[dict(variable=var)] = _prcp_norm(
                        pred.sel(variable=var).to_numpy(),
                        self.mean_prcp,
                        to_norm=False,
                    )
                else:
                    pred.loc[dict(variable=var)] = pred.sel(variable=var)
        # add attrs for units
        pred.attrs.update(self.data_target.attrs)
        return pred.to_dataset(dim="variable")

    def cal_stat_all(self, vars_data):
        """
        Calculate statistics of outputs(streamflow etc), and inputs(forcing and attributes)
        Parameters
        ----------
        vars_data: dict
            data for all variables used

        Returns
        -------
        dict
            a dict with statistic values
        """
        stat_dict = {}
        for k, v in vars_data.items():
            for i in range(len(v.coords["variable"].values)):
                var_name = v.coords["variable"].values[i]
                if var_name in self.prcp_norm_cols:
                    stat_dict[var_name] = cal_stat_prcp_norm(
                        v.sel(variable=var_name).to_numpy(),
                        self.mean_prcp,
                    )
                elif var_name in self.gamma_norm_cols:
                    stat_dict[var_name] = cal_stat_gamma(
                        v.sel(variable=var_name).to_numpy()
                    )
                else:
                    stat_dict[var_name] = cal_stat(v.sel(variable=var_name).to_numpy())

        return stat_dict

    def get_data_norm(self, data, to_norm: bool = True) -> np.array:
        """
        Get normalized values

        Parameters
        ----------
        data
            origin data
        to_norm
            if true, perform normalization
            if false, perform denormalization

        Returns
        -------
        np.array
            the output value for modeling
        """
        stat_dict = self.stat_dict
        out = xr.full_like(data, np.nan)
        # if we don't set a copy() here, the attrs of data will be changed, which is not our wish
        out.attrs = copy.deepcopy(data.attrs)
        _vars = data.coords["variable"].values
        if "units" not in out.attrs:
            Warning("The attrs of output data does not contain units")
            out.attrs["units"] = {}
        for i in range(len(_vars)):
            var = _vars[i]
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
            _vars,
            stat_dict,
            log_norm_cols=self.log_norm_cols,
            to_norm=to_norm,
        )
        return out

    def load_norm_data(self, vars_data):
        """
        Read data and perform normalization for DL models
        Parameters
        ----------
        vars_data: dict
            data for all variables used

        Returns
        -------
        tuple
            x: 3-d  gages_num*time_num*var_num
            y: 3-d  gages_num*time_num*1
            c: 2-d  gages_num*var_num
        """
        return {k: self.get_data_norm(v) for k, v in vars_data.items()}
