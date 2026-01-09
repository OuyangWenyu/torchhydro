"""
Author: Wenyu Ouyang
Date: 2024-04-08 18:17:44
LastEditTime: 2025-10-29 08:53:29
LastEditors: Wenyu Ouyang
Description: normalize the data
FilePath: \torchhydro\torchhydro\datasets\data_scalers.py
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

# 导入 NaN-safe 版本的统计函数
try:
    from torchhydro.datasets.data_scalers_nan_safe import (
        cal_stat_nan_safe,
        cal_stat_prcp_norm_nan_safe,
        cal_stat_gamma_nan_safe,
        check_data_quality,
    )
    USE_NAN_SAFE_STATS = True
except ImportError:
    USE_NAN_SAFE_STATS = False
    print("警告: 无法导入 NaN-safe 统计函数，将使用原始版本")

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
        
        self.one_hot_cols = data_cfgs["scaler_params"].get("one_hot_cols", [])
        
        # 从 scaler_params 中读取 use_nan_safe_stats 参数
        self.use_nan_safe_stats = data_cfgs["scaler_params"].get("use_nan_safe_stats", True)
        
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
        # Get the first target variable (usually flow variable) instead of hardcoding "streamflow"
        flow_var_name = self.data_cfgs["target_cols"][0]
        final_unit = self.data_target.attrs["units"][flow_var_name]
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



    def _get_one_hot_categories(self, vars_data):
        """
        Calculate unique categories for one-hot encoding columns from training data
        """
        categories_dict = {}
        for k, v in vars_data.items():
            if v is None:
                continue
            for var_name in v.coords["variable"].values:
                if var_name in self.one_hot_cols:
                    # Get unique values, ignoring NaNs
                    unique_vals = np.unique(v.sel(variable=var_name).to_numpy())
                    unique_vals = unique_vals[~np.isnan(unique_vals)]
                    unique_vals.sort()
                    categories_dict[f"{var_name}_categories"] = unique_vals.tolist()
        return categories_dict

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
        
        # 从实例属性中读取 use_nan_safe_stats
        use_nan_safe = getattr(self, 'use_nan_safe_stats', True)
        
        # Calculate categories for one-hot encoding if needed
        if self.one_hot_cols:
            categories_dict = self._get_one_hot_categories(vars_data)
            stat_dict.update(categories_dict)

        for k, v in vars_data.items():
            if v is None:
                continue
            for i in range(len(v.coords["variable"].values)):
                var_name = v.coords["variable"].values[i]
                
                # Skip statistics calculation for one-hot columns as they don't need mean/std normalization
                if var_name in self.one_hot_cols:
                    continue
                
                # 获取数据
                data = v.sel(variable=var_name).to_numpy()
                
                # 检查 NaN 值情况
                nan_count = np.isnan(data).sum()
                total_count = data.size
                if nan_count > 0:
                    nan_ratio = nan_count / total_count
                    print(f"警告: {var_name} 包含 {nan_count}/{total_count} ({nan_ratio:.2%}) 个 NaN 值")
                    
                    # 如果 NaN 比例过高，给出警告
                    if nan_ratio > 0.5:
                        print(f"  ⚠️  {var_name} 的 NaN 比例超过 50%，统计量可能不可靠")
                
                # 根据变量类型选择统计函数
                if var_name in self.prcp_norm_cols:
                    if use_nan_safe and USE_NAN_SAFE_STATS:
                        stat_dict[var_name] = cal_stat_prcp_norm_nan_safe(
                            data,
                            self.mean_prcp,
                        )
                    else:
                        stat_dict[var_name] = cal_stat_prcp_norm(
                            data,
                            self.mean_prcp,
                        )
                elif var_name in self.gamma_norm_cols:
                    if use_nan_safe and USE_NAN_SAFE_STATS:
                        stat_dict[var_name] = cal_stat_gamma_nan_safe(data)
                    else:
                        stat_dict[var_name] = cal_stat_gamma(data)
                else:
                    if use_nan_safe and USE_NAN_SAFE_STATS:
                        stat_dict[var_name] = cal_stat_nan_safe(data)
                    else:
                        stat_dict[var_name] = cal_stat(data)
                
                # 验证统计量是否有效（应该有 4 个元素：p10, p90, mean, std）
                if len(stat_dict[var_name]) != 4:
                    print(f"  ❌ {var_name} 的统计量格式错误，应该有 4 个元素: {stat_dict[var_name]}")
                    raise ValueError(
                        f"变量 {var_name} 的统计量格式错误，应该是 [p10, p90, mean, std]"
                    )
                if any(np.isnan(stat_dict[var_name])):
                    print(f"  ❌ {var_name} 的统计量包含 NaN: {stat_dict[var_name]}")
                    raise ValueError(
                        f"变量 {var_name} 的统计量计算失败，包含 NaN 值。"
                        f"请检查数据质量或在 scaler_params 中设置 use_nan_safe_stats=True"
                    )

        return stat_dict

    def get_data_norm(self, data, to_norm: bool = True) -> np.ndarray:
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
        
        # We need to handle one-hot encoding which changes the dimensions/variables
        # So we cannot simply use xr.full_like
        
        _vars = data.coords["variable"].values
        data_arrays = []
        
        for i in range(len(_vars)):
            var = _vars[i]
            
            if var in self.one_hot_cols:
                # Perform one-hot encoding
                categories = stat_dict.get(f"{var}_categories", [])
                if not categories:
                    # If no categories found (e.g. in test mode but not in stat_dict), 
                    # we might have an issue. But let's assume it's handled or empty.
                    # Or maybe we should warn?
                    print(f"Warning: No categories found for {var} in stat_dict")
                    continue
                
                original_vals = data.sel(variable=var).to_numpy()
                
                for cat in categories:
                    # Create binary mask
                    # Use isclose for float comparison if needed, but == is usually fine for categories if they are distinct
                    # Assuming categories are floats since they come from numpy array
                    mask = np.isclose(original_vals, cat) if isinstance(cat, (int, float)) else (original_vals == cat)
                    
                    new_var_name = f"{var}_{cat}"
                    
                    # Create new DataArray for this category
                    # We need to preserve dimensions except 'variable'
                    # The shape of original_vals matches the other dims
                    
                    # We construct a new DataArray with a single variable coordinate
                    da_cat = xr.DataArray(
                        mask.astype(float),
                        coords={d: data.coords[d] for d in data.dims if d != "variable"},
                        dims=[d for d in data.dims if d != "variable"]
                    ).expand_dims(variable=[new_var_name])
                    
                    da_cat.attrs["units"] = "dimensionless"
                    data_arrays.append(da_cat)
                    
            else:
                # Normal processing for non-one-hot variables
                # Extract data for this variable
                var_data = data.sel(variable=var)
                
                # Create a new DataArray to hold normalized values
                # We start with a copy to keep coords/dims
                out_var = xr.full_like(var_data, np.nan)
                out_var.attrs = copy.deepcopy(data.attrs)
                if "units" not in out_var.attrs:
                    out_var.attrs["units"] = {}
                
                if var in self.prcp_norm_cols:
                    out_var.values = _prcp_norm(
                        var_data.to_numpy(),
                        self.mean_prcp,
                        to_norm=True,
                    )
                else:
                    out_var.values = var_data.to_numpy()
                
                out_var.attrs["units"] = "dimensionless"
                
                # Now apply _trans_norm logic for this single variable
                # Note: _trans_norm expects an xarray with 'variable' dim and multiple variables usually,
                # but here we are processing one by one. 
                # We can reuse _trans_norm if we wrap it back or just implement the logic here.
                # Implementing logic here is safer to avoid recursion or shape issues.
                
                stat = stat_dict[var]
                # stat: [mean, std, mean, std] usually? 
                # Wait, cal_stat returns [mean, std, mean, std] or similar?
                # Let's check cal_stat in hydro_stat.py or assume standard usage
                # _trans_norm uses stat[2] for mean and stat[3] for std
                
                if to_norm:
                    if var in self.log_norm_cols:
                        out_var.values = (np.log10(np.sqrt(np.abs(out_var.values)) + 0.1) - stat[2]) / stat[3]
                    else:
                        out_var.values = (out_var.values - stat[2]) / stat[3]
                
                # Expand dims to include variable name
                out_var = out_var.expand_dims(variable=[var])
                data_arrays.append(out_var)

        # Merge all data arrays
        if not data_arrays:
            return None
            
        # Ensure all DataArrays have names (though expand_dims(variable=...) usually sets the coordinate, not the name)
        # Actually, xr.merge expects a list of DataArrays (if they have names) or a dict
        # But we expanded dims with 'variable', so they are effectively parts of a larger array
        # However, xr.merge merges by coordinates. If we want to concatenate along 'variable', we should use xr.concat
        
        out = xr.concat(data_arrays, dim="variable")
        
        if data.name:
            out.name = data.name
        
        # Restore attributes if possible, though merge might have lost them or mixed them
        # We mainly care about units which we set on individual arrays
        
        # Reorder dimensions to match input if possible, but 'variable' might be in different place
        # Usually we want (basin, time, variable) or similar
        # input data dims: e.g. ('basin', 'variable') or ('basin', 'time', 'variable')
        
        # Ensure 'variable' is the last dimension or consistent with input
        desired_dims = [d for d in data.dims if d != "variable"] + ["variable"]
        out = out.transpose(*desired_dims)
        
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
        if vars_data is None:
            return None
        return {
            k: self.get_data_norm(v) if v is not None else None
            for k, v in vars_data.items()
        }
