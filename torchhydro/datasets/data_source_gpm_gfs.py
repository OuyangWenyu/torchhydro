"""
Author: Xinzhuo Wu
Date: 2023-09-30 1:20:18
LastEditTime: 2023-12-29 11:05:57
LastEditors: Xinzhuo Wu
Description: data source
FilePath: \torchhydro\torchhydro\datasets\data_source_gpm_gfs.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
import os
from datetime import datetime, timedelta
from typing import Union

import numpy as np
import xarray as xr
import yaml
from hydrodataset import HydroDataset
import pathlib as pl

GPM_GFS_NO_DATASET_ERROR_LOG = (
    "We cannot read this dataset now. Please check if you choose correctly:\n"
)

with open(os.path.join(pl.Path.home(), 'privacy_config.yml'), "r") as f:
    privacy_cfg = yaml.safe_load(f)


class GPM_GFS(HydroDataset):
    def __init__(
        self,
        data_path=None,
        download=False,
    ):
        super().__init__(data_path)
        if download:
            raise NotImplementedError(
                "We don't provide methods for downloading data at present\n"
            )

    def read_waterlevel_xrdataset(
        self,
        gage_id_lst,
        t_range: list,
        var_list: list,
        forecast_length: int,
        water_level_source_path: Union[str | os.PathLike | xr.Dataset],
        user: str,
        **kwargs,
    ):
        if var_list is None or len(var_list) == 0:
            return None
        if user in privacy_cfg['trainer']:
            waterlevel = xr.open_dataset(water_level_source_path)
        elif user in privacy_cfg['tester']:
            waterlevel = water_level_source_path
        else:
             waterlevel = xr.Dataset()
        all_vars = waterlevel.data_vars
        if any(var not in waterlevel.variables for var in var_list):
            raise ValueError(f"var_lst must all be in {all_vars}")
        subset_list = []
        for period in t_range:
            start_date = period["start"]
            end_date = datetime.strptime(period["end"], "%Y-%m-%d")
            new_end_date = end_date + timedelta(hours=forecast_length)
            end_date_str = new_end_date.strftime("%Y-%m-%d")
            subset = waterlevel.sel(basin=gage_id_lst).sel(
                time=slice(start_date, end_date_str)
            )
            subset_list.append(subset)
        return xr.concat(subset_list, dim="time")

    def read_streamflow_xrdataset(
        self,
        gage_id_lst,
        t_range: list,
        var_list: list,
        forecast_length: int,
        user: str,
        streamflow_source_path: Union[str | os.PathLike | xr.Dataset],
        **kwargs,
    ):
        if var_list is None or len(var_list) == 0:
            return None
        if user in privacy_cfg['trainer']:
            streamflow = xr.open_dataset(streamflow_source_path)
        elif user in privacy_cfg['tester']:
            streamflow = streamflow_source_path
        else:
            streamflow = xr.Dataset()
        all_vars = streamflow.data_vars
        if any(var not in streamflow.variables for var in var_list):
            raise ValueError(f"var_lst must all be in {all_vars}")
        subset_list = []
        for period in t_range:
            start_date = period["start"]
            end_date = datetime.strptime(period["end"], "%Y-%m-%d")
            new_end_date = end_date + timedelta(hours=forecast_length)
            end_date_str = new_end_date.strftime("%Y-%m-%d")
            subset = streamflow.sel(basin=gage_id_lst).sel(
                time=slice(start_date, end_date_str)
            )
            subset_list.append(subset)
        return xr.concat(subset_list, dim="time")

    def read_rainfall_xrdataset(
        self,
        gage_id_lst: list = None,
        t_range: list = None,
        var_lst: list = None,
        rainfall_source_path: str = None,
        **kwargs,
    ):
        if var_lst is None:
            return None
        rainfall_dict = {}
        # 如何将流域号与一系列对象组合起来
        for basin in gage_id_lst:
            rainfall = xr.open_dataset(
                os.path.join(rainfall_source_path, f"{basin}.nc")
            )
            subset_list = []
            for period in t_range:
                start_date = period["start"]
                end_date = period["end"]
                subset = rainfall.sel(time_now=slice(start_date, end_date))
                subset_list.append(subset)
            merged_dataset_tp = xr.concat(subset_list, dim="time_now")
            rainfall_dict[basin] = merged_dataset_tp.to_array(dim="variable")
        return rainfall_dict

    def read_gfs_xrdataset(
        self,
        gage_id_lst: list,
        t_range: list,
        var_lst: list,
        gfs_source_path: str,
        user: str,
        **kwargs,
    ):
        if var_lst is None:
            return None
        gfs_dict = {}
        # 如何将流域号与一系列对象组合起来
        for basin in gage_id_lst:
            gfs = xr.open_dataset(os.path.join(gfs_source_path, f"{basin}.nc"))
            subset_list = []
            for period in t_range:
                start_date = period["start"]
                end_date = period["end"]
                subset = gfs[var_lst].sel(time_now=slice(start_date, end_date))
                subset_list.append(subset)
            merged_dataset_gfs = xr.concat(subset_list, dim="time")
            gfs_dict[basin] = merged_dataset_gfs.to_array(dim="variable")
        return gfs_dict

    def read_attr_xrdataset(
        self, gage_id_lst, var_lst, attributes_path, user, **kwargs
    ):
        if var_lst is None or len(var_lst) == 0:
            return None
        if user in privacy_cfg['trainer']:
            attr = xr.open_dataset(attributes_path)
        elif user in privacy_cfg['tester']:
            attr = attributes_path
        else:
            attr = xr.Dataset()
        if "all" in var_lst:
            return attr.sel(basin=gage_id_lst)
        return attr[var_lst].sel(basin=gage_id_lst)

    def read_mean_prcp(self, gage_id_lst, mean_prep_path, user) -> np.array:
        if user in privacy_cfg['trainer']:
            mean_prep = xr.open_dataset(mean_prep_path)
        elif user in privacy_cfg['tester']:
            mean_prep = mean_prep_path
        else:
            mean_prep = xr.Dataset()
        return mean_prep["p_mean"].sel(basin=gage_id_lst)
