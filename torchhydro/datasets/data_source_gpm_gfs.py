"""
Author: Xinzhuo Wu
Date: 2023-09-30 1:20:18
LastEditTime: 2024-01-11 14:21:00
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

with open(os.path.join(pl.Path.home(), "privacy_config.yml"), "r") as f:
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
        water_level_source_path: Union[str, os.PathLike, xr.Dataset],
        user: str,
        **kwargs,
    ):
        """
        Reads water level data as an xarray Dataset based on specified parameters.

        This function opens a dataset of water levels from the given path or dataset object,
        and selects a subset of the data based on the specified gauge IDs, time range, and variables.
        It supports different access levels for 'trainer' and 'tester' users.

        Parameters:
        - gage_id_lst: List of gauge IDs for which data needs to be read.
        - t_range: List of dictionaries with 'start' and 'end' dates specifying the time range for each period.
        - var_list: List of variables to be included in the resulting dataset.
        - forecast_length: An integer representing the forecast length in hours.
        - water_level_source_path: Path or xarray Dataset where water level data is stored.
        - user: User type, which can be 'trainer' or 'tester', to control data access.

        Returns:
        An xarray Dataset containing the selected subsets of water level data concatenated over the time dimension.

        Raises:
        ValueError: If any variable in var_list is not present in the dataset.
        """

        if var_list is None or len(var_list) == 0:
            return None
        if user in privacy_cfg["trainer"]:
            waterlevel = xr.open_dataset(water_level_source_path)
        elif user in privacy_cfg["tester"]:
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
        streamflow_source_path: Union[str, os.PathLike, xr.Dataset],
        **kwargs,
    ):
        """
        Reads streamflow data as an xarray Dataset based on specified parameters.

        This function opens a dataset of streamflows from the provided path or dataset object,
        and selects a subset of data based on the given gauge IDs, time range, and variables.
        It handles different data access for 'trainer' and 'tester' user types.

        Parameters:
        - gage_id_lst: List of gauge IDs for which data is to be read.
        - t_range: List of dictionaries with 'start' and 'end' dates specifying the time range for each period.
        - var_list: List of variables to be included in the resulting dataset.
        - forecast_length: An integer indicating the forecast length in hours.
        - user: Type of user ('trainer' or 'tester') which determines the data access level.
        - streamflow_source_path: Path or xarray Dataset from where streamflow data is sourced.

        Returns:
        An xarray Dataset containing the selected subsets of streamflow data concatenated over the time dimension.

        Raises:
        ValueError: If any of the specified variables in var_list are not found in the dataset.
        """

        if var_list is None or len(var_list) == 0:
            return None
        if user in privacy_cfg["trainer"]:
            streamflow = xr.open_dataset(streamflow_source_path)
        elif user in privacy_cfg["tester"]:
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
        gage_id_lst: list,
        t_range: list,
        var_lst: list,
        rainfall_source_path: str | os.PathLike | list,
        user: str,
        **kwargs,
    ):
        """
        Reads rainfall data and returns a dictionary of xarray Datasets keyed by basin IDs.

        This function opens multiple rainfall datasets from a specified source path, each corresponding to a different basin.
        It then selects subsets of data based on the given time range and combines them into a single dataset for each basin.
        The resulting datasets are stored in a dictionary, with basin IDs as keys.

        Parameters:
        - gage_id_lst: List of gauge IDs (basin IDs) for which rainfall data is to be read. Defaults to None.
        - t_range: List of dictionaries with 'start' and 'end' dates specifying the time range for each period. Defaults to None.
        - var_lst: List of variables to be included in the resulting dataset. Defaults to None.
        - rainfall_source_path: String path where individual rainfall .nc files are stored for each basin. Defaults to None.

        Returns:
        A dictionary with basin IDs as keys and xarray Datasets as values. Each Dataset represents combined rainfall data for the corresponding basin over the specified time ranges.

        Note:
        The function returns None if var_lst is not provided.
        """

        if var_lst is None:
            return None
        rainfall_dict = {}
        if user in privacy_cfg["trainer"]:
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
        elif user in privacy_cfg["tester"]:
            # 为保证结果正确，rainfall_source_path应该和gage_id一一对应
            for basin in gage_id_lst:
                rainfall = rainfall_source_path[gage_id_lst.index(basin)]
                subset_list = []
                for period in t_range:
                    start_date = period["start"]
                    end_date = period["end"]
                    subset = rainfall.sel(time_now=slice(start_date, end_date))
                    subset_list.append(subset)
                merged_dataset_tp = xr.concat(subset_list, dim="time_now")
                rainfall_dict[basin] = merged_dataset_tp.to_array(dim="variable")
        else:
            raise NotImplementedError
        return rainfall_dict

    def read_gfs_xrdataset(
        self,
        gage_id_lst: list,
        t_range: list,
        var_lst: list,
        gfs_source_path: str|os.PathLike|list,
        user: str,
        **kwargs,
    ):
        """
        Reads Global Forecast System (GFS) data except rainfall and returns a dictionary of xarray Datasets keyed by basin IDs.

        This function opens multiple GFS datasets from a specified source path, with each dataset corresponding to a different basin.
        It selects subsets of data based on a given time range and list of variables, and combines these subsets into a single dataset for each basin.
        The resulting datasets are stored in a dictionary, where each key is a basin ID.

        Parameters:
        - gage_id_lst: List of gauge IDs (basin IDs) for which GFS data is to be read.
        - t_range: List of dictionaries with 'start' and 'end' dates specifying the time range for each period.
        - var_lst: List of variables to be included in the resulting dataset.
        - gfs_source_path: String path where individual GFS .nc files are stored for each basin.
        - user: Type of user, which might affect data access or processing.

        Returns:
        A dictionary with basin IDs as keys and xarray Datasets as values. Each Dataset contains combined GFS data for the corresponding basin over the specified time ranges.

        Note:
        The function returns None if var_lst is not provided.
        """

        if var_lst is None:
            return None
        gfs_dict = {}
        if user in privacy_cfg["trainer"]:
            for basin in gage_id_lst:
                gfs = xr.open_dataset(os.path.join(gfs_source_path, f"{basin}.nc"))
                subset_list = []
                for period in t_range:
                    start_date = period["start"]
                    end_date = period["end"]
                    subset = gfs[var_lst].sel(time=slice(start_date, end_date))
                    subset_list.append(subset)
                merged_dataset_gfs = xr.concat(subset_list, dim="time")
                gfs_dict[basin] = merged_dataset_gfs.to_array(dim="variable")
        elif user in privacy_cfg["tester"]:
            # 为保证结果正确，gfs_source_path应该和gage_id一一对应
            for basin in gage_id_lst:
                gfs = gfs_source_path[gage_id_lst.index(basin)]
                subset_list = []
                for period in t_range:
                    start_date = period["start"]
                    end_date = period["end"]
                    subset = gfs[var_lst].sel(time=slice(start_date, end_date))
                    subset_list.append(subset)
                merged_dataset_gfs = xr.concat(subset_list, dim="time")
                gfs_dict[basin] = merged_dataset_gfs.to_array(dim="variable")
        else:
            raise NotImplementedError
        return gfs_dict

    def read_attr_xrdataset(
        self, gage_id_lst, var_lst, attributes_path, user, **kwargs
    ):
        """
        Reads basin attributes data from an xarray Dataset and returns a subset based on specified basin IDs and variables.

        This function opens an attributes dataset from a provided path or directly uses the dataset object based on the user type.
        It then filters the dataset for selected basin IDs and variables. The function handles different access levels for 'trainer' and 'tester' users.

        Parameters:
        - gage_id_lst: List of basin IDs for which attributes data is to be read.
        - var_lst: List of variables to be included in the resulting dataset. If 'all' is included in var_lst, all variables are returned.
        - attributes_path: Path to the dataset file or the dataset object itself containing basin attributes.
        - user: Type of user ('trainer' or 'tester') which determines data access level.

        Returns:
        An xarray Dataset containing the selected attributes for the specified basins.

        Note:
        - The function returns None if var_lst is not provided or is empty.
        - The user's access level (trainer or tester) determines how the attributes dataset is accessed.
        """

        if var_lst is None or len(var_lst) == 0:
            return None
        if user in privacy_cfg["trainer"]:
            attr = xr.open_dataset(attributes_path)
        elif user in privacy_cfg["tester"]:
            attr = attributes_path
        else:
            attr = xr.Dataset()
        if "all" in var_lst:
            return attr.sel(basin=gage_id_lst)
        return attr[var_lst].sel(basin=gage_id_lst)

    def read_mean_prcp(self, gage_id_lst, mean_prep_path, user) -> np.array:
        """
        Reads mean precipitation data for specified basin IDs from an xarray Dataset.

        This function opens a dataset containing mean precipitation data from a given path or dataset object,
        depending on the user type. It then extracts precipitation data for the specified basin IDs.
        The function handles different data access levels for 'trainer' and 'tester' users.

        Parameters:
        - gage_id_lst: List of basin IDs for which mean precipitation data is to be read.
        - mean_prep_path: Path to the dataset file or the dataset object itself containing mean precipitation data.
        - user: Type of user ('trainer' or 'tester') which determines data access level.

        Returns:
        A NumPy array containing mean precipitation values for the specified basins.

        Note:
        - The function adapts the data source based on the user's access level (trainer or tester).
        """

        if user in privacy_cfg["trainer"]:
            mean_prep = xr.open_dataset(mean_prep_path)
        elif user in privacy_cfg["tester"]:
            mean_prep = mean_prep_path
        else:
            mean_prep = xr.Dataset()
        return mean_prep["p_mean"].sel(basin=gage_id_lst)
