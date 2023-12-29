"""
Author: Xinzhuo Wu
Date: 2023-09-30 1:20:18
LastEditTime: 2023-12-29 11:05:57
LastEditors: Xinzhuo Wu
Description: A pytorch dataset class; references to https://github.com/neuralhydrology/neuralhydrology
FilePath: \torchhydro\torchhydro\datasets\data_sets.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
import os
import hydrodataset as hds
from hydrodataset import HydroDataset, CACHE_DIR
from hydrodataset.camels import map_string_vars
import numpy as np
from netCDF4 import Dataset as ncdataset
import collections
import pandas as pd
import xarray as xr

GPM_GFS_NO_DATASET_ERROR_LOG = (
    "We cannot read this dataset now. Please check if you choose correctly:\n"
)


class GPM_GFS(HydroDataset):
    def __init__(
        self,
        data_path=os.path.join("gpm_gfs_data"),
        download=False,
        region: str = "US",
    ):
        super().__init__(data_path)
        self.region = region
        self.data_source_description = self.set_data_source_describe()
        if download:
            raise NotImplementedError(
                "We don't provide methods for downloading data at present\n"
            )
        self.sites = self.read_site_info()

    def get_name(self):
        return "GPM_GFS_" + self.region

    def set_data_source_describe(self) -> collections.OrderedDict:
        """
        the files in the dataset and their location in file system

        Returns
        -------
        collections.OrderedDict
            the description for GPM and GFS dataset
        """
        gpm_gfs_db = self.data_source_dir
        if self.region == "US":
            return self._set_data_source_GpmGfsUS_describe(gpm_gfs_db)

        else:
            raise NotImplementedError(GPM_GFS_NO_DATASET_ERROR_LOG)

    def _set_data_source_GpmGfsUS_describe(self, gpm_gfs_db):
        # water_level of basins
        camels_water_level = gpm_gfs_db.joinpath("water_level")

        # gpm
        gpm_data = gpm_gfs_db.joinpath("gpm")

        # gfs
        gfs_data = gpm_gfs_db.joinpath("gfs")

        # basin id
        gauge_id_file = gpm_gfs_db.joinpath("camels_name.txt")

        return collections.OrderedDict(
            GPM_GFS_DIR=gpm_gfs_db,
            CAMELS_WATER_LEVEL=camels_water_level,
            GPM_DATA=gpm_data,
            GFS_DATA=gfs_data,
            CAMELS_GAUGE_FILE=gauge_id_file,
        )

    def read_site_info(self) -> pd.DataFrame:
        """
        Read the basic information of gages in a CAMELS dataset

        Returns
        -------
        pd.DataFrame
            basic info of gages
        """
        camels_gauge_file = self.data_source_description["CAMELS_GAUGE_FILE"]
        if self.region == "US":
            data = pd.read_csv(
                camels_gauge_file, sep=";", dtype={"gauge_id": str, "huc_02": str}
            )
        else:
            raise NotImplementedError(GPM_GFS_NO_DATASET_ERROR_LOG)
        return data

    def read_object_ids(self, **kwargs) -> np.array:
        """
        read station ids

        Parameters
        ----------
        **kwargs
            optional params if needed

        Returns
        -------
        np.array
            gage/station ids
        """
        if self.region in ["US"]:
            return self.sites["gauge_id"].values
        else:
            raise NotImplementedError(GPM_GFS_NO_DATASET_ERROR_LOG)

    def read_waterlevel_xrdataset(
        self, gage_id_lst=None, t_range: list = None, var_list=None, **kwargs
    ):
        if var_list is None or len(var_list) == 0:
            return None

        waterlevel = xr.open_dataset(
            os.path.join("/ftproot", "gpm_gfs_data", "water_level_total.nc")
        )
        all_vars = waterlevel.data_vars
        if any(var not in waterlevel.variables for var in var_list):
            raise ValueError(f"var_lst must all be in {all_vars}")
        return waterlevel[["waterlevel"]].sel(
            time=slice(t_range[0], t_range[1]), basin=gage_id_lst
        )

    def read_streamflow_xrdataset(
        self, gage_id_lst=None, t_range: list = None, var_list=None, **kwargs
    ):
        if var_list is None or len(var_list) == 0:
            return None

        streamflow = xr.open_dataset(
            os.path.join("/ftproot", "biliuhe", "streamflow_UTC0.nc")
        )
        all_vars = streamflow.data_vars
        if any(var not in streamflow.variables for var in var_list):
            raise ValueError(f"var_lst must all be in {all_vars}")

        subset_list = []

        for period in t_range:
            start_date = period["start"]
            end_date = period["end"]
            subset = streamflow.sel(time=slice(start_date, end_date))
            subset_list.append(subset)

        return xr.concat(subset_list, dim="time")

    def read_gpm_xrdataset(
        self,
        gage_id_lst: list = None,
        t_range: list = None,
        var_lst: list = None,
        **kwargs,
    ):
        if var_lst is None:
            return None

        gpm_dict = {}
        for basin in gage_id_lst:
            gpm = xr.open_dataset(
                os.path.join("/ftproot", "biliuhe", "gpm_gfs_full_re2.nc")
            )
            subset_list = []

            for period in t_range:
                start_date = period["start"]
                end_date = period["end"]
                subset = gpm.sel(time_now=slice(start_date, end_date))
                subset_list.append(subset)

            merged_dataset_tp = xr.concat(subset_list, dim="time_now")
            gpm_dict[basin] = merged_dataset_tp

        return gpm_dict

    def read_attr_xrdataset(self, gage_id_lst=None, var_lst=None, **kwargs):
        if var_lst is None or len(var_lst) == 0:
            return None
        attr = xr.open_dataset(
            os.path.join("/home", "wuxinzhuo", "camelsus_attributes.nc")
        )
        if "all_number" in list(kwargs.keys()) and kwargs["all_number"]:
            attr_num = map_string_vars(attr)
            return attr_num[var_lst].sel(basin=gage_id_lst)
        return attr[var_lst].sel(basin=gage_id_lst)

    def read_mean_prcp(self, gage_id_lst) -> np.array:
        if self.region in ["US", "AUS", "BR", "GB"]:
            if self.region == "US":
                return self.read_attr_xrdataset(gage_id_lst, ["p_mean"])
            return self.read_constant_cols(
                gage_id_lst, ["p_mean"], is_return_dict=False
            )
        elif self.region == "CL":
            return self.read_constant_cols(
                gage_id_lst, ["p_mean_cr2met"], is_return_dict=False
            )
        else:
            raise NotImplementedError(GPM_GFS_NO_DATASET_ERROR_LOG)

    def read_area(self, gage_id_lst) -> np.array:
        if self.region == "US":
            return self.read_attr_xrdataset(gage_id_lst, ["area_gages2"])
        else:
            raise NotImplementedError(GPM_GFS_NO_DATASET_ERROR_LOG)
