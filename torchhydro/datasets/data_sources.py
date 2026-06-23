"""
Author: Wenyu Ouyang
Date: 2024-04-02 14:37:09
LastEditTime: 2025-04-19 20:28:38
LastEditors: Wenyu Ouyang
Description: A module for different data sources
FilePath: /HydroForecastEval/mnt/disk1/owen/code/torchhydro/torchhydro/datasets/data_sources.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import json
import warnings
import collections
import os
import re
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Union

from hydroutils.hydro_stat import cal_fdc
from pandas.core.dtypes.common import is_string_dtype, is_numeric_dtype
import xarray as xr
import pint_xarray  # noqa but it is used in the code
from tqdm import tqdm

from hydrodataset import HydroDataset, Camels
from hydroutils import (
    hydro_time,
    hydro_arithmetric,
    hydro_file,
    hydro_logger,
)
from hydrodataset.grdc_caravan import GrdcCaravan
from hydrodataset.camelsh import Camelsh
from hydrodataset.caravan import Caravan
from hydrodataset.camels_aus import CamelsAus
from hydrodataset.camels_br import CamelsBr
from hydrodataset.camels_cl import CamelsCl
from hydrodataset.camels_gb import CamelsGb
from hydrodataset.camels_col import CamelsCol
from hydrodatasource.reader.data_source import (
    SelfMadeHydroDataset,
    SelfMadeForecastDataset,
    LongTermDataset,
    StationHydroDataset,
    TgHydroDatasource,
)
from hydrodatasource.reader.floodevent import FloodEventDatasource

from torchhydro import CACHE_DIR, SETTING


class SupData4Camels:
    """A parent class for different data sources for CAMELS-US
    and also a class for reading streamflow data after 2014-12-31"""

    def __init__(self, supdata_dir=None) -> None:
        self.camels = Camels(
            data_path=os.path.join(
                SETTING["local_data_path"]["datasets-origin"], "camels", "camels_us"
            )
        )
        self.camels671_sites = self.camels.read_site_info()

        if supdata_dir is None:
            supdata_dir = os.path.join(
                SETTING["local_data_path"]["datasets-interim"],
                "camels_us",
            )
        self.data_source_dir = supdata_dir
        self.data_source_description = self.set_data_source_describe()

    def set_data_source_describe(self):
        return self.camels.data_source_description

    def read_ts_table(self, gage_id_lst=None, t_range=None, var_lst=None, **kwargs):
        """A parent function for reading camels timeseries data from csv or txt files.
        For different data sources, we need to implement this function.
        Here it is also a function for reading streamflow data after 2014-12-31.


        Parameters
        ----------
        gage_id_lst : _type_, optional
            basin ids, by default None
        t_range : _type_, optional
            time range, by default None
        var_lst : _type_, optional
            all variables including forcing and streamflow, by default None

        Raises
        ------
        NotImplementedError
            _description_
        """
        if gage_id_lst is None:
            gage_id_lst = self.all_basins
        if t_range is None:
            t_range = self.all_t_range
        if var_lst is None:
            var_lst = self.vars
        return self.camels.read_target_cols(
            gage_id_lst=gage_id_lst,
            t_range=t_range,
            target_cols=var_lst,
        )

    @property
    def all_basins(self):
        return self.camels671_sites["gauge_id"].values

    @property
    def all_t_range(self):
        # this is a left-closed right-open interval
        return ["1980-01-01", "2022-01-01"]

    @property
    def units(self):
        return ["ft^3/s"]

    @property
    def vars(self):
        return ["streamflow"]

    @property
    def ts_xrdataset_path(self):
        return CACHE_DIR.joinpath("camelsus_streamflow.nc")

    def cache_ts_xrdataset(self):
        """Save all timeseries data in a netcdf file in the cache directory"""
        basins = self.all_basins
        t_range = self.all_t_range
        times = hydro_time.t_range_days(t_range).tolist()
        variables = self.vars
        ts_data = self.read_ts_table(
            gage_id_lst=basins,
            t_range=t_range,
            var_lst=variables,
        )
        # All units' names are from Pint https://github.com/hgrecco/pint/blob/master/pint/default_en.txt
        units = self.units
        xr_data = xr.Dataset(
            data_vars={
                **{
                    variables[i]: (
                        ["basin", "time"],
                        ts_data[:, :, i],
                        {"units": units[i]},
                    )
                    for i in range(len(variables))
                }
            },
            coords={
                "basin": basins,
                "time": times,
            },
        )
        xr_data.to_netcdf(self.ts_xrdataset_path)

    def read_ts_xrdataset(self, gage_id_lst=None, t_range=None, var_lst=None):
        """Read all timeseries data from a netcdf file in the cache directory"""
        if not self.ts_xrdataset_path.exists():
            self.cache_ts_xrdataset()
        if var_lst is None:
            return None
        ts = xr.open_dataset(self.ts_xrdataset_path)
        all_vars = ts.data_vars
        if any(var not in ts.variables for var in var_lst):
            raise ValueError(f"var_lst must all be in {all_vars}")
        return ts[var_lst].sel(basin=gage_id_lst, time=slice(t_range[0], t_range[1]))

    def read_attr_xrdataset(self, gage_id_lst=None, var_lst=None):
        return self.camels.read_attr_xrdataset(gage_id_lst=gage_id_lst, var_lst=var_lst)


# the following is a dict for different data sources
class ModisEt4Camels(SupData4Camels):
    """
    A datasource class for MODIS ET data of basins in CAMELS.

    Attributes data come from CAMELS.
    ET data include:
        PMLV2 (https://doi.org/10.1016/j.rse.2018.12.031)
        MODIS16A2v006 (https://developers.google.com/earth-engine/datasets/catalog/MODIS_006_MOD16A2?hl=en#bands)
        MODIS16A2v105 (https://developers.google.com/earth-engine/datasets/catalog/MODIS_NTSG_MOD16A2_105?hl=en#description)
    """

    def __init__(self, supdata_dir=None):
        """
        Initialize a ModisEt4Camels instance.

        Parameters
        ----------
        supdata_dir
            a list including the data file directory for the instance and CAMELS's path

        """
        if supdata_dir is None:
            supdata_dir = os.path.join(
                SETTING["local_data_path"]["datasets-interim"],
                "camels_us",
                "modiset4camels",
            )
        super().__init__(supdata_dir)

    @property
    def all_t_range(self):
        # this is a left-closed right-open interval
        return ["2001-01-01", "2022-01-01"]

    @property
    def units(self):
        return [
            "gC/m^2/d",
            "mm/day",
            "mm/day",
            "mm/day",
            "mm/day",
            "mm/day",
            "mm/day",
            "mm/day",
            "mm/day",
            "mm/day",
            "dimensionless",
        ]

    @property
    def vars(self):
        return [
            # PMLV2
            "GPP",
            "Ec",
            "Es",
            "Ei",
            "ET_water",
            # PML_V2's ET = Ec + Es + Ei
            "ET_sum",
            # MODIS16A2
            "ET",
            "LE",
            "PET",
            "PLE",
            "ET_QC",
        ]

    @property
    def ts_xrdataset_path(self):
        return CACHE_DIR.joinpath("camelsus_modiset.nc")

    def set_data_source_describe(self):
        et_db = self.data_source_dir
        # ET
        et_basin_mean_dir = os.path.join(et_db, "basin_mean_forcing")
        modisa16v105_dir = os.path.join(et_basin_mean_dir, "MOD16A2_105_CAMELS")
        modisa16v006_dir = os.path.join(et_basin_mean_dir, "MOD16A2_006_CAMELS")
        pmlv2_dir = os.path.join(et_basin_mean_dir, "PML_V2_CAMELS")
        if not os.path.isdir(et_basin_mean_dir):
            raise NotADirectoryError(
                "Please check if you have downloaded the data and put it in the correct dir"
            )
        return collections.OrderedDict(
            MODIS_ET_CAMELS_DIR=et_db,
            MODIS_ET_CAMELS_MEAN_DIR=et_basin_mean_dir,
            MOD16A2_CAMELS_DIR=modisa16v006_dir,
            PMLV2_CAMELS_DIR=pmlv2_dir,
        )

    def read_ts_table(
        self,
        gage_id_lst=None,
        t_range=None,
        var_lst=None,
        reduce_way="mean",
        **kwargs,
    ):
        """
        Read ET data.

        Parameters
        ----------
        gage_id_lst
            the ids of gages in CAMELS
        t_range
            the start and end periods
        target_cols
            the forcing var types
        reduce_way
            how to do "reduce" -- mean or sum; the default is "mean"

        Returns
        -------
        np.array
            return an np.array
        """

        assert len(t_range) == 2
        assert all(x < y for x, y in zip(gage_id_lst, gage_id_lst[1:]))
        # Data is not daily. For convenience, we fill NaN values in gap periods.
        # For example, the data is in period 1 (1-8 days), then there is one data in the 1st day while the rest are NaN
        t_range_list = hydro_time.t_range_days(t_range)
        nt = t_range_list.shape[0]
        x = np.empty([len(gage_id_lst), nt, len(var_lst)])
        for k in tqdm(range(len(gage_id_lst)), desc="Read MODIS ET data for CAMELS-US"):
            # two way to read data are provided:
            # 1. directly read data: the data is sum of 8 days
            # 2. calculate daily mean value of 8 days
            data = self.read_basin_mean_modiset(
                gage_id_lst[k], var_lst, t_range_list, reduce_way=reduce_way
            )
            x[k, :, :] = data
        return x

    def read_basin_mean_modiset(
        self, usgs_id, var_lst, t_range_list, reduce_way
    ) -> np.array:
        """
        Read modis ET from PMLV2 and MOD16A2

        Parameters
        ----------
        usgs_id
            ids of basins
        var_lst
            et variables from PMLV2 or/and MOD16A2
        t_range_list
            daily datetime list
        reduce_way
            how to do "reduce" -- mean or sum; the default is "sum"

        Returns
        -------
        np.array
            ET data
        """
        gage_id_df = self.camels671_sites
        huc = gage_id_df[gage_id_df["gauge_id"] == usgs_id]["huc_02"].values[0]

        modis16a2_data_folder = self.data_source_description["MOD16A2_CAMELS_DIR"]
        pmlv2_data_folder = self.data_source_description["PMLV2_CAMELS_DIR"]
        pmlv2_data_file = os.path.join(
            pmlv2_data_folder, huc, f"{usgs_id}_lump_pmlv2_et.txt"
        )
        modis16a2_data_file = os.path.join(
            modis16a2_data_folder, huc, f"{usgs_id}_lump_modis16a2v006_et.txt"
        )
        pmlv2_data_temp = pd.read_csv(pmlv2_data_file, header=None, skiprows=1)
        modis16a2_data_temp = pd.read_csv(modis16a2_data_file, header=None, skiprows=1)
        pmlv2_lst = [
            "Year",
            "Mnth",
            "Day",
            "Hr",
            "GPP",
            "Ec",
            "Es",
            "Ei",
            "ET_water",
            "ET_sum",
        ]  # PMLV2
        modis16a2_lst = [
            "Year",
            "Mnth",
            "Day",
            "Hr",
            "ET",
            "LE",
            "PET",
            "PLE",
            "ET_QC",
        ]  # MODIS16A2
        df_date_pmlv2 = pmlv2_data_temp[[0, 1, 2]]
        df_date_pmlv2.columns = ["year", "month", "day"]
        df_date_modis16a2 = modis16a2_data_temp[[0, 1, 2]]
        df_date_modis16a2.columns = ["year", "month", "day"]
        ind1_pmlv2, ind2_pmlv2, t_range_final_pmlv2 = self.date_intersect(
            df_date_pmlv2, t_range_list
        )
        (
            ind1_modis16a2,
            ind2_modis16a2,
            t_range_final_modis16a2,
        ) = self.date_intersect(df_date_modis16a2, t_range_list)

        nf = len(var_lst)
        nt = len(t_range_list)
        out = np.full([nt, nf], np.nan)

        for k in range(nf):
            if var_lst[k] in pmlv2_lst:
                if len(t_range_final_pmlv2) == 0:
                    # no data, just skip this var
                    continue
                if var_lst[k] == "ET_sum":
                    # No such item in original PML_V2 data
                    et_3components = self.read_basin_mean_modiset(
                        usgs_id, ["Ec", "Es", "Ei"], t_range_list, reduce_way
                    )
                    # it si equal to sum of 3 components
                    out[:, k] = np.sum(et_3components, axis=-1)
                    continue
                ind = pmlv2_lst.index(var_lst[k])
                if reduce_way == "sum":
                    out[ind2_pmlv2, k] = pmlv2_data_temp[ind].values[ind1_pmlv2]
                elif reduce_way == "mean":
                    days_interval = [y - x for x, y in zip(ind2_pmlv2, ind2_pmlv2[1:])]
                    if (
                        t_range_final_pmlv2[-1].item().month == 12
                        and t_range_final_pmlv2[-1].item().day == 31
                    ):
                        final_timedelta = (
                            t_range_final_pmlv2[-1].item()
                            - t_range_final_pmlv2[ind2_pmlv2[-1]].item()
                        )
                        final_day_interval = [final_timedelta.days]
                    else:
                        final_day_interval = [8]
                    days_interval = np.array(days_interval + final_day_interval)
                    # there may be some missing data, so that some interval will be larger than 8
                    days_interval[np.where(days_interval > 8)] = 8
                    out[ind2_pmlv2, k] = (
                        pmlv2_data_temp[ind].values[ind1_pmlv2] / days_interval
                    )
                else:
                    raise NotImplementedError("We don't have such a reduce way")
            elif var_lst[k] in modis16a2_lst:
                if len(t_range_final_modis16a2) == 0:
                    # no data, just skip this var
                    continue
                ind = modis16a2_lst.index(var_lst[k])
                if reduce_way == "sum":
                    out[ind2_modis16a2, k] = modis16a2_data_temp[ind].values[
                        ind1_modis16a2
                    ]
                elif reduce_way == "mean":
                    days_interval = [
                        y - x for x, y in zip(ind2_modis16a2, ind2_modis16a2[1:])
                    ]
                    if (
                        t_range_final_modis16a2[-1].item().month == 12
                        and t_range_final_modis16a2[-1].item().day == 31
                    ):
                        final_timedelta = (
                            t_range_final_modis16a2[-1].item()
                            - t_range_final_modis16a2[ind2_modis16a2[-1]].item()
                        )
                        final_day_interval = [final_timedelta.days]
                    else:
                        final_day_interval = [8]
                    days_interval = np.array(days_interval + final_day_interval)
                    # there may be some missing data, so that some interval will be larger than 8
                    days_interval[np.where(days_interval > 8)] = 8
                    out[ind2_modis16a2, k] = (
                        modis16a2_data_temp[ind].values[ind1_modis16a2] / days_interval
                    )
                else:
                    raise NotImplementedError("We don't have such a reduce way")
            else:
                raise NotImplementedError("No such var type now")
        # unit is 0.1mm/day(or 8/5/6days), so multiply it with 0.1 to transform to mm/day(or 8/5/6days))
        # TODO: only valid for MODIS, for PMLV2, we need to check the unit
        out = out * 0.1
        return out

    @staticmethod
    def date_intersect(df_date, t_range_list):
        date = pd.to_datetime(df_date).values.astype("datetime64[D]")
        if (
            np.datetime64(f"{str(date[-1].astype(object).year)}-12-31")
            > date[-1]
            > np.datetime64(f"{str(date[-1].astype(object).year)}-12-24")
        ):
            final_date = np.datetime64(f"{str(date[-1].astype(object).year + 1)}-01-01")
        else:
            final_date = date[-1] + np.timedelta64(8, "D")
        date_all = hydro_time.t_range_days(
            hydro_time.t_days_lst2range([date[0], final_date])
        )
        t_range_final = np.intersect1d(date_all, t_range_list)
        [c, ind1, ind2] = np.intersect1d(date, t_range_final, return_indices=True)
        return ind1, ind2, t_range_final


class Nldas4Camels(SupData4Camels):
    """
    A datasource class for geo attributes data, NLDAS v2 forcing data, and streamflow data of basins in CAMELS.

    The forcing data are basin mean values. Attributes and streamflow data come from CAMELS.
    """

    def __init__(self, supdata_dir=None):
        """
        Initialize a Nldas4Camels instance.

        Parameters
        ----------
        supdata_dir
            a list including the data file directory for the instance and CAMELS's path

        """
        if supdata_dir is None:
            supdata_dir = os.path.join(
                SETTING["local_data_path"]["datasets-interim"],
                "camels_us",
                "nldas4camels",
            )
        super().__init__(supdata_dir=supdata_dir)

    @property
    def units(self):
        return [
            "°C",
            "dimensionless",
            "Pa",
            "m/s",
            "m/s",
            "W/m^2",
            "dimensionless",
            "W/m^2",
            "J/kg",
            # unit of potential_evaporation and total_precipitation is kg/m^2 (for a day),
            # we use rho=1000kg/m^3 as water's density to transform these two variables‘ unit to mm/day
            # so it's 10^-3 m /day and it is just mm/day, hence we don't need to transform actually
            "mm/day",
            "mm/day",
        ]

    @property
    def vars(self):
        return [
            "temperature",
            "specific_humidity",
            "pressure",
            "wind_u",
            "wind_v",
            "longwave_radiation",
            "convective_fraction",
            "shortwave_radiation",
            "potential_energy",
            "potential_evaporation",
            "total_precipitation",
        ]

    @property
    def ts_xrdataset_path(self):
        return CACHE_DIR.joinpath("camelsus_nldas.nc")

    def set_data_source_describe(self):
        nldas_db = self.data_source_dir
        # forcing
        nldas_forcing_basin_mean_dir = os.path.join(nldas_db, "basin_mean_forcing")
        if not os.path.isdir(nldas_forcing_basin_mean_dir):
            raise NotADirectoryError(
                "Please check if you have downloaded the data and put it in the correct dir"
            )
        return collections.OrderedDict(
            NLDAS_CAMELS_DIR=nldas_db,
            NLDAS_CAMELS_MEAN_DIR=nldas_forcing_basin_mean_dir,
        )

    def read_basin_mean_nldas(self, usgs_id, var_lst, t_range_list):
        gage_id_df = self.camels671_sites
        huc = gage_id_df[gage_id_df["gauge_id"] == usgs_id]["huc_02"].values[0]

        data_folder = self.data_source_description["NLDAS_CAMELS_MEAN_DIR"]
        data_file = os.path.join(
            data_folder, huc, f"{usgs_id}_lump_nldas_forcing_leap.txt"
        )
        data_temp = pd.read_csv(data_file, sep=r"\s+", header=None, skiprows=1)
        forcing_lst = ["Year", "Mnth", "Day", "Hr"] + self.vars
        df_date = data_temp[[0, 1, 2]]
        df_date.columns = ["year", "month", "day"]
        date = pd.to_datetime(df_date).values.astype("datetime64[D]")

        nf = len(var_lst)
        [c, ind1, ind2] = np.intersect1d(date, t_range_list, return_indices=True)
        nt = c.shape[0]
        out = np.empty([nt, nf])

        for k in range(nf):
            ind = forcing_lst.index(var_lst[k])
            if "potential_evaporation" in var_lst[k]:
                pet = data_temp[ind].values
                # there are a few negative values for pet, set them 0
                pet[pet < 0] = 0.0
                out[ind2, k] = pet[ind1]
            else:
                out[ind2, k] = data_temp[ind].values[ind1]
        return out

    def read_ts_table(
        self,
        gage_id_lst: list = None,
        t_range: list = None,
        var_lst: list = None,
        **kwargs,
    ) -> np.array:
        """
        Read forcing data.

        Parameters
        ----------
        gage_id_lst
            the ids of gages in CAMELS
        t_range
            the start and end periods
        var_lst
            the forcing var types

        Returns
        -------
        np.array
            return an np.array
        """

        assert len(t_range) == 2
        assert all(x < y for x, y in zip(gage_id_lst, gage_id_lst[1:]))

        t_range_list = hydro_time.t_range_days(t_range)
        nt = t_range_list.shape[0]
        x = np.empty([len(gage_id_lst), nt, len(var_lst)])
        for k in tqdm(
            range(len(gage_id_lst)), desc="Read NLDAS forcing data for CAMELS-US"
        ):
            data = self.read_basin_mean_nldas(gage_id_lst[k], var_lst, t_range_list)
            x[k, :, :] = data
        return x


class Smap4Camels(SupData4Camels):
    """
    A datasource class for geo attributes data, forcing data, and SMAP data of basins in CAMELS.
    """

    def __init__(self, supdata_dir=None):
        """
        Parameters
        ----------
        supdata_dir
            a list including the data file directory for the instance and CAMELS's path

        """
        if supdata_dir is None:
            supdata_dir = os.path.join(
                SETTING["local_data_path"]["datasets-interim"],
                "camels_us",
                "smap4camels",
            )
        super().__init__(supdata_dir=supdata_dir)

    @property
    def all_t_range(self):
        # this is a left-closed right-open interval
        return ["2015-04-01", "2021-10-04"]

    @property
    def units(self):
        return ["mm", "mm", "dimensionless", "dimensionless", "dimensionless"]

    @property
    def vars(self):
        return ["ssm", "susm", "smp", "ssma", "susma"]

    @property
    def ts_xrdataset_path(self):
        return CACHE_DIR.joinpath("camelsus_smap.nc")

    def set_data_source_describe(self):
        # forcing
        smap_db = self.data_source_dir
        if not os.path.isdir(smap_db):
            raise NotADirectoryError(
                "Please check if you have downloaded the data and put it in the correct dir"
            )
        smap_data_dir = os.path.join(smap_db, "NASA_USDA_SMAP_CAMELS")
        return collections.OrderedDict(
            SMAP_CAMELS_DIR=smap_db, SMAP_CAMELS_MEAN_DIR=smap_data_dir
        )

    def read_ts_table(self, gage_id_lst=None, t_range=None, var_lst=None, **kwargs):
        """
        Read SMAP basin mean data

        More detials about NASA-USDA Enhanced SMAP data could be seen in:
        https://explorer.earthengine.google.com/#detail/NASA_USDA%2FHSL%2FSMAP10KM_soil_moisture

        Parameters
        ----------
        gage_id_lst
            the ids of gages in CAMELS
        t_range
            the start and end periods
        target_cols
            the var types

        Returns
        -------
        np.array
            return an np.array
        """
        # Data is not daily. For convenience, we fill NaN values in gap periods.
        # For example, the data is in period 1 (1-3 days), then there is one data in the 1st day while the rest are NaN
        t_range_list = hydro_time.t_range_days(t_range)
        nt = t_range_list.shape[0]
        x = np.empty([len(gage_id_lst), nt, len(var_lst)])
        for k in tqdm(
            range(len(gage_id_lst)), desc="Read NSDA-SMAP data for CAMELS-US"
        ):
            # two way to read data are provided:
            # 1. directly read data: the data is sum of 8 days
            # 2. calculate daily mean value of 8 days
            data = self.read_basin_mean_smap(gage_id_lst[k], var_lst, t_range_list)
            x[k, :, :] = data
        return x

    def read_basin_mean_smap(self, usgs_id, var_lst, t_range_list):
        gage_id_df = self.camels671_sites
        huc = gage_id_df[gage_id_df["gauge_id"] == usgs_id]["huc_02"].values[0]

        data_folder = self.data_source_description["SMAP_CAMELS_MEAN_DIR"]
        data_file = os.path.join(data_folder, huc, f"{usgs_id}_lump_nasa_usda_smap.txt")
        data_temp = pd.read_csv(data_file, sep=",", header=None, skiprows=1)
        smap_var_lst = [
            "Year",
            "Mnth",
            "Day",
            "Hr",
            "ssm",
            "susm",
            "smp",
            "ssma",
            "susma",
        ]
        df_date = data_temp[[0, 1, 2]]
        df_date.columns = ["year", "month", "day"]
        date = pd.to_datetime(df_date).values.astype("datetime64[D]")
        [c, ind1, ind2] = np.intersect1d(date, t_range_list, return_indices=True)

        nf = len(var_lst)
        nt = len(t_range_list)
        out = np.full([nt, nf], np.nan)

        for k in range(nf):
            ind = smap_var_lst.index(var_lst[k])
            out[ind2, k] = data_temp[ind].values[ind1]
        return out


class Reservoirs(HydroDataset):
    def __init__(self, data_path=os.path.join("reservoirs"), download=False):
        """
        Initialization for CAMELS series dataset

        Parameters
        ----------
        data_path
            where we put the dataset.
            we already set the ROOT directory for hydrodataset,
            so here just set it as a relative path,
            by default "reservoirs"
            download
                if true, download, by default False
        """
        super().__init__(data_path)
        self.data_source_description = self.set_data_source_describe()
        if download:
            self.download_data_source()
        self.reservoir_sites = self.read_site_info()

    def get_name(self):
        return "Reservoirs"

    def set_data_source_describe(self):
        """
        the files in the dataset and their location in file system

        Returns
        -------
        collections.OrderedDict
            the description for a Reservoirs dataset
        """
        reservoirs_dir = self.data_source_dir
        if not os.path.isdir(reservoirs_dir):
            raise NotADirectoryError(
                "Please check if you have downloaded the data and put it in the correct dir"
            )
        reservoirs_attr_file = os.path.join(reservoirs_dir, "reservoirs_static.nc")
        reservoirs_time_series_file = os.path.join(
            reservoirs_dir, "reservoirs_timeseries.nc"
        )
        return collections.OrderedDict(
            RESERVOIRS_DIR=reservoirs_dir,
            RESERVOIRS_ATTR_FILE=reservoirs_attr_file,
            RESERVOIRS_TIME_SERIES_FILE=reservoirs_time_series_file,
        )

    def get_constant_cols(self) -> np.array:
        """all readable attrs in Reservoirs"""
        reservoir_attr_file = self.data_source_description["RESERVOIRS_ATTR_FILE"]
        ts = xr.open_dataset(reservoir_attr_file)
        all_cols = list(ts.data_vars.keys())
        other_cols = [col for col in all_cols if col != "Reservoir_ID"]
        return other_cols

    def get_relevant_cols(self):
        return np.array(["NetInflow", "Storage"])

    def get_target_cols(self):
        return np.array(["Release"])

    def read_site_info(self):
        """
        Read the site information of reservoirs

        Returns
        -------
        pd.DataFrame
            the site information of reservoirs
        """
        reservoir_attr_file = self.data_source_description["RESERVOIRS_ATTR_FILE"]
        ts = xr.open_dataset(reservoir_attr_file)
        reservoir_ids = ts["Reservoir_ID"].values
        reservoir_ids.sort()
        assert all(x < y for x, y in zip(reservoir_ids, reservoir_ids[1:]))
        return reservoir_ids

    def read_forcing_reservoir(self, reservoir_id, var_lst, t_range_list):
        reservoir_time_series_file = self.data_source_description[
            "RESERVOIRS_TIME_SERIES_FILE"
        ]
        ts = xr.open_dataset(reservoir_time_series_file)

        # 确保时间是升序排列
        assert all(
            ts["Time"].values[i] < ts["Time"].values[i + 1]
            for i in range(len(ts["Time"].values) - 1)
        )

        # 获取所有时间和水库索引
        time_all = ts["Time"].values.astype("datetime64[D]")
        rid_all = ts["Reservoir_ID"].values
        assert reservoir_id in rid_all

        # 找到 reservoir_id 对应的索引
        rid_index = int(np.where(rid_all == reservoir_id)[0][0])

        # 时间匹配：获取匹配时间在数组中的位置
        _, ind1, ind2 = np.intersect1d(time_all, t_range_list, return_indices=True)
        assert time_all[0] <= t_range_list[0] and time_all[-1] >= t_range_list[-1]

        # 输出初始化
        nt = len(t_range_list)
        nf = len(var_lst)
        out = np.empty((nt, nf))

        for k, var in enumerate(var_lst):
            # 取出每个变量在指定水库和时间段下的值
            values = ts[var].values  # shape: (n_reservoir, n_time)
            out[:, k] = values[rid_index, ind1]

        return out

    def read_relevant_cols(
        self, reservoir_id_lst=None, t_range=None, var_lst=None, **kwargs
    ) -> np.array:
        assert all(x < y for x, y in zip(reservoir_id_lst, reservoir_id_lst[1:]))
        assert all(x < y for x, y in zip(t_range, t_range[1:]))
        print("reading formatted data:")
        t_lst = hydro_time.t_range_days(t_range)
        nt = t_lst.shape[0]
        x = np.empty([len(reservoir_id_lst), nt, len(var_lst)])
        for k in range(len(reservoir_id_lst)):
            data = self.read_forcing_reservoir(
                reservoir_id_lst[k],
                var_lst,
                t_lst,
            )
            x[k, :, :] = data
        return x

    def read_object_ids(self, object_params=None) -> np.array:
        return self.reservoir_sites

    def read_ts_xrdataset(
        self,
        reservoir_id_lst: list = None,
        t_range: list = None,
        var_lst: list = None,
        **kwargs,
    ):
        if var_lst is None:
            return None
        ts = xr.open_dataset(
            self.data_source_description["RESERVOIRS_TIME_SERIES_FILE"]
        )
        all_vars = ts.data_vars
        if any(var not in ts.variables for var in var_lst):
            raise ValueError(f"var_lst must all be in {all_vars}")
        return ts[var_lst].sel(
            Reservoir_ID=reservoir_id_lst, Time=slice(t_range[0], t_range[1])
        )

    def read_attr_xrdataset(self, reservoir_id_lst=None, var_lst=None, **kwargs):
        if var_lst is None or len(var_lst) == 0:
            return None
        attr = xr.open_dataset(self.data_source_description["RESERVOIRS_ATTR_FILE"])
        if "all_number" in list(kwargs.keys()) and kwargs["all_number"]:
            attr_num = map_string_vars(attr)
            reservoir_id_lst = [int(id) for id in reservoir_id_lst]
            return attr_num[var_lst].sel(Reservoir_ID=reservoir_id_lst)
        return attr[var_lst].sel(Reservoir_ID=reservoir_id_lst)

    def read_forcing_ts(
        self,
        reservoir_id_lst: list = None,
        t_range: list = None,
        var_lst: list = None,
        **kwargs,
    ):
        # NetInflow
        netinflow_reservoirs_ds = self.read_ts_xrdataset(
            reservoir_id_lst,
            t_range,
            [var_lst[0]],
        )
        # Storage
        storage_reservoirs_ds = self.read_ts_xrdataset(
            reservoir_id_lst,
            t_range,
            [var_lst[1]],
        )
        # 拼接下 netinflow_reservoirs_ds storage_reservoirs_ds
        data_forcing_all_ds = xr.merge(
            [
                netinflow_reservoirs_ds,
                storage_reservoirs_ds,
            ]
        )
        return data_forcing_all_ds

    def read_release_ts(
        self,
        reservoir_id_lst: list = None,
        t_range: list = None,
        var_lst: list = None,
        **kwargs,
    ):
        data_output_ds = self.read_ts_xrdataset(
            reservoir_id_lst,
            t_range,
            var_lst,
        )
        return data_output_ds

    def read_attr(self, reservoir_id_lst: list = None, var_lst=None, **kwargs):
        data_attr_ds = self.read_attr_xrdataset(
            reservoir_id_lst,
            var_lst,
            all_number=True,
        )
        return data_attr_ds


class Gages(HydroDataset):
    def __init__(self, data_path=os.path.join("gages"), download=False):
        """
        Initialization for CAMELS series dataset

        Parameters
        ----------
        data_path
            where we put the dataset.
            we already set the ROOT directory for hydrodataset,
            so here just set it as a relative path,
            by default "gages"
            download
                if true, download, by default False
        """
        super().__init__(data_path)
        self.data_source_description = self.set_data_source_describe()
        if download:
            self.download_data_source()
        self.gages_sites = self.read_site_info()

    def get_name(self):
        return "GAGES"

    def get_constant_cols(self) -> np.array:
        """all readable attrs in GAGES-II"""
        dir_gage_attr = self.data_source_description["GAGES_ATTR_DIR"]
        var_desc_file = os.path.join(dir_gage_attr, "variable_descriptions.txt")
        var_desc = pd.read_csv(var_desc_file)
        return var_desc["VARIABLE_NAME"].values

    def get_relevant_cols(self):
        return np.array(["dayl", "prcp", "srad", "swe", "tmax", "tmin", "vp"])

    def get_target_cols(self):
        return np.array(["usgsFlow"])

    def get_other_cols(self) -> dict:
        return {
            "FDC": {"time_range": ["1980-01-01", "2000-01-01"], "quantile_num": 100}
        }

    def set_data_source_describe(self):
        """
        the files in the dataset and their location in file system

        Returns
        -------
        collections.OrderedDict
            the description for a GAGES dataset
        """
        gages_db = self.data_source_dir
        gage_region_dir = os.path.join(
            gages_db,
            "boundaries_shapefiles_by_aggeco",
            "boundaries-shapefiles-by-aggeco",
        )
        gages_regions = [
            "bas_ref_all",
            "bas_nonref_CntlPlains",
            "bas_nonref_EastHghlnds",
            "bas_nonref_MxWdShld",
            "bas_nonref_NorthEast",
            "bas_nonref_SECstPlain",
            "bas_nonref_SEPlains",
            "bas_nonref_WestMnts",
            "bas_nonref_WestPlains",
            "bas_nonref_WestXeric",
        ]
        # point shapefile
        gagesii_points_file = os.path.join(
            gages_db, "gagesII_9322_point_shapefile", "gagesII_9322_sept30_2011.shp"
        )

        # config of flow data
        flow_dir = os.path.join(gages_db, "gages_streamflow", "gages_streamflow")
        # forcing
        forcing_dir = os.path.join(gages_db, "basin_mean_forcing", "basin_mean_forcing")
        forcing_types = ["daymet"]
        # attr
        attr_dir = os.path.join(
            gages_db, "basinchar_and_report_sept_2011", "spreadsheets-in-csv-format"
        )
        gauge_id_file = os.path.join(attr_dir, "conterm_basinid.txt")

        download_url_lst = [
            "https://water.usgs.gov/GIS/dsdl/basinchar_and_report_sept_2011.zip",
            "https://water.usgs.gov/GIS/dsdl/gagesII_9322_point_shapefile.zip",
            "https://water.usgs.gov/GIS/dsdl/boundaries_shapefiles_by_aggeco.zip",
            "https://www.sciencebase.gov/catalog/file/get/59692a64e4b0d1f9f05fbd39",
        ]
        usgs_streamflow_url = "https://waterdata.usgs.gov/nwis/dv?cb_00060=on&format=rdb&site_no={}&referred_module=sw&period=&begin_date={}-{}-{}&end_date={}-{}-{}"
        # GAGES-II time series data_source dir
        gagests_dir = os.path.join(gages_db, "59692a64e4b0d1f9f05f")
        population_file = os.path.join(
            gagests_dir,
            "Dataset8_Population-Housing",
            "Dataset8_Population-Housing",
            "PopulationHousing.txt",
        )
        wateruse_file = os.path.join(
            gagests_dir,
            "Dataset10_WaterUse",
            "Dataset10_WaterUse",
            "WaterUse_1985-2010.txt",
        )
        return collections.OrderedDict(
            GAGES_DIR=gages_db,
            GAGES_FLOW_DIR=flow_dir,
            GAGES_FORCING_DIR=forcing_dir,
            GAGES_FORCING_TYPE=forcing_types,
            GAGES_ATTR_DIR=attr_dir,
            GAGES_GAUGE_FILE=gauge_id_file,
            GAGES_DOWNLOAD_URL_LST=download_url_lst,
            GAGES_REGIONS_SHP_DIR=gage_region_dir,
            GAGES_REGION_LIST=gages_regions,
            GAGES_POINT_SHP_FILE=gagesii_points_file,
            GAGES_POPULATION_FILE=population_file,
            GAGES_WATERUSE_FILE=wateruse_file,
            USGS_FLOW_URL=usgs_streamflow_url,
        )

    def read_other_cols(self, object_ids=None, other_cols=None, **kwargs) -> dict:
        # TODO: not finish
        out_dict = {}
        for key, value in other_cols.items():
            if key == "FDC":
                assert "time_range" in value.keys()
                if "quantile_num" in value.keys():
                    quantile_num = value["quantile_num"]
                    out = cal_fdc(
                        self.read_target_cols(
                            object_ids, value["time_range"], "usgsFlow"
                        ),
                        quantile_num=quantile_num,
                    )
                else:
                    out = cal_fdc(
                        self.read_target_cols(
                            object_ids, value["time_range"], "usgsFlow"
                        )
                    )
            else:
                raise NotImplementedError("No this item yet!!")
            out_dict[key] = out
        return out_dict

    def read_attr_all(self, gages_ids: Union[list, np.ndarray]):
        """
        read all attr data for some sites in GAGES-II
        TODO: now it is not same as functions in CAMELS where read_attr_all has no "gages_ids" parameter

        Parameters
        ----------
        gages_ids : Union[list, np.ndarray]
            gages sites' ids

        Returns
        -------
        ndarray
            all attr data for gages_ids
        """
        dir_gage_attr = self.data_source_description["GAGES_ATTR_DIR"]
        f_dict = {}  # factorize dict
        # each key-value pair for atts in a file (list）
        var_dict = {}
        # all attrs
        var_lst = []
        out_lst = []
        # read all attrs
        var_des = pd.read_csv(
            os.path.join(dir_gage_attr, "variable_descriptions.txt"), sep=","
        )
        var_des_map_values = var_des["VARIABLE_TYPE"].tolist()
        for i in range(len(var_des)):
            var_des_map_values[i] = var_des_map_values[i].lower()
        # sort by type
        key_lst = list(set(var_des_map_values))
        key_lst.sort(key=var_des_map_values.index)
        # remove x_region_names
        key_lst.remove("x_region_names")

        for key in key_lst:
            # in "spreadsheets-in-csv-format" directory, the name of "flow_record" file is conterm_flowrec.txt
            if key == "flow_record":
                key = "flowrec"
            data_file = os.path.join(dir_gage_attr, "conterm_" + key + ".txt")
            # remove some unused atttrs in bas_classif
            if key == "bas_classif":
                # https://stackoverflow.com/questions/22216076/unicodedecodeerror-utf8-codec-cant-decode-byte-0xa5-in-position-0-invalid-s
                data_temp = pd.read_csv(
                    data_file,
                    sep=",",
                    dtype={"STAID": str},
                    usecols=range(4),
                    encoding="unicode_escape",
                )
            else:
                data_temp = pd.read_csv(data_file, sep=",", dtype={"STAID": str})
            if key == "flowrec":
                # remove final column which is nan
                data_temp = data_temp.iloc[:, range(data_temp.shape[1] - 1)]
            # all attrs in files
            var_lst_temp = list(data_temp.columns[1:])
            var_dict[key] = var_lst_temp
            var_lst.extend(var_lst_temp)
            k = 0
            n_gage = len(gages_ids)
            out_temp = np.full(
                [n_gage, len(var_lst_temp)], np.nan
            )  # 1d:sites，2d: attrs in current data_file
            # sites intersection，ind2 is the index of sites in conterm_ files，set them in out_temp
            range1 = gages_ids
            range2 = data_temp.iloc[:, 0].astype(str).tolist()
            assert all(x < y for x, y in zip(range2, range2[1:]))
            # Notice the sequence of station ids ! Some id_lst_all are not sorted, so don't use np.intersect1d
            ind2 = [range2.index(tmp) for tmp in range1]
            for field in var_lst_temp:
                if is_string_dtype(data_temp[field]):  # str vars -> categorical vars
                    value, ref = pd.factorize(data_temp.loc[ind2, field], sort=True)
                    out_temp[:, k] = value
                    f_dict[field] = ref.tolist()
                elif is_numeric_dtype(data_temp[field]):
                    out_temp[:, k] = data_temp.loc[ind2, field].values
                k = k + 1
            out_lst.append(out_temp)
        out = np.concatenate(out_lst, 1)
        return out, var_lst, var_dict, f_dict

    def read_constant_cols(
        self, object_ids=None, constant_cols: list = None, **kwargs
    ) -> np.array:
        """
        read some attrs of some sites

        Parameters
        ----------
        object_ids : [type], optional
            sites_ids, by default None
        constant_cols : list, optional
            attrs' names, by default None

        Returns
        -------
        np.array
            attr data for object_ids
        """
        # assert all(x < y for x, y in zip(object_ids, object_ids[1:]))
        attr_all, var_lst_all, var_dict, f_dict = self.read_attr_all(object_ids)
        ind_var = list()
        for var in constant_cols:
            ind_var.append(var_lst_all.index(var))
        out = attr_all[:, ind_var]
        return out

    def read_area(self, gage_id_lst) -> np.array:
        return self.read_attr_xrdataset(gage_id_lst, ["DRAIN_SQKM"])

    def read_mean_prcp(self, gage_id_lst) -> np.array:
        return self.read_attr_xrdataset(gage_id_lst, ["PPTAVG_BASIN"])

    def read_attr_origin(self, gages_ids, attr_lst) -> np.ndarray:
        """
        this function read the attrs data in GAGES-II but not transform them to int when they are str

        Parameters
        ----------
        gages_ids : [type]
            [description]
        attr_lst : [type]
            [description]

        Returns
        -------
        np.ndarray
            the first dim is types of attrs, and the second one is sites
        """
        dir_gage_attr = self.data_source_description["GAGES_ATTR_DIR"]
        var_des = pd.read_csv(
            os.path.join(dir_gage_attr, "variable_descriptions.txt"), sep=","
        )
        var_des_map_values = var_des["VARIABLE_TYPE"].tolist()
        for i in range(len(var_des)):
            var_des_map_values[i] = var_des_map_values[i].lower()
        key_lst = list(set(var_des_map_values))
        key_lst.sort(key=var_des_map_values.index)
        key_lst.remove("x_region_names")
        out_lst = []
        for i in range(len(attr_lst)):
            out_lst.append([])
        range1 = gages_ids
        gage_id_file = self.data_source_description["GAGES_GAUGE_FILE"]
        data_all = pd.read_csv(gage_id_file, sep=",", dtype={0: str})
        range2 = data_all["STAID"].values.tolist()
        assert all(x < y for x, y in zip(range2, range2[1:]))
        # Notice the sequence of station ids ! Some id_lst_all are not sorted, so don't use np.intersect1d
        ind2 = [range2.index(tmp) for tmp in range1]

        for key in key_lst:
            # in "spreadsheets-in-csv-format" directory, the name of "flow_record" file is conterm_flowrec.txt
            if key == "flow_record":
                key = "flowrec"
            data_file = os.path.join(dir_gage_attr, "conterm_" + key + ".txt")
            if key == "bas_classif":
                data_temp = pd.read_csv(
                    data_file,
                    sep=",",
                    dtype={
                        "STAID": str,
                        "WR_REPORT_REMARKS": str,
                        "ADR_CITATION": str,
                        "SCREENING_COMMENTS": str,
                    },
                    engine="python",
                    encoding="unicode_escape",
                )
            elif key == "bound_qa":
                # "DRAIN_SQKM" already exists
                data_temp = pd.read_csv(
                    data_file,
                    sep=",",
                    dtype={"STAID": str},
                    usecols=[
                        "STAID",
                        "BASIN_BOUNDARY_CONFIDENCE",
                        "NWIS_DRAIN_SQKM",
                        "PCT_DIFF_NWIS",
                        "HUC10_CHECK",
                    ],
                )
            else:
                data_temp = pd.read_csv(data_file, sep=",", dtype={"STAID": str})
            if key == "flowrec":
                data_temp = data_temp.iloc[:, range(0, data_temp.shape[1] - 1)]
            var_lst_temp = list(data_temp.columns[1:])
            do_exist, idx_lst = hydro_arithmetric.is_any_elem_in_a_lst(
                attr_lst, var_lst_temp, return_index=True
            )
            if do_exist:
                for idx in idx_lst:
                    idx_in_var = (
                        var_lst_temp.index(attr_lst[idx]) + 1
                    )  # +1 because the first col of data_temp is ID
                    out_lst[idx] = data_temp.iloc[ind2, idx_in_var].values
            else:
                continue
        out = np.array(out_lst)
        return out

    def read_forcing_gage(self, usgs_id, var_lst, t_range_list, forcing_type="daymet"):
        gage_dict = self.gages_sites
        ind = np.argwhere(gage_dict["STAID"] == usgs_id)[0][0]
        huc = gage_dict["HUC02"][ind]

        data_folder = os.path.join(
            self.data_source_description["GAGES_FORCING_DIR"], forcing_type
        )
        # original daymet file not for leap year, there is no data in 12.31 in leap year,
        # so files which have been interpolated for nan value have name "_leap"
        data_file = os.path.join(
            data_folder, huc, f"{usgs_id}_lump_{forcing_type}_forcing_leap.txt"
        )
        print("reading", forcing_type, "forcing data ", usgs_id)
        data_temp = pd.read_csv(data_file, sep=r"\s+", header=None, skiprows=1)

        df_date = data_temp[[0, 1, 2]]
        df_date.columns = ["year", "month", "day"]
        date = pd.to_datetime(df_date).values.astype("datetime64[D]")
        nf = len(var_lst)
        assert all(x < y for x, y in zip(date, date[1:]))
        [c, ind1, ind2] = np.intersect1d(date, t_range_list, return_indices=True)
        assert date[0] <= t_range_list[0] and date[-1] >= t_range_list[-1]
        nt = t_range_list.size
        out = np.empty([nt, nf])
        var_lst_in_file = [
            "dayl(s)",
            "prcp(mm/day)",
            "srad(W/m2)",
            "swe(mm)",
            "tmax(C)",
            "tmin(C)",
            "vp(Pa)",
        ]
        for k in range(nf):
            # assume all files are of same columns. May check later.
            ind = [
                i
                for i in range(len(var_lst_in_file))
                if var_lst[k] in var_lst_in_file[i]
            ][0]
            out[ind2, k] = data_temp[ind + 4].values[ind1]
        return out

    def read_relevant_cols(
        self, gage_id_lst=None, t_range=None, var_lst=None, **kwargs
    ) -> np.array:
        assert all(x < y for x, y in zip(gage_id_lst, gage_id_lst[1:]))
        assert all(x < y for x, y in zip(t_range, t_range[1:]))
        print("reading formatted data:")
        t_lst = hydro_time.t_range_days(t_range)
        nt = t_lst.shape[0]
        x = np.empty([len(gage_id_lst), nt, len(var_lst)])
        for k in range(len(gage_id_lst)):
            data = self.read_forcing_gage(
                gage_id_lst[k],
                var_lst,
                t_lst,
                forcing_type=self.data_source_description["GAGES_FORCING_TYPE"][0],
            )
            x[k, :, :] = data
        return x

    def read_target_cols(
        self, gage_id_lst=None, t_range=None, target_cols=None, **kwargs
    ) -> np.array:
        """
        Read USGS daily average streamflow data according to id and time

        Parameters
        ----------
        usgs_id_lst
            site information
        t_range_list
            must be time range for downloaded data
        target_cols

        kwargs
            optional

        Returns
        -------
        np.array
            streamflow data, 1d-axis: gages, 2d-axis: day, 3d-axis: streamflow
        """
        t_lst = hydro_time.t_range_days(t_range)
        nt = t_lst.shape[0]
        y = np.empty([len(gage_id_lst), nt, 1])
        for k in range(len(gage_id_lst)):
            data_obs = self.read_usgs_gage(gage_id_lst[k], t_lst)
            y[k, :, 0] = data_obs
        return y

    def read_usgs_gage(self, usgs_id, t_lst):
        """
        read data for one gage

        Parameters
        ----------
        usgs_id : [type]
            [description]
        t_lst : [type]
            [description]

        Returns
        -------
        [type]
            [description]
        """
        print(usgs_id)
        dir_gage_flow = self.data_source_description["GAGES_FLOW_DIR"]
        gage_id_df = pd.DataFrame(self.gages_sites)
        huc = gage_id_df[gage_id_df["STAID"] == usgs_id]["HUC02"].values[0]
        usgs_file = os.path.join(dir_gage_flow, str(huc), usgs_id + ".txt")
        # ignore the comment lines and the first non-value row
        df_flow = pd.read_csv(
            usgs_file, comment="#", sep="\t", dtype={"site_no": str}
        ).iloc[1:, :]
        # change the original column names
        columns_names = df_flow.columns.tolist()
        columns_flow = [
            column_name
            for column_name in columns_names
            if "_00060_00003" in column_name and "_00060_00003_cd" not in column_name
        ]
        columns_flow_cd = [
            column_name
            for column_name in columns_names
            if "_00060_00003_cd" in column_name
        ]
        if len(columns_flow) > 1:
            self._format_flow_data(df_flow, t_lst, columns_flow, columns_flow_cd)
        else:
            for column_name in columns_names:
                if (
                    "_00060_00003" in column_name
                    and "_00060_00003_cd" not in column_name
                ):
                    df_flow.rename(columns={column_name: "flow"}, inplace=True)
                    break
            for column_name in columns_names:
                if "_00060_00003_cd" in column_name:
                    df_flow.rename(columns={column_name: "mode"}, inplace=True)
                    break

        columns = ["agency_cd", "site_no", "datetime", "flow", "mode"]
        if df_flow.empty:
            df_flow = pd.DataFrame(columns=columns)
        if "flow" not in df_flow.columns.intersection(columns):
            data_temp = df_flow.loc[:, df_flow.columns.intersection(columns)]
            # add nan column to data_temp
            data_temp = pd.concat([data_temp, pd.DataFrame(columns=["flow", "mode"])])
        else:
            data_temp = df_flow.loc[:, columns]
        self._check_flow_data(data_temp, "flow")
        # set negative value -- nan
        obs = data_temp["flow"].astype("float").values
        obs[obs < 0] = np.nan
        # time range intersection. set points without data nan values
        nt = len(t_lst)
        out = np.full([nt], np.nan)
        # date in df is str，so transform them to datetime
        df_date = data_temp["datetime"]
        date = pd.to_datetime(df_date).values.astype("datetime64[D]")
        c, ind1, ind2 = np.intersect1d(date, t_lst, return_indices=True)
        out[ind2] = obs[ind1]
        return out

    def _format_flow_data(self, df_flow, t_lst, columns_flow, columns_flow_cd):
        print("there are some columns for flow, choose one\n")
        df_date_temp = df_flow["datetime"]
        date_temp = pd.to_datetime(df_date_temp).values.astype("datetime64[D]")
        c_temp, ind1_temp, ind2_temp = np.intersect1d(
            date_temp, t_lst, return_indices=True
        )
        num_nan_lst = []
        for item in columns_flow:
            out_temp = np.full([len(t_lst)], np.nan)

            self._check_flow_data(df_flow, item)
            df_flow_temp = df_flow[item].copy().values
            out_temp[ind2_temp] = df_flow_temp[ind1_temp]
            num_nan = np.isnan(out_temp).sum()
            num_nan_lst.append(num_nan)
        num_nan_np = np.array(num_nan_lst)
        index_flow_num = np.argmin(num_nan_np)
        df_flow.rename(columns={columns_flow[index_flow_num]: "flow"}, inplace=True)
        df_flow.rename(columns={columns_flow_cd[index_flow_num]: "mode"}, inplace=True)

    def _check_flow_data(self, arg0, arg1):
        arg0.loc[arg0[arg1] == "Ice", arg1] = np.nan
        arg0.loc[arg0[arg1] == "Ssn", arg1] = np.nan
        arg0.loc[arg0[arg1] == "Tst", arg1] = np.nan
        arg0.loc[arg0[arg1] == "Eqp", arg1] = np.nan
        arg0.loc[arg0[arg1] == "Rat", arg1] = np.nan
        arg0.loc[arg0[arg1] == "Dis", arg1] = np.nan
        arg0.loc[arg0[arg1] == "Bkw", arg1] = np.nan
        arg0.loc[arg0[arg1] == "***", arg1] = np.nan
        arg0.loc[arg0[arg1] == "Mnt", arg1] = np.nan
        arg0.loc[arg0[arg1] == "ZFL", arg1] = np.nan

    def read_object_ids(self, object_params=None) -> np.array:
        return self.gages_sites["STAID"]

    def read_basin_area(self, object_ids) -> np.array:
        return self.read_constant_cols(object_ids, ["DRAIN_SQKM"], is_return_dict=False)

    def read_mean_prep(self, object_ids) -> np.array:
        mean_prep = self.read_constant_cols(
            object_ids, ["PPTAVG_BASIN"], is_return_dict=False
        )
        mean_prep = mean_prep / 365 * 10
        return mean_prep

    def download_data_source(self):
        print("Please download data manually!")
        if not os.path.isdir(self.data_source_description["GAGES_DIR"]):
            os.makedirs(self.data_source_description["GAGES_DIR"])
        zip_files = [
            "59692a64e4b0d1f9f05fbd39",
            "basin_mean_forcing.zip",
            "basinchar_and_report_sept_2011.zip",
            "boundaries_shapefiles_by_aggeco.zip",
            "gages_streamflow.zip",
            "gagesII_9322_point_shapefile.zip",
        ]
        download_zip_files = [
            os.path.join(self.data_source_description["GAGES_DIR"], zip_file)
            for zip_file in zip_files
        ]
        for download_zip_file in download_zip_files:
            if not os.path.isfile(download_zip_file):
                raise RuntimeError(
                    download_zip_file + " not found! Please download the data"
                )
        unzip_dirs = [
            os.path.join(self.data_source_description["GAGES_DIR"], zip_file[:-4])
            for zip_file in zip_files
        ]
        for i in range(len(unzip_dirs)):
            if not os.path.isdir(unzip_dirs[i]):
                print("unzip directory:" + unzip_dirs[i])
                hydro_file.unzip_nested_zip(download_zip_files[i], unzip_dirs[i])
            else:
                print("unzip directory -- " + unzip_dirs[i] + " has existed")

    def read_site_info(self):
        gage_id_file = self.data_source_description["GAGES_GAUGE_FILE"]
        data_all = pd.read_csv(gage_id_file, sep=",", dtype={0: str})
        gage_fld_lst = data_all.columns.values
        out = {}
        df_id_region = data_all.iloc[:, 0].values
        assert all(x < y for x, y in zip(df_id_region, df_id_region[1:]))
        for s in gage_fld_lst:
            if s is gage_fld_lst[1]:
                out[s] = data_all[s].values.tolist()
            else:
                out[s] = data_all[s].values
        return out

    def cache_forcing_np_json(self):
        """
        Save all daymet basin-forcing data in a numpy array file in the cache directory.

        Because it takes much time to read data from txt files,
        it is a good way to cache data as a numpy file to speed up the reading.

        """
        cache_npy_file = CACHE_DIR.joinpath("gages_daymet_forcing.npy")
        json_file = CACHE_DIR.joinpath("gages_daymet_forcing.json")
        variables = self.get_relevant_cols()
        basins = self.gages_sites["STAID"]
        daymet_t_range = ["1980-01-01", "2019-12-31"]
        times = [
            hydro_time.t2str(tmp)
            for tmp in hydro_time.t_range_days(daymet_t_range).tolist()
        ]
        data_info = collections.OrderedDict(
            {
                "dim": ["basin", "time", "variable"],
                "basin": basins.tolist(),
                "time": times,
                "variable": variables.tolist(),
            }
        )
        with open(json_file, "w") as FP:
            json.dump(data_info, FP, indent=4)
        data = self.read_relevant_cols(
            gage_id_lst=basins.tolist(),
            t_range=daymet_t_range,
            var_lst=variables.tolist(),
        )
        np.save(cache_npy_file, data)

    def cache_streamflow_np_json(self):
        """
        Save all basins' streamflow data in a numpy array file in the cache directory
        """
        cache_npy_file = CACHE_DIR.joinpath("gages_streamflow.npy")
        json_file = CACHE_DIR.joinpath("gages_streamflow.json")
        variables = self.get_target_cols()
        basins = self.gages_sites["STAID"]
        t_range = ["1980-01-01", "2020-01-01"]
        times = [
            hydro_time.t2str(tmp) for tmp in hydro_time.t_range_days(t_range).tolist()
        ]
        data_info = collections.OrderedDict(
            {
                "dim": ["basin", "time", "variable"],
                "basin": basins.tolist(),
                "time": times,
                "variable": variables.tolist(),
            }
        )
        with open(json_file, "w") as FP:
            json.dump(data_info, FP, indent=4)
        data = self.read_target_cols(
            gage_id_lst=basins,
            t_range=t_range,
            target_cols=variables,
        )
        np.save(cache_npy_file, data)

    def cache_attributes_xrdataset(self):
        """Convert all the attributes to a single dataframe

        Returns
        -------
        None
        """
        # dir_gage_attr = self.data_source_description["GAGES_ATTR_DIR"]
        # attrs_lst = []
        # # read all attrs
        # var_des = pd.read_csv(
        #     os.path.join(dir_gage_attr, "variable_descriptions.txt"), sep=","
        # )
        # var_des_map_values = var_des["VARIABLE_TYPE"].tolist()
        # for i in range(len(var_des)):
        #     var_des_map_values[i] = var_des_map_values[i].lower()
        # # sort by type
        # key_lst = list(set(var_des_map_values))
        # key_lst.sort(key=var_des_map_values.index)
        # # remove x_region_names
        # key_lst.remove("x_region_names")
        # for key in key_lst:
        #     if key == "flow_record":
        #         key = "flowrec"
        #     attr_file = os.path.join(dir_gage_attr, "conterm_" + key + ".txt")
        #     # remove some unused atttrs in bas_classif
        #     if key == "bas_classif":
        #         # https://stackoverflow.com/questions/22216076/unicodedecodeerror-utf8-codec-cant-decode-byte-0xa5-in-position-0-invalid-s
        #         attrs = pd.read_csv(
        #             attr_file,
        #             sep=",",
        #             dtype={"STAID": str},
        #             usecols=range(4),
        #             encoding="unicode_escape",
        #             index_col=0,
        #         )
        #     else:
        #         attrs = pd.read_csv(attr_file, sep=",", dtype={"STAID": str}, index_col=0)
        #     if key == "flowrec":
        #         # remove final column which is nan
        #         attrs = attrs.iloc[:, range(attrs.shape[1] - 1)]
        #     attrs_lst.append(attrs)
        # attrs_df = np.concatenate(attrs_lst, 1)
        gages_id = self.read_site_info()["STAID"]
        attrs, var_lst, var_dict, f_dict = self.read_attr_all(gages_id)
        # unify id to basin
        attrs_df = pd.DataFrame(attrs, columns=var_lst)
        attrs_df["basin"] = gages_id
        attrs_df.set_index("basin", inplace=True)
        attrs_df = attrs_df.loc[:, ~attrs_df.columns.duplicated()]
        # attrs_df.key.name = "basin"
        # We use xarray dataset to cache all data
        duplicates = attrs_df.columns[attrs_df.columns.duplicated()].unique()
        print("Duplicate column names:", duplicates)
        ds_from_df = attrs_df.to_xarray()
        units_dict = {
            "STANAME": "dimensionless",
            "DRAIN_SQKM": "km**2",
            "HUC02": "dimensionless",
            "LAT_GAGE": "degree",
            "LNG_GAGE": "degree",
            "LAT_CENT": "degree",
            "LONG_CENT": "degree",
            "PPTAVG_BASIN": "cm",
            "PPTAVG_SITE": "cm",
            "T_AVG_BASIN": "degC",
            "T_AVG_SITE": "degC",
            "T_MAX_BASIN": "degC",
            "T_MAXSTD_BASIN": "degC",
            "T_MAX_SITE": "degC",
            "T_MIN_BASIN": "degC",
            "T_MINSTD_BASIN": "degC",
            "T_MIN_SITE": "degC",
            "RH_BASIN": "percent",
            "RH_SITE": "percent",
            "FST32F_BASIN": "day",
            "LST32F_BASIN": "day",
            "FST32F_SITE": "day",
            "LST32F_SITE": "day",
            "PET": "mm/year",
            "SNOW_PCT_PRECIP": "percent",
            "PRECIP_SEAS_IND": "dimensionless",
            "JAN_PPT7100_CM": "cm",
            "FEB_PPT7100_CM": "cm",
            "MAR_PPT7100_CM": "cm",
            "APR_PPT7100_CM": "cm",
            "MAY_PPT7100_CM": "cm",
            "JUN_PPT7100_CM": "cm",
            "JUL_PPT7100_CM": "cm",
            "AUG_PPT7100_CM": "cm",
            "SEP_PPT7100_CM": "cm",
            "OCT_PPT7100_CM": "cm",
            "NOV_PPT7100_CM": "cm",
            "DEC_PPT7100_CM": "cm",
            "JAN_TMP7100_DEGC": "degC",
            "FEB_TMP7100_DEGC": "degC",
            "MAR_TMP7100_DEGC": "degC",
            "APR_TMP7100_DEGC": "degC",
            "MAY_TMP7100_DEGC": "degC",
            "JUN_TMP7100_DEGC": "degC",
            "JUL_TMP7100_DEGC": "degC",
            "AUG_TMP7100_DEGC": "degC",
            "SEP_TMP7100_DEGC": "degC",
            "OCT_TMP7100_DEGC": "degC",
            "NOV_TMP7100_DEGC": "degC",
            "DEC_TMP7100_DEGC": "degC",
            "PPT1950_AVG thru PPT2009_avg (60 values)": "cm",
            "TMP1950_AVG thru TMP2009_avg (60 values)": "degC",
            "GEOL_REEDBUSH_DOM": "dimensionless",
            "GEOL_REEDBUSH_DOM_PCT": "percent",
            "GEOL_REEDBUSH_SITE": "dimensionless",
            "STREAMS_KM_SQ_KM": "km/km**2",
            "STRAHLER_MAX": "dimensionless",
            "MAINSTEM_SINUOUSITY": "dimensionless",
            "ARTIFPATH_PCT": "percent",
            "ARTIFPATH_MAINSTEM_PCT": "percent",
            "HIRES_LENTIC_PCT": "percent",
            "BFI_AVE": "percent",
            "PERDUN": "percent",
            "PERHOR": "percent",
            "TOPWET": "dimensionless",
            "CONTACT": "days",
            "RUNAVE7100": "mm/year",
            "WB5100_JAN_MM": "mm/month",
            "WB5100_FEB_MM": "mm/month",
            "WB5100_MAR_MM": "mm/month",
            "WB5100_APR_MM": "mm/month",
            "WB5100_MAY_MM": "mm/month",
            "WB5100_JUN_MM": "mm/month",
            "WB5100_JUL_MM": "mm/month",
            "WB5100_AUG_MM": "mm/month",
            "WB5100_SEP_MM": "mm/month",
            "WB5100_OCT_MM": "mm/month",
            "WB5100_NOV_MM": "mm/month",
            "WB5100_DEC_MM": "mm/month",
            "WB5100_ANN_MM": "mm/year",
            "PCT_1ST_ORDER": "percent",
            "PCT_2ND_ORDER": "percent",
            "PCT_3RD_ORDER": "percent",
            "PCT_4TH_ORDER": "percent",
            "PCT_5TH_ORDER": "percent",
            "PCT_6TH_ORDER_OR_MORE": "percent",
            "PCT_NO_ORDER": "percent",
            "NDAMS_2009": "dimensionless",
            "DDENS_2009": "dimensionless/km**2",
            "STOR_NID_2009": "megaliters/km**2",
            "STOR_NOR_2009": "megaliters/km**2",
            "MAJ_NDAMS_2009": "dimensionless",
            "MAJ_DDENS_2009": "dimensionless/km**2",
            "pre1940_NDAMS": "dimensionless",
            "pre1950_NDAMS": "dimensionless",
            "pre1960_NDAMS": "dimensionless",
            "pre1970_NDAMS": "dimensionless",
            "pre1980_NDAMS": "dimensionless",
            "pre1990_NDAMS": "dimensionless",
            "pre1940_DDENS": "dimensionless/km**2",
            "pre1950_DDENS": "dimensionless/km**2",
            "pre1960_DDENS": "dimensionless/km**2",
            "pre1970_DDENS": "dimensionless/km**2",
            "pre1980_DDENS": "dimensionless/km**2",
            "pre1990_DDENS": "dimensionless/km**2",
            "pre1940_STOR": "megaliters/km**2",
            "pre1950_STOR": "megaliters/km**2",
            "pre1960_STOR": "megaliters/km**2",
            "pre1970_STOR": "megaliters/km**2",
            "pre1980_STOR": "megaliters/km**2",
            "pre1990_STOR": "megaliters/km**2",
            "RAW_DIS_NEAREST_DAM": "km",
            "RAW_AVG_DIS_ALLDAMS": "km",
            "RAW_DIS_NEAREST_MAJ_DAM": "km",
            "RAW_AVG_DIS_ALL_MAJ_DAMS": "km",
            "CANALS_PCT": "percent",
            "RAW_DIS_NEAREST_CANAL": "km",
            "RAW_AVG_DIS_ALLCANALS": "km",
            "CANALS_MAINSTEM_PCT": "percent",
            "NPDES_MAJ_DENS": "dimensionless/km**2",
            "RAW_DIS_NEAREST_MAJ_NPDES": "km",
            "RAW_AVG_DIS_ALL_MAJ_NPDES": "km",
            "FRESHW_WITHDRAWAL": "megaliters/year/km**2",
            "MINING92_PCT": "percent",
            "PCT_IRRIG_AG": "percent",
            "POWER_NUM_PTS": "dimensionless",
            "POWER_SUM_MW": "MW",
            "FRAGUN_BASIN": "dimensionless",
            "HIRES_LENTIC_NUM": "dimensionless",
            "HIRES_LENTIC_DENS": "dimensionless/km**2",
            "HIRES_LENTIC_MEANSIZ": "hectares",
            "DEVNLCD06": "percent",
            "FORESTNLCD06": "percent",
            "PLANTNLCD06": "percent",
            "WATERNLCD06": "percent",
            "SNOWICENLCD06": "percent",
            "DEVOPENNLCD06": "percent",
            "DEVLOWNLCD06": "percent",
            "DEVMEDNLCD06": "percent",
            "DEVHINLCD06": "percent",
            "BARRENNLCD06": "percent",
            "DECIDNLCD06": "percent",
            "EVERGRNLCD06": "percent",
            "MIXEDFORNLCD06": "percent",
            "SHRUBNLCD06": "percent",
            "GRASSNLCD06": "percent",
            "PASTURENLCD06": "percent",
            "CROPSNLCD06": "percent",
            "WOODYWETNLCD06": "percent",
            "EMERGWETNLCD06": "percent",
            "MAINS100_DEV": "percent",
            "MAINS100_FOREST": "percent",
            "MAINS100_PLANT": "percent",
            "MAINS100_11": "percent",
            "MAINS100_12": "percent",
            "MAINS100_21": "percent",
            "MAINS100_22": "percent",
            "MAINS100_23": "percent",
            "MAINS100_24": "percent",
            "MAINS100_31": "percent",
            "MAINS100_41": "percent",
            "MAINS100_42": "percent",
            "MAINS100_43": "percent",
            "MAINS100_52": "percent",
            "MAINS100_71": "percent",
            "MAINS100_81": "percent",
            "MAINS100_82": "percent",
            "MAINS100_90": "percent",
            "MAINS100_95": "percent",
            "MAINS800_DEV": "percent",
            "MAINS800_FOREST": "percent",
            "MAINS800_PLANT": "percent",
            "MAINS800_11": "percent",
            "MAINS800_12": "percent",
            "MAINS800_21": "percent",
            "MAINS800_22": "percent",
            "MAINS800_23": "percent",
            "MAINS800_24": "percent",
            "MAINS800_31": "percent",
            "MAINS800_41": "percent",
            "MAINS800_42": "percent",
            "MAINS800_43": "percent",
            "MAINS800_52": "percent",
            "MAINS800_71": "percent",
            "MAINS800_81": "percent",
            "MAINS800_82": "percent",
            "MAINS800_90": "percent",
            "MAINS800_95": "percent",
            "RIP100_DEV": "percent",
            "RIP100_FOREST": "percent",
            "RIP100_PLANT": "percent",
            "RIP100_11": "percent",
            "RIP100_12": "percent",
            "RIP100_21": "percent",
            "RIP100_22": "percent",
            "RIP100_23": "percent",
            "RIP100_24": "percent",
            "RIP100_31": "percent",
            "RIP100_41": "percent",
            "RIP100_42": "percent",
            "RIP100_43": "percent",
            "RIP100_52": "percent",
            "RIP100_71": "percent",
            "RIP100_81": "percent",
            "RIP100_82": "percent",
            "RIP100_90": "percent",
            "RIP100_95": "percent",
            "RIP800_DEV": "percent",
            "RIP800_FOREST": "percent",
            "RIP800_PLANT": "percent",
            "RIP800_11": "percent",
            "RIP800_12": "percent",
            "RIP800_21": "percent",
            "RIP800_22": "percent",
            "RIP800_23": "percent",
            "RIP800_24": "percent",
            "RIP800_31": "percent",
            "RIP800_41": "percent",
            "RIP800_42": "percent",
            "RIP800_43": "percent",
            "RIP800_52": "percent",
            "RIP800_71": "percent",
            "RIP800_81": "percent",
            "RIP800_82": "percent",
            "RIP800_90": "percent",
            "RIP800_95": "percent",
            "CDL_CORN": "percent",
            "CDL_COTTON": "percent",
            "CDL_RICE": "percent",
            "CDL_SORGHUM": "percent",
            "CDL_SOYBEANS": "percent",
            "CDL_SUNFLOWERS": "percent",
            "CDL_PEANUTS": "percent",
            "CDL_BARLEY": "percent",
            "CDL_DURUM_WHEAT": "percent",
            "CDL_SPRING_WHEAT": "percent",
            "CDL_WINTER_WHEAT": "percent",
            "CDL_WWHT_SOY_DBL_CROP": "percent",
            "CDL_OATS": "percent",
            "CDL_ALFALFA": "percent",
            "CDL_OTHER_HAYS": "percent",
            "CDL_DRY_BEANS": "percent",
            "CDL_POTATOES": "percent",
            "CDL_FALLOW_IDLE": "percent",
            "CDL_PASTURE_GRASS": "percent",
            "CDL_ORANGES": "percent",
            "CDL_OTHER_CROPS": "percent",
            "CDL_ALL_OTHER_LAND": "percent",
            "PDEN_2000_BLOCK": "persons/km**2",
            "PDEN_DAY_LANDSCAN_2007": "persons/km**2",
            "PDEN_NIGHT_LANDSCAN_2007": "persons/km**2",
            "ROADS_KM_SQ_KM": "km/km**2",
            "RD_STR_INTERS": "dimensionless/km",
            "IMPNLCD06": "percent",
            "NLCD01_06_DEV": "percent",
            "PADCAT1_PCT_BASIN": "percent",
            "PADCAT2_PCT_BASIN": "percent",
            "PADCAT3_PCT_BASIN": "percent",
            "ECO3_BAS_PCT": "percent",
            "NUTR_BAS_PCT": "percent",
            "HLR_BAS_PCT_100M": "percent",
            "PNV_BAS_PCT": "percent",
            "HGA": "percent",
            "HGB": "percent",
            "HGAD": "percent",
            "HGC": "percent",
            "HGD": "percent",
            "HGAC": "percent",
            "HGBD": "percent",
            "HGCD": "percent",
            "HGBC": "percent",
            "HGVAR": "percent",
            "AWCAVE": "dimensionless",
            "PERMAVE": "inch/hour",
            "BDAVE": "g/cm**3",
            "OMAVE": "percent",
            "WTDEPAVE": "feet",
            "ROCKDEPAVE": "inches",
            "CLAYAVE": "percent",
            "SILTAVE": "percent",
            "SANDAVE": "percent",
            "KFACT_UP": "dimensionless",
            "RFACT": "ft*tonf*inch/h/ac/year",
            "ELEV_MEAN_M_BASIN": "meter",
            "ELEV_MAX_M_BASIN": "meter",
            "ELEV_MIN_M_BASIN": "meter",
            "ELEV_MEDIAN_M_BASIN": "meter",
            "ELEV_STD_M_BASIN": "meter",
            "ELEV_SITE_M": "meter",
            "RRMEAN": "dimensionless",
            "RRMEDIAN": "dimensionless",
            "SLOPE_PCT": "percent",
            "ASPECT_DEGREES": "degree",
            "ASPECT_NORTHNESS": "dimensionless",
            "ASPECT_EASTNESS": "dimensionless",
        }

        # Assign units to the variables in the Dataset
        for var_name in units_dict:
            if var_name in ds_from_df.data_vars:
                ds_from_df[var_name].attrs["units"] = units_dict[var_name]

        # Assign categorical mappings to the variables in the Dataset
        for column in f_dict:
            mapping_str = f_dict[column]
            ds_from_df[column].attrs["category_mapping"] = str(mapping_str)
        return ds_from_df

    def cache_streamflow_xrdataset(self):
        """Save all basins' streamflow data in a netcdf file in the cache directory"""
        cache_npy_file = CACHE_DIR.joinpath("gages_streamflow.npy")
        json_file = CACHE_DIR.joinpath("gages_streamflow.json")
        if (not os.path.isfile(cache_npy_file)) or (not os.path.isfile(json_file)):
            self.cache_streamflow_np_json()
        streamflow = np.load(cache_npy_file)
        with open(json_file, "r") as fp:
            streamflow_dict = json.load(fp, object_pairs_hook=collections.OrderedDict)

        basins = streamflow_dict["basin"]
        times = pd.date_range(
            streamflow_dict["time"][0], periods=len(streamflow_dict["time"])
        )
        return xr.Dataset(
            {
                "streamflow": (
                    ["basin", "time"],
                    streamflow.reshape(streamflow.shape[0], streamflow.shape[1]),
                    {"units": self.streamflow_unit},
                )
            },
            coords={
                "basin": basins,
                "time": times,
            },
        )

    def cache_forcing_xrdataset(self):
        """Save all daymet basin-forcing data in a netcdf file in the cache directory."""
        cache_npy_file = CACHE_DIR.joinpath("gages_daymet_forcing.npy")
        json_file = CACHE_DIR.joinpath("gages_daymet_forcing.json")
        if (not os.path.isfile(cache_npy_file)) or (not os.path.isfile(json_file)):
            self.cache_forcing_np_json()
        daymet_forcing = np.load(cache_npy_file)
        with open(json_file, "r") as fp:
            daymet_forcing_dict = json.load(
                fp, object_pairs_hook=collections.OrderedDict
            )

        basins = daymet_forcing_dict["basin"]
        times = pd.date_range(
            daymet_forcing_dict["time"][0], periods=len(daymet_forcing_dict["time"])
        )
        variables = daymet_forcing_dict["variable"]
        # All units' names are from Pint https://github.com/hgrecco/pint/blob/master/pint/default_en.txt
        # final is PET's unit. PET comes from the model output of CAMELS-US
        units = ["s", "mm/day", "W/m^2", "mm", "°C", "°C", "Pa"]
        return xr.Dataset(
            data_vars={
                **{
                    variables[i]: (
                        ["basin", "time"],
                        daymet_forcing[:, :, i],
                        {"units": units[i]},
                    )
                    for i in range(len(variables))
                }
            },
            coords={
                "basin": basins,
                "time": times,
            },
            attrs={"forcing_type": "daymet"},
        )

    def cache_xrdataset(self):
        """Save all data in a netcdf file in the cache directory"""
        warnings.warn("Check you units of all variables")
        ds_attr = self.cache_attributes_xrdataset()
        ds_attr.to_netcdf(CACHE_DIR.joinpath("gages_attributes.nc"))
        ds_streamflow = self.cache_streamflow_xrdataset()
        ds_forcing = self.cache_forcing_xrdataset()
        ds = xr.merge([ds_streamflow, ds_forcing])
        ds.to_netcdf(CACHE_DIR.joinpath("gages_timeseries.nc"))

    def read_ts_xrdataset(
        self,
        gage_id_lst: list = None,
        t_range: list = None,
        var_lst: list = None,
        **kwargs,
    ):
        if var_lst is None:
            return None
        ts = xr.open_dataset(CACHE_DIR.joinpath("gages_timeseries.nc"))
        all_vars = ts.data_vars
        if any(var not in ts.variables for var in var_lst):
            raise ValueError(f"var_lst must all be in {all_vars}")
        return ts[var_lst].sel(basin=gage_id_lst, time=slice(t_range[0], t_range[1]))

    def read_attr_xrdataset(self, gage_id_lst=None, var_lst=None, **kwargs):
        if var_lst is None or len(var_lst) == 0:
            return None
        try:
            attr = xr.open_dataset(CACHE_DIR.joinpath("gages_attributes.nc"))
        except FileNotFoundError:
            attr = self.cache_attributes_xrdataset()
            attr.to_netcdf(CACHE_DIR.joinpath("gages_attributes.nc"))
        if "all_number" in list(kwargs.keys()) and kwargs["all_number"]:
            attr_num = map_string_vars(attr)
            return attr_num[var_lst].sel(basin=gage_id_lst)
        return attr[var_lst].sel(basin=gage_id_lst)

    @property
    def streamflow_unit(self):
        return "foot^3/s"


def map_string_vars(ds):
    # Iterate over all variables in the dataset
    for var in ds.data_vars:
        # Check if the variable contains string data
        if ds[var].dtype == object:
            # Convert the DataArray to a pandas Series
            var_series = ds[var].to_series()

            # Get all unique strings and create a mapping to integers
            unique_strings = sorted(var_series.unique())
            mapping = {value: i for i, value in enumerate(unique_strings)}

            # Apply the mapping to the series
            mapped_series = var_series.map(mapping)

            # Convert the series back to a DataArray and replace the old one in the Dataset
            ds[var] = xr.DataArray(mapped_series)

    return ds


def prepare_usgs_data(
    data_source_description: Dict, t_download_range: Union[tuple, list]
):
    hydro_logger.info("NOT all data_source could be downloaded from website directly!")
    # download zip files
    [
        hydro_file.download_one_zip(attr_url, data_source_description["GAGES_DIR"])
        for attr_url in data_source_description["GAGES_DOWNLOAD_URL_LST"]
    ]
    # download streamflow data from USGS website
    dir_gage_flow = data_source_description["GAGES_FLOW_DIR"]
    streamflow_url = data_source_description["USGS_FLOW_URL"]
    if not os.path.isdir(dir_gage_flow):
        os.makedirs(dir_gage_flow)
    dir_list = os.listdir(dir_gage_flow)
    # if no streamflow data for the usgs_id_lst, then download them from the USGS website
    data_all = pd.read_csv(
        data_source_description["GAGES_GAUGE_FILE"], sep=",", dtype={0: str}
    )
    usgs_id_lst = data_all.iloc[:, 0].values.tolist()
    gage_fld_lst = data_all.columns.values
    for ind in range(len(usgs_id_lst)):  # different hucs different directories
        huc_02 = data_all[gage_fld_lst[3]][ind]
        dir_huc_02 = str(huc_02)
        if dir_huc_02 not in dir_list:
            dir_huc_02 = os.path.join(dir_gage_flow, str(huc_02))
            os.mkdir(dir_huc_02)
            dir_list = os.listdir(dir_gage_flow)
        dir_huc_02 = os.path.join(dir_gage_flow, str(huc_02))
        file_list = os.listdir(dir_huc_02)
        file_usgs_id = f"{str(usgs_id_lst[ind])}.txt"
        if file_usgs_id not in file_list:
            # download data and save as txt file
            start_time_str = datetime.strptime(t_download_range[0], "%Y-%m-%d")
            end_time_str = datetime.strptime(
                t_download_range[1], "%Y-%m-%d"
            ) - timedelta(days=1)
            url = streamflow_url.format(
                usgs_id_lst[ind],
                start_time_str.year,
                start_time_str.month,
                start_time_str.day,
                end_time_str.year,
                end_time_str.month,
                end_time_str.day,
            )

            # save in its HUC02 dir
            temp_file = os.path.join(dir_huc_02, f"{str(usgs_id_lst[ind])}.txt")
            hydro_file.download_small_file(url, temp_file)
            print("successfully download " + temp_file + " streamflow data！")


def get_dor_values(gages: Gages, usgs_id) -> np.array:
    """
    get dor values from gages for the usgs_id-sites

    """

    assert all(x < y for x, y in zip(usgs_id, usgs_id[1:]))
    # mm/year 1-km grid,  megaliters total storage per sq km  (1 megaliters = 1,000,000 liters = 1,000 cubic meters)
    # attr_lst = ["RUNAVE7100", "STOR_NID_2009"]
    attr_lst = ["RUNAVE7100", "STOR_NOR_2009"]
    data_attr = gages.read_constant_cols(usgs_id, attr_lst)
    run_avg = data_attr[:, 0] * (10 ** (-3)) * (10**6)  # m^3 per year
    nor_storage = data_attr[:, 1] * 1000  # m^3
    return nor_storage / run_avg


def get_diversion(gages: Gages, usgs_id) -> np.array:
    diversion_strs = ["diversion", "divert"]
    assert all(x < y for x, y in zip(usgs_id, usgs_id[1:]))
    attr_lst = ["WR_REPORT_REMARKS", "SCREENING_COMMENTS"]
    data_attr = gages.read_attr_origin(usgs_id, attr_lst)
    diversion_strs_lower = [elem.lower() for elem in diversion_strs]
    data_attr0_lower = np.array(
        [elem.lower() if type(elem) == str else elem for elem in data_attr[0]]
    )
    data_attr1_lower = np.array(
        [elem.lower() if type(elem) == str else elem for elem in data_attr[1]]
    )
    data_attr_lower = np.vstack((data_attr0_lower, data_attr1_lower)).T
    diversions = [
        hydro_arithmetric.is_any_elem_in_a_lst(
            diversion_strs_lower, data_attr_lower[i], include=True
        )
        for i in range(len(usgs_id))
    ]
    return np.array(diversions)


class Mopex(HydroDataset):
    def __init__(
        self,
        data_path=os.path.join("mopex"),
    ):
        """
        Initialization for MOPEX series dataset

        Parameters
        ----------
        data_path
            where we put the dataset.
            we already set the ROOT directory for hydrodataset,
            so here just set it as a relative path,
            by default "mopex"
        """
        super().__init__(data_path)
        self.data_source_description = self.set_data_source_describe()
        self.mopex_sites = self.read_site_info()

    def get_name(self):
        return "MOPEX"

    def get_relevant_cols(self):
        return np.array(["map", "pe", "streamflow", "tmax", "tmin"])

    def set_data_source_describe(self):
        """
        the files in the dataset and their location in file system

        Returns
        -------
        collections.OrderedDict
            the description for a Precipitation4Fusion dataset
        """
        mopex_db = self.data_source_dir
        forcing_dir = os.path.join(mopex_db, "basin_mean_forcing")
        return collections.OrderedDict(
            MOPEX_DIR=mopex_db,
            MOPEXFORCING_DIR=forcing_dir,
        )

    def read_site_info(self):
        mopex_id_file = self.data_source_description["MOPEXFORCING_DIR"]
        gage_ids = [os.path.splitext(f)[0] for f in os.listdir(mopex_id_file)]
        gage_ids.sort()
        assert all(x < y for x, y in zip(gage_ids, gage_ids[1:]))
        return gage_ids

    def read_forcing_gage(self, usgs_id, var_lst, t_range_list):
        data_folder = self.data_source_description["MOPEXFORCING_DIR"]
        data_file = os.path.join(data_folder, f"{usgs_id}.dly")
        print("reading", "forcing data ", usgs_id)
        date_temp = pd.read_fwf(data_file, widths=[8], names=["date"])

        columns = [
            "mean_areal_precipitation",
            "climatic_potential_evaporation",
            "daily_streamflow_discharge",
            "daily_max_air_temperature",
            "daily_min_air_temperature",
        ]

        data_temp = []

        with open(data_file, "r") as file:
            for line in file:
                # 从第9个字符开始截取字符串，然后分割
                data_parts = line[8:].strip().split()
                if len(data_parts) >= len(columns):  # 确保行有足够的数据
                    data_temp.append(data_parts[: len(columns)])  # 只添加需要的列数

        data_temp = pd.DataFrame(data_temp, columns=columns)
        standardize_date = [re.sub(r"\s", "0", date) for date in date_temp["date"]]

        # df_date = date_temp["date"]

        # df_date = pd.to_datetime(standardize_date, format='%Y%m%d')
        date = pd.to_datetime(standardize_date).values.astype("datetime64[D]")

        data_temp = pd.concat([pd.DataFrame(date, columns=["date"]), data_temp], axis=1)

        # data_temp = pd.read_csv(data_file, sep=r"\s+", header=None, usecols=range(1, 6))
        data_temp.columns = [
            "date",
            "mean_areal_precipitation",
            "climatic_potential_evaporation",
            "daily_streamflow_discharge",
            "daily_max_air_temperature",
            "daily_min_air_temperature",
        ]
        # df_date = data_temp["date"]
        # date = pd.to_datetime(df_date).values.astype("datetime64[D]")
        nf = len(var_lst)
        assert all(x < y for x, y in zip(date, date[1:]))
        [c, ind1, ind2] = np.intersect1d(date, t_range_list, return_indices=True)
        assert date[0] <= t_range_list[0] and date[-1] >= t_range_list[-1]
        nt = t_range_list.size
        out = np.empty([nt, nf])
        var_lst_in_file = [
            "map(mm/day)",
            "pe(mm/day)",
            "streamflow(mm/day)",
            "tmax(C/day)",
            "tmin(C/day)",
        ]
        for k in range(nf):
            # assume all files are of same columns. May check later.
            ind = [
                i
                for i in range(len(var_lst_in_file))
                if var_lst[k] in var_lst_in_file[i]
            ][0]
            out[ind2, k] = data_temp.iloc[:, ind + 1].values[ind1]
        return out

    def read_relevant_cols(
        self, gage_id_lst=None, t_range=None, var_lst=None, **kwargs
    ) -> np.array:
        assert all(x < y for x, y in zip(gage_id_lst, gage_id_lst[1:]))
        assert all(x < y for x, y in zip(t_range, t_range[1:]))
        print("reading formatted data:")
        t_lst = hydro_time.t_range_days(t_range)
        nt = t_lst.shape[0]
        x = np.empty([len(gage_id_lst), nt, len(var_lst)])
        for k in range(len(gage_id_lst)):
            data = self.read_forcing_gage(
                gage_id_lst[k],
                var_lst,
                t_lst,
            )
            x[k, :, :] = data
        return x

    def read_object_ids(self, object_params=None) -> np.array:
        return self.mopex_sites

    def cache_forcing_np_json(self):
        """
        Save all basin-forcing data in a numpy array file in the cache directory.

        Because it takes much time to read data from txt files,
        it is a good way to cache data as a numpy file to speed up the reading.

        """
        cache_npy_file = CACHE_DIR.joinpath("mopex_forcing.npy")
        json_file = CACHE_DIR.joinpath("mopex_forcing.json")
        variables = self.get_relevant_cols()
        basins = self.mopex_sites
        mopex_t_range = ["1948-01-01", "2003-12-31"]
        times = [
            hydro_time.t2str(tmp)
            for tmp in hydro_time.t_range_days(mopex_t_range).tolist()
        ]
        data_info = collections.OrderedDict(
            {
                "dim": ["basin", "time", "variable"],
                "basin": basins,
                "time": times,
                "variable": variables.tolist(),
            }
        )
        with open(json_file, "w") as FP:
            json.dump(data_info, FP, indent=4)
        data = self.read_relevant_cols(
            gage_id_lst=basins,
            t_range=mopex_t_range,
            var_lst=variables.tolist(),
        )
        np.save(cache_npy_file, data)

    def cache_forcing_xrdataset(self):
        """Save all basin-forcing data in a netcdf file in the cache directory."""
        cache_npy_file = CACHE_DIR.joinpath("mopex_forcing.npy")
        json_file = CACHE_DIR.joinpath("mopex_forcing.json")
        if (not os.path.isfile(cache_npy_file)) or (not os.path.isfile(json_file)):
            self.cache_forcing_np_json()
        mopex_forcing = np.load(cache_npy_file)
        with open(json_file, "r") as fp:
            mopex_forcing_dict = json.load(
                fp, object_pairs_hook=collections.OrderedDict
            )

        basins = mopex_forcing_dict["basin"]
        times = pd.date_range(
            mopex_forcing_dict["time"][0], periods=len(mopex_forcing_dict["time"])
        )
        variables = mopex_forcing_dict["variable"]
        # All units' names are from Pint https://github.com/hgrecco/pint/blob/master/pint/default_en.txt
        # final is PET's unit. PET comes from the model output of CAMELS-US
        units = ["mm/day", "mm/day", "mm/day", "°C", "°C"]
        return xr.Dataset(
            data_vars={
                **{
                    variables[i]: (
                        ["basin", "time"],
                        mopex_forcing[:, :, i],
                        {"units": units[i]},
                    )
                    for i in range(len(variables))
                }
            },
            coords={
                "basin": basins,
                "time": times,
            },
            attrs={"forcing_type": "mopex"},
        )

    def cache_xrdataset(self):
        """Save all data in a netcdf file in the cache directory"""
        warnings.warn("Check you units of all variables")
        ds = self.cache_forcing_xrdataset()
        ds.to_netcdf(CACHE_DIR.joinpath("mopex_timeseries.nc"))

    def read_ts_xrdataset(
        self,
        gage_id_lst: list = None,
        t_range: list = None,
        var_lst: list = None,
        **kwargs,
    ):
        if var_lst is None:
            return None
        ts = xr.open_dataset(CACHE_DIR.joinpath("mopex_timeseries.nc"))
        all_vars = ts.data_vars
        if any(var not in ts.variables for var in var_lst):
            raise ValueError(f"var_lst must all be in {all_vars}")
        return ts[var_lst].sel(basin=gage_id_lst, time=slice(t_range[0], t_range[1]))


class GagesMopexPrepFusion(HydroDataset):
    def __init__(
        self,
        gages_data_path=os.path.join("gages"),
        mopex_data_path=os.path.join("mopex"),
    ):
        """
        Initialization for GagesMopexPrepFusion series dataset

        Parameters
        ----------
        gages_data_path
            where we put the GAGES dataset.
            we already set the ROOT directory for hydrodataset,
            so here just set it as a relative path,
            by default "gages/"
        mopex_data_path
            where we put the MOPEX dataset.
            we already set the ROOT directory for hydrodataset,
            so here just set it as a relative path,
            by default "mopex/"
        download
            if true, download, by default False
        """
        self.gages_data_path = gages_data_path
        self.mopex_data_path = mopex_data_path
        self.gages = Gages(gages_data_path)
        self.mopex = Mopex(mopex_data_path)
        self.gages_sites = list(
            set(self.gages.read_site_info()["STAID"]).intersection(
                set(self.mopex.read_site_info())
            )
        )

    def get_name(self):
        return "GagesMopexPrepFusion"

    def read_forcing_ts(
        self,
        gage_id_lst: list = None,
        t_range: list = None,
        var_lst: list = None,
        **kwargs,
    ):
        # p_gages
        precipitation_gages_ds = self.gages.read_ts_xrdataset(
            gage_id_lst,
            t_range,
            [var_lst[0]],
        )
        # p_mopex
        precipitation_mopex_ds = self.mopex.read_ts_xrdataset(
            gage_id_lst,
            t_range,
            [var_lst[1]],
        )
        # x
        data_forcing_ds = self.gages.read_ts_xrdataset(
            gage_id_lst,
            t_range,
            var_lst[2:],
        )
        # 拼接下 precipitation_gages_ds precipitation_mopex_ds data_forcing_ds
        data_forcing_all_ds = xr.merge(
            [
                precipitation_gages_ds,
                precipitation_mopex_ds,
                data_forcing_ds,
            ]
        )
        return data_forcing_all_ds

    def read_streamflow_ts(
        self,
        gage_id_lst: list = None,
        t_range: list = None,
        var_lst: list = None,
        **kwargs,
    ):
        data_output_ds = self.gages.read_ts_xrdataset(
            gage_id_lst,
            t_range,
            var_lst,
        )
        return data_output_ds

    def read_attr(self, gage_id_lst: list = None, var_lst=None, **kwargs):
        data_attr_ds = self.gages.read_attr_xrdataset(
            gage_id_lst,
            var_lst,
            all_number=True,
        )
        return data_attr_ds

    def get_relevant_cols(self) -> np.array:
        return np.array(self.gages.get_relevant_cols() + self.mopex.get_relevant_cols())

    def read_object_ids(self, **kwargs) -> np.array:
        return self.gages_sites

    def read_forcing_gage(self, usgs_id, var_lst, t_range_list) -> np.array:
        gages_data = self.gages.read_forcing_gage(usgs_id, var_lst, t_range_list)
        mopex_data = self.mopex.read_forcing_gage(usgs_id, var_lst, t_range_list)
        return np.concatenate((gages_data, mopex_data), axis=1)

    def read_relevant_cols(
        self,
        gage_id_lst: list = None,
        t_range: list = None,
        var_lst: list = None,
    ) -> np.array:
        gages_data = self.gages.read_relevant_cols(
            gage_id_lst,
            t_range,
            var_lst,
        )
        mopex_data = self.mopex.read_relevant_cols(
            gage_id_lst,
            t_range,
            var_lst,
        )
        return np.concatenate((gages_data, mopex_data), axis=1)

    def streamflow_unit(self):
        return self.gages.streamflow_unit()

    def read_area(self, gage_id_lst) -> np.array:
        return self.gages.read_area(gage_id_lst)

    def read_mean_prcp(self, gage_id_lst) -> np.array:
        return self.gages.read_mean_prcp(gage_id_lst)


class MopexPrepGagesAttrFusion(HydroDataset):
    def __init__(
        self,
        gages_data_path=os.path.join("gages"),
        mopex_data_path=os.path.join("mopex"),
    ):
        """
        Initialization for GagesMopexPrepFusion series dataset

        Parameters
        ----------
        gages_data_path
            where we put the GAGES dataset.
            we already set the ROOT directory for hydrodataset,
            so here just set it as a relative path,
            by default "gages/"
        mopex_data_path
            where we put the MOPEX dataset.
            we already set the ROOT directory for hydrodataset,
            so here just set it as a relative path,
            by default "mopex/"
        download
            if true, download, by default False
        """
        self.gages_data_path = gages_data_path
        self.mopex_data_path = mopex_data_path
        self.gages = Gages(gages_data_path)
        self.mopex = Mopex(mopex_data_path)
        self.gages_sites = list(
            set(self.gages.read_site_info()["STAID"]).intersection(
                set(self.mopex.read_site_info())
            )
        )

    def get_name(self):
        return "MopexPrepGagesAttrFusion"

    def read_forcing_ts(
        self,
        gage_id_lst: list = None,
        t_range: list = None,
        var_lst: list = None,
        **kwargs,
    ):
        # p_mopex
        precipitation_mopex_ds = self.mopex.read_ts_xrdataset(
            gage_id_lst,
            t_range,
            [var_lst[0]],
        )
        # x
        data_forcing_ds = self.gages.read_ts_xrdataset(
            gage_id_lst,
            t_range,
            var_lst[1:],
        )
        # 拼接下 precipitation_gages_ds precipitation_mopex_ds data_forcing_ds
        data_forcing_all_ds = xr.merge(
            [
                precipitation_mopex_ds,
                data_forcing_ds,
            ]
        )
        return data_forcing_all_ds

    def read_streamflow_ts(
        self,
        gage_id_lst: list = None,
        t_range: list = None,
        var_lst: list = None,
        **kwargs,
    ):
        data_output_ds = self.gages.read_ts_xrdataset(
            gage_id_lst,
            t_range,
            var_lst,
        )
        return data_output_ds

    def read_attr(self, gage_id_lst: list = None, var_lst=None, **kwargs):
        data_attr_ds = self.gages.read_attr_xrdataset(
            gage_id_lst,
            var_lst,
            all_number=True,
        )
        return data_attr_ds

    def get_relevant_cols(self) -> np.array:
        return np.array(self.gages.get_relevant_cols() + self.mopex.get_relevant_cols())

    def read_object_ids(self, **kwargs) -> np.array:
        return self.gages_sites

    def read_forcing_gage(self, usgs_id, var_lst, t_range_list) -> np.array:
        gages_data = self.gages.read_forcing_gage(usgs_id, var_lst, t_range_list)
        mopex_data = self.mopex.read_forcing_gage(usgs_id, var_lst, t_range_list)
        return np.concatenate((gages_data, mopex_data), axis=1)

    def read_relevant_cols(
        self,
        gage_id_lst: list = None,
        t_range: list = None,
        var_lst: list = None,
    ) -> np.array:
        gages_data = self.gages.read_relevant_cols(
            gage_id_lst,
            t_range,
            var_lst,
        )
        mopex_data = self.mopex.read_relevant_cols(
            gage_id_lst,
            t_range,
            var_lst,
        )
        return np.concatenate((gages_data, mopex_data), axis=1)

    def streamflow_unit(self):
        return self.gages.streamflow_unit()

    def read_area(self, gage_id_lst) -> np.array:
        return self.gages.read_area(gage_id_lst)

    def read_mean_prcp(self, gage_id_lst) -> np.array:
        return self.gages.read_mean_prcp(gage_id_lst)


class Reservoir(HydroDataset):
    def __init__(self, data_path=os.path.join("reservoir"), download=False):
        """
        Initialization for CAMELS series dataset

        Parameters
        ----------
        data_path
            where we put the dataset.
            we already set the ROOT directory for hydrodataset,
            so here just set it as a relative path,
            by default "Reservoir"
            download
                if true, download, by default False
        """
        super().__init__(data_path)
        self.data_source_description = self.set_data_source_describe()
        if download:
            self.download_data_source()
        self.reservoir_sites = self.read_site_info()

    def get_name(self):
        return "Reservoir"

    def set_data_source_describe(self):
        """
        the files in the dataset and their location in file system

        Returns
        -------
        collections.OrderedDict
            the description for a Reservoir dataset
        """
        reservoir_dir = self.data_source_dir
        if not os.path.isdir(reservoir_dir):
            raise NotADirectoryError(
                "Please check if you have downloaded the data and put it in the correct dir"
            )
        # reservoirs_attr_file = os.path.join(reservoirs_dir, "reservoirs_static.nc")
        reservoir_time_series_file = os.path.join(reservoir_dir, "rainflow_21100150.nc")
        return collections.OrderedDict(
            RESERVOIR_DIR=reservoir_dir,
            # RESERVOIRS_ATTR_FILE=reservoirs_attr_file,
            RESERVOIR_TIME_SERIES_FILE=reservoir_time_series_file,
        )

    # def get_constant_cols(self) -> np.array:
    #     """all readable attrs in DHFReservoir"""
    #     reservoir_attr_file = self.data_source_description["RESERVOIRS_ATTR_FILE"]
    #     ts = xr.open_dataset(reservoir_attr_file)
    #     all_cols = list(ts.data_vars.keys())
    #     other_cols = [col for col in all_cols if col != "Reservoir_ID"]
    #     return other_cols

    def get_relevant_cols(self):
        return np.array(["precip", "year"])

    def get_target_cols(self):
        return np.array(["flow"])

    def read_site_info(self):
        """
        Read the site information of Reservoir

        Returns
        -------
        pd.DataFrame
            the site information of Reservoir
        """
        # reservoir_attr_file = self.data_source_description["RESERVOIRS_ATTR_FILE"]
        # ts = xr.open_dataset(reservoir_attr_file)
        reservoir_ids = [
            "21100150",
        ]
        reservoir_ids.sort()
        assert all(x < y for x, y in zip(reservoir_ids, reservoir_ids[1:]))
        return reservoir_ids

    def read_forcing_reservoir(self, reservoir_id, var_lst, t_range_list):
        reservoir_time_series_file = self.data_source_description[
            "RESERVOIR_TIME_SERIES_FILE"
        ]
        ts = xr.open_dataset(reservoir_time_series_file)

        # 确保时间是升序排列
        assert all(
            ts["time"].values[i] < ts["time"].values[i + 1]
            for i in range(len(ts["time"].values) - 1)
        )

        # 获取所有时间和水库索引
        time_all = ts["time"].values.astype("datetime64[D]")
        rid_all = ts["station"].values
        assert reservoir_id in rid_all

        # 找到 reservoir_id 对应的索引
        rid_index = int(np.where(rid_all == reservoir_id)[0][0])

        # 时间匹配：获取匹配时间在数组中的位置
        _, ind1, ind2 = np.intersect1d(time_all, t_range_list, return_indices=True)
        assert time_all[0] <= t_range_list[0] and time_all[-1] >= t_range_list[-1]

        # 输出初始化
        nt = len(t_range_list)
        nf = len(var_lst)
        out = np.empty((nt, nf))

        for k, var in enumerate(var_lst):
            # 取出每个变量在指定水库和时间段下的值
            values = ts[var].values  # shape: (n_reservoir, n_time)
            out[:, k] = values[rid_index, ind1]

        return out

    def read_relevant_cols(
        self, reservoir_id_lst=None, t_range=None, var_lst=None, **kwargs
    ) -> np.array:
        assert all(x < y for x, y in zip(reservoir_id_lst, reservoir_id_lst[1:]))
        assert all(x < y for x, y in zip(t_range, t_range[1:]))
        print("reading formatted data:")
        t_lst = hydro_time.t_range_days(t_range)
        nt = t_lst.shape[0]
        x = np.empty([len(reservoir_id_lst), nt, len(var_lst)])
        for k in range(len(reservoir_id_lst)):
            data = self.read_forcing_reservoir(
                reservoir_id_lst[k],
                var_lst,
                t_lst,
            )
            x[k, :, :] = data
        return x

    def read_object_ids(self, object_params=None) -> np.array:
        return self.reservoir_sites

    def read_ts_xrdataset(
        self,
        reservoir_id_lst: list = None,
        t_range: list = None,
        var_lst: list = None,
        **kwargs,
    ):
        if var_lst is None:
            return None
        ts = xr.open_dataset(self.data_source_description["RESERVOIR_TIME_SERIES_FILE"])
        all_vars = ts.data_vars
        if any(var not in ts.variables for var in var_lst):
            raise ValueError(f"var_lst must all be in {all_vars}")
        return ts[var_lst].sel(
            station=reservoir_id_lst, time=slice(t_range[0], t_range[1])
        )

    # def read_attr_xrdataset(self, reservoir_id_lst=None, var_lst=None, **kwargs):
    #     if var_lst is None or len(var_lst) == 0:
    #         return None
    #     attr = xr.open_dataset(self.data_source_description["RESERVOIRS_ATTR_FILE"])
    #     if "all_number" in list(kwargs.keys()) and kwargs["all_number"]:
    #         attr_num = map_string_vars(attr)
    #         reservoir_id_lst = [int(id) for id in reservoir_id_lst]
    #         return attr_num[var_lst].sel(Reservoir_ID=reservoir_id_lst)
    #     return attr[var_lst].sel(Reservoir_ID=reservoir_id_lst)

    def read_forcing_ts(
        self,
        reservoir_id_lst: list = None,
        t_range: list = None,
        var_lst: list = None,
        **kwargs,
    ):
        # Precip
        precip_reservoir_ds = self.read_ts_xrdataset(
            reservoir_id_lst,
            t_range,
            [var_lst[0]],
        )
        # Year
        year_reservoir_ds = self.read_ts_xrdataset(
            reservoir_id_lst,
            t_range,
            [var_lst[1]],
        )
        # 拼接下 precip_reservoir_ds year_reservoir_ds
        data_forcing_all_ds = xr.merge(
            [
                precip_reservoir_ds,
                year_reservoir_ds,
            ]
        )
        return data_forcing_all_ds

    def read_mean_prcp(self, gage_id_lst) -> np.array:
        mean_year_prep = 812
        mean_day_prep = mean_year_prep / 365
        return mean_day_prep

    def read_flow_ts(
        self,
        reservoir_id_lst: list = None,
        t_range: list = None,
        var_lst: list = None,
        **kwargs,
    ):
        data_output_ds = self.read_ts_xrdataset(
            reservoir_id_lst,
            t_range,
            var_lst,
        )
        return data_output_ds

    # def read_attr(
    #     self,
    #     reservoir_id_lst: list = None,
    #     var_lst=None,
    #     **kwargs
    # ):
    #     data_attr_ds = self.read_attr_xrdataset(
    #         reservoir_id_lst,
    #         var_lst,
    #         all_number=True,
    #     )
    #     return data_attr_ds


data_sources_dict = {
    "camels_us": Camels,
    "camelsh": Camelsh,
    "caravan": Caravan,
    "grdc_caravan": GrdcCaravan,
    "selfmadehydrodataset": SelfMadeHydroDataset,
    "selfmadeforecastdataset": SelfMadeForecastDataset,
    "longtermdataset": LongTermDataset,
    "floodeventdatasource": FloodEventDatasource,
    "stationhydrodataset": StationHydroDataset,
    "tghydrodatasource": TgHydroDatasource,
    "camels_aus": CamelsAus,
    "camels_br": CamelsBr,
    "camels_cl": CamelsCl,
    "camels_gb": CamelsGb,
    "camels_col": CamelsCol,
    "usgs4camels": SupData4Camels,
    "modiset4camels": ModisEt4Camels,
    "nldas4camels": Nldas4Camels,
    "smap4camels": Smap4Camels,
    "gagesmopexprepfusion": GagesMopexPrepFusion,
    "mopexprepgagesattrfusion": MopexPrepGagesAttrFusion,
    "gages": Gages,
    "mopex": Mopex,
    "reservoirs": Reservoirs,
    "reservoir": Reservoir,
}
