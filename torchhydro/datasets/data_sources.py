"""
Author: Wenyu Ouyang
Date: 2024-04-02 14:37:09
LastEditTime: 2024-07-10 09:26:07
LastEditors: Wenyu Ouyang
Description: A module for different data sources
FilePath: /torchhydro/torchhydro/datasets/data_sources.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import collections
import os
import numpy as np
import pandas as pd
import xarray as xr
import pint_xarray  # noqa but it is used in the code
from tqdm import tqdm

from hydroutils import hydro_time
from hydrodataset import Camels
from hydrodatasource.reader.data_source import SelfMadeHydroDataset


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


data_sources_dict = {
    "camels_us": Camels,
    "selfmadehydrodataset": SelfMadeHydroDataset,
    "usgs4camels": SupData4Camels,
    "modiset4camels": ModisEt4Camels,
    "nldas4camels": Nldas4Camels,
    "smap4camels": Smap4Camels,
}
