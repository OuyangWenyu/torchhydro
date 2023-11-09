from abc import ABC
from pathlib import Path
from typing import Union
import hydrodataset as hds
from hydrodataset import HydroDataset
import numpy as np


class HydroData(ABC):
    """An interface for reading multi-modal data sources.

    Parameters
    ----------
    ABC : _type_
        _description_
    """

    def __init__(self, data_path):
        self.data_source_dir = Path(hds.ROOT_DIR, data_path)
        if not self.data_source_dir.is_dir():
            self.data_source_dir.mkdir(parents=True)

    def get_name(self):
        raise NotImplementedError

    def set_data_source_describe(self):
        raise NotImplementedError

    def check_data_ready(self):
        raise NotImplementedError

    def read_input(self):
        raise NotImplementedError

    def read_target(self):
        raise NotImplementedError


class HydroOpendata(HydroData):
    """A class for reading public data sources.

    Typically, we read GPM/GFS/ERA5/SMAP/NOAA weather station data as forcing data for hydrological models.
    read USGS NWIS data as input/target data for hydrological models.

    Parameters
    ----------
    HydroData : _type_
        _description_
    """

    def __init__(self, data_path):
        super().__init__(data_path)

    def get_name(self):
        return "HydroOpendata"

    def set_data_source_describe(self):
        self.grid_data_source = "MINIO"
        self.grid_data_source_url = (
            "https://s3.us-east-2.amazonaws.com/minio.t-integration.cloud.tibco.com"
        )
        self.grid_data_source_bucket = "test"
        self.ts_data_source = "Local"

    def check_data_ready(self):
        raise NotImplementedError

    def read_input(self):
        raise NotImplementedError

    def read_target(self):
        raise NotImplementedError


class HydroDatasetSim(HydroDataset):
    """A class for reading hydrodataset, but not really ready datasets,
    just some data directorys organized like a ready dataset.

    Typically, we read data from our self-made data.

    Parameters
    ----------
    HydroData : _type_
        _description_
    """

    def __init__(self, data_path):
        super().__init__(data_path)
        # the naming convention for basin ids are needed
        # we use GRDC station's ids as our default coding convention
        # GRDC station ids are 7 digits, the first 1 digit is continent code,
        # the second 4 digits are sub-region related code
        # 

    def get_name(self):
        return "HydroDatasetSim"

    def set_data_source_describe(self):
        self.attr_data_dir = Path(self.data_source_dir, "attr")
        self.forcing_data_dir = Path(self.data_source_dir, "forcing")
        self.streamflow_data_dir = Path(self.data_source_dir, "streamflow")

    def read_object_ids(self, object_params=None) -> np.array:
        
        raise NotImplementedError

    def read_target_cols(
        self, object_ids=None, t_range_list=None, target_cols=None, **kwargs
    ) -> np.array:
        raise NotImplementedError

    def read_relevant_cols(
        self, object_ids=None, t_range_list: list = None, relevant_cols=None, **kwargs
    ) -> Union[np.array, list]:
        """3d data (site_num * time_length * var_num), time-series data"""
        raise NotImplementedError

    def read_constant_cols(
        self, object_ids=None, constant_cols=None, **kwargs
    ) -> np.array:
        """2d data (site_num * var_num), non-time-series data"""
        raise NotImplementedError

    def read_other_cols(
        self, object_ids=None, other_cols: dict = None, **kwargs
    ) -> dict:
        """some data which cannot be easily treated as constant vars or time-series with same length as relevant vars
        CONVENTION: other_cols is a dict, where each item is also a dict with all params in it"""
        raise NotImplementedError

    def get_constant_cols(self) -> np.array:
        """the constant cols in this data_source"""
        raise NotImplementedError

    def get_relevant_cols(self) -> np.array:
        """the relevant cols in this data_source"""
        raise NotImplementedError

    def get_target_cols(self) -> np.array:
        """the target cols in this data_source"""
        raise NotImplementedError

    def get_other_cols(self) -> dict:
        """the other cols in this data_source"""
        raise NotImplementedError

    def cache_xrdataset(self, **kwargs):
        """cache xarray dataset and pandas feather for faster reading"""
        raise NotImplementedError

    def read_ts_xrdataset(
        self,
        gage_id_lst: list = None,
        t_range: list = None,
        var_lst: list = None,
        **kwargs
    ):
        """read time-series xarray dataset"""
        raise NotImplementedError

    def read_attr_xrdataset(self, gage_id_lst=None, var_lst=None, **kwargs):
        """read attribute pandas feather"""
        raise NotImplementedError

    def read_area(self, gage_id_lst=None):
        """read area of each basin/unit"""
        raise NotImplementedError

    def read_mean_prcp(self, gage_id_lst=None):
        """read mean precipitation of each basin/unit"""
        raise NotImplementedError
