"""
Author: Wenyu Ouyang
Date: 2024-04-02 14:37:09
LastEditTime: 2025-10-30 16:45:34
LastEditors: Wenyu Ouyang
Description: A module for different data sources
FilePath: \torchhydro\torchhydro\datasets\data_sources.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

from hydrodataset import HydroDataset, Camels
from hydroutils import (
    hydro_time,
    hydro_file,
    hydro_logger,
)
from hydroutils import hydro_time
from hydrodataset.camels_us import CamelsUs
from hydrodataset.grdc_caravan import GrdcCaravan
from hydrodatasource.reader.data_source import (
    SelfMadeHydroDataset,
    SelfMadeForecastDataset,
    # StationHydroDataset
)

# from hydrodatasource.reader.floodevent import FloodEventDatasource

from tqdm import tqdm
from torchhydro import CACHE_DIR, SETTING

data_sources_dict = {
    "camels_us": CamelsUs,
    "grdc_caravan": GrdcCaravan,
    "selfmadehydrodataset": SelfMadeHydroDataset,
    "selfmadeforecastdataset": SelfMadeForecastDataset,
    # "floodeventdatasource": FloodEventDatasource,
    # "stationhydrodataset": StationHydroDataset
}
