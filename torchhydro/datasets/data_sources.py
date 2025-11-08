"""
Author: Wenyu Ouyang
Date: 2024-04-02 14:37:09
LastEditTime: 2025-11-08 09:58:42
LastEditors: Wenyu Ouyang
Description: A module for different data sources
FilePath: \torchhydro\torchhydro\datasets\data_sources.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

from hydrodataset.camels_us import CamelsUs
from hydrodataset.camelsh import Camelsh
from hydrodataset.caravan import Caravan
from hydrodataset.grdc_caravan import GrdcCaravan
from hydrodatasource.reader.data_source import (
    SelfMadeHydroDataset,
    SelfMadeForecastDataset,
    StationHydroDataset,
)

from hydrodatasource.reader.floodevent import FloodEventDatasource


data_sources_dict = {
    "camels_us": CamelsUs,
    "camelsh": Camelsh,
    "caravan": Caravan,
    "grdc_caravan": GrdcCaravan,
    "selfmadehydrodataset": SelfMadeHydroDataset,
    "selfmadeforecastdataset": SelfMadeForecastDataset,
    "floodeventdatasource": FloodEventDatasource,
    "stationhydrodataset": StationHydroDataset,
}
