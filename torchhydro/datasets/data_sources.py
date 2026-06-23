"""
Author: Wenyu Ouyang
Date: 2024-04-02 14:37:09
LastEditTime: 2025-11-22 11:10:32
LastEditors: Wenyu Ouyang
Description: A module for different data sources
FilePath: \torchhydro\torchhydro\datasets\data_sources.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

from hydrodataset.camels_us import CamelsUs
from hydrodataset.camelsh import Camelsh
from hydrodataset.caravan import Caravan
from hydrodataset.grdc_caravan import GrdcCaravan
from hydrodataset.camels_aus import CamelsAus
from hydrodataset.camels_br import CamelsBr
from hydrodataset.camels_cl import CamelsCl
from hydrodataset.camels_gb import CamelsGb
from hydrodataset.camels_col import CamelsCol
from hydrodatasource.reader.data_source import (
    SelfMadeHydroDataset,
    SelfMadeForecastDataset,
    StationHydroDataset,
    TgHydroDatasource,
)
from hydrodataset.camels import Camels

from hydrodatasource.reader.floodevent import FloodEventDatasource


data_sources_dict = {
    "camels": Camels,
    "camels_us": CamelsUs,
    "camelsh": Camelsh,
    "caravan": Caravan,
    "grdc_caravan": GrdcCaravan,
    "selfmadehydrodataset": SelfMadeHydroDataset,
    "selfmadeforecastdataset": SelfMadeForecastDataset,
    "floodeventdatasource": FloodEventDatasource,
    "stationhydrodataset": StationHydroDataset,
    "tghydrodatasource": TgHydroDatasource,
    "camels_aus": CamelsAus,
    "camels_br": CamelsBr,
    "camels_cl": CamelsCl,
    "camels_gb": CamelsGb,
    "camels_col": CamelsCol,
}
