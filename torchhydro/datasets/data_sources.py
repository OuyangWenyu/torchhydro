"""
Author: Wenyu Ouyang
Date: 2024-04-02 14:37:09
LastEditTime: 2025-10-28 16:22:46
LastEditors: Wenyu Ouyang
Description: A module for different data sources
FilePath: \torchhydro\torchhydro\datasets\data_sources.py
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
from hydrodataset.camels_us import CamelsUs
from hydrodataset.camelsh import Camelsh
from hydrodataset.grdc_caravan import GrdcCaravan
from hydrodatasource.reader.data_source import (
    SelfMadeHydroDataset,
    SelfMadeForecastDataset,
    # StationHydroDataset
)

# from hydrodatasource.reader.floodevent import FloodEventDatasource


from torchhydro import CACHE_DIR, SETTING

data_sources_dict = {
    "camels_us": CamelsUs,
    "camelsh": Camelsh,
    "grdc_caravan": GrdcCaravan,
    "selfmadehydrodataset": SelfMadeHydroDataset,
    "selfmadeforecastdataset": SelfMadeForecastDataset,
    # "floodeventdatasource": FloodEventDatasource,
    # "stationhydrodataset": StationHydroDataset
}
