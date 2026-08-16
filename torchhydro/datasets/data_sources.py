"""
Author: Wenyu Ouyang
Date: 2024-04-02 14:37:09
LastEditTime: 2025-04-19 20:28:38
LastEditors: Wenyu Ouyang
Description: A module for different data sources
FilePath: /HydroForecastEval/mnt/disk1/owen/code/torchhydro/torchhydro/datasets/data_sources.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

# torchhydro's own data-source classes (SupData4Camels family, Gages/Mopex,
# GAGES-II/MOPEX fusion, Reservoirs/Reservoir) were removed.  These belong in
# the data layer and are provided by hydrodatasource; torchhydro no longer
# re-implements them.
#
# Standard datasets (camels_us, caravan, ...) resolve through hydrodatasource's
# unified resolver (via ``resolve_dataset``), and the self-made readers below
# come directly from hydrodatasource.

from hydrodatasource.reader.data_source import (
    SelfMadeHydroDataset,
    SelfMadeForecastDataset,
    LongTermDataset,
    StationHydroDataset,
    TgHydroDatasource,
)
from hydrodatasource.reader.floodevent import FloodEventDatasource

from torchhydro.configs.data_resolver import resolve_dataset


data_sources_dict = {
    # Standard hydrodataset datasets resolve through the unified hydrodatasource
    # resolver; torchhydro-specific sources below keep their own classes.
    "camels_us": resolve_dataset("camels_us"),
    "camelsh": resolve_dataset("camelsh"),
    "caravan": resolve_dataset("caravan"),
    "grdc_caravan": resolve_dataset("grdc_caravan"),
    "selfmadehydrodataset": SelfMadeHydroDataset,
    "selfmadeforecastdataset": SelfMadeForecastDataset,
    "longtermdataset": LongTermDataset,
    "floodeventdatasource": FloodEventDatasource,
    "stationhydrodataset": StationHydroDataset,
    "tghydrodatasource": TgHydroDatasource,
    "camels_aus": resolve_dataset("camels_aus"),
    "camels_br": resolve_dataset("camels_br"),
    "camels_cl": resolve_dataset("camels_cl"),
    "camels_gb": resolve_dataset("camels_gb"),
    "camels_col": resolve_dataset("camels_col"),
}
