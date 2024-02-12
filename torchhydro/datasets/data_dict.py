"""
Author: Wenyu Ouyang
Date: 2021-12-31 11:08:29
LastEditTime: 2024-02-12 18:50:13
LastEditors: Wenyu Ouyang
Description: A dict used for data source and data loader
FilePath: \torchhydro\torchhydro\datasets\data_dict.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
from hydrodataset import Camels
from torchhydro.datasets.data_source_gpm_gfs import GPM_GFS
from torchhydro.datasets.data_source_mean import MEAN
from hydrodataset.caravan import Caravan
from torchhydro.datasets.data_sources import SelfMadeCamels

from torchhydro.datasets.data_sets import (
    BaseDataset,
    BasinSingleFlowDataset,
    DplDataset,
    GPM_GFS_Dataset,
)

data_sources_dict = {
    "CAMELS": Camels,
    "GPM_GFS": GPM_GFS,
    "Mean": MEAN,
    "Caravan": Caravan,
    "SelfMadeCAMELS": SelfMadeCamels,
}

datasets_dict = {
    "StreamflowDataset": BaseDataset,
    "SingleflowDataset": BasinSingleFlowDataset,
    "DplDataset": DplDataset,
    "GPM_GFS_Dataset": GPM_GFS_Dataset,
    "MEAN_Dataset": GPM_GFS_Dataset
}
