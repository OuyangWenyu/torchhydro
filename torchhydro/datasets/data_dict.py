"""
Author: Wenyu Ouyang
Date: 2021-12-31 11:08:29
LastEditTime: 2023-11-20 19:14:52
LastEditors: Wenyu Ouyang
Description: A dict used for data source and data loader
FilePath: /torchhydro/torchhydro/datasets/data_dict.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
from hydrodataset import Camels
from hydrodataset.caravan import Caravan
from torchhydro.datasets.data_sources import SelfMadeCamels

from torchhydro.datasets.data_sets import (
    BaseDataset,
    BasinSingleFlowDataset,
    DplDataset,
)

data_sources_dict = {
    "CAMELS": Camels,
    "Caravan": Caravan,
    "SelfMadeCAMELS": SelfMadeCamels,
}

datasets_dict = {
    "StreamflowDataset": BaseDataset,
    "SingleflowDataset": BasinSingleFlowDataset,
    "DplDataset": DplDataset,
}
