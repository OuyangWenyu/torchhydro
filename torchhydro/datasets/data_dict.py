"""
Author: Wenyu Ouyang
Date: 2021-12-31 11:08:29
LastEditTime: 2023-07-30 14:41:23
LastEditors: Wenyu Ouyang
Description: A dict used for data source and data loader
FilePath: \torchhydro\torchhydro\datasets\data_dict.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
from hydrodataset import Camels

# more data types which cannot be easily treated same with attribute or forcing data
from torchhydro.datasets.data_sets import (
    KuaiDataset,
    BasinFlowDataset,
    BasinSingleFlowDataset,
    DplDataset,
)

data_sources_dict = {
    "CAMELS": Camels,
}

dataloaders_dict = {
    "StreamflowDataset": BasinFlowDataset,
    "SingleflowDataset": BasinSingleFlowDataset,
    # 不完全遍历
    "KuaiDataset": KuaiDataset,
    "DplDataset": DplDataset
}
