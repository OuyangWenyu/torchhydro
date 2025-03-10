"""
Author: Wenyu Ouyang
Date: 2021-12-31 11:08:29
LastEditTime: 2024-11-02 21:17:44
LastEditors: Wenyu Ouyang
Description: A dict used for data source and data loader
FilePath: /torchhydro/torchhydro/datasets/data_dict.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""

from torchhydro.datasets.data_sets import (
    BaseDataset,
    BasinSingleFlowDataset,
    DplDataset,
    FlexibleDataset,
    Seq2SeqDataset,
    TransformerDataset,
)


datasets_dict = {
    "StreamflowDataset": BaseDataset,
    "SingleflowDataset": BasinSingleFlowDataset,
    "DplDataset": DplDataset,
    "FlexDataset": FlexibleDataset,
    "Seq2SeqDataset": Seq2SeqDataset,
    "TransformerDataset": TransformerDataset,
}


# temp note
#
# 然后在torchhydro里面就能写具体的torch dataset了，就是按照和模型对接的数据类型来编写dataset，dataset整体也有一个dict 来记录，然后就是具体的dataset类的module了。
# datasets里面还有一些归一化的、通用处理数据的工具的module
# datasource is the original data, e.g. camelsus_attributes.nc and camelsus_timeseries.nc
# dataset is the specific data for a specific model?  made from camelsxx_xxxx.nc?  can be used directly by a specific model?

