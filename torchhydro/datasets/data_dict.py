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

# datasets：首先，我们设置了几个datasource的仓库工具，来提供数据源，包括公开数据集的hydrodataset（比如CAMELS）、处理自己数据的hydrodatasource（不像CAMELS这样做好的数据集，而是需要自己整理的），
# 这些数据源主要提供的功能就是对数据的访问，然后在torchhydro里面就能写具体的torch dataset了，就是按照和模型对接的数据类型来编写dataset，dataset整体也有一个dict 来记录，
# 然后就是具体的dataset类的module了。datasets里面还有一些归一化的、通用处理数据的工具的module
#



datasets_dict = {
    "StreamflowDataset": BaseDataset,
    "SingleflowDataset": BasinSingleFlowDataset,
    "DplDataset": DplDataset,
    "FlexDataset": FlexibleDataset,
    "Seq2SeqDataset": Seq2SeqDataset,
    "TransformerDataset": TransformerDataset,
}
