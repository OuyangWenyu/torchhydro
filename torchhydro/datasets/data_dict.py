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
    HoDataset,
    FoDataset,
    HFDataset,
    ForecasetDataset,
    HoSameDataset,
    OffsetForecasetDataset,
    OffsetForecasetDataset2,
    MultiInputOffsetForecasetDataset,
    MultiInputOffsetForecasetDataset2,
    BasinSingleFlowDataset,
    DplDataset,
    FlexibleDataset,
    Seq2SeqDataset,
    Seq2SeqDataset2,
    SeqForecastDataset,
    TransformerDataset,
)


datasets_dict = {
    "StreamflowDataset": BaseDataset,
    "HoDataset": HoDataset,
    "HoSameDataset": HoSameDataset,
    "FoDataset": FoDataset,
    "HFDataset": HFDataset,
    "ForecasetDataset": ForecasetDataset,
    "OffsetForecasetDataset": OffsetForecasetDataset,
    "OffsetForecasetDataset2": OffsetForecasetDataset2,
    "MultiInputOffsetForecasetDataset": MultiInputOffsetForecasetDataset,
    "MultiInputOffsetForecasetDataset2": MultiInputOffsetForecasetDataset2,
    "SingleflowDataset": BasinSingleFlowDataset,
    "DplDataset": DplDataset,
    "FlexDataset": FlexibleDataset,
    "Seq2SeqDataset": Seq2SeqDataset,
    "Seq2SeqDataset2": Seq2SeqDataset2,
    "SeqForecastDataset": SeqForecastDataset,
    "TransformerDataset": TransformerDataset,
}
