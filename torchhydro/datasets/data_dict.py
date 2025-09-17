"""
Author: Wenyu Ouyang
Date: 2021-12-31 11:08:29
LastEditTime: 2025-07-13 15:40:07
LastEditors: Wenyu Ouyang
Description: A dict used for data source and data loader
FilePath: \torchhydro\torchhydro\datasets\data_dict.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""

from torchhydro.datasets.data_sets import (
    BaseDataset,
    FloodEventDplDataset,
    ForecastDataset,
    HFDataset,
    BasinSingleFlowDataset,
    DplDataset,
    FlexibleDataset,
    ObsForeDataset,
    Seq2SeqDataset,
    SeqForecastDataset,
    TransformerDataset,
    GNNDataset,
    FloodEventDataset,
    AugmentedFloodEventDataset,
    AEFDataset,
    CamelsHourlyDataset
)


datasets_dict = {
    "StreamflowDataset": BaseDataset,
    "ForecastDataset": ForecastDataset,
    "HFDataset": HFDataset,
    "SingleflowDataset": BasinSingleFlowDataset,
    "DplDataset": DplDataset,
    "FlexDataset": FlexibleDataset,
    "Seq2SeqDataset": Seq2SeqDataset,
    "GNNDataset": GNNDataset,
    "SeqForecastDataset": SeqForecastDataset,
    "TransformerDataset": TransformerDataset,
    "ObsForeDataset": ObsForeDataset,
    "FloodEventDataset": FloodEventDataset,
    "FloodEventDplDataset": FloodEventDplDataset,
    "AugmentedFloodEventDataset": AugmentedFloodEventDataset,
    "AlphaEarthDataset": AEFDataset,
    "CamelsHourlyDataset": CamelsHourlyDataset,
}
