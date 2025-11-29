"""
Author: Wenyu Ouyang
Date: 2021-12-31 11:08:29
LastEditTime: 2025-11-22 16:41:37
LastEditors: Wenyu Ouyang
Description: Dicts including models (which are seq-first), losses, and optims
FilePath: \torchhydro\torchhydro\models\model_dict_function.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""

from torchhydro.models.gconv_gru import MultiFeatureStreamGCGRU
from torchhydro.models.mtslstm import MTSLSTM
from torchhydro.models.dpl4hbv import DplLstmHbv, DplAnnHbv
from torchhydro.models.cudnnlstm import (
    CudnnLstmModel,
    LinearCudnnLstmModel,
    CNN1dLCmodel,
    CudnnLstmModelLstmKernel,
    CudnnLstmModelMultiOutput,
    CpuLstmModel,
)

from torchhydro.models.simple_lstm import (
    LinearMultiLayerLSTMModel,
    LinearSimpleLSTMModel,
    MultiLayerLSTM,
    SimpleLSTM,
    SimpleLSTMForecast,
    HFLSTM,
)
from torchhydro.models.seqforecast import SequentialForecastLSTM
from torchhydro.models.seq2seq import (
    GeneralSeq2Seq,
    DataEnhancedModel,
    DataFusionModel,
    Transformer,
)
from torchhydro.models.gnn import (
    GNNBaseModel,
    GNNMLP,
    GCN,
    ResGCN,
    GCNII,
    ResGAT,
)
from torch.optim import Adam, SGD, Adadelta
from torchhydro.models.crits import (
    RMSELoss,
    RmseLoss,
    MultiOutLoss,
    UncertaintyWeights,
    DynamicTaskPrior,
    MultiOutWaterBalanceLoss,
    NSELoss,
    MAPELoss,
    MASELoss,
    MAELoss,
    QuantileLoss,
    PenalizedMSELoss,
    FloodLoss,
    HybridLoss,
    HybridFloodloss,
)
from torchhydro.models.dpl4xaj import DplLstmXaj, DplAnnXaj
from torchhydro.models.dpl4xaj_nn4et import DplLstmNnModuleXaj
from torchhydro.models.spplstm import SPP_LSTM_Model, SPP_LSTM_Model_2
from torchhydro.models.dpl4gr4j import DplLstmGr4j, DplAnnGr4j
from torchhydro.models.moe import MoE
from torchhydro.models.ann import SimpleAnn

"""
Utility dictionaries to map a string to a class.
"""
pytorch_model_dict = {
    # MoE model
    "MoE": MoE,
    # ANN models
    "SimpleAnn": SimpleAnn,
    # LSTM models from Group MHPI
    "KuaiLSTM": CudnnLstmModel,
    "CpuLSTM": CpuLstmModel,
    "KaiLSTM": LinearCudnnLstmModel,
    "DapengCNNLSTM": CNN1dLCmodel,
    "LSTMKernel": CudnnLstmModelLstmKernel,
    "KuaiLSTMMultiOut": CudnnLstmModelMultiOutput,
    # Differentiable models
    "DplLstmXaj": DplLstmXaj,
    "DplAttrXaj": DplAnnXaj,
    "DplNnModuleXaj": DplLstmNnModuleXaj,
    "DplLstmHbv": DplLstmHbv,
    "DplAnnHbv": DplAnnHbv,
    "DplLstmGr4j": DplLstmGr4j,
    "DplAnnGr4j": DplAnnGr4j,
    # LSTMs
    "SimpleLSTM": SimpleLSTM,
    "LinearSimpleLSTMModel": LinearSimpleLSTMModel,
    "MultiLayerLSTM": MultiLayerLSTM,
    "LinearMultiLayerLSTMModel": LinearMultiLayerLSTMModel,
    "SPPLSTM": SPP_LSTM_Model,
    "SimpleLSTMForecast": SimpleLSTMForecast,
    "HFLSTM": HFLSTM,
    "SPPLSTM2": SPP_LSTM_Model_2,
    "SeqForecastLSTM": SequentialForecastLSTM,
    "Seq2Seq": GeneralSeq2Seq,
    "DataEnhanced": DataEnhancedModel,
    "DataFusion": DataFusionModel,
    "MTSLSTM": MTSLSTM,
    # Transformer
    "Transformer": Transformer,
    # GNN models
    "GNNBaseModel": GNNBaseModel,
    "GNNMLP": GNNMLP,
    "GCN": GCN,
    "ResGCN": ResGCN,
    "GCNII": GCNII,
    "ResGAT": ResGAT,
    "AutoRegressiveGCGRU": MultiFeatureStreamGCGRU,
}

pytorch_criterion_dict = {
    "RMSE": RMSELoss,
    # xxxSum means that calculate the criterion for each "feature"(the final dim of output), then sum them up
    "RMSESum": RmseLoss,
    "MultiOutLoss": MultiOutLoss,
    "UncertaintyWeights": UncertaintyWeights,
    "DynamicTaskPrior": DynamicTaskPrior,
    "MultiOutWaterBalanceLoss": MultiOutWaterBalanceLoss,
    "NSELoss": NSELoss,
    "MAPELoss": MAPELoss,
    "MASELoss": MASELoss,
    "MAELoss": MAELoss,
    "QuantileLoss": QuantileLoss,
    "MSELoss": PenalizedMSELoss,
    "FloodLoss": FloodLoss,
    "HybridLoss": HybridLoss,
    "HybridFloodloss": HybridFloodloss,  # Alias for backward compatibility
}

pytorch_opt_dict = {"Adam": Adam, "SGD": SGD, "Adadelta": Adadelta}
