"""
Author: Wenyu Ouyang
Date: 2021-12-31 11:08:29
LastEditTime: 2024-09-10 10:56:55
LastEditors: Wenyu Ouyang
Description: Dicts including models (which are seq-first), losses, and optims
FilePath: \torchhydro\torchhydro\models\model_dict_function.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""

from torchhydro.models.cudnnlstm import (
    CudnnLstmModel,
    LinearCudnnLstmModel,
    CNN1dLCmodel,
    CudnnLstmModelLstmKernel,
    CudnnLstmModelMultiOutput,
    CpuLstmModel,
)

from torchhydro.models.simple_lstm import SimpleLSTMForecast
from torchhydro.models.seq2seq import (
    GeneralSeq2Seq,
    DataEnhancedModel,
    DataFusionModel,
    Transformer,
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
)
from torchhydro.models.dpl4xaj import DplLstmXaj, DplAnnXaj
from torchhydro.models.dpl4xaj_nn4et import DplLstmNnModuleXaj
from torchhydro.models.spplstm import SPP_LSTM_Model, SPP_LSTM_Model_2

"""
Utility dictionaries to map a string to a class.
"""
pytorch_model_dict = {
    "KuaiLSTM": CudnnLstmModel,
    "CpuLSTM": CpuLstmModel,
    "KaiLSTM": LinearCudnnLstmModel,
    "DapengCNNLSTM": CNN1dLCmodel,
    "LSTMKernel": CudnnLstmModelLstmKernel,
    "KuaiLSTMMultiOut": CudnnLstmModelMultiOutput,
    "DplLstmXaj": DplLstmXaj,
    "DplAttrXaj": DplAnnXaj,
    "SPPLSTM": SPP_LSTM_Model,
    "SimpleLSTMForecast": SimpleLSTMForecast,
    "SPPLSTM2": SPP_LSTM_Model_2,
    "Seq2Seq": GeneralSeq2Seq,
    "DataEnhanced": DataEnhancedModel,
    "DataFusion": DataFusionModel,
    "Transformer": Transformer,
    "DplNnModuleXaj": DplLstmNnModuleXaj,
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
}

pytorch_opt_dict = {"Adam": Adam, "SGD": SGD, "Adadelta": Adadelta}
