"""
Author: Wenyu Ouyang
Date: 2021-12-31 11:08:29
LastEditTime: 2023-07-11 20:43:43
LastEditors: Wenyu Ouyang
Description: Dicts including models (which are seq-first), losses, and optims
FilePath: \HydroTL\hydrotl\models\model_dict_function.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
import torch.nn

from torchhydro.models.cudnnlstm import (
    CudnnLstmModel,
    LinearCudnnLstmModel,
    CNN1dLCmodel,
    CudnnLstmModelLstmKernel,
    CudnnLstmModelMultiOutput,
    KuaiLstm, CpuLstmModel,
)
from torch.optim import Adam, SGD, Adadelta
from torchhydro.models.crits import (
    RMSELoss,
    RmseLoss,
    MultiOutLoss,
    UncertaintyWeights,
    DynamicTaskPrior,
    MultiOutWaterBalanceLoss,
)
from torchhydro.models.dpl4xaj import DplLstmXaj

"""
Utility dictionaries to map a string to a class.
"""
pytorch_model_dict = {
    "KuaiLSTM": CudnnLstmModel,
    "KuaiLstm": KuaiLstm,
    "KaiTlLSTM": LinearCudnnLstmModel,
    "DapengCNNLSTM": CNN1dLCmodel,
    "LSTMKernel": CudnnLstmModelLstmKernel,
    "KuaiLSTMMultiOut": CudnnLstmModelMultiOutput,
    # Uncompleted
    "DplLstmXaj": DplLstmXaj,
    "CpuLSTM": CpuLstmModel
}

pytorch_model_wrapper_dict = {}

pytorch_criterion_dict = {
    "RMSE": RMSELoss,
    # xxxSum means that calculate the criterion for each "feature"(the final dim of output), then sum them up
    "RMSESum": RmseLoss,
    "MultiOutLoss": MultiOutLoss,
    "UncertaintyWeights": UncertaintyWeights,
    "DynamicTaskPrior": DynamicTaskPrior,
    "MultiOutWaterBalanceLoss": MultiOutWaterBalanceLoss,
}

pytorch_opt_dict = {"Adam": Adam, "SGD": SGD, "Adadelta": Adadelta}
