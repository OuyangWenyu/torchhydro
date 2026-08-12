"""
Author: Wenyu Ouyang
Date: 2021-12-31 11:08:29
<LastEditTime: 2025-11-22 16:41:37
LastEditors: Wenyu Ouyang
Description: Dicts including models (which are seq-first), losses, and optims
FilePath: \torchhydro\torchhydro\models\model_dict_function.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
from torchhydro.models.coupledlstm import CoupledLSTMModel
try:
    from torchhydro.models.gconv_gru import MultiFeatureStreamGCGRU, MultiFeatureStreamGCLSTM
except (ImportError, OSError) as e:
    MultiFeatureStreamGCGRU = None
    MultiFeatureStreamGCLSTM = None
    print(f"Warning: gconv_gru not available due to: {e}")
from torchhydro.models.diffusion import ConditionalDiffusionModel
from torchhydro.models.mtslstm import MTSLSTM
from torchhydro.models.regulstm import REGULstmModel
from torchhydro.models.inflowlstm import InflowLstmModel
from torchhydro.models.dpl4hbv import DplLstmHbv, DplAnnHbv
from torchhydro.models.cudnnlstm import (
    CudnnLstmModel,
    LinearCudnnLstmModel,
    CNN1dLCmodel,
    CudnnLstmModelLstmKernel,
    CudnnLstmModelMultiOutput,
    CpuLstmModel,
)
from torchhydro.models.cnnlstm import CNN_LSTM_Model

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
    GeneralSeq2SeqNoForecastInput,
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
from torchhydro.models.adaptive_gnn import WDNEAdapter
from torchhydro.models.gnn_st_flood_3d_batch import (
    SpatioTemporalFloodNet3D,
    MeanAggregationFloodNet,
    AttentionAggregationFloodNet,
    MultiScaleAggregationFloodNet,
    SpatialFirstTemporalFloodNet3D,
)
from torchhydro.models.STGCN import (
    STGCN,
    STGCNModel,
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
    GenerativeLoss,
    PyGraphLoss,
    PeakShapeLoss,
)
from torchhydro.models.dpl4xaj import DplLstmXaj, DplAnnXaj
from torchhydro.models.dpl4xaj_nn4et import DplLstmNnModuleXaj
from torchhydro.models.spplstm import SPP_LSTM_Model, SPP_LSTM_Model_2
from torchhydro.models.dpl4gr4j import DplLstmGr4j, DplAnnGr4j
from torchhydro.models.balstm_model import BALSTM, VanillaLSTM, SimpleBALSTM
from torchhydro.models.ed_balstm import SimpleBALSTM_EncDec
from torchhydro.models.attn_ed_balstm import Attn_SimpleBALSTM_EncDec
from torchhydro.models.moe import (
    MoE,
    DenseMoE,
    SparseMoE,
    SwitchMoE,
    HydrologyTopKMoE,
    GlobalTopKMoE,
)
from torchhydro.models.ann import SimpleAnn
try:
    from torchhydro.models.fno import FNO
except ImportError:
    FNO = None
    print("Warning: torchhydro.models.fno not available")

"""
Utility dictionaries to map a string to a class.
"""
pytorch_model_dict = {
    # FNO
    # "FNO": FNO,
    # Diffusion model
    "ConditionalDiffusionModel": ConditionalDiffusionModel,
    # MoE model
    "MoE": MoE,
    "DenseMoE": DenseMoE,
    "SparseMoE": SparseMoE,
    "SwitchMoE": SwitchMoE,
    "HydrologyTopKMoE": HydrologyTopKMoE,
    "GlobalTopKMoE": GlobalTopKMoE,
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
    "CNNLSTM": CNN_LSTM_Model,
    "CoupledLSTM": CoupledLSTMModel,
    "Seq2SeqNoForecast": GeneralSeq2SeqNoForecastInput,
    "DataEnhanced": DataEnhancedModel,
    "DataFusion": DataFusionModel,
    "MTSLSTM": MTSLSTM,
    "BALSTM": BALSTM,
    "VanillaLSTM": VanillaLSTM,
    "SimpleBALSTM": SimpleBALSTM,
    "SimpleBALSTM_EncDec": SimpleBALSTM_EncDec,
    "Attn_SimpleBALSTM_EncDec": Attn_SimpleBALSTM_EncDec,
    "REGULSTM": REGULstmModel,
    "INFLOWLSTM": InflowLstmModel,
    # Transformer
    "Transformer": Transformer,
    # GNN models
    "GNNBaseModel": GNNBaseModel,
    "GNNMLP": GNNMLP,
    "GCN": GCN,
    "ResGCN": ResGCN,
    "GCNII": GCNII,
    "ResGAT": ResGAT,
    # "AutoRegressiveGCGRU": MultiFeatureStreamGCGRU,
    # "AutoRegressiveGCLSTM": MultiFeatureStreamGCLSTM,
    # Spatio-temporal GNN models
    # 3D Batch Flood Models (with different aggregation strategies)
    "SpatioTemporalFloodNet3D": SpatioTemporalFloodNet3D,
    "MeanAggregationFloodNet": MeanAggregationFloodNet,
    "AttentionAggregationFloodNet": AttentionAggregationFloodNet,
    "MultiScaleAggregationFloodNet": MultiScaleAggregationFloodNet,
    "FloodGCN3D": SpatioTemporalFloodNet3D,  # Alias for easier access
    # Standard STGCN implementation
    "STGCN": STGCNModel,  # Standard Spatio-Temporal Graph Convolutional Network
    # Spatial-First Temporal Models (Space → Time + Transformer)
    "SpatialFirstTemporalFloodNet3D": SpatialFirstTemporalFloodNet3D,
    # 自适应图网络模型 (WDNE family)
    "WDNEInspiredFloodNet": WDNEAdapter,
    "WDNEAdaptiveRoutingNet": WDNEAdapter,
    #"ASTGCN": ASTGCN,
}

if FNO is not None:
    pytorch_model_dict["FNO"] = FNO
if MultiFeatureStreamGCGRU is not None:
    pytorch_model_dict["AutoRegressiveGCGRU"] = MultiFeatureStreamGCGRU
if MultiFeatureStreamGCLSTM is not None:
    pytorch_model_dict["AutoRegressiveGCLSTM"] = MultiFeatureStreamGCLSTM

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
    # Generative model loss (loss computed inside model, this is a placeholder)
    "GenerativeLoss": GenerativeLoss,
    "PyGraphLoss": PyGraphLoss,
    "PeakShapeLoss": PeakShapeLoss,
}

pytorch_opt_dict = {"Adam": Adam, "SGD": SGD, "Adadelta": Adadelta}
