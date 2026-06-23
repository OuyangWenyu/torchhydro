import torch
import torch.nn as nn
import torch.nn.functional as F
import torchhydro.models.cudnnlstm


class CoupledLSTMModel(nn.Module):
    """
    创建降雨融合模型，耦合两个LSTM模型
    第一个模型的输入是两个降雨序列，输出是融合降雨，后一个模型输入是融合后的降雨以及温度、风速和气压等数据序列，输出是径流序列
    """

    def __init__(self,
                 precipitation_fusion_input_feature,
                 precipitation_fusion_output_feature,
                 precipitation_fusion_hidden_states,
                 flow_prediction_input_feature,
                 flow_prediction_output_feature,
                 flow_prediction_hidden_states):
        super(CoupledLSTMModel, self).__init__()
        self.precipitation_fusion_model = torchhydro.models.cudnnlstm.CudnnLstmModel(n_input_features=precipitation_fusion_input_feature,
                                                                                   n_output_features=precipitation_fusion_output_feature,
                                                                                   n_hidden_states=precipitation_fusion_hidden_states)
        self.flow_prediction_model = torchhydro.models.cudnnlstm.CudnnLstmModel(n_input_features=flow_prediction_input_feature,
                                                                              n_output_features=flow_prediction_output_feature,
                                                                              n_hidden_states=flow_prediction_hidden_states)

    def forward(self, x):
        precipitation = x[:, :, :2]  # precipitation_gages,precipitation_mopex
        lstm_precipitation_fusion_output = self.precipitation_fusion_model(precipitation)
        lstm_flow_prediction_input = torch.cat((lstm_precipitation_fusion_output, x[:, :, 2:]), dim=2)
        flow_prediction_output = self.flow_prediction_model(lstm_flow_prediction_input)
        return flow_prediction_output
