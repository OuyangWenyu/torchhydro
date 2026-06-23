import torch
import torch.nn as nn
import torch.nn.functional as F
import torchhydro.models.cudnnlstm


class REGULstmModel(nn.Module):
    """
    创建水库影响的径流预报模型，耦合两个LSTM模型
    第一个模型的输入是15天累积降雨、前20天累积降雨、前10天累积降雨序列，输出是径流衰减系数，后一个模型输入是降雨和年份序列，输出是径流序列
    """

    def __init__(
        self,
        regulation_input_features,
        regulation_output_features,
        regulation_hidden_states,
        res_inflow_input_feature,
        res_inflow_output_feature,
        res_inflow__hidden_states,
    ):
        super(REGULstmModel, self).__init__()
        self.res_regulation_model = torchhydro.models.cudnnlstm.CudnnLstmModel(
            n_input_features=regulation_input_features,
            n_output_features=regulation_output_features,
            n_hidden_states=regulation_hidden_states,
        )

        self.res_inflow_model = torchhydro.models.cudnnlstm.CudnnLstmModel(
            n_input_features=res_inflow_input_feature,
            n_output_features=res_inflow_output_feature,
            n_hidden_states=res_inflow__hidden_states,
        )

    def forward(self, x_normalized, x_origin):
        """_summary_

        Args:
            x_normalized (_type_): _description_
            x_origin (_type_): _description_

        Returns:
            the flow prediction output, the regulation coefficient output, the inflow output
        """
        precip_sum = x_normalized[:, :, 2:3]  # 前15天累积降雨
        is_all_nan = torch.isnan(precip_sum).all().item()
        # print(f"precip_sum 是否全为 NaN: {is_all_nan}")
        # lstm_res_regulation_output = torch.nn.functional.softplus(self.res_regulation_model(precip_sum)) + 1
        # limit the range of lstm_res_regulation_output to (0, 2)
        lstm_res_regulation_output = (
            torch.tanh(self.res_regulation_model(precip_sum)) + 1
        )
        lstm_res_inflow_output = self.res_inflow_model(
            x_normalized[:, :, :2]
        )  # 降雨和年份序列
        flow_prediction_output = lstm_res_regulation_output * lstm_res_inflow_output
        # lstm_flow_prediction_input = torch.cat((lstm_precipitation_fusion_output, x[:, :, 2:]), dim=2)
        # flow_prediction_output = self.flow_prediction_model(lstm_flow_prediction_input)
        return flow_prediction_output, lstm_res_regulation_output, lstm_res_inflow_output
