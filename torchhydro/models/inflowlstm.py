import torch
import torch.nn as nn
import torch.nn.functional as F
import torchhydro.models.cudnnlstm


class InflowLstmModel(nn.Module):
    """
    创建水库影响的径流预报模型，一个LSTM模型，一个水库蓄泄判断
    蓄泄判断指标为前15天累积降雨序列，输出是径流衰减系数，lstm模型输入是降雨和年份序列，输出是径流序列
    """

    def __init__(self,
                 n_input_features,
                 n_output_features,
                 n_hidden_states):
        super(InflowLstmModel, self).__init__()
        self.res_inflow_model = torchhydro.models.cudnnlstm.CudnnLstmModel(n_input_features=n_input_features,
                                                                           n_output_features=n_output_features,
                                                                           n_hidden_states=n_hidden_states)

    def forward(self, x_normalized, x_origin):
        precip_15d_sum = x_origin[:, :, 2]  # 前15天累积降雨
        # regulation_factor = torch.where(
        #     rainfall_15d_sum > 50,
        #     torch.tensor(1.5, device=x.device),
        #     torch.tensor(1.0, device=x.device)
        # )

        # 默认 1.0
        regulation_factor = torch.ones_like(precip_15d_sum, device=x_origin.device)

        # regulation_factor = torch.where(rainfall_15d_sum > 100, 1.0, regulation_factor)
        regulation_factor = torch.where(precip_15d_sum > 150, 1.2, regulation_factor)
        regulation_factor = torch.where(precip_15d_sum < 100, 0.8, regulation_factor)

        num_1 = torch.sum(regulation_factor == 1.0).item()
        print(f"1.0的个数: {num_1}")
        num_1_2 = torch.sum(regulation_factor == 1.2).item()
        print(f"1.2的个数: {num_1_2}")
        num_0_8 = torch.sum(regulation_factor == 0.8).item()
        print(f"0.8的个数: {num_0_8}")

        num_above_150 = torch.sum(precip_15d_sum > 150).item()
        print(f"超过150的个数: {num_above_150}")
        num_below_100 = torch.sum(precip_15d_sum < 100).item()
        print(f"小于100的个数: {num_below_100}")
        num_between_100_and_150 = torch.sum((precip_15d_sum >= 100) & (precip_15d_sum <= 150)).item()
        print(f"100-150之间的个数: {num_between_100_and_150}")


        regulation_factor = regulation_factor.unsqueeze(-1)  # [batch_size, seq_len, 1]

        lstm_res_inflow_output = self.res_inflow_model(x_normalized[:, :, :2])  # 降雨和年份序列

        flow_prediction_output = regulation_factor * lstm_res_inflow_output
        return flow_prediction_output, regulation_factor, lstm_res_inflow_output
