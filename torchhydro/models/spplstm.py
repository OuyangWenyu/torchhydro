"""
Author: Xinzhuo Wu
Date: 2023-09-30 1:20:18
LastEditTime: 2024-05-27 16:26:06
LastEditors: Wenyu Ouyang
Description: spp lstm model
FilePath: \torchhydro\torchhydro\models\spplstm.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""

import torch
from torch import nn


class TimeDistributed(nn.Module):
    def __init__(self, layer):
        super(TimeDistributed, self).__init__()
        self.layer = layer

    def forward(self, x):
        outputs = []
        for t in range(x.size(1)):
            xt = x[:, t, :]
            output = self.layer(xt)
            outputs.append(output.unsqueeze(1))
        outputs = torch.cat(outputs, dim=1)
        return outputs


class SppLayer(nn.Module):
    def __init__(self, out_pool_size):
        """
        out_pool_size: a int vector of expected output size of max pooling layer
        """
        super(SppLayer, self).__init__()
        self.out_pool_size = out_pool_size
        self.pools = []
        for i in range(len(out_pool_size)):
            pool_i = nn.AdaptiveMaxPool2d(out_pool_size[i])
            self.pools.append(pool_i)

    def forward(self, previous_conv):
        """
        Parameters
        ----------
        previous_conv
            a tensor vector of previous convolution layer

        Returns
        -------
        torch.Tensor
            a tensor vector with shape [1 x n] is the concentration of multi-level pooling
        """

        num_sample = previous_conv.size(0)
        channel_size = previous_conv.size(1)
        out_pool_size = self.out_pool_size
        for i in range(len(out_pool_size)):
            maxpool = self.pools[i]
            x = maxpool(previous_conv)
            if i == 0:
                spp = x.view(num_sample, channel_size, -1)
            else:
                spp = torch.cat((spp, x.view(num_sample, channel_size, -1)), -1)
        return spp


class SPP_LSTM_Model(nn.Module):
    def __init__(
        self, forecast_history, forecast_length, n_output, n_hidden_states, dropout
    ):
        super(SPP_LSTM_Model, self).__init__()

        self.conv1 = TimeDistributed(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=1,
                padding="same",
            )
        )

        self.conv2 = TimeDistributed(
            nn.Conv2d(
                in_channels=32,
                out_channels=16,
                kernel_size=3,
                padding="same",
                bias=True,
            )
        )

        self.conv3 = TimeDistributed(
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=3,
                padding="same",
                bias=True,
            )
        )

        self.maxpool1 = TimeDistributed(nn.MaxPool2d(kernel_size=2, stride=(2, 2)))

        self.conv4 = TimeDistributed(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding="same",
                bias=True,
            )
        )

        self.conv5 = TimeDistributed(
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                padding="same",
                bias=True,
            )
        )

        self.maxpool2 = TimeDistributed(SppLayer([4, 2, 1]))

        self.dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=21 * 32, hidden_size=n_hidden_states, batch_first=True
        )

        self.dense = nn.Linear(in_features=n_hidden_states, out_features=n_output)

        self.forecast_length = forecast_length

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.conv3(x)
        x = torch.relu(x)
        x = self.maxpool1(x)
        x = self.conv4(x)
        x = torch.relu(x)
        x = self.conv5(x)
        x = torch.relu(x)
        x = self.maxpool2(x)
        x = self.dropout(x)
        x = x.view(x.shape[0], x.shape[1], -1)
        x, _ = self.lstm(x)
        x = self.dense(x)
        x = x[:, -self.forecast_length :, :]
        return x


class SPP_LSTM_Model_2(nn.Module):
    def __init__(
        self,
        forecast_history,
        forecast_length,
        p_n_output,
        p_n_hidden_states,
        p_dropout,
        p_in_channels,
        p_out_channels,
        len_c=None,
        s_forecast_history=None,
        s_n_output=None,
        s_n_hidden_states=None,
        s_dropout=None,
        s_in_channels=None,
        s_out_channels=None,
    ):
        """
        A custom neural network model designed for handling and integrating various types of meteorological and geographical data.
        This model incorporates data from different sources: precipitation (p), soil (s), basin attributes (c), and Global Forecast System (g).

        Parameters:
        - forecast_history: The length of the input sequence for precipitation data.
        - forecast_length: The length of the forecast period.
        - p_n_output: Output dimension for the precipitation (p) data path.
        - p_n_hidden_states: Number of hidden states in the LSTM for the precipitation path.
        - p_dropout: Dropout rate applied in the precipitation path.
        - p_in_channels: Number of input channels for the convolutional layer in the precipitation path.
        - p_out_channels: Number of output channels for the convolutional layer in the precipitation path.
        - len_c: Optional, the number of basin attribute (c) features.
        - s_forecast_history, s_n_output, s_n_hidden_states, s_dropout, s_in_channels, s_out_channels: Similar parameters for the soil (s) data path.

        Methods:
        - forward(*x_lst): Processes the input data depending on its composition:
            - If only precipitation data is provided, processes it through the primary precipitation path.
            - If precipitation and either basin attributes or soil data are provided, integrates these with precipitation data for processing.
            - If all three types of data (precipitation, basin attributes, soil) are provided, combines them effectively for prediction.

        This model's architecture is versatile, enabling it to adapt to various data availability scenarios, making it suitable for complex meteorological and geographical data processing tasks.
        """
        super(SPP_LSTM_Model_2, self).__init__()
        self.conv_p = nn.Conv2d(
            in_channels=p_in_channels,
            out_channels=p_out_channels,
            kernel_size=(3, 3),
            padding="same",
        )

        self.leaky_relu_p = nn.LeakyReLU(0.01)

        self.lstm_p = nn.LSTM(
            input_size=p_out_channels * 5 + len_c,
            hidden_size=p_n_hidden_states,
            batch_first=True,
        )

        self.dropout_p = nn.Dropout(p_dropout)

        self.fc_p = nn.Linear(p_n_hidden_states, p_n_output)

        self.spp_p = SppLayer([2, 1])

        self.p_length = forecast_history + forecast_length
        self.forecast_length = forecast_length

        if s_forecast_history is not None:
            self.conv_s = nn.Conv2d(
                in_channels=s_in_channels,
                out_channels=s_out_channels,
                kernel_size=(3, 3),
                padding="same",
            )

            self.leaky_relu_s = nn.LeakyReLU(0.01)
            self.sigmoid_s = nn.Sigmoid()

            self.lstm_s = nn.LSTM(
                input_size=s_out_channels * 5,
                hidden_size=s_n_hidden_states,
                batch_first=True,
            )

            self.dropout_s = nn.Dropout(s_dropout)

            self.fc_s = nn.Linear(s_n_hidden_states, s_n_output)

            self.spp_s = SppLayer([2, 1])

            self.s_length = s_forecast_history

    def forward(self, *x_lst):
        # c and s must be None, g might be None
        if len(x_lst) == 1:
            x = x_lst[0]
            x = x.view(-1, x.shape[2], x.shape[3], x.shape[4])
            x = self.conv_p(x)
            x = self.leaky_relu_p(x)
            x = self.spp_p(x)
            x = x.view(x.shape[0], -1)
            x = x.view(int(x.shape[0] / (self.p_length)), self.p_length, -1)
            x, _ = self.lstm_p(x)
            x = self.dropout_p(x)
            x = self.fc_p(x)
        # g might be None. either c or s must be None, but not both
        elif len(x_lst) == 2:
            p = x_lst[0]
            m = x_lst[1].permute(1, 0, 2)
            # c is not None
            if m.dim() == 3:
                p = p.view(-1, p.shape[2], p.shape[3], p.shape[4])
                p = self.conv_p(p)
                p = self.leaky_relu_p(p)
                p = self.spp_p(p)
                p = p.view(p.shape[0], -1)
                p = p.view(int(p.shape[0] / (self.p_length)), self.p_length, -1)
                x = torch.cat([p, m], dim=2)
                x, _ = self.lstm_p(x)
                x = self.dropout_p(x)
                x = self.fc_p(x)
            # s is not None
            else:
                p = p.view(-1, p.shape[2], p.shape[3], p.shape[4])
                p = self.conv_p(p)
                p = self.leaky_relu_p(p)
                p = self.spp_p(p)
                p = p.view(p.shape[0], -1)
                p = p.view(int(p.shape[0] / (self.p_length)), self.p_length, -1)
                p, _ = self.lstm_p(p)
                p = self.dropout_p(p)
                p = self.fc_p(p)

                m = m.view(-1, m.shape[2], m.shape[3], m.shape[4])
                m = self.conv_s(m)
                m = self.leaky_relu_s(m)
                m = self.spp_s(m)
                m = m.view(m.shape[0], -1)
                m = m.view(int(m.shape[0] / (self.s_length)), self.s_length, -1)
                m, _ = self.lstm_s(m)
                m = m[:, -1:, :]
                m = self.dropout_s(m)
                m = self.fc_s(m)
                m = self.sigmoid_s(m)

                x = m * p
        # g might be None. Both s and c are not None
        elif len(x_lst) == 3:
            p = x_lst[0]
            c = x_lst[1].permute(1, 0, 2)
            s = x_lst[2]

            p = p.view(-1, p.shape[2], p.shape[3], p.shape[4])
            p = self.conv_p(p)
            p = self.leaky_relu_p(p)
            p = self.spp_p(p)
            p = p.view(p.shape[0], -1)
            p = p.view(int(p.shape[0] / (self.p_length)), self.p_length, -1)
            p_c = torch.cat([p, c], dim=2)
            p_c, _ = self.lstm_p(p_c)
            p_c = self.dropout_p(p_c)
            p_c = self.fc_p(p_c)

            s = s.view(-1, s.shape[2], s.shape[3], s.shape[4])
            s = self.conv_s(s)
            s = self.leaky_relu_s(s)
            s = self.spp_s(s)
            s = s.view(s.shape[0], -1)
            s = s.view(int(s.shape[0] / (self.s_length)), self.s_length, -1)
            s, _ = self.lstm_s(s)
            s = s[:, -1:, :]
            s = self.dropout_s(s)
            s = self.fc_s(s)
            s = self.sigmoid_s(s)

            x = s * p_c
        return x[:, -self.forecast_length :, :]
