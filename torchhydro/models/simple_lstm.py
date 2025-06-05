"""
Author: Wenyu Ouyang
Date: 2023-09-19 09:36:25
LastEditTime: 2025-06-05 09:06:53
LastEditors: Wenyu Ouyang
Description: Some self-made LSTMs
FilePath: \torchhydro\torchhydro\models\simple_lstm.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import math
import torch as th
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor as T
from torch.nn import Parameter as P
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SimpleLSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, dr=0.0):
        super(SimpleLSTM, self).__init__()
        self.linearIn = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
        )
        self.dropout = nn.Dropout(p=dr)
        self.linearOut = nn.Linear(hidden_size, output_size)

    def forward(self, x, *args):
        """
        Forward pass of SimpleLSTM

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor
        *args : optional
            If provided, args[0] should be sequence lengths for PackedSequence
        """
        x0 = F.relu(self.linearIn(x))

        # if args is not None and args[0] is not None, use PackedSequence
        if args and args[0] is not None:
            seq_lengths = args[0]
            packed_x = pack_padded_sequence(
                x0, seq_lengths, batch_first=False, enforce_sorted=False
            )
            packed_out, (hn, cn) = self.lstm(packed_x)
            out_lstm, _ = pad_packed_sequence(packed_out, batch_first=False)
        else:
            # standard processing, directly use LSTM
            out_lstm, (hn, cn) = self.lstm(x0)

        out_lstm_dr = self.dropout(out_lstm)
        return self.linearOut(out_lstm_dr)


class HFLSTM(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size,
        dr=0.0,
        teacher_forcing_ratio=0,
        hindcast_with_output=True,
    ):
        """

        Parameters
        ----------
        input_size : int
            without streamflow
        output_size : int
            streamflow
        hidden_size : int
        dr : float, optional
            dropout, by default 0.0
        teacher_forcing_ratio : float, optional
            by default 0
        hindcast_with_output : bool, optional
            whether to use the output of the model as input for the next time step, by default True
        """
        super(HFLSTM, self).__init__()
        self.linearIn = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
        )
        self.dropout = nn.Dropout(p=dr)
        self.linearOut = nn.Linear(hidden_size, output_size)
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.hindcast_with_output = hindcast_with_output
        self.hidden_size = hidden_size
        self.output_size = output_size

    def _teacher_forcing_preparation(self, xq_hor):
        # teacher forcing preparation
        valid_mask = ~torch.isnan(xq_hor)
        random_vals = torch.rand_like(valid_mask, dtype=torch.float)
        return (random_vals < self.teacher_forcing_ratio) * valid_mask

    def _rho_forward(self, x_rho):
        x0_rho = F.relu(self.linearIn(x_rho))
        out_lstm_rho, (hn_rho, cn_rho) = self.lstm(x0_rho)
        out_lstm_rho_dr = self.dropout(out_lstm_rho)
        out_lstm_rho_lnout = self.linearOut(out_lstm_rho_dr)
        prev_output = out_lstm_rho_lnout[-1:, :, :]
        return out_lstm_rho_lnout, hn_rho, cn_rho, prev_output

    def forward(self, *x):
        xfc_rho, xfc_hor, xq_rho, xq_hor = x

        x_rho = torch.cat((xfc_rho, xq_rho), dim=-1)
        hor_len, batch_size, _ = xfc_hor.size()

        # hindcast-forecast, we do not have forecast-hindcast situation
        # do rho forward first, prev_output is the last output of rho (seq_length = 1, batch_size, feature = output_size)
        if self.hindcast_with_output:
            _, h_n, c_n, prev_output = self._rho_forward(x_rho)
            seq_len = hor_len
        else:
            # TODO: need more test
            seq_len = xfc_rho.shape[0] + hor_len
            xfc_hor = torch.cat((xfc_rho, xfc_hor), dim=0)
            xq_hor = torch.cat((xq_rho, xq_hor), dim=0)
            h_n = torch.randn(1, batch_size, self.hidden_size).to(xfc_rho.device) * 0.1
            c_n = torch.randn(1, batch_size, self.hidden_size).to(xfc_rho.device) * 0.1
            prev_output = (
                torch.randn(1, batch_size, self.output_size).to(xfc_rho.device) * 0.1
            )

        use_teacher_forcing = self._teacher_forcing_preparation(xq_hor)

        # do hor forward
        outputs = torch.zeros(seq_len, batch_size, self.output_size).to(xfc_rho.device)
        # TODO: too slow here when seq_len is large, need to optimize
        for t in range(seq_len):
            real_streamflow_input = xq_hor[t : t + 1, :, :]
            prev_output = torch.where(
                use_teacher_forcing[t : t + 1, :, :],
                real_streamflow_input,
                prev_output,
            )
            input_concat = torch.cat((xfc_hor[t : t + 1, :, :], prev_output), dim=-1)

            # Pass through the initial linear layer
            x0 = F.relu(self.linearIn(input_concat))

            # LSTM step
            out_lstm, (h_n, c_n) = self.lstm(x0, (h_n, c_n))

            # Generate the current output
            prev_output = self.linearOut(out_lstm)
            outputs[t, :, :] = prev_output.squeeze(0)
        # Return the outputs
        return outputs[-hor_len:, :, :]


class MultiLayerLSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers=1, dr=0.0):
        super(MultiLayerLSTM, self).__init__()
        self.linearIn = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers,
            dropout=dr,
        )
        self.linearOut = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x0 = F.relu(self.linearIn(x))
        out_lstm, (hn, cn) = self.lstm(x0)
        return self.linearOut(out_lstm)


class LinearSimpleLSTMModel(SimpleLSTM):
    """
    This model is nonlinear layer + SimpleLSTM.
    """

    def __init__(self, linear_size, **kwargs):
        """

        Parameters
        ----------
        linear_size
            the number of input features for the first input linear layer
        """
        super(LinearSimpleLSTMModel, self).__init__(**kwargs)
        self.former_linear = nn.Linear(linear_size, kwargs["input_size"])

    def forward(self, x):
        x0 = F.relu(self.former_linear(x))
        return super(LinearSimpleLSTMModel, self).forward(x0)


class LinearMultiLayerLSTMModel(MultiLayerLSTM):
    """
    This model is nonlinear layer + MultiLayerLSTM.
    """

    def __init__(self, linear_size, **kwargs):
        """

        Parameters
        ----------
        linear_size
            the number of input features for the first input linear layer
        """
        super(LinearMultiLayerLSTMModel, self).__init__(**kwargs)
        self.former_linear = nn.Linear(linear_size, kwargs["input_size"])

    def forward(self, x):
        x0 = F.relu(self.former_linear(x))
        return super(LinearMultiLayerLSTMModel, self).forward(x0)


class SimpleLSTMForecast(SimpleLSTM):
    def __init__(self, input_size, output_size, hidden_size, forecast_length, dr=0.0):
        super(SimpleLSTMForecast, self).__init__(
            input_size, output_size, hidden_size, dr
        )
        self.forecast_length = forecast_length

    def forward(self, x):
        # 调用父类的forward方法获取完整的输出
        full_output = super(SimpleLSTMForecast, self).forward(x)

        return full_output[-self.forecast_length :, :, :]


class SlowLSTM(nn.Module):
    """
    A pedagogic implementation of Hochreiter & Schmidhuber:
    'Long-Short Term Memory'
    http://www.bioinf.jku.at/publications/older/2604.pdf
    """

    def __init__(self, input_size, hidden_size, bias=True, dropout=0.0):
        super(SlowLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.dropout = dropout
        # input to hidden weights
        self.w_xi = P(T(hidden_size, input_size))
        self.w_xf = P(T(hidden_size, input_size))
        self.w_xo = P(T(hidden_size, input_size))
        self.w_xc = P(T(hidden_size, input_size))
        # hidden to hidden weights
        self.w_hi = P(T(hidden_size, hidden_size))
        self.w_hf = P(T(hidden_size, hidden_size))
        self.w_ho = P(T(hidden_size, hidden_size))
        self.w_hc = P(T(hidden_size, hidden_size))
        # bias terms
        self.b_i = T(hidden_size).fill_(0)
        self.b_f = T(hidden_size).fill_(0)
        self.b_o = T(hidden_size).fill_(0)
        self.b_c = T(hidden_size).fill_(0)

        # Wrap biases as parameters if desired, else as variables without gradients
        W = P if bias else (lambda x: P(x, requires_grad=False))
        self.b_i = W(self.b_i)
        self.b_f = W(self.b_f)
        self.b_o = W(self.b_o)
        self.b_c = W(self.b_c)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        h, c = hidden
        h = h.view(h.size(0), -1)
        c = c.view(h.size(0), -1)
        x = x.view(x.size(0), -1)
        # Linear mappings
        i_t = th.mm(x, self.w_xi) + th.mm(h, self.w_hi) + self.b_i
        f_t = th.mm(x, self.w_xf) + th.mm(h, self.w_hf) + self.b_f
        o_t = th.mm(x, self.w_xo) + th.mm(h, self.w_ho) + self.b_o
        # activations
        i_t.sigmoid_()
        f_t.sigmoid_()
        o_t.sigmoid_()
        # cell computations
        c_t = th.mm(x, self.w_xc) + th.mm(h, self.w_hc) + self.b_c
        c_t.tanh_()
        c_t = th.mul(c, f_t) + th.mul(i_t, c_t)
        h_t = th.mul(o_t, th.tanh(c_t))
        # Reshape for compatibility
        h_t = h_t.view(h_t.size(0), 1, -1)
        c_t = c_t.view(c_t.size(0), 1, -1)
        if self.dropout > 0.0:
            F.dropout(h_t, p=self.dropout, training=self.training, inplace=True)
        return h_t, (h_t, c_t)

    def sample_mask(self):
        pass
