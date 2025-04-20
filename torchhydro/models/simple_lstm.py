"""
Author: Wenyu Ouyang
Date: 2023-09-19 09:36:25
LastEditTime: 2025-04-16 19:03:46
LastEditors: Wenyu Ouyang
Description: Some self-made LSTMs
FilePath: /torchhydro/torchhydro/models/simple_lstm.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import math
import torch as th
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor as T
from torch.nn import Parameter as P


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

    def forward(self, x):
        x0 = F.relu(self.linearIn(x))
        out_lstm, (hn, cn) = self.lstm(x0)
        out_lstm_dr = self.dropout(out_lstm)
        return self.linearOut(out_lstm_dr)


# class SimpleLSTM(nn.Module):
#     def __init__(self, input_size, output_size, hidden_size, dr=0.0):
#         super(SimpleLSTM, self).__init__()
#         self.linearIn = nn.Linear(input_size, hidden_size)
#         self.lstm = nn.LSTM(
#             hidden_size,
#             hidden_size,
#             1,
#             dropout=dr,
#         )
#         self.linearOut = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         x0 = F.relu(self.linearIn(x))
#         out_lstm, (hn, cn) = self.lstm(x0)
#         return self.linearOut(out_lstm)


class HoLSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, dr=0.0):
        super(HoLSTM, self).__init__()
        self.linearIn = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            1,
            dropout=dr,
        )
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.linearOut = nn.Linear(hidden_size, output_size)

    def forward(self, *x):
        if len(x) != 2:
            return self._eval_forward(*x)
        xf, xy = x
        x = torch.cat((xf, xy), dim=-1)
        x0 = F.relu(self.linearIn(x))
        out_lstm, (hn, cn) = self.lstm(x0)
        return self.linearOut(out_lstm)

    def _eval_forward(self, *x):
        x = x[0]
        seq_len, batch_size, _ = x.size()
        outputs = torch.zeros(seq_len, batch_size, self.output_size).to(x.device)
        h_n = torch.randn(1, batch_size, self.hidden_size).to(x.device) * 0.1
        c_n = torch.randn(1, batch_size, self.hidden_size).to(x.device) * 0.1

        # Initialize the previous output with zeros for the first time step
        prev_output = torch.randn(1, batch_size, self.output_size).to(x.device) * 0.1

        for t in range(seq_len):
            input_concat = torch.cat((x[t : t + 1, :, :], prev_output), dim=-1)

            # Pass through the initial linear layer
            x0 = F.relu(self.linearIn(input_concat))

            # LSTM step
            out_lstm, (h_n, c_n) = self.lstm(x0, (h_n, c_n))

            # Generate the current output
            prev_output = self.linearOut(out_lstm)
            outputs[t, :, :] = prev_output.squeeze(0)

        return outputs


class FoLSTM(nn.Module):
    def __init__(
        self, input_size, output_size, hidden_size, dr=0.0, teacher_forcing_ratio=0
    ):
        super(FoLSTM, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.dr = dr
        self.teacher_forcing_ratio = teacher_forcing_ratio

        self.linearIn = nn.Linear(
            input_size, hidden_size
        )  # Input now includes previous output
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            1,
            dropout=dr,
        )
        self.linearOut = nn.Linear(hidden_size, output_size)

    def forward(self, *x):
        """
        Args:
            x (torch.Tensor): Input sequence of shape (seq_len, batch_size, input_size)

        Returns:
            torch.Tensor: Output sequence of shape (seq_len, batch_size, output_size)
        """
        if len(x) == 2:
            x, xy = x
        else:
            x = x[0]
            device = x.device
            xy = torch.full(
                (
                    x.shape[0],  # seq
                    x.shape[1],  # batch_size
                    self.output_size,  # features
                ),
                float("nan"),
            ).to(device)
        seq_len, batch_size, _ = x.size()
        outputs = torch.zeros(seq_len, batch_size, self.output_size).to(x.device)
        h_n = torch.randn(1, batch_size, self.hidden_size).to(x.device) * 0.1
        c_n = torch.randn(1, batch_size, self.hidden_size).to(x.device) * 0.1

        # Initialize the previous output with zeros for the first time step
        prev_output = torch.randn(1, batch_size, self.output_size).to(x.device) * 0.1
        valid_mask = ~torch.isnan(xy)
        random_vals = torch.rand_like(valid_mask, dtype=torch.float)
        use_teacher_forcing = (random_vals < self.teacher_forcing_ratio) * valid_mask

        for t in range(seq_len):
            # Concatenate the current input with the previous output
            real_streamflow_input = xy[t : t + 1, :, :]
            prev_output = torch.where(
                use_teacher_forcing[t : t + 1, :, :], real_streamflow_input, prev_output
            )
            input_concat = torch.cat((x[t : t + 1, :, :], prev_output), dim=-1)

            # Pass through the initial linear layer
            x0 = F.relu(self.linearIn(input_concat))

            # LSTM step
            out_lstm, (h_n, c_n) = self.lstm(x0, (h_n, c_n))

            # Generate the current output
            prev_output = self.linearOut(out_lstm)
            outputs[t, :, :] = prev_output.squeeze(0)

        return outputs


class HFLSTM(nn.Module):
    def __init__(
        self, input_size, output_size, hidden_size, dr=0.0, teacher_forcing_ratio=0
    ):
        super(HFLSTM, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.dr = dr
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.linearIn = nn.Linear(input_size, hidden_size)

        self.fc_hidden = nn.Linear(hidden_size, hidden_size)
        self.fc_cell = nn.Linear(hidden_size, hidden_size)
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            1,
            dropout=dr,
        )
        self.linearOut = nn.Linear(hidden_size, output_size)

    def forward(self, *x):
        # train
        if len(x) == 3:
            x_hind, xf_fore, xq_fore = x
            x0_hind = F.relu(self.linearIn(x_hind))
            hind_out, (h_n, c_n) = self.lstm(x0_hind)
            hind_out_streamflow = self.linearOut(hind_out)
            prev_output = hind_out_streamflow[-1, :, :].unsqueeze(0)
            seq_len, batch_size, _ = xf_fore.size()
            outputs = torch.zeros(seq_len, batch_size, self.output_size).to(
                xf_fore.device
            )
            h_n = torch.tanh(self.fc_hidden(h_n))
            c_n = torch.tanh(self.fc_cell(c_n))

            valid_mask = ~torch.isnan(xq_fore)
            random_vals = torch.rand_like(valid_mask, dtype=torch.float)
            use_teacher_forcing = (
                random_vals < self.teacher_forcing_ratio
            ) * valid_mask

            for t in range(seq_len):
                # Concatenate the current input with the previous output
                real_streamflow_input = xq_fore[t : t + 1, :, :]
                prev_output = torch.where(
                    use_teacher_forcing[t : t + 1, :, :],
                    real_streamflow_input,
                    prev_output,
                )
                input_concat = torch.cat(
                    (xf_fore[t : t + 1, :, :], prev_output), dim=-1
                )

                # Pass through the initial linear layer
                x0 = F.relu(self.linearIn(input_concat))

                # LSTM step
                out_lstm, (h_n, c_n) = self.lstm(x0, (h_n, c_n))

                # Generate the current output
                prev_output = self.linearOut(out_lstm)
                outputs[t, :, :] = prev_output.squeeze(0)
        else:
            x_hind, xf_fore = x
            x0_hind = F.relu(self.linearIn(x_hind))
            hind_out, (h_n, c_n) = self.lstm(x0_hind)
            hind_out_streamflow = self.linearOut(hind_out)
            prev_output = hind_out_streamflow[-1, :, :].unsqueeze(0)
            seq_len, batch_size, _ = xf_fore.size()
            outputs = torch.zeros(seq_len, batch_size, self.output_size).to(
                xf_fore.device
            )
            for t in range(seq_len):
                input_concat = torch.cat(
                    (xf_fore[t : t + 1, :, :], prev_output), dim=-1
                )

                # Pass through the initial linear layer
                x0 = F.relu(self.linearIn(input_concat))

                # LSTM step
                out_lstm, (h_n, c_n) = self.lstm(x0, (h_n, c_n))

                # Generate the current output
                prev_output = self.linearOut(out_lstm)
                outputs[t, :, :] = prev_output.squeeze(0)

        outputs = torch.cat((hind_out_streamflow, outputs), dim=0)
        return outputs


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
