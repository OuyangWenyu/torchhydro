"""slstm model"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.rnn import RNNBase
from torch import Tensor
from torch.nn import Parameter

class sLSTM(nn.Module):
    """
    stacked lstm model.
    """
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        device=None,
        dtype=None,
    ):
        """

        Parameters
        ----------
        input_size
        output_size
        hidden_size
        num_layers
        dropout
        device
        dtype
        """
        super(sLSTM, self).__init__()
        self.linearIn = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers,
            dropout=dropout,
        )
        self.linearOut = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x0 = F.relu(self.linearIn(x))
        out_lstm, (hn, cn) = self.lstm(x0)
        out = self.linearOut(out_lstm)
        return out

class MI_STL_sLSTM(nn.Module):
    """

    """
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        num_layers: int = 10,
        dropout: float = 0.0,
        device=None,
        dtype=None,
    ):
        """

        Parameters
        ----------
        input_size
        output_size
        hidden_size
        num_layers
        dropout
        device
        dtype
        """
        super(MI_STL_sLSTM, self).__init__()
        self.slstm = sLSTM(input_size, output_size, hidden_size, num_layers, dropout)


class pcLSTM(RNNBase):
    """

    """
    def __init__(
        self,
        # mode: str = "LSTM",
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        nonlinearity: str = "tanh",
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ) -> None:
        """

        Parameters
        ----------
        input_size
        hidden_size
        num_layers
        nonlinearity
        bias
        batch_first
        dropout
        bidirectional
        """
        super(pcLSTM, self).__init__("LSTM")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.bias = bias
        self.dropout = dropout
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.gate_size = 4 * hidden_size  # lstm 4

        # self.w_ih = Parameter(torch.Tensor(hidden_size * 4, input_size))
        # self.w_hh = Parameter(torch.Tensor(hidden_size * 4, hidden_size))
        # self.b_ih = Parameter(torch.Tensor(hidden_size * 4))
        # self.b_hh = Parameter(torch.Tensor(hidden_size * 4))
    def forward(self, input, hx=None):
        """
        narx forward function
        Parameters
        ----------
        input
            input time series
        hx
            hidden state
        Returns
        -------

        """
        if self.bidirectional:
            num_directions = 2
        else:
            num_directions = 1
        h = torch.zeros(  # hidden space
            self.num_layers * num_directions,
            self.batch_size,
            self.hidden_size,
            dtype=input.dtype,
            device=input.device,
        )
        c = torch.zeros(  # cell space
            self.num_layers * num_directions,
            self.batch_size,
            self.hidden_size,
            dtype=input.dtype,
            device=input.device,
        )
        b = None
        gates = F.linear(input, hx, self.b_ih) + F.linear(h0, w_hh, self.b_hh)
        gate_i, gate_f, gate_c, gate_o = gates.chunk(4, 1)
