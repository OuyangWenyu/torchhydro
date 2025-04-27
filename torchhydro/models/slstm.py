"""slstm model"""

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.slstm = nn.Linear(input_size, hidden_size)
