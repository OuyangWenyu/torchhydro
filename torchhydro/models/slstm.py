"""slstm model"""
import torch
import torch.nn as nn


class sLstm(nn.module):
    def __init__(
            self, 
            input_size, 
            output_size, 
            hidden_size, 
            dr=0.0
    ):
        """
        Initiate a sLstm
        """
        super(sLstm, self).__init__()
        self.linearIn = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            1,
            dropout=dr,
        )
        self.linearOut = nn.Linear(hidden_size, output_size)
