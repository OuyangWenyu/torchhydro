import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Optional


class MCLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        bias: bool = True,
        initial_forget_bias: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout if num_layers > 1 else 0.0
        self.bias = bias
        self.initial_forget_bias = initial_forget_bias

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bias=self.bias,
        )
        self.linearOut = nn.Linear(self.hidden_size, self.output_size)

        self._init_forget_bias(self.initial_forget_bias)

    def _init_forget_bias(self, forget_bias_value: float) -> None:
        if forget_bias_value == 0.0:
            return
        for layer in range(self.num_layers):
            bias_ih = getattr(self.lstm, f"bias_ih_l{layer}")
            bias_hh = getattr(self.lstm, f"bias_hh_l{layer}")
            bias_ih.data[self.hidden_size : 2 * self.hidden_size] = forget_bias_value
            bias_hh.data[self.hidden_size : 2 * self.hidden_size] = forget_bias_value

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        seq_lengths: Optional[torch.Tensor] = kwargs.get("seq_lengths", None)
        mask: Optional[torch.Tensor] = kwargs.get("mask", None)
        use_manual_mask: bool = kwargs.get("use_manual_mask", False)

        x_in = x
        if use_manual_mask and mask is not None:
            x_in = x.masked_fill(mask == 0, 0.0)
        if x_in.dtype != torch.float32:
            x_in = x_in.float()
        x_in = x_in.contiguous()

        if seq_lengths is not None and not use_manual_mask:
            packed = nn.utils.rnn.pack_padded_sequence(
                x_in.contiguous(), seq_lengths.cpu(), enforce_sorted=False
            )
            self.lstm.flatten_parameters()
            out_packed, _ = self.lstm(packed)
            out_lstm, _ = nn.utils.rnn.pad_packed_sequence(out_packed)
        else:
            self.lstm.flatten_parameters()
            out_lstm, _ = self.lstm(x_in.contiguous())

        out = self.linearOut(out_lstm)
        return out
