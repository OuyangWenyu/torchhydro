
from typing import overload, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.modules.rnn import RNNBase
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence

__all__ = [
    "RNNBase",
    "narx",
]

_narx_impls = {
    "narx_tanh",
    "narx_relu",
}

class narx(RNNBase):
    """
    narx model
    nonlinear autoregressive with exogenous inputs neural network.
    """
    @overload
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        nonlinearity: str = "tanh",
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        proj_size: int = 0,
        device=None,
        dtype=None,
    ) -> None:
        ...

    @overload
    def __init__(self, *args, **kwargs):
        ...

    def __init__(self, *args, **kwargs):
        if "proj_size" in kwargs:
            raise ValueError(
                "proj_size argument is not supported for narx"
            )
        if len(args) > 3:
            self.nonlinearity = args[3]
            args = args[:3] + args[4:]
        else:
            self.nonlinearity = kwargs.pop("nonlinearity", "tanh")
        if self.nonlinearity == "tanh":
            mode = "narx_tanh"
        elif self.nonlinearity == "relu":
            mode = "narx_relu"
        else:
            raise ValueError(
                f"Unknown nonlinearity '{self.nonlinearity}'. Select from 'tanh' or 'relu'."
            )
        super().__init__(mode, *args, **kwargs)
