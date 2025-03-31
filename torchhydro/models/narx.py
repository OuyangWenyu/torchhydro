import torch.nn as nn
from torch.nn.modules.rnn import RNNBase
from typing import overload

__all__ = [
    "RNNBase",
    "narx",
]

class narx(RNNBase):
    """
    narx model
    nonlinear autoregressive with exogenous inputs
    """
    @overload
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
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
        super().__init__("narx", *args, **kwargs)

    def __init__(self):
        super(narx, self).__init__()
