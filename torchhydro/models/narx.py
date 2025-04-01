
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
        close_loop: bool = False,
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

    @overload
    @torch._jit_internal._overload_method
    def forward(
        self, input: Tensor, hx: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        pass

    @overload
    @torch._jit_internal._overload_method
    def forward(
        self, input: PackedSequence, hx: Optional[Tensor] = None
    ) -> Tuple[PackedSequence, Tensor]:
        pass

    def forward(self, input, hx=None):
        """
        narx forward function
        Parameters
        ----------
        input
            input time series
        hx

        Returns
        -------

        """
        self._update_flat_weights()

        num_directions = 2 if self.bidirectional else 1
        orig_input = input

        if isinstance(orig_input, PackedSequence):
            input, batch_sizes, sorted_indices, unsorted_indices = input
            max_batch_size = batch_sizes[0]

            if hx is None:
                hx = torch.zeros(
                    self.num_layers * num_directions,
                    max_batch_size,
                    self.hidden_size,
                    dtype=input.dtype,
                    device=input.device,
                )
            else:
                hx = self.permute_hidden(hx, sorted_indices)
        else:
            batch_sizes = None
            if input.dim() not in (2, 3):
                raise ValueError(
                    f"narx: Expected input to be 2D or 3D, got {input.dim()}D tensor instead"
                )
            is_batched = input.dim() == 3
            batch_dim = 0 if self.batch_first else 1
            if not is_batched:
                input = input.unsqueeze(batch_dim)
                if hx is not None:
                    if hx.dim() != 2:
                        raise RuntimeError(
                            f"For unbatched 2-D input, hx should also be 2-D but got {hx.dim()}-D tensor"
                        )
                    hx = hx.unsqueeze(1)
            else:
                if hx is not None and hx.dim() != 3:
                    raise RuntimeError(
                        f"For batched 3-D input, hx should also be 3-D but got {hx.dim()}-D tensor"
                    )
