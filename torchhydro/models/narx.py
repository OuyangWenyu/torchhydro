
from typing import overload, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.modules.rnn import RNNBase
from torch import _VF, Tensor
from torch.nn.utils.rnn import PackedSequence

__all__ = [
    "RNNBase",
    "Narx",
]

_narx_impls = {
    "RNN_TANH": _VF.rnn_tanh,
    "RNN_RELU": _VF.rnn_relu,
}

class Narx(RNNBase):
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
        """

        Parameters
        ----------
        input_size
        hidden_size
        num_layers
            the number of recurrent stack layers
        nonlinearity
        bias
        batch_first
        dropout
        bidirectional
        proj_size
        close_loop
            default false when train period.
        device
        dtype
        """
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
            mode = "RNN_TANH"
        elif self.nonlinearity == "relu":
            mode = "RNN_RELU"
        else:
            raise ValueError(
                f"Unknown nonlinearity '{self.nonlinearity}'. Select from 'tanh' or 'relu'."
            )
        super().__init__(mode, *args, **kwargs)
        self.linearOut = nn.Linear(hidden_size, output_size)

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
            hidden state
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
        else:  #
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
            else:  #
                if hx is not None and hx.dim() != 3:
                    raise RuntimeError(
                        f"For batched 3-D input, hx should also be 3-D but got {hx.dim()}-D tensor"
                    )
            max_batch_size = input.size(0) if self.batch_first else input.size(1)
            sorted_indices = None
            unsorted_indices = None
            if hx is None:  #
                hx = torch.zeros(
                    self.num_layers * num_directions,
                    max_batch_size,
                    self.hidden_size,
                    dtype=input.dtype,
                    device=input.device,
                )
            else:
                hx = self.permute_hidden(hx, sorted_indices)

        assert hx is not None
        self.check_forward_args(input, hx, batch_sizes)
        assert self.mode == "RNN_TANH" or self.mode == "RNN_RELU"
        if batch_sizes is None:  #
            if self.mode == "RNN_TANH":  #
                result = _VF.rnn_tanh(
                    input,
                    hx,
                    self._flat_weights,
                    self.bias,
                    self.num_layers,
                    self.dropout,
                    self.training,
                    self.bidirectional,
                    self.batch_first,
                )
            else:
                result = _VF.rnn_relu(
                    input,
                    hx,
                    self._flat_weights,
                    self.bias,
                    self.num_layers,
                    self.dropout,
                    self.training,
                    self.bidirectional,
                    self.batch_first,
                )
        else:
            if self.mode == "RNN_TANH":
                result = _VF.rnn_tanh(
                    input,
                    batch_sizes,
                    hx,
                    self._flat_weights,
                    self.bias,
                    self.num_layers,
                    self.dropout,
                    self.training,
                    self.bidirectional,
                )
            else:
                result = _VF.rnn_relu(
                    input,
                    batch_sizes,
                    hx,
                    self._flat_weights,
                    self.bias,
                    self.num_layers,
                    self.dropout,
                    self.training,
                    self.bidirectional,
                )

        output = result[0]
        hidden = result[1]

        if isinstance(orig_input, PackedSequence):
            output_packed = PackedSequence(
                output, batch_sizes, sorted_indices, unsorted_indices
            )
            return output_packed, self.permute_hidden(hidden, unsorted_indices)

        if not is_batched:
            output = output.squeeze(batch_dim)
            hidden = hidden.squeeze(1)

        return output, self.permute_hidden(hidden, unsorted_indices)

    def close_loop(self):
        """
        close loop when prediction period.
        Returns
        -------

        """

        return 0
