"""
Author: Wenyu Ouyang
Date: 2021-12-17 18:02:27
LastEditTime: 2025-06-25 14:07:58
LastEditors: Wenyu Ouyang
Description: ANN model
FilePath: \torchhydro\torchhydro\models\ann.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""

from typing import Union

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class SimpleAnn(nn.Module):
    def __init__(
        self,
        nx: int,
        ny: int,
        hidden_size: Union[int, tuple, list] = None,
        dr: Union[float, tuple, list] = 0.0,
        activation: str = "relu",
        normalization: str = None,
    ):
        """
        A simple multi-layer NN model with final linear layer

        Parameters
        ----------
        nx
            number of input neurons
        ny
            number of output neurons
        hidden_size
            a list/tuple which contains number of neurons in each hidden layer;
            if int, only one hidden layer except for hidden_size=0
        dr
            dropout rate of layers, default is 0.0 which means no dropout;
            here we set number of dropout layers to (number of nn layers - 1)
        activation
            activation function for hidden layers, default is "relu"
        normalization
            normalization type, "LayerNorm" or "RMSNorm" or None
        """
        super(SimpleAnn, self).__init__()
        linear_list = nn.ModuleList()
        dropout_list = nn.ModuleList()
        normalization_list = nn.ModuleList()
        self.normalization = normalization
        if (
            hidden_size is None
            or (type(hidden_size) is int and hidden_size == 0)
            or (type(hidden_size) in [tuple, list] and len(hidden_size) < 1)
        ):
            self._add_normalization(normalization_list, nx, normalization)
            linear_list.add_module("linear1", nn.Linear(nx, ny))
            if type(dr) in [tuple, list]:
                dr = dr[0]
            if dr > 0.0:
                dropout_list.append(nn.Dropout(dr))
        elif type(hidden_size) is int:
            self._add_normalization(normalization_list, nx, normalization)
            linear_list.add_module("linear1", nn.Linear(nx, hidden_size))
            self._add_normalization(normalization_list, hidden_size, normalization)
            linear_list.add_module("linear2", nn.Linear(hidden_size, ny))
            if type(dr) in [tuple, list]:
                # dropout layer do not have additional weights, so we do not name them here
                dropout_list.append(nn.Dropout(dr[0]))
                if len(dr) > 1:
                    dropout_list.append(nn.Dropout(dr[1]))
            else:
                # dr must be a float
                dropout_list.append(nn.Dropout(dr))
        else:
            self._add_normalization(normalization_list, nx, normalization)
            linear_list.add_module("linear1", nn.Linear(nx, hidden_size[0]))

            if type(dr) is float:
                dr = [dr] * len(hidden_size)
            elif len(dr) > len(hidden_size) + 1:
                raise ArithmeticError(
                    "At most, we set dropout layer for each nn layer, please check the number of dropout layers"
                )
            # dropout_list.add_module("dropout1", torch.nn.Dropout(dr[0]))
            dropout_list.append(nn.Dropout(dr[0]))
            for i in range(len(hidden_size) - 1):
                self._add_normalization(
                    normalization_list, hidden_size[i], normalization
                )
                linear_list.add_module(
                    "linear%d" % (i + 1 + 1),
                    nn.Linear(hidden_size[i], hidden_size[i + 1]),
                )
                dropout_list.append(
                    nn.Dropout(dr[i + 1]),
                )
            self._add_normalization(normalization_list, hidden_size[-1], normalization)
            linear_list.add_module(
                "linear%d" % (len(hidden_size) + 1),
                nn.Linear(hidden_size[-1], ny),
            )
            if len(dr) == len(linear_list):
                # if final linear also need a dr
                dropout_list.append(nn.Dropout(dr[-1]))
        self.linear_list = linear_list
        self.dropout_list = dropout_list
        self.normalization_list = normalization_list
        self.activation = self._get_activation(activation)

    def _add_normalization(self, normalization_list, hidden_size, normalization):
        if normalization == "LayerNorm":
            normalization_list.append(nn.LayerNorm(hidden_size))
        elif normalization == "RMSNorm":
            normalization_list.append(RMSNorm(hidden_size))
        else:
            pass

    def forward(self, x):
        # Refactored order: Norm -> Activation -> Linear -> Dropout
        out = x
        norm_idx = 0

        for i, model in enumerate(self.linear_list):
            # Apply Normalization
            if self.normalization is not None and norm_idx < len(
                self.normalization_list
            ):
                out = self.normalization_list[norm_idx](out)
                norm_idx += 1

            # Apply Activation
            # Note: Applying activation to raw input (for first layer) or normalized input
            out = self.activation(out)

            # Apply Linear
            out = model(out)

            # Apply Dropout
            if i < len(self.dropout_list):
                out = self.dropout_list[i](out)

        return out

    def _get_activation(self, name: str) -> nn.Module:
        """a function to get activation function by name, reference from:
        https://github.com/neuralhydrology/neuralhydrology/blob/master/neuralhydrology/modelzoo/fc.py

        Parameters
        ----------
        name : str
            _description_

        Returns
        -------
        nn.Module
            _description_

        Raises
        ------
        NotImplementedError
            _description_
        """
        if name.lower() == "tanh":
            activation = nn.Tanh()
        elif name.lower() == "sigmoid":
            activation = nn.Sigmoid()
        elif name.lower() == "relu":
            activation = nn.ReLU()
        elif name.lower() == "linear":
            activation = nn.Identity()
        elif name.lower() == "gelu":
            activation = nn.GELU()
        else:
            raise NotImplementedError(
                f"{name} currently not supported as activation in this class"
            )
        return activation


class Mlp(SimpleAnn):
    def __init__(
        self,
        nx: int,
        ny: int,
        hidden_size: Union[int, tuple, list] = None,
        dr: Union[float, tuple, list] = 0.0,
        activation: str = "relu",
        normalization: str = None,
    ):
        """
        MLP model inherited from SimpleAnn, using activation + dropout after each layer.
        The final layer also goes through activation+dropout if there's a corresponding
        dropout layer in dropout_list.
        """
        if type(dr) is float:
            if type(hidden_size) in [tuple, list]:
                dr = [dr] * (len(hidden_size) + 1)
            elif hidden_size is not None and hidden_size > 0:
                dr = [dr] * 2
        super(Mlp, self).__init__(
            nx=nx,
            ny=ny,
            hidden_size=hidden_size,
            dr=dr,
            activation=activation,
            normalization=normalization,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with Norm -> Activation -> Linear -> Dropout
        """
        # Raise an error if the number of linear layers and dropout layers do not match
        # and do not satisfy "number of linear layers = number of dropout layers + 1"
        if len(self.linear_list) != len(self.dropout_list) and (
            len(self.linear_list) - 1
        ) != len(self.dropout_list):
            raise ValueError(
                "Mlp: linear_list and dropout_list sizes do not match. "
                "They either have the same length or linear_list has exactly one more."
            )

        out = x
        norm_idx = 0
        for i, layer in enumerate(self.linear_list):
            # Apply Normalization
            if self.normalization is not None and norm_idx < len(
                self.normalization_list
            ):
                out = self.normalization_list[norm_idx](out)
                norm_idx += 1

            # Apply Activation
            out = self.activation(out)

            # Apply Linear
            out = layer(out)

            # Apply Dropout
            if i < len(self.dropout_list):
                out = self.dropout_list[i](out)

        return out
