"""
Author: Wenyu Ouyang
Date: 2021-12-17 18:02:27
LastEditTime: 2025-01-01 18:20:54
LastEditors: Wenyu Ouyang
Description: ANN model
FilePath: \torchhydro\torchhydro\models\ann.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""

from typing import Union

import torch
from torch import nn


class SimpleAnn(nn.Module):
    def __init__(
        self,
        nx: int,
        ny: int,
        hidden_size: Union[int, tuple, list] = None,
        dr: Union[float, tuple, list] = 0.0,
        activation: str = "relu",
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
        """
        super(SimpleAnn, self).__init__()
        linear_list = nn.ModuleList()
        dropout_list = nn.ModuleList()
        if (
            hidden_size is None
            or (type(hidden_size) is int and hidden_size == 0)
            or (type(hidden_size) in [tuple, list] and len(hidden_size) < 1)
        ):
            linear_list.add_module("linear1", nn.Linear(nx, ny))
            if type(dr) in [tuple, list]:
                dr = dr[0]
            if dr > 0.0:
                dropout_list.append(nn.Dropout(dr))
        elif type(hidden_size) is int:
            linear_list.add_module("linear1", nn.Linear(nx, hidden_size))
            linear_list.add_module("linear2", nn.Linear(hidden_size, ny))
            if type(dr) in [tuple, list]:
                # dropout layer do not have additional weights, so we do not name them here
                dropout_list.append(nn.Dropout(dr[0]))
                if len(dr) > 1:
                    dropout_list.append(nn.Dropout(dr[1]))
            else:
                dropout_list.append(dr)
        else:
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
                linear_list.add_module(
                    "linear%d" % (i + 1 + 1),
                    nn.Linear(hidden_size[i], hidden_size[i + 1]),
                )
                dropout_list.append(
                    nn.Dropout(dr[i + 1]),
                )
            linear_list.add_module(
                "linear%d" % (len(hidden_size) + 1),
                nn.Linear(hidden_size[-1], ny),
            )
            if len(dr) == len(linear_list):
                # if final linear also need a dr
                dropout_list.append(nn.Dropout(dr[-1]))
        self.linear_list = linear_list
        self.dropout_list = dropout_list
        self.activation = self._get_activation(activation)

    def forward(self, x):
        for i, model in enumerate(self.linear_list):
            if i == 0:
                if len(self.linear_list) == 1:
                    # final layer must be a linear layer
                    return (
                        model(x)
                        if len(self.dropout_list) < len(self.linear_list)
                        else self.dropout_list[i](model(x))
                    )
                else:
                    out = self.activation(self.dropout_list[i](model(x)))
            elif i == len(self.linear_list) - 1:
                # in final layer, no relu again
                return (
                    model(out)
                    if len(self.dropout_list) < len(self.linear_list)
                    else self.dropout_list[i](model(out))
                )
            else:
                out = self.activation(self.dropout_list[i](model(out)))

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
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with activation followed by dropout for each layer in linear_list.
        The number of dropout layers must match or be exactly one less than
        the number of linear layers.
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
        for i, layer in enumerate(self.linear_list):
            out = layer(out)
            out = self.activation(out)
            if i < len(self.dropout_list):
                out = self.dropout_list[i](out)

        return out
