"""
Author: Wenyu Ouyang
Date: 2021-12-17 18:02:27
LastEditTime: 2023-10-06 19:26:09
LastEditors: Wenyu Ouyang
Description: ANN model
FilePath: \torchhydro\torchhydro\models\ann.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""

from typing import Union

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
        elif type(hidden_size) is int:
            if type(dr) in [tuple, list]:
                dr = dr[0]
            linear_list.add_module("linear1", nn.Linear(nx, hidden_size))
            # dropout layer do not have additional weights, so we do not name them here
            dropout_list.append(nn.Dropout(dr))
            linear_list.add_module("linear2", nn.Linear(hidden_size, ny))
        else:
            linear_list.add_module("linear1", nn.Linear(nx, hidden_size[0]))
            if type(dr) is float:
                dr = [dr] * len(hidden_size)
            elif len(dr) != len(hidden_size):
                raise ArithmeticError(
                    "We set dropout layer for each nn layer, please check the number of dropout layers"
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
        self.linear_list = linear_list
        self.dropout_list = dropout_list
        self.activation = self._get_activation(activation)

    def forward(self, x):
        for i, model in enumerate(self.linear_list):
            if i == 0:
                if len(self.linear_list) == 1:
                    return model(x)
                else:
                    out = self.activation(self.dropout_list[i](model(x)))
            elif i == len(self.linear_list) - 1:
                # in final layer, no relu again
                return model(out)
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
