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

import torch
import torch.nn.functional as F


class SimpleAnn(torch.nn.Module):
    def __init__(self, nx: int, ny: int, hidden_size: Union[int, tuple, list] = None,  # Union,联合。hidden_size可以是int,tuple,list中的任意一种。 [tuple, list]；隐藏的层数及各层的神经元个数。
                 dr: Union[float, tuple, list] = 0.0):
        """
        A simple multi-layer NN model with final linear layer  简单多层神经网络模型，最后一层是线性层。

        Parameters
        ----------
        nx
            number of input neurons  输入神经元个数
        ny
            number of output neurons  输出神经元个数
        hidden_size  隐藏大小，二维。 e.g. [3,4,5,4,3], 五层，各层的
            a list/tuple which contains number of neurons in each hidden layer; 一个列或一个元组，每层隐藏层包含一些神经元。
            if int, only one hidden layer except for hidden_size=0 除了等于零外，如果是 int, 表示只有一层隐藏层。
        dr
            dropout rate of layers, default is 0.0 which means no dropout;
            here we set number of dropout layers to (number of nn layers - 1)
        """
        super(SimpleAnn, self).__init__()
        linear_list = torch.nn.ModuleList()
        dropout_list = torch.nn.ModuleList()
        if (
                hidden_size is None
                or (type(hidden_size) is int and hidden_size == 0)
                or (type(hidden_size) in [tuple, list] and len(hidden_size) < 1)
        ):
            linear_list.add_module("linear1", torch.nn.Linear(nx, ny))
        elif type(hidden_size) is int:
            if type(dr) in [tuple, list]:
                dr = dr[0]
            linear_list.add_module("linear1", torch.nn.Linear(nx, hidden_size))
            # dropout layer do not have additional weights, so we do not name them here
            dropout_list.append(torch.nn.Dropout(dr))
            linear_list.add_module("linear2", torch.nn.Linear(hidden_size, ny))
        else:
            linear_list.add_module("linear1", torch.nn.Linear(nx, hidden_size[0]))
            if type(dr) is float:
                dr = [dr] * len(hidden_size)
            elif len(dr) != len(hidden_size):
                raise ArithmeticError(
                    "We set dropout layer for each nn layer, please check the number of dropout layers")
            # dropout_list.add_module("dropout1", torch.nn.Dropout(dr[0]))
            dropout_list.append(torch.nn.Dropout(dr[0]))
            for i in range(len(hidden_size) - 1):
                linear_list.add_module(
                    "linear%d" % (i + 1 + 1),
                    torch.nn.Linear(hidden_size[i], hidden_size[i + 1]),
                )
                dropout_list.append(
                    torch.nn.Dropout(dr[i + 1]),
                )
            linear_list.add_module(
                "linear%d" % (len(hidden_size) + 1),
                torch.nn.Linear(hidden_size[-1], ny),
            )
        self.linear_list = linear_list
        self.dropout_list = dropout_list

    def forward(self, x):
        for i, model in enumerate(self.linear_list):
            if i == 0:
                if len(self.linear_list) == 1:
                    return model(x)
                else:
                    out = F.relu(self.dropout_list[i](model(x)))
            elif i == len(self.linear_list) - 1:
                # in final layer, no relu again
                return model(out)
            else:
                out = F.relu(self.dropout_list[i](model(out)))
