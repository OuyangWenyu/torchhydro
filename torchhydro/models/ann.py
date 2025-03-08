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
    def __init__(self, nx: int, ny: int, hidden_size: Union[int, tuple, list] = None,  # union,联合。hidden_size可以是int,tuple,list中的任意一种。 [tuple, list]；隐藏的层数及各层的神经元个数。
                 dr: Union[float, tuple, list] = 0.0):
        """
        A simple multi-layer NN model with final linear layer  简单多层神经网络模型，最后一层是线性层。

        Parameters
        ----------
        nx
            number of input neurons  输入神经元个数
        ny
            number of output neurons  输出神经元个数
        hidden_size  隐藏大小，表达的是二维大小。 e.g. [3, 4, 5, 4, 3], 五层隐藏层，各层的神经元个数分别为 3、4、5、4、3。
            a list/tuple which contains number of neurons in each hidden layer; 一个列或一个元组，每层隐藏层包含一些神经元。
            if int, only one hidden layer except for hidden_size=0 除了等于零外，如果是 int, 表示只有一层隐藏层。 e.g. 5, 表示一层隐藏层，此层有5个神经元。
        dr          同hidden_size, 表达的是二维大小。   e.g. [0.1, 0.2, 0.3, 0.2], 4层可暂退层，各层暂退率分别为 0.1, 0.2, 0.3, 0.2。
            dropout rate of layers, default is 0.0 which means no dropout;  同hidden_size,
            here we set number of dropout layers to (number of nn layers - 1)  设置暂退层数为 神经网络层数-1。 除最后一层输出层不设置为可以暂退的层外，其余层均设置为可暂退层，包括第一层输入层。
        """
        super(SimpleAnn, self).__init__()
        linear_list = torch.nn.ModuleList()  # 模块列表  线性列表
        dropout_list = torch.nn.ModuleList()  # 暂退列表
        if (
                hidden_size is None
                or (type(hidden_size) is int and hidden_size == 0)
                or (type(hidden_size) in [tuple, list] and len(hidden_size) < 1)  # 没有隐藏层
        ):
            linear_list.add_module("linear1", torch.nn.Linear(nx, ny))  # 第一层，输入层  输入层->输出层 线性变换，输入层nx个神经元，输出层ny个神经元
        elif type(hidden_size) is int:  # 如果隐藏层只有一层 输入层|隐藏层|输出层，总共三层
            if type(dr) in [tuple, list]:  # 同时暂退率是元组或列表
                dr = dr[0]    # 设置暂退率为元组或列表的第一个元素
            linear_list.add_module("linear1", torch.nn.Linear(nx, hidden_size))  # 输入层->隐藏层
            # dropout layer do not have additional weights, so we do not name them here
            dropout_list.append(torch.nn.Dropout(dr))  # 第二层，隐藏层的暂退率
            linear_list.add_module("linear2", torch.nn.Linear(hidden_size, ny))  # 隐藏层->输出层
        else:
            linear_list.add_module("linear1", torch.nn.Linear(nx, hidden_size[0]))
            if type(dr) is float:  # 各隐藏层暂退率相同
                dr = [dr] * len(hidden_size)
            elif len(dr) != len(hidden_size):  # 各隐藏层暂退率不同时，应保证暂退率的长度与hidden_size的长度一致
                raise ArithmeticError(
                    "We set dropout layer for each nn layer, please check the number of dropout layers")
            # dropout_list.add_module("dropout1", torch.nn.Dropout(dr[0]))
            dropout_list.append(torch.nn.Dropout(dr[0]))  # 第二层，第一层隐藏层的暂退率
            for i in range(len(hidden_size) - 1):  # 添加隐藏层
                linear_list.add_module(
                    "linear%d" % (i + 1 + 1),  # 数组下标从0开始+1，已经创建了第一层+1
                    torch.nn.Linear(hidden_size[i], hidden_size[i + 1]), # 第i+1层、第i+2层的神经元个数
                )
                dropout_list.append(
                    torch.nn.Dropout(dr[i + 1]),  # 第i+1层的暂退率
                )
            linear_list.add_module(   # 输出层
                "linear%d" % (len(hidden_size) + 1),
                torch.nn.Linear(hidden_size[-1], ny),
            )
        self.linear_list = linear_list  # 注册到模型
        self.dropout_list = dropout_list

    def forward(self, x):  # 输入数据单次通过整个神经网络模型的传播结果
        for i, model in enumerate(self.linear_list):
            if i == 0:  # 第一层  不是是输入层，输入层不做任何运算，输入层到第一层隐藏层开始运算。
                if len(self.linear_list) == 1:
                    return model(x)  # model 是 W*x+b 的线性模型，只有输入输出层，只进行一次线性变换，输入层->输出层。
                else:
                    out = F.relu(self.dropout_list[i](model(x)))  # 只有一层隐藏层，输入层到隐藏层，进行一次线性换，使用暂退率和激活函数
            elif i == len(self.linear_list) - 1:  # 最后一层，输出层。
                # in final layer, no relu again
                return model(out)  # 最后一层隐藏层到输出层，进行一次线性变换，不使用暂退和激活函数
            else:  # 隐藏层数>1
                out = F.relu(self.dropout_list[i](model(out)))   # 中间隐藏层，各层运算，采用暂退和激活。 各层的out依次传向下一层。
