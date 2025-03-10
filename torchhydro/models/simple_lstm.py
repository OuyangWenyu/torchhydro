"""
Author: Wenyu Ouyang
Date: 2023-09-19 09:36:25
LastEditTime: 2024-04-09 15:29:35
LastEditors: Wenyu Ouyang
Description: Some self-made LSTMs
FilePath: \torchhydro\torchhydro\models\simple_lstm.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import math
import torch as th
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor as T
from torch.nn import Parameter as P


class SimpleLSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, dr=0.0):
        """
        A simple multi-layer LSTM - NN model
        循环神经网络
        LSTM是 nn 里面已经封装好了的模型，这里再一次封装
        Parameters
        ----------
        input_size
            number of input neurons  输入神经元个数  输入数据的特征维数  input(seq_len,batch_size,input_size)
        output_size
            number of output neurons  输出神经元个数    output(seq_len,batch_size,num_directions*hidden_size)
        hidden_size
            number of neurons in each hidden layer    隐藏层的维数   隐藏层节点特征维度
        dr: float
            dropout rate of layers, default is 0.0 which means no dropout;
            here we set number of dropout layers to (number of nn layers - 1)
        """
        super(SimpleLSTM, self).__init__()
        self.linearIn = nn.Linear(input_size, hidden_size)   # 第一层，即输入层为线性层
        self.lstm = nn.LSTM(  #
            hidden_size,  # input_size 输入特征的维度
            hidden_size,  #  隐藏层神经元个数
            1,  # 循环神经网络的层数
            dropout=dr,  # 这里使用暂退。
        )
        self.linearOut = nn.Linear(hidden_size, output_size)  # 最后一层，即输出层也为线性层

    def forward(self, x):  # 传播。 todo: 这个传播是仅前向传播还是包括前向和反向？
        x0 = F.relu(self.linearIn(x))  # 输入层经过激活函数后，传入长短期记忆模型
        out_lstm, (hn, cn) = self.lstm(x0)  # 经过长短期记忆模型传播后，传入输出层
        return self.linearOut(out_lstm)  # 输出层经过一次线性变换后传出一次传播结果，不断循环往复传播。


class SimpleLSTMForecast(SimpleLSTM):
    def __init__(self, input_size, output_size, hidden_size, forecast_length, dr=0.0):
        super(SimpleLSTMForecast, self).__init__(
            input_size, output_size, hidden_size, dr
        )
        self.forecast_length = forecast_length

    def forward(self, x):
        # 调用父类的forward方法获取完整的输出
        full_output = super(SimpleLSTMForecast, self).forward(x)

        return full_output[-self.forecast_length :, :, :]


class SlowLSTM(nn.Module):
    """
    A pedagogic implementation of Hochreiter & Schmidhuber:
    'Long-Short Term Memory'
    http://www.bioinf.jku.at/publications/older/2604.pdf
    """

    def __init__(self, input_size, hidden_size, bias=True, dropout=0.0):
        super(SlowLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.dropout = dropout
        # input to hidden weights
        self.w_xi = P(T(hidden_size, input_size))
        self.w_xf = P(T(hidden_size, input_size))
        self.w_xo = P(T(hidden_size, input_size))
        self.w_xc = P(T(hidden_size, input_size))
        # hidden to hidden weights
        self.w_hi = P(T(hidden_size, hidden_size))
        self.w_hf = P(T(hidden_size, hidden_size))
        self.w_ho = P(T(hidden_size, hidden_size))
        self.w_hc = P(T(hidden_size, hidden_size))
        # bias terms
        self.b_i = T(hidden_size).fill_(0)
        self.b_f = T(hidden_size).fill_(0)
        self.b_o = T(hidden_size).fill_(0)
        self.b_c = T(hidden_size).fill_(0)

        # Wrap biases as parameters if desired, else as variables without gradients
        W = P if bias else (lambda x: P(x, requires_grad=False))
        self.b_i = W(self.b_i)
        self.b_f = W(self.b_f)
        self.b_o = W(self.b_o)
        self.b_c = W(self.b_c)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        h, c = hidden
        h = h.view(h.size(0), -1)
        c = c.view(h.size(0), -1)
        x = x.view(x.size(0), -1)
        # Linear mappings
        i_t = th.mm(x, self.w_xi) + th.mm(h, self.w_hi) + self.b_i
        f_t = th.mm(x, self.w_xf) + th.mm(h, self.w_hf) + self.b_f
        o_t = th.mm(x, self.w_xo) + th.mm(h, self.w_ho) + self.b_o
        # activations
        i_t.sigmoid_()
        f_t.sigmoid_()
        o_t.sigmoid_()
        # cell computations
        c_t = th.mm(x, self.w_xc) + th.mm(h, self.w_hc) + self.b_c
        c_t.tanh_()
        c_t = th.mul(c, f_t) + th.mul(i_t, c_t)
        h_t = th.mul(o_t, th.tanh(c_t))
        # Reshape for compatibility
        h_t = h_t.view(h_t.size(0), 1, -1)
        c_t = c_t.view(c_t.size(0), 1, -1)
        if self.dropout > 0.0:
            F.dropout(h_t, p=self.dropout, training=self.training, inplace=True)
        return h_t, (h_t, c_t)

    def sample_mask(self):
        pass
