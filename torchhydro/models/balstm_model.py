from typing import Tuple
import torch
import torch.nn as nn


class BALSTM(nn.Module):
    def __init__(
        self,
        input_size_sta,
        input_size_dyn,
        input_size_glo,
        hidden_size,
        output_size,
        num_layers,
        dropout=0.5,
        hindcast_output_window=0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # 使用SingleBALSTM作为子模块
        self.lstm = SingleBALSTM(
            input_size_sta,
            input_size_dyn,
            input_size_glo,
            hidden_size,
            batch_first=True,
            initial_forget_bias=0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.act = nn.ReLU()

    # def forward(self, data):
    #     x_s, x_d, x_g = data
    def forward(self, x_s, x_d, x_g):
        if x_s.dim() == 3:
            # x_s = x_s.view(32,-1)
            x_s = x_s.squeeze(1)
        # 前向传播LSTM
        # out: tensor的形状为(batch_size, seq_length, hidden_size)
        out, _ = self.lstm(x_s, x_d, x_g)
        # # 解码最后一个时间步的隐藏状态
        # out = self.fc(self.dropout(out[:, -1, :]))
        out = self.fc(self.dropout(out))
        # # 使用特定激活函数
        out = self.act(out)
        return out


class SingleBALSTM(nn.Module):
    """实现基于 Basin-Aware-LSTM (BA-LSTM) 的模型
    This model is written by Xiangyi Le, with support from Wenyu Ouyang.
    Yicheng Sun move it from Xiangyi's code to here.
    """

    def __init__(
        self,
        input_size_sta: int,
        input_size_dyn: int,
        input_size_glo: int,
        hidden_size: int,
        batch_first: bool = True,
        initial_forget_bias: int = 0,
    ):
        super().__init__()

        self.input_size_dyn = input_size_dyn
        self.input_size_sta = input_size_sta
        self.input_size_glo = input_size_glo
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.initial_forget_bias = initial_forget_bias

        # 创建可学习参数的张量
        self.weight_ih = nn.Parameter(
            torch.FloatTensor(input_size_dyn, 3 * hidden_size)
        )
        self.weight_hh = nn.Parameter(torch.FloatTensor(hidden_size, 3 * hidden_size))
        self.weight_sh = nn.Parameter(
            torch.FloatTensor(input_size_sta, 2 * hidden_size)
        )
        self.weight_gh = nn.Parameter(torch.FloatTensor(input_size_glo, hidden_size))
        self.bias = nn.Parameter(torch.FloatTensor(3 * hidden_size))
        self.bias_s = nn.Parameter(torch.FloatTensor(2 * hidden_size))
        self.bias_g = nn.Parameter(torch.FloatTensor(hidden_size))

        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        """初始化 LSTM 的所有可学习参数"""
        nn.init.orthogonal_(self.weight_ih.data)
        nn.init.orthogonal_(self.weight_gh.data)
        nn.init.orthogonal_(self.weight_sh)

        weight_hh_data = torch.eye(self.hidden_size)
        weight_hh_data = weight_hh_data.repeat(1, 3)
        self.weight_hh.data = weight_hh_data

        nn.init.constant_(self.bias.data, val=0)
        nn.init.constant_(self.bias_s.data, val=0)

        if self.initial_forget_bias != 0:
            self.bias.data[: self.hidden_size] = self.initial_forget_bias

    def forward(
        self, x_s: torch.Tensor, x_d: torch.Tensor, x_g: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x_d : torch.Tensor
            包含动态特征序列的批量张量。形状必须与 batch_first 指定的格式匹配。
        x_s : torch.Tensor
            包含静态特征的批量张量。
        x_g : torch.Tensor
            包含全局特征的批量张量。

        Returns
        -------
        h_n : torch.Tensor
            批量中每个样本每个时间步的隐藏状态。
        c_n : torch.Tensor
            批量中每个样本每个时间步的记忆状态。
        """
        if self.batch_first:
            # 交换维度顺序为 (seq_len, batch_size, feature_size)
            x_d = x_d.transpose(0, 1)
            x_g = x_g.transpose(0, 1)

        seq_len, batch_size, _ = x_d.size()

        h_0 = x_d.data.new(batch_size, self.hidden_size).zero_()
        c_0 = x_d.data.new(batch_size, self.hidden_size).zero_()
        h_x = (h_0, c_0)

        h_n, c_n = [], []

        bias_batch = self.bias.unsqueeze(0).expand(batch_size, *self.bias.size())
        bias_g_batch = self.bias_g.unsqueeze(0).expand(batch_size, *self.bias_g.size())

        bias_s_batch = self.bias_s.unsqueeze(0).expand(batch_size, *self.bias_s.size())
        # 0920修改
        # bias_s_batch = self.bias_s.unsqueeze(0).unsqueeze(0).expand(seq_len, batch_size, -1)
        i = torch.addmm(bias_s_batch, x_s, self.weight_sh)
        i1, i2 = i.chunk(2, 1)
        i1 = torch.sigmoid(i1)
        i2 = torch.sigmoid(i2)

        for t in range(seq_len):
            h_0, c_0 = h_x

            g = torch.addmm(bias_g_batch, x_g[t], self.weight_gh)
            gates = torch.addmm(bias_batch, h_0, self.weight_hh) + torch.mm(
                x_d[t], self.weight_ih
            )
            f, d, o_h = gates.chunk(3, 1)
            o = o_h + torch.mm(x_g[t], self.weight_gh)
            c_1 = torch.sigmoid(f) * c_0 + i1 * torch.tanh(d) + i2 * torch.tanh(g)
            h_1 = torch.sigmoid(o) * torch.tanh(c_1)

            h_n.append(h_1)
            c_n.append(c_1)

            h_x = (h_1, c_1)

        h_n = torch.stack(h_n, 0)
        c_n = torch.stack(c_n, 0)

        if self.batch_first:
            # 根据 batch_first 参数调整输出维度顺序
            return h_n.transpose(1, 0), c_n.transpose(1, 0)
        else:
            return h_n, c_n


class VanillaLSTM(nn.Module):
    def __init__(
        self,
        input_size_sta,
        input_size_dyn,
        hidden_size,
        output_size,
        num_layers,
        drop_prob=0.5,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        if num_layers == 1:
            self.lstm = nn.LSTM(
                input_size_dyn + input_size_sta,
                hidden_size,
                num_layers,
                batch_first=True,
            )
        else:
            self.lstm = nn.LSTM(
                input_size_dyn + input_size_sta,
                hidden_size,
                num_layers,
                dropout=drop_prob,
                batch_first=True,
            )
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_size, output_size)
        self.act = nn.ReLU()

    # def forward(self, data):
    def forward(self, x_s, x_d):
        # x_s, x_d = data
        x = torch.cat(
            (
                x_d,
                x_s.repeat_interleave(x_d.shape[1], dim=0).reshape(*x_d.shape[:2], -1),
            ),
            dim=-1,
        )
        # Set initial hidden and cell states

        # Forward propagate LSTM
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        out, _ = self.lstm(x)

        # Decode the hidden state of the last time step
        # out = self.dropout(out[:, -1, :])
        out = self.dropout(out)
        out = self.fc(out)

        # use specific activation
        out = self.act(out)
        return out


# class SimpleBALSTM(nn.Module):
#     '''
#     A simplified version of balstm
#     '''
#     def __init__(self, input_size_sta, input_size_dyn, input_size_glo, hidden_size, output_size, dropout=0.5,num_layers=0,hindcast_output_window=0):
#         super(SimpleBALSTM, self).__init__()
#         self.hidden_size = hidden_size
#         self.W_xs1 = nn.Linear(input_size_sta, hidden_size)
#         self.W_xs2 = nn.Linear(input_size_sta, hidden_size)
#         self.W_xg = nn.Linear(input_size_glo + hidden_size, hidden_size)
#         self.W_xt = nn.Linear(input_size_dyn + hidden_size, hidden_size)
#         self.W_f = nn.Linear(input_size_dyn + hidden_size, hidden_size)
#         self.W_o = nn.Linear(input_size_dyn + hidden_size, hidden_size)
#         self.fc = nn.Linear(hidden_size, output_size)
#         self.dropout = nn.Dropout(p=dropout)
#     def forward(self, xs, xt, xg, init_states=None):
#         batch_size = xt.size(0)
#         seq_length = xt.size(1)
#         if xs.dim() == 3:
#             xs=xs.squeeze(1)
#         if init_states is None:
#             h_t = torch.zeros(batch_size, self.hidden_size).to(xt.device)
#             c_t = torch.zeros(batch_size, self.hidden_size).to(xt.device)
#         else:
#             h_t, c_t = init_states

#         i1 = torch.sigmoid(self.W_xs1(xs))
#         i2 = torch.sigmoid(self.W_xs2(xs))
#         outputs = []
#         for t in range(seq_length):
#             x_g = xg[:, t, :]
#             x_t = xt[:, t, :]
#             xg_combined = torch.cat((x_g, h_t), dim=1)
#             xt_combined = torch.cat((x_t, h_t), dim=1)
#             f_t = torch.sigmoid(self.W_f(xt_combined))
#             g_t = torch.tanh(self.W_xg(xg_combined))
#             i_t = torch.tanh(self.W_xt(xt_combined))
#             o_t = torch.sigmoid(self.W_o(xt_combined))
#             c_t = f_t * c_t + i1 * g_t + i2 * i_t
#             h_t = o_t * torch.tanh(c_t)
#             h_t = self.dropout(h_t)
#             outputs.append(h_t.unsqueeze(1))
#         outputs = torch.cat(outputs, dim=1)
#         out = self.fc(outputs)
#         return out, (h_t, c_t)


class SimpleBALSTM(nn.Module):
    def __init__(
        self,
        input_size_sta,
        input_size_dyn,
        input_size_glo,
        hidden_size,
        output_size,
        dropout=0.5,
        num_layers=1,
        hindcast_output_window=0,
    ):
        super(SimpleBALSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 为每一层创建权重矩阵
        self.layers = nn.ModuleList([])
        for i in range(num_layers):
            layer_input_size = input_size_dyn if i == 0 else hidden_size
            layer = nn.ModuleDict(
                {
                    "W_xs1": nn.Linear(input_size_sta, hidden_size),
                    "W_xs2": nn.Linear(input_size_sta, hidden_size),
                    "W_xg": nn.Linear(input_size_glo + hidden_size, hidden_size),
                    "W_xt": nn.Linear(layer_input_size + hidden_size, hidden_size),
                    "W_f": nn.Linear(layer_input_size + hidden_size, hidden_size),
                    "W_o": nn.Linear(layer_input_size + hidden_size, hidden_size),
                }
            )
            self.layers.append(layer)

        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, xs, xt, xg, init_states=None):
        batch_size = xt.size(0)
        seq_length = xt.size(1)
        if xs.dim() == 3:
            xs = xs.squeeze(1)

        # 初始化所有层的隐藏状态
        if init_states is None:
            h_t = [
                torch.zeros(batch_size, self.hidden_size).to(xt.device)
                for _ in range(self.num_layers)
            ]
            c_t = [
                torch.zeros(batch_size, self.hidden_size).to(xt.device)
                for _ in range(self.num_layers)
            ]
        else:
            h_t, c_t = init_states

        # 每层的i1, i2只需计算一次
        i1 = [torch.sigmoid(layer["W_xs1"](xs)) for layer in self.layers]
        i2 = [torch.sigmoid(layer["W_xs2"](xs)) for layer in self.layers]

        outputs = []
        for t in range(seq_length):
            x_input = xt[:, t, :]
            x_g = xg[:, t, :]

            # 通过每一层
            for l in range(self.num_layers):
                xg_combined = torch.cat((x_g, h_t[l]), dim=1)
                xt_combined = torch.cat((x_input, h_t[l]), dim=1)

                f_t = torch.sigmoid(self.layers[l]["W_f"](xt_combined))
                g_t = torch.tanh(self.layers[l]["W_xg"](xg_combined))
                i_t = torch.tanh(self.layers[l]["W_xt"](xt_combined))
                o_t = torch.sigmoid(self.layers[l]["W_o"](xt_combined))

                c_t[l] = f_t * c_t[l] + i1[l] * g_t + i2[l] * i_t
                h_t[l] = o_t * torch.tanh(c_t[l])
                h_t[l] = self.dropout(h_t[l])

                # 下一层的输入是当前层的输出
                x_input = h_t[l]

            outputs.append(h_t[-1].unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)
        out = self.fc(outputs)

        # 返回所有层的最终隐藏状态和细胞状态
        h_t = torch.stack(h_t, dim=0)  # [num_layers, batch_size, hidden_size]
        c_t = torch.stack(c_t, dim=0)  # [num_layers, batch_size, hidden_size]

        return out, (h_t, c_t)
