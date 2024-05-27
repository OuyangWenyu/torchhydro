"""
Author: MHPI group, Wenyu Ouyang
Date: 2021-12-31 11:08:29
LastEditTime: 2024-05-27 16:01:38
LastEditors: Wenyu Ouyang
Description: LSTM with dropout implemented by Kuai Fang and more LSTMs using it
FilePath: \torchhydro\torchhydro\models\cudnnlstm.py
Copyright (c) 2021-2022 MHPI group, Wenyu Ouyang. All rights reserved.
"""

import math
from typing import Union

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

from torchhydro.models.ann import SimpleAnn
from torchhydro.models.dropout import DropMask, create_mask


class LstmCellTied(nn.Module):
    """
    LSTM with dropout implemented by Kuai Fang: https://github.com/mhpi/hydroDL/blob/release/hydroDL/model/rnn.py

    the name of "Tied" comes from this paper:
    http://papers.nips.cc/paper/6241-a-theoretically-grounded-application-of-dropout-in-recurrent-neural-networks.pdf
    which means the weights of all gates will be tied together to be used (eq. 6 in this paper).
    this code is mainly used as a CPU version of CudnnLstm
    """

    def __init__(
        self,
        *,
        input_size,
        hidden_size,
        mode="train",
        dr=0.5,
        dr_method="drX+drW+drC",
        gpu=1
    ):
        super(LstmCellTied, self).__init__()

        self.inputSize = input_size
        self.hiddenSize = hidden_size
        self.dr = dr

        self.w_ih = Parameter(torch.Tensor(hidden_size * 4, input_size))
        self.w_hh = Parameter(torch.Tensor(hidden_size * 4, hidden_size))
        self.b_ih = Parameter(torch.Tensor(hidden_size * 4))
        self.b_hh = Parameter(torch.Tensor(hidden_size * 4))

        self.drMethod = dr_method.split("+")
        self.gpu = gpu
        self.mode = mode
        if mode == "train":
            self.train(mode=True)
        elif mode in ["test", "drMC"]:
            self.train(mode=False)
        if gpu >= 0:
            self = self.cuda()
            self.is_cuda = True
        else:
            self.is_cuda = False
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hiddenSize)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def reset_mask(self, x, h, c):
        self.mask_x = create_mask(x, self.dr)
        self.mask_h = create_mask(h, self.dr)
        self.mask_c = create_mask(c, self.dr)
        self.mask_w_ih = create_mask(self.w_ih, self.dr)
        self.mask_w_hh = create_mask(self.w_hh, self.dr)

    def forward(self, x, hidden, *, do_reset_mask=True, do_drop_mc=False):
        do_drop = self.dr > 0 and (do_drop_mc is True or self.training is True)
        batch_size = x.size(0)
        h0, c0 = hidden
        if h0 is None:
            h0 = x.new_zeros(batch_size, self.hiddenSize, requires_grad=False)
        if c0 is None:
            c0 = x.new_zeros(batch_size, self.hiddenSize, requires_grad=False)

        if self.dr > 0 and self.training is True and do_reset_mask is True:
            self.reset_mask(x, h0, c0)

        if do_drop and "drH" in self.drMethod:
            h0 = DropMask.apply(h0, self.mask_h, True)

        if do_drop and "drX" in self.drMethod:
            x = DropMask.apply(x, self.mask_x, True)

        if do_drop and "drW" in self.drMethod:
            w_ih = DropMask.apply(self.w_ih, self.mask_w_ih, True)
            w_hh = DropMask.apply(self.w_hh, self.mask_w_hh, True)
        else:
            # self.w are parameters, while w are not
            w_ih = self.w_ih
            w_hh = self.w_hh

        gates = F.linear(x, w_ih, self.b_ih) + F.linear(h0, w_hh, self.b_hh)
        gate_i, gate_f, gate_c, gate_o = gates.chunk(4, 1)

        gate_i = torch.sigmoid(gate_i)
        gate_f = torch.sigmoid(gate_f)
        gate_c = torch.tanh(gate_c)
        gate_o = torch.sigmoid(gate_o)

        if self.training is True and "drC" in self.drMethod:
            gate_c = gate_c.mul(self.mask_c)

        c1 = (gate_f * c0) + (gate_i * gate_c)
        h1 = gate_o * torch.tanh(c1)

        return h1, c1


class CpuLstmModel(nn.Module):
    """Cpu version of CudnnLstmModel"""

    def __init__(self, *, n_input_features, n_output_features, n_hidden_states, dr=0.5):
        super(CpuLstmModel, self).__init__()
        self.nx = n_input_features
        self.ny = n_output_features
        self.hiddenSize = n_hidden_states
        self.ct = 0
        self.nLayer = 1
        self.linearIn = torch.nn.Linear(n_input_features, n_hidden_states)
        self.lstm = LstmCellTied(
            input_size=n_hidden_states,
            hidden_size=n_hidden_states,
            dr=dr,
            dr_method="drW",
            gpu=-1,
        )
        self.linearOut = torch.nn.Linear(n_hidden_states, n_output_features)
        self.gpu = -1

    def forward(self, x, do_drop_mc=False):
        # x0 = F.relu(self.linearIn(x))
        # outLSTM, (hn, cn) = self.lstm(x0, do_drop_mc=do_drop_mc)
        # out = self.linearOut(outLSTM)
        # return out
        nt, ngrid, nx = x.shape
        yt = torch.zeros(ngrid, 1)
        out = torch.zeros(nt, ngrid, self.ny)
        ht = None
        ct = None
        reset_mask = True
        for t in range(nt):
            xt = x[t, :, :]
            xt = torch.where(torch.isnan(xt), torch.full_like(xt, 0), xt)
            x0 = F.relu(self.linearIn(xt))
            ht, ct = self.lstm(x0, hidden=(ht, ct), do_reset_mask=reset_mask)
            yt = self.linearOut(ht)
            reset_mask = False
            out[t, :, :] = yt
        return out


class CudnnLstm(nn.Module):
    """
    LSTM with dropout implemented by Kuai Fang: https://github.com/mhpi/hydroDL/blob/release/hydroDL/model/rnn.py

    Only run in GPU; the CPU version is LstmCellTied in this file
    """

    def __init__(self, *, input_size, hidden_size, dr=0.5):
        """

        Parameters
        ----------
        input_size
            number of neurons in input layer
        hidden_size
            number of neurons in hidden layer
        dr
            dropout rate
        """
        super(CudnnLstm, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dr = dr
        self.w_ih = Parameter(torch.Tensor(hidden_size * 4, input_size))
        self.w_hh = Parameter(torch.Tensor(hidden_size * 4, hidden_size))
        self.b_ih = Parameter(torch.Tensor(hidden_size * 4))
        self.b_hh = Parameter(torch.Tensor(hidden_size * 4))
        self._all_weights = [["w_ih", "w_hh", "b_ih", "b_hh"]]
        # self.cuda()

        self.reset_mask()
        self.reset_parameters()

    def _apply(self, fn):
        return super()._apply(fn)

    def __setstate__(self, d):  # this func will be called when loading the model
        super().__setstate__(d)
        self.__dict__.setdefault("_data_ptrs", [])
        if "all_weights" in d:
            self._all_weights = d["all_weights"]
        if isinstance(self._all_weights[0][0], str):
            return
        self._all_weights = [["w_ih", "w_hh", "b_ih", "b_hh"]]

    def reset_mask(self):
        self.mask_w_ih = create_mask(self.w_ih, self.dr)
        self.mask_w_hh = create_mask(self.w_hh, self.dr)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx=None, cx=None, do_drop_mc=False, dropout_false=False):
        # dropout_false: it will ensure do_drop is false, unless do_drop_mc is true
        if dropout_false and (not do_drop_mc):
            do_drop = False
        elif self.dr > 0 and (do_drop_mc is True or self.training is True):
            do_drop = True
        else:
            do_drop = False

        batch_size = input.size(1)

        if hx is None:
            hx = input.new_zeros(1, batch_size, self.hidden_size, requires_grad=False)
        if cx is None:
            cx = input.new_zeros(1, batch_size, self.hidden_size, requires_grad=False)

        # handle = torch.backends.cudnn.get_handle()
        if do_drop is True:
            # cuDNN backend - disabled flat weight
            freeze_mask = False
            if not freeze_mask:
                self.reset_mask()
            weight = [
                DropMask.apply(self.w_ih, self.mask_w_ih, True),
                DropMask.apply(self.w_hh, self.mask_w_hh, True),
                self.b_ih,
                self.b_hh,
            ]
        else:
            weight = [self.w_ih, self.w_hh, self.b_ih, self.b_hh]
        if torch.__version__ < "1.8":
            output, hy, cy, reserve, new_weight_buf = torch._cudnn_rnn(
                input,
                weight,
                4,
                None,
                hx,
                cx,
                2,  # 2 means LSTM
                self.hidden_size,
                1,
                False,
                0,
                self.training,
                False,
                (),
                None,
            )
        else:
            output, hy, cy, reserve, new_weight_buf = torch._cudnn_rnn(
                input,
                weight,
                4,
                None,
                hx,
                cx,
                2,  # 2 means LSTM
                self.hidden_size,
                0,
                1,
                False,
                0,
                self.training,
                False,
                (),
                None,
            )
        return output, (hy, cy)

    @property
    def all_weights(self):
        return [
            [getattr(self, weight) for weight in weights]
            for weights in self._all_weights
        ]


class CudnnLstmModel(nn.Module):
    def __init__(self, n_input_features, n_output_features, n_hidden_states, dr=0.5):
        """
        An LSTM model writen by Kuai Fang from this paper: https://doi.org/10.1002/2017GL075619

        only gpu version

        Parameters
        ----------
        n_input_features
            the number of input features
        n_output_features
            the number of output features
        n_hidden_states
            the number of hidden features
        dr
            dropout rate and its default is 0.5
        """
        super(CudnnLstmModel, self).__init__()
        self.nx = n_input_features
        self.ny = n_output_features
        self.hidden_size = n_hidden_states
        self.ct = 0
        self.nLayer = 1
        self.linearIn = torch.nn.Linear(self.nx, self.hidden_size)
        self.lstm = CudnnLstm(
            input_size=self.hidden_size, hidden_size=self.hidden_size, dr=dr
        )
        self.linearOut = torch.nn.Linear(self.hidden_size, self.ny)

    def forward(self, x, do_drop_mc=False, dropout_false=False, return_h_c=False):
        x0 = F.relu(self.linearIn(x))
        out_lstm, (hn, cn) = self.lstm(
            x0, do_drop_mc=do_drop_mc, dropout_false=dropout_false
        )
        out = self.linearOut(out_lstm)
        return (out, (hn, cn)) if return_h_c else out


class LinearCudnnLstmModel(CudnnLstmModel):
    """This model is nonlinear layer + CudnnLSTM/CudnnLstm-MultiOutput-Model.
    kai_tl: model from this paper by Ma et al. -- https://doi.org/10.1029/2020WR028600
    """

    def __init__(self, linear_size, **kwargs):
        """

        Parameters
        ----------
        linear_size
            the number of input features for the first input linear layer
        """
        super(LinearCudnnLstmModel, self).__init__(**kwargs)
        self.former_linear = torch.nn.Linear(linear_size, kwargs["n_input_features"])

    def forward(self, x, do_drop_mc=False, dropout_false=False):
        x0 = F.relu(self.former_linear(x))
        return super(LinearCudnnLstmModel, self).forward(
            x0, do_drop_mc=do_drop_mc, dropout_false=dropout_false
        )


def cal_conv_size(lin, kernel, stride, padding=0, dilation=1):
    lout = (lin + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1
    return int(lout)


def cal_pool_size(lin, kernel, stride=None, padding=0, dilation=1):
    if stride is None:
        stride = kernel
    lout = (lin + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1
    return int(lout)


class CNN1dKernel(torch.nn.Module):
    def __init__(self, *, ninchannel=1, nkernel=3, kernelSize=3, stride=1, padding=0):
        super(CNN1dKernel, self).__init__()
        self.cnn1d = torch.nn.Conv1d(
            in_channels=ninchannel,
            out_channels=nkernel,
            kernel_size=kernelSize,
            padding=padding,
            stride=stride,
        )
        self.name = "CNN1dkernel"
        self.is_legacy = True

    def forward(self, x):
        return F.relu(self.cnn1d(x))


class CNN1dLCmodel(nn.Module):
    # Directly add the CNN extracted features into LSTM inputSize
    def __init__(
        self,
        nx,
        ny,
        nobs,
        hidden_size,
        n_kernel: Union[list, tuple] = (10, 5),
        kernel_size: Union[list, tuple] = (3, 3),
        stride: Union[list, tuple] = (2, 1),
        dr=0.5,
        pool_opt=None,
        cnn_dr=0.5,
        cat_first=True,
    ):
        """cat_first means: we will concatenate the CNN output with the x, then input them to the CudnnLstm model;
        if not cat_first, it is relu_first, meaning we will relu the CNN output firstly, then concatenate it with x
        """
        # two convolutional layer
        super(CNN1dLCmodel, self).__init__()
        # N_cnn_out代表输出的特征数量
        # nx代表历史的输入
        # ny代表最后线性层输出的维度，如果只预报流量，则为1
        # nobs代表要输入到CNN的维度
        # hidden_size是线性层的隐藏层的节点数
        self.nx = nx
        self.ny = ny
        self.obs = nobs
        self.hiddenSize = hidden_size
        n_layer = len(n_kernel)
        self.features = nn.Sequential()
        n_in_chan = 1
        lout = nobs
        for ii in range(n_layer):
            conv_layer = CNN1dKernel(
                ninchannel=n_in_chan,
                nkernel=n_kernel[ii],
                kernelSize=kernel_size[ii],
                stride=stride[ii],
            )
            self.features.add_module("CnnLayer%d" % (ii + 1), conv_layer)
            if cnn_dr != 0.0:
                self.features.add_module("dropout%d" % (ii + 1), nn.Dropout(p=cnn_dr))
            n_in_chan = n_kernel[ii]
            lout = cal_conv_size(lin=lout, kernel=kernel_size[ii], stride=stride[ii])
            self.features.add_module("Relu%d" % (ii + 1), nn.ReLU())
            if pool_opt is not None:
                self.features.add_module(
                    "Pooling%d" % (ii + 1), nn.MaxPool1d(pool_opt[ii])
                )
                lout = cal_pool_size(lin=lout, kernel=pool_opt[ii])
        self.N_cnn_out = int(
            lout * n_kernel[-1]
        )  # total CNN feature number after convolution
        self.cat_first = cat_first
        # 要不要先拼接？
        # 先拼接，则代表线性层中，输入的维度是未来的降水等输入输出的CNN特征维度，和历史观测的等时间序列的特征数量，通过线性层合并成一个，然后再把这些特征输出到一个线性层中
        # 如果不拼接，那么历史观测数据先进入一个线性层
        if cat_first:
            nf = self.N_cnn_out + nx
            self.linearIn = torch.nn.Linear(nf, hidden_size)
            # CudnnLstm除了最基础的部分以外，主要是有个h和c两个门为空的纠错，这个在论文里讲述的是因为可能输入缺失，但是又不想用插值处理
            # 不想用插值处理是因为认为会暴露未来信息
            # 采用了置零操作，原文的表述是这种缺失点较少，在模型的不断更新参数后，这种置零的影响对于模型的输出影响很小
            self.lstm = CudnnLstm(
                input_size=hidden_size, hidden_size=hidden_size, dr=dr
            )
        else:
            nf = self.N_cnn_out + hidden_size
            self.linearIn = torch.nn.Linear(nx, hidden_size)
            self.lstm = CudnnLstm(input_size=nf, hidden_size=hidden_size, dr=dr)
        self.linearOut = torch.nn.Linear(hidden_size, ny)
        self.gpu = 1

    def forward(self, x, z, do_drop_mc=False):
        # z = n_grid*nVar add a channel dimension
        # z = z.t()
        n_grid, nobs, _ = z.shape
        z = z.reshape(n_grid * nobs, 1)
        n_t, bs, n_var = x.shape
        # add a channel dimension
        z = torch.unsqueeze(z, dim=1)
        z0 = self.features(z)
        # z0 = (n_grid) * n_kernel * sizeafterconv
        z0 = z0.view(n_grid, self.N_cnn_out).repeat(n_t, 1, 1)
        if self.cat_first:
            x = torch.cat((x, z0), dim=2)
            x0 = F.relu(self.linearIn(x))
        else:
            x = F.relu(self.linearIn(x))
            x0 = torch.cat((x, z0), dim=2)
        out_lstm, (hn, cn) = self.lstm(x0, do_drop_mc=do_drop_mc)
        return self.linearOut(out_lstm)


class CudnnLstmModelLstmKernel(nn.Module):
    """use a trained/un-trained CudnnLstm as a kernel generator before another CudnnLstm."""

    def __init__(
        self,
        nx,
        ny,
        hidden_size,
        nk=None,
        hidden_size_later=None,
        cut=False,
        dr=0.5,
        delta_s=False,
    ):
        """delta_s means we will use the difference of the first lstm's output and the second's as the final output"""
        super(CudnnLstmModelLstmKernel, self).__init__()
        # These three layers are same with CudnnLstmModel to be used for transfer learning or just vanilla-use
        self.linearIn = torch.nn.Linear(nx, hidden_size)
        self.lstm = CudnnLstm(input_size=hidden_size, hidden_size=hidden_size, dr=dr)
        self.linearOut = torch.nn.Linear(hidden_size, ny)
        # if cut is True, we will only select the final index in nk, and repeat it, then concatenate with x
        self.cut = cut
        # the second lstm has more input than the previous
        if nk is None:
            nk = ny
        if hidden_size_later is None:
            hidden_size_later = hidden_size
        self.linear_in_later = torch.nn.Linear(nx + nk, hidden_size_later)
        self.lstm_later = CudnnLstm(
            input_size=hidden_size_later, hidden_size=hidden_size_later, dr=dr
        )
        self.linear_out_later = torch.nn.Linear(hidden_size_later, ny)

        self.delta_s = delta_s
        # when delta_s is true, cut cannot be true, because they have to have same number params
        assert not (cut and delta_s)

    def forward(self, x, do_drop_mc=False, dropout_false=False):
        x0 = F.relu(self.linearIn(x))
        out_lstm1, (hn1, cn1) = self.lstm(
            x0, do_drop_mc=do_drop_mc, dropout_false=dropout_false
        )
        gen = self.linearOut(out_lstm1)
        if self.cut:
            gen = gen[-1, :, :].repeat(x.shape[0], 1, 1)
        x1 = torch.cat((x, gen), dim=len(gen.shape) - 1)
        x2 = F.relu(self.linear_in_later(x1))
        out_lstm2, (hn2, cn2) = self.lstm_later(
            x2, do_drop_mc=do_drop_mc, dropout_false=dropout_false
        )
        out = self.linear_out_later(out_lstm2)
        return gen - out if self.delta_s else (out, gen)


class CudnnLstmModelMultiOutput(nn.Module):
    def __init__(
        self,
        n_input_features,
        n_output_features,
        n_hidden_states,
        layer_hidden_size=(128, 64),
        dr=0.5,
        dr_hidden=0.0,
    ):
        """
        Multiple output CudnnLSTM.

        It has multiple output layers, each for one output, so that we can easily freeze any output layer.

        Parameters
        ----------
        n_input_features
            the size of input features
        n_output_features
            the size of output features; in this model, we set different nonlinear layer for each output
        n_hidden_states
            the size of LSTM's hidden features
        layer_hidden_size
            hidden_size for multi-layers
        dr
            dropout rate
        dr_hidden
            dropout rates of hidden layers
        """
        super(CudnnLstmModelMultiOutput, self).__init__()
        self.ct = 0
        multi_layers = torch.nn.ModuleList()
        for i in range(n_output_features):
            multi_layers.add_module(
                "layer%d" % (i + 1),
                SimpleAnn(n_hidden_states, 1, layer_hidden_size, dr=dr_hidden),
            )
        self.multi_layers = multi_layers
        self.linearIn = torch.nn.Linear(n_input_features, n_hidden_states)
        self.lstm = CudnnLstm(
            input_size=n_hidden_states, hidden_size=n_hidden_states, dr=dr
        )

    def forward(self, x, do_drop_mc=False, dropout_false=False, return_h_c=False):
        x0 = F.relu(self.linearIn(x))
        out_lstm, (hn, cn) = self.lstm(
            x0, do_drop_mc=do_drop_mc, dropout_false=dropout_false
        )
        outs = [mod(out_lstm) for mod in self.multi_layers]
        final = torch.cat(outs, dim=-1)
        return (final, (hn, cn)) if return_h_c else final
