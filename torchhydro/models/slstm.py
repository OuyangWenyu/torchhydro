"""slstm model"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, Parameter

class sLSTM(nn.Module):
    """
    stacked lstm model.
    """
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        device=None,
        dtype=None,
    ):
        """

        Parameters
        ----------
        input_size
        output_size
        hidden_size
        num_layers
        dropout
        device
        dtype
        """
        super(sLSTM, self).__init__()
        self.linearIn = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers,
            dropout=dropout,
        )
        self.linearOut = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x0 = F.relu(self.linearIn(x))
        out_lstm, (hn, cn) = self.lstm(x0)
        out = self.linearOut(out_lstm)
        return out

class MI_STL_sLSTM(nn.Module):
    """

    """
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        num_layers: int = 10,
        dropout: float = 0.0,
        device=None,
        dtype=None,
    ):
        """

        Parameters
        ----------
        input_size
        output_size
        hidden_size
        num_layers
        dropout
        device
        dtype
        """
        super(MI_STL_sLSTM, self).__init__()
        self.slstm = sLSTM(input_size, output_size, hidden_size, num_layers, dropout)


class pcLSTMCell(Module):
    r"""
    single step LSTM cell.
    .. math::
        \begin{array}{ll} \\
            i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi} + W_{ci} c_{t-1}) \\  input gate
            f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf} + W_{cf} c_{t-1}) \\  forget gate
            g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg}) \\  input operate
            o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho} + W_{co} c_{t-1}) \\  output gate
            c_t = f_t \odot c_{t-1} + i_t \odot g_t \\  output, cell state
            h_t = o_t \odot \tanh(c_t) \\  output，hidden state
        \end{array}
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        # batch_size: int,
        # num_layers: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
    ) -> None:
        """

        Parameters
        ----------
        input_size
        hidden_size
        num_layers
        bias
        dropout
        """
        super(pcLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # self.batch_size = batch_size
        # self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout

        # input to hidden weights
        self.w_xi = Parameter(Tensor(hidden_size, input_size))
        self.w_xf = Parameter(Tensor(hidden_size, input_size))
        self.w_xo = Parameter(Tensor(hidden_size, input_size))
        self.w_xc = Parameter(Tensor(hidden_size, input_size))
        # hidden to hidden weights
        self.w_hi = Parameter(Tensor(hidden_size, hidden_size))
        self.w_hf = Parameter(Tensor(hidden_size, hidden_size))
        self.w_ho = Parameter(Tensor(hidden_size, hidden_size))
        self.w_hc = Parameter(Tensor(hidden_size, hidden_size))
        # bias terms
        self.b_i = Tensor(hidden_size).fill_(0)
        self.b_f = Tensor(hidden_size).fill_(0)
        self.b_o = Tensor(hidden_size).fill_(0)
        self.b_c = Tensor(hidden_size).fill_(0)
        # cell stat weight
        self.c_i = Parameter(Tensor(hidden_size, input_size))   # todo: pay attention to dimension match
        self.c_f = Parameter(Tensor(hidden_size, hidden_size))
        self.c_o = Parameter(Tensor(hidden_size, hidden_size))

        # Wrap biases as parameters if desired, else as variables without gradients
        W = Parameter if bias else (lambda x: Parameter(x, requires_grad=False))
        self.b_i = W(self.b_i)
        self.b_f = W(self.b_f)
        self.b_o = W(self.b_o)
        self.b_c = W(self.b_c)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)  # uniform distribution

    def forward(self, x: Tensor, hx: Tensor) -> Tensor:
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
        neural network
        weight spac, data space in python
        parameter space
        h, b
        data space

        """
        # if self.bidirectional:
        #     num_directions = 2
        # else:
        #     num_directions = 1
        batch_size = x.size(0)
        h, c = hx
        if h is None:
            h = torch.zeros(  # hidden space
                batch_size,
                self.hidden_size,
                dtype=x.dtype,
                device=x.device,
                requires_grad=False
            )
            c = torch.zeros(  # cell space
                batch_size,
                self.hidden_size,
                dtype=x.dtype,
                device=x.device,
                requires_grad=False,
            )
        h = h.view(h.size(0), -1)
        c = c.view(h.size(0), -1)
        x = x.view(x.size(0), -1)
        # forget gate
        f_t = torch.mm(x, self.w_xf) + torch.mm(h, self.w_hf) + torch.mm(c, self.c_f) + self.b_f
        f_t.sigmoid_()
        # input gate
        i_t = torch.mm(x, self.w_xi) + torch.mm(h, self.w_hi) + torch.mm(c, self.c_i) + self.b_i
        i_t.sigmoid_()
        # cell computations
        g_t = torch.mm(x, self.w_xc) + torch.mm(h, self.w_hc) + self.b_c
        g_t.tanh_()
        c_t = torch.mul(c, f_t) + torch.mul(i_t, g_t)
        # output gate
        o_t = torch.mm(x, self.w_xo) + torch.mm(h, self.w_ho) + torch.mm(c_t, self.c_o) + self.b_o
        o_t.sigmoid_()
        h_t = torch.mul(o_t, torch.tanh(c_t))    # hidden state update
        # Reshape for compatibility
        h_t = h_t.view(h_t.size(0), 1, -1)
        c_t = c_t.view(c_t.size(0), 1, -1)
        if self.dropout > 0.0:
            F.dropout(h_t, p=self.dropout, training=self.training, inplace=True)

        h_t = torch.squeeze(h_t, dim=1)
        c_t = torch.squeeze(c_t, dim=1)

        return h_t, c_t

class pcLSTM(Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        num_layers: int = 3,
        # nonlinearity: str = "tanh",
        bias: bool = True,
        # batch_first: bool = False,
        dropout: float = 0.0,
        # bidirectional: bool = False,
    ):
        super(pcLSTM, self).__init__()
        self.nx = input_size
        self.ny = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # self.nonlinearity = nonlinearity
        self.bias = bias
        # self.batch_first = batch_first
        self.dropout = dropout
        # self.bidirectional = bidirectional
        self.linearIn = torch.nn.Linear(self.nx, self.hidden_size)
        self.lstm = []
        pclstmcell = pcLSTMCell(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
        )
        for i in range(self.num_layers):
            self.lstm.append(pclstmcell)
        print("pclstmcell model list")
        for i in range(self.num_layers):
            print(self.lstm[i])
        self.linearOut = torch.nn.Linear(self.hidden_size, self.ny)



# Epoch 2 Valid Loss 0.7846 Valid Metric {'NSE of streamflow': [-0.07179832458496094, 0.6020937860012054, 0.3605666160583496, 0.49114787578582764, 0.5146141648292542, 0.5641768872737885, 0.4332682490348816, 0.5208058059215546, 0.3445504307746887, 0.5849301815032959], 'RMSE of streamflow': [0.937727689743042, 0.6115314364433289, 0.8725977540016174, 0.8397765159606934, 0.721656322479248, 0.6320325136184692, 0.7655382752418518, 0.7387174367904663, 0.8698011636734009, 0.7924478054046631], 'R2 of streamflow': [-0.07179832458496094, 0.6020937860012054, 0.3605666160583496, 0.49114787578582764, 0.5146141648292542, 0.5641768872737885, 0.4332682490348816, 0.5208058059215546, 0.3445504307746887, 0.5849301815032959], 'KGE of streamflow': [-4.17055214968388, 0.2613345451191861, -4.6996091150268215, -12.4683304821144, 0.1698882335962545, 0.5628409933861886, -0.1697408942596299, -0.04265819030779183, -48.41112055692167, -0.047906949678860444], 'FHV of streamflow': [-19.597141444683075, -30.103501677513123, -24.726715683937073, -41.080063581466675, -47.5444495677948, -44.422733783721924, -58.1508994102478, -52.74372100830078, -60.483938455581665, -42.67994165420532], 'FLV of streamflow': [-119.58931684494019, -55.743175745010376, -75.80413222312927, -61.86025142669678, -23.665884137153625, -44.34462785720825, -18.388770520687103, -29.776406288146973, -8.400652557611465, -50.57823061943054]}
# Weights sucessfully loaded
# All processes are finished!
# 100%|██████████| 44/44 [02:15<00:00,  3.08s/it]
# 100%|██████████| 44/44 [00:37<00:00,  1.17it/s]
