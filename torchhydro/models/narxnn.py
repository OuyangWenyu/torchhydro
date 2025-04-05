from typing import Union
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchhydro.models import narx

class Narxnn(nn.Module):
    """
    nonlinear autoregressive with exogenous inputs neural network model
    """
    def __init__(
        self,
        n_input_features: int,
        n_output_features: int,
        n_hidden_states: int,
        input_delay: int,
        feedback_delay: int,
        num_layers: int = 10,
        close_loop: bool = False,
    ):
        """

        Parameters
        ----------
        n_input_features: int, number of input features.
        n_output_features: int, number of output features.
        n_hidden_states: int, number of hidden states.
        input_delay: int, the maximum input delay time-step.
        feedback_delay: int, the maximum feedback delay time-step.
        num_layers: int, the number of recurrent layers.
        close_loop: bool, whether to close the loop when feeding in.
        """
        super(Narxnn, self).__init__()
        self.nx = n_input_features  #
        self.ny = n_output_features  # n_output_features = n_feedback_features in narx nn model
        self.num_layers = num_layers
        self.hidden_size = n_hidden_states
        self.input_delay = input_delay
        self.feedback_delay = feedback_delay
        self.close_loop = close_loop
        self.linearIn = nn.Linear(n_input_features, n_hidden_states)
        self.narx = nn.RNNCell(
            input_size=n_hidden_states,
            hidden_size=n_hidden_states,
            num_layers = self.num_layers,
        )
        self.linearOut = nn.Linear(n_hidden_states, n_output_features)

    def forward(self, x):
        """
        forward propagation function
        Parameters
        ----------
        x
            the input time sequence
        Returns
        -------
        out
            the output sequence of the model
        """
        nt, ngrid, nx = x.shape  # (time,basins,features)
        out = torch.zeros(nt, ngrid, self.ny)  # (time,basins,output_features)
        for t in range(self.max_delay,nt):
            x0 = x[(t-self.input_delay):t, :, :(self.nx-self.ny)]
            x0_t = x0[-1,:,:]
            for i in range(self.input_delay):
                x0_i = x0[i,:,:]
                x0_t = torch.cat((x0_t,x0_i),dim=-1)
            y0 = x[(t-self.feedback_delay):t, :, -self.ny:]
            y0_t = y0[-1,:,:]
            for i in range(self.feedback_delay):
                y0i = y0[i,:,:]
                y0_t = torch.cat((y0_t,y0i),dim=-1)
            xt = torch.cat([x0_t, y0_t], dim=-1)  # (basins,features)  single step, no time dimension.
            x_l = F.relu(self.linearIn(xt))
            out_t, hxt = self.narx(x_l)  # single step
            yt = self.linearOut(out_t)  # (basins,output_features) single step output value, e.g. streamflow
            out[t, :, :] = yt
            if self.close_loop:
                x[t+1, :, -self.ny:] = yt  #
        return out
