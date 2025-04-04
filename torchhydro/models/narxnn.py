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
        num_layers: int = 1,
        delay: Union[list, tuple] = None,
        close_loop: bool = False,
    ):
        """

        Parameters
        ----------
        input_size
        output_size
        hidden_size
        num_layers
        """
        super(Narxnn, self).__init__()
        self.nx = n_input_features
        self.ny = n_output_features
        self.hidden_size = n_hidden_states
        self.delay = delay
        self.close_loop = close_loop
        self.linearIn = nn.Linear(n_input_features, n_hidden_states)
        self.narx = narx.Narx(
            input_size=n_hidden_states,
            hidden_size=n_hidden_states,
            num_layers=num_layers,
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
        yt = torch.zeros(ngrid, self.ny)  # (basins,output_features) single step output value, e.g. streamflow
        out = torch.zeros(nt, ngrid, self.ny)  # (time,basins,output_features)
        for t in range(nt):
            xt = x[t, :, :]
            x0 = F.relu(self.linearIn(x))
            out_t, hxt = self.narx(x0)
            yt = self.linearOut(out_t)
            out[t, :, :] = yt
            if self.close_loop:
                x[t+1, :, -self.ny:] = yt  # delay = 1
        return out
