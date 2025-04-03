
import torch.nn as nn
from torch.nn import functional as F
from torchhydro.models import narx

class SimpleNarx(nn.Module):
    """
    a simple multi-layer nonlinear autoregressive with exogenous inputs neural network model
    """
    def __init__(self, n_input_features, n_output_features, n_hidden_states, num_layers=1):
        """

        Parameters
        ----------
        input_size
        output_size
        hidden_size
        num_layers
        """
        super(SimpleNarx, self).__init__()
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
        x0 = F.relu(self.linearIn(x))
        out_narx, hxn = self.narx(x0)
        out = self.linearOut(out_narx)
        return out
