
import torch
import torch.nn as nn
from torch.nn import functional as F

from datasets.narxdataset import NarxDataset
from models.basintree import Basin, BasinTree


class Narx(nn.Module):
    """
    nonlinear autoregressive with exogenous inputs neural network model.
        y(t) = f(y(t-1),...,y(t-ny),x(t),...,x(t-nx))
    """
    def __init__(
        self,
        n_input_features: int,
        n_output_features: int,
        n_hidden_states: int,
        input_delay: int,
        feedback_delay: int,
        # num_layers: int = 10,
        close_loop: bool = False,
    ):
        """
        Initialize the Narx model instance.

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
        super(Narx, self).__init__()
        self.nx = n_input_features
        self.ny = n_output_features  # n_output_features = n_feedback_features in narx nn model
        # self.num_layers = num_layers
        self.hidden_size = n_hidden_states
        self.input_delay = input_delay
        self.feedback_delay = feedback_delay
        self.max_delay = max(self.input_delay, self.feedback_delay)
        self.close_loop = close_loop
        in_features = (self.nx-self.ny) * (self.input_delay + 1) + self.ny * (self.feedback_delay + 1)
        self.linearIn = nn.Linear(in_features, self.hidden_size)
        self.narx = nn.RNNCell(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
        )
        self.linearOut = nn.Linear(self.hidden_size, self.ny)

    def close_loop(self):
        self.close_loop = True

    def forward(self, x):
        """
        forward propagation function
        /home/yulili/.conda/envs/torchhydro/lib/python3.13/site-packages/torch/nn/modules/module.py
        /torchhydro/trainers/train_utils.py  torch_single_train() model.train()
        /torchhydro/torchhydro/trainers/train_utils.py  model_infer() output = model(*xs)
        Parameters
        ----------
        x
            the input time sequence. note, the input features need to contain the history output(target) features data in train and test period.
            e.g. streamflow is an output(target) feature in a task of flood forcasting.
        Returns
        -------
        out
            the output sequence of model.
        """
        nt, ngrid, nx = x.shape  # (time, basins, features(forcing, streamflow))
        out = torch.zeros(nt, ngrid, self.ny)  # (time,basins, output_features)
        for t in range(self.max_delay):
            out[t, :, :] = x[t, :, -self.ny:]
        for t in range(self.max_delay, nt):
            x0 = x[(t - self.input_delay): t, :, :(self.nx - self.ny)]
            x0_t = x0[-1, :, :]
            for i in range(self.input_delay):
                x0_i = x0[i, :, :]
                x0_t = torch.cat((x0_t, x0_i), dim=-1)
            y0 = x[(t-self.feedback_delay):t, :, -self.ny:]
            y0_t = y0[-1, :, :]
            for i in range(self.feedback_delay):
                y0i = y0[i, :, :]
                y0_t = torch.cat((y0_t, y0i), dim=-1)
            xt = torch.cat([x0_t, y0_t], dim=-1)  # (basins, features)  single step, no time dimension.
            x_l = F.relu(self.linearIn(xt))
            out_t = self.narx(x_l)  # single step
            yt = self.linearOut(out_t)  # (basins, output_features) single step output value, e.g. streamflow
            out[t, :, :] = yt
            if self.close_loop:
                x[t+1, :, -self.ny:] = yt
        return out


class NestedNarx(nn.Module):
    """NestedNarx model

    """
    def __init__(
            self,
            n_input_features: int,
            n_output_features: int,
            n_hidden_states: int,
            input_delay: int,
            feedback_delay: int,
            # num_layers: int = 10,
            close_loop: bool = False,
            nested_model: dict = None,
        ):
        """Initialize NestedNarx model
        
        """
        super(NestedNarx, self).__init__()
        self.dl_model = Narx(
            n_input_features,
            n_output_features,
            n_hidden_states,
            input_delay,
            feedback_delay,
            # num_layers,
            close_loop,
        )
        self.Nested_model = nested_model
        self.basin_trees = self.Nested_model["basin_tree"]
        self.basin_tree_max_order = self.Nested_model["basin_tree_max_order"]
        self.basin_list = self.Nested_model["basin_list"]
        self.order_list = self.Nested_model["order_list"]

    def forward(self, x):
        """
        implement netsed calculation here.
        x
            input data.  (forcing, target)/(prcp,pet,streamflow)   [sequence, batch, feature]/[time, basin, (prcp,pet,streamflow)]  sequence first.
        """
        n_step, n_basin, n_feature = x.size()  # split in basin dimension.  self.basin_list
        basin_list_x = []
        if n_basin == len(self.basin_list):
            for i in range(n_basin):
                x_i = x[:, i, :]   # time|(prcp,pet,streamflow)
                basin_list_x.append(x_i)
        n_basintrees = len(self.basintress)
        # calculate along basintree
        # basins with a same order calculate together
        # meanwhile take the link relationship between basins into count.
        # means call narx for each basin
        basin_tree_i = [Basin]
        for i in range(n_basintrees):
            # root, limb, single_basin
            basin_tree_i = self.basin_trees[i]
            n_basin_i = len(basin_tree_i)
            max_order_i = basin_tree_i[-1].basin_order
            for j in range(max_order_i):
                y = self.dl_model(x)  # streamflow



