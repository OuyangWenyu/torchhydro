"""Narx and NestedNarx model"""

import torch
import torch.nn as nn
from torch.nn import functional as F


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
        nt, ngrid, nx = x.shape  # (time, basins, features(forcing, streamflow))  nx = self.nx + self.ny
        out = torch.zeros(nt, ngrid, self.ny)  # (time, basins, output_features)
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
            yt = self.linearOut(out_t)  # (basins, output_features) single step output value, e.g. streamflow  # todo:
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
        self.nx = n_input_features
        self.ny = n_output_features
        self.close_loop = close_loop
        self.Nested_model = nested_model
        self.basin_trees = self.Nested_model["basin_trees"]  # 3 dimension
        self.basin_tree_max_order = self.Nested_model["basin_tree_max_order"]
        self.basin_list = self.Nested_model["basin_list"]  # basin_id list, one dimension.
        self.basin_list_array = self.Nested_model["basin_list_array"]  # basin_id list, 2 dimension.
        self.order_list = self.Nested_model["order_list"]  # order list, one dimension.
        self.n_basin_per_order_list = self.Nested_model["n_basin_per_order_list"]  # 2 dimension
        self.n_basin_per_order = self.Nested_model["n_basin_per_order"]  # 1 dimension
        self.n_call_froward = 0


    def forward(self, x):
        """
        implement netsed calculation here.
        deal with data order -> done.
        calculate along basintree -> may can calculate by order 
        basins with a same order calculate together -> for each tree
        meanwhile take the link relationship between basins into count.  -> done. use basin, node and basintree.
        means call narx for each basin -> may can call naxr for each order
        fit to tensor and backforward
        backforward 
        x
            input data.  (forcing, target)/(prcp,pet,streamflow)   [sequence, batch, feature]/[time, basin, (prcp,pet,streamflow)]  sequence first.
        """

        nested_model_device = x.device

        n_t, n_basin, n_feature = x.size()  # split in basin dimension.  self.basin_list    n_feature = self.nx + self.ny
        self.n_call_froward = self.n_call_froward + 1
        if self.n_call_froward > 12:
            print("self.n_call_froward > 12" )
            print("n_basin = " + format(n_basin, 'd'))
            print("len(self.basin_list) = " + format(len(self.basin_list), 'd'))
        n_basintrees = len(self.basin_trees)
        if n_basin != len(self.basin_list):
            raise ValueError("The dimension of input data x dismatch with basintree, please check both." \
            "\nself.n_call_froward = " + format(self.n_call_froward, 'd') + \
            "\nn_basin = " + format(n_basin, 'd') + \
            "\nlen(self.basin_list) = " + format(len(self.basin_list), 'd'))
        else:
            # remove data in basin before calculation.
            m = 0
            for i in range(n_basintrees):  # basintrees
                max_order_i = len(self.n_basin_per_order_list[i])
                for j in range(max_order_i - 1, -1, -1):  # order
                    n_basin_j = self.n_basin_per_order_list[i][j]
                    for k in range(n_basin_j-1, -1, -1):  # per order
                        self.basin_trees[i][j][k].remove_data()  # inflow coming from upstream basin             
                        m = m + 1
                        if j > 0:
                            # the last/root basin outflow directly, no node_ds.
                            self.basin_trees[i][j][k].node_ds.remove_data()  # basin output

            out = torch.Tensor([])
            # set data
            m = 0
            for i in range(n_basintrees):  # basintrees
                n_order_i = len(self.basin_trees[i])
                for j in range(n_order_i):  # order
                    n_basin_j = len(self.basin_trees[i][j])
                    for k in range(n_basin_j):  # per order
                        self.basin_trees[i][j][k].set_device(nested_model_device)
                        self.basin_trees[i][j][k].set_x_data(torch.unsqueeze(x[:, m, :self.nx], 1))
                        self.basin_trees[i][j][k].set_y_data(torch.unsqueeze(x[:, m, -self.ny:], 1))
                        self.basin_trees[i][j][k].set_model(self.dl_model)
                        m = m + 1
            # run model
            # try:
            # y = torch.zeros(n_t, n_basin, self.ny)
            m = 0
            for i in range(n_basintrees):  # basintrees
                max_order_i = len(self.n_basin_per_order_list[i])
                for j in range(max_order_i - 1, -1, -1):  # order
                    n_basin_j = self.n_basin_per_order_list[i][j]
                    for k in range(n_basin_j-1, -1, -1):  # per order
                        self.basin_trees[i][j][k].node_us.refresh_y_output()
                        self.basin_trees[i][j][k].get_y_us_data()  # inflow coming from upstream basin
                        # y[:, m, :] = self.basin_trees[i][j][k].run_model()  # call narx for each basin.   # todo: RuntimeError
                        # out = torch.cat((out, y[:, m, :]), dim=1)
                        out = torch.cat((out, self.basin_trees[i][j][k].run_model()), dim=1)               
                        m = m + 1
                        if j > 0 :
                            # the last/root basin outflow directly, no node_ds.
                            self.basin_trees[i][j][k].set_output()  # basin output
            # return result
            return out
            # except:
            #     raise RuntimeError("backward error.")   #

# using 0 workers

#   0%|          | 0/38 [00:00<?, ?it/s]
#   5%|▌         | 2/38 [00:00<00:02, 13.89it/s]
#  11%|█         | 4/38 [00:00<00:02, 13.89it/s]
#  16%|█▌        | 6/38 [00:00<00:02, 12.96it/s]
#  21%|██        | 8/38 [00:00<00:02, 13.53it/s]
#  26%|██▋       | 10/38 [00:00<00:01, 14.42it/s]
#  32%|███▏      | 12/38 [00:00<00:01, 15.40it/s]
#  37%|███▋      | 14/38 [00:00<00:01, 15.52it/s]
#  42%|████▏     | 16/38 [00:01<00:01, 15.01it/s]
#  47%|████▋     | 18/38 [00:01<00:01, 14.95it/s]
#  53%|█████▎    | 20/38 [00:01<00:01, 14.71it/s]
#  58%|█████▊    | 22/38 [00:01<00:01, 14.90it/s]
#  63%|██████▎   | 24/38 [00:01<00:00, 14.79it/s]
#  68%|██████▊   | 26/38 [00:01<00:00, 14.58it/s]
#  74%|███████▎  | 28/38 [00:01<00:00, 14.73it/s]
#  79%|███████▉  | 30/38 [00:02<00:00, 14.94it/s]
#  84%|████████▍ | 32/38 [00:02<00:00, 14.70it/s]
#  89%|████████▉ | 34/38 [00:02<00:00, 15.13it/s]
#  95%|█████████▍| 36/38 [00:02<00:00, 14.80it/s]
# 100%|██████████| 38/38 [00:02<00:00, 14.27it/s]
# 100%|██████████| 38/38 [00:02<00:00, 14.60it/s]
# Epoch 1 Loss 1.0455 time 14.95 lr 1.0
# NestedNarx(
#   (dl_model): Narx(
#     (linearIn): Linear(in_features=5, out_features=64, bias=True)
#     (narx): RNNCell(64, 64)
#     (linearOut): Linear(in_features=64, out_features=1, bias=True)
#   )
# )
# F
# E           ValueError: The dimension of input data x dismatch with basintree, please check both.
# E           self.n_call_froward = 39
# E           n_basin = 4
# E           len(self.basin_list) = 18

# Torch is using cpu
# I0424 11:49:32.086000 14708 site-packages/torch/distributed/nn/jit/instantiator.py:22] Created a temporary directory at /tmp/tmph3zbu7dy
# I0424 11:49:32.090000 14708 site-packages/torch/distributed/nn/jit/instantiator.py:73] Writing /tmp/tmph3zbu7dy/_remote_module_non_scriptable.py
# using 0 workers

#   0%|          | 0/12 [00:00<?, ?it/s]
#   8%|▊         | 1/12 [00:02<00:22,  2.03s/it]
#  17%|█▋        | 2/12 [00:02<00:12,  1.29s/it]
#  25%|██▌       | 3/12 [00:03<00:09,  1.06s/it]
#  33%|███▎      | 4/12 [00:04<00:07,  1.06it/s]
#  42%|████▏     | 5/12 [00:05<00:06,  1.15it/s]
#  50%|█████     | 6/12 [00:05<00:04,  1.22it/s]
#  58%|█████▊    | 7/12 [00:06<00:04,  1.19it/s]
#  67%|██████▋   | 8/12 [00:07<00:03,  1.28it/s]
#  75%|███████▌  | 9/12 [00:08<00:02,  1.34it/s]
#  83%|████████▎ | 10/12 [00:09<00:01,  1.22it/s]
#  92%|█████████▏| 11/12 [00:09<00:00,  1.30it/s]
# self.n_call_froward = 12
# n_basin = 18
# len(self.basin_list) = 18

# 100%|██████████| 12/12 [00:14<00:00,  2.12s/it]
# 100%|██████████| 12/12 [00:14<00:00,  1.24s/it]
# Epoch 1 Loss 0.8252 time 28.11 lr 1.0
# NestedNarx(
#   (dl_model): Narx(
#     (linearIn): Linear(in_features=2, out_features=64, bias=True)
#     (narx): RNNCell(64, 64)
#     (linearOut): Linear(in_features=64, out_features=1, bias=True)
#   )
# )
# F

#             # first dim is batch
# >           obs_final = torch.cat(obs, dim=0)
# E           RuntimeError: torch.cat(): expected a non-empty list of Tensors

# -- chack the configuration --   seems the batch_size.  when load data.
# train period:
# kuaisampler n_basin=18
# valid period:
# SequentiaSampler  n_basin=4


# using 0 workers

#   0%|          | 0/12 [00:00<?, ?it/s]
#   8%|▊         | 1/12 [00:02<00:25,  2.30s/it]
#  17%|█▋        | 2/12 [00:02<00:10,  1.07s/it]
#  25%|██▌       | 3/12 [00:02<00:06,  1.47it/s]
#  33%|███▎      | 4/12 [00:02<00:04,  1.99it/s]
#  42%|████▏     | 5/12 [00:03<00:02,  2.49it/s]
#  50%|█████     | 6/12 [00:03<00:02,  2.91it/s]
#  58%|█████▊    | 7/12 [00:03<00:01,  3.25it/s]
#  67%|██████▋   | 8/12 [00:03<00:01,  3.58it/s]
#  75%|███████▌  | 9/12 [00:04<00:00,  3.75it/s]
#  83%|████████▎ | 10/12 [00:04<00:00,  3.96it/s]
#  92%|█████████▏| 11/12 [00:04<00:00,  4.12it/s]
# 100%|██████████| 12/12 [00:04<00:00,  4.23it/s]
# 100%|██████████| 12/12 [00:04<00:00,  2.52it/s]
# Epoch 1 Loss 0.8252 time 11.50 lr 1.0
# NestedNarx(
#   (dl_model): Narx(
#     (linearIn): Linear(in_features=2, out_features=64, bias=True)
#     (narx): RNNCell(64, 64)
#     (linearOut): Linear(in_features=64, out_features=1, bias=True)
#   )
# )
# self.n_call_froward > 12
# n_basin = 4
# len(self.basin_list) = 18
# F

#            ValueError: The dimension of input data x dismatch with basintree, please check both.
# E           self.n_call_froward = 13
# E           n_basin = 4
# E           len(self.basin_list) = 18