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
        close_loop: bool, whether to close the loop when feeding in.      open loop for training, close loop for forecasting.
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
            yt = self.linearOut(out_t)  # (basins, output_features) single step output value, e.g. streamflow
            out[t, :, :] = yt
            if self.close_loop:
                if t < (nt-1):
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
        self.n_basintrees = len(self.basin_trees)
        self.device = None
        self.b_set_device = False
        self.set_dl_model()

    def set_dl_model(self):
        """set dl model into each basin"""
        for i in range(self.n_basintrees):  # basintrees
            n_order_i = len(self.basin_trees[i])
            for j in range(n_order_i):  # order
                n_basin_j = len(self.basin_trees[i][j])
                for k in range(n_basin_j):  # per order
                    self.basin_trees[i][j][k].set_model(self.dl_model)

    def set_device(self):
        """set device into each basin"""
        for i in range(self.n_basintrees):  # basintrees
            n_order_i = len(self.basin_trees[i])
            for j in range(n_order_i):  # order
                n_basin_j = len(self.basin_trees[i][j])
                for k in range(n_basin_j):  # per order
                    self.basin_trees[i][j][k].set_device(self.device)
                    if j > 0:
                        # the last/root basin outflow directly, no node_ds.
                        self.basin_trees[i][j][k].node_ds.set_device(self.device)  # basin output  
        self.b_set_device = True

    def remove_dl_model(self):
        """remove dl model of each basin"""
        for i in range(self.n_basintrees):  # basintrees
            n_order_i = len(self.basin_trees[i])
            for j in range(n_order_i):  # order
                n_basin_j = len(self.basin_trees[i][j])
                for k in range(n_basin_j):  # per order
                    self.basin_trees[i][j][k].remove_model()

    def remove_device(self):
        """remove device of each basin and node"""
        for i in range(self.n_basintrees):  # basintrees
            n_order_i = len(self.basin_trees[i])
            for j in range(n_order_i):  # order
                n_basin_j = len(self.basin_trees[i][j])
                for k in range(n_basin_j):  # per order
                    self.basin_trees[i][j][k].remove_device()
                    if j > 0:
                        # the last/root basin outflow directly, no node_ds.
                        self.basin_trees[i][j][k].node_ds.remove_device()  # basin output  
        self.b_set_device = False
    
    def remove_memory(self):
        for i in range(self.n_basintrees):  # basintrees
            max_order_i = len(self.n_basin_per_order_list[i])
            for j in range(max_order_i - 1, -1, -1):  # order
                n_basin_j = self.n_basin_per_order_list[i][j]
                for k in range(n_basin_j-1, -1, -1):  # per order
                    self.basin_trees[i][j][k].remove_data()  # inflow coming from upstream basin             
                    if j > 0:
                        # the last/root basin outflow directly, no node_ds.
                        self.basin_trees[i][j][k].node_ds.remove_data()  # basin output

    def remove_model_and_memory(self):
        self.remove_dl_model()
        self.remove_device()
        self.remove_memory()


    def forward(self, x):
        """
        implement netsed calculation here.
        deal with data order -> done.
        calculate along basintree -> may can calculate by order   seems cannot.
        basins with a same order calculate together -> for each tree
        meanwhile take the link relationship between basins into count.  -> done. use basin, node and basintree.
        means call narx for each basin -> may can call naxr for each order   seems cannot.
        x
            input data.  (forcing, target)/(prcp,pet,streamflow)   [sequence, batch, feature]/[time, basin, (prcp,pet,streamflow)]  sequence first.
        """
        self.device = x.device

        if not self.b_set_device:
            self.set_device()

        n_t, n_basin, n_feature = x.size()  # split in basin dimension.  self.basin_list    n_feature = self.nx + self.ny
        if n_basin != len(self.basin_list):
            raise ValueError("The dimension of input data x dismatch with basintree, please check both.")
        else:
            # remove data in basin before calculation.
            self.remove_memory()

            out = torch.Tensor([])
            # set data
            m = 0
            for i in range(self.n_basintrees):  # basintrees
                n_order_i = len(self.basin_trees[i])
                for j in range(n_order_i):  # order
                    n_basin_j = len(self.basin_trees[i][j])
                    for k in range(n_basin_j):  # per order
                        self.basin_trees[i][j][k].set_x_data(torch.unsqueeze(x[:, m, :self.nx], 1))
                        self.basin_trees[i][j][k].set_y_data(torch.unsqueeze(x[:, m, -self.ny:], 1))
                        m = m + 1
            # run model
            for i in range(self.n_basintrees):  # basintrees
                max_order_i = len(self.n_basin_per_order_list[i])
                for j in range(max_order_i - 1, -1, -1):  # order
                    n_basin_j = self.n_basin_per_order_list[i][j]
                    for k in range(n_basin_j-1, -1, -1):  # per order
                        self.basin_trees[i][j][k].node_us.refresh_y_output()
                        self.basin_trees[i][j][k].get_y_us_data()  # inflow coming from upstream basin
                        out = torch.cat((out, self.basin_trees[i][j][k].run_model()), dim=1)               
                        if j > 0:
                            # the last/root basin outflow directly, no node_ds.
                            self.basin_trees[i][j][k].set_output()  # basin output
            # return result
            return out


# ============================= test session starts ==============================
# platform linux -- Python 3.13.3, pytest-8.3.5, pluggy-1.5.0
# rootdir: /home/yulili/code/torchhydro/tests
# configfile: ../setup.cfg
# plugins: mock-3.14.0
# Backend tkagg is interactive backend. Turning interactive mode on.
# collected 1 item

# test_camelsfr_netsednarx.py update config file
# !!!!!!NOTE!!!!!!!!
# -------Please make sure the PRECIPITATION variable is in the 1st location in var_t setting!!---------
# If you have POTENTIAL_EVAPOTRANSPIRATION, please set it the 2nd!!!-
# !!!!!!NOTE!!!!!!!!
# -------Please make sure the STREAMFLOW variable is in the 1st location in var_out setting!!---------
# Finish Normalization


#   0%|          | 0/18 [00:00<?, ?it/s]
# 100%|██████████| 18/18 [00:00<00:00, 1021.22it/s]
# Finish Normalization


#   0%|          | 0/18 [00:00<?, ?it/s]
# 100%|██████████| 18/18 [00:00<00:00, 17954.21it/s]
# Finish Normalization


#   0%|          | 0/18 [00:00<?, ?it/s]
# 100%|██████████| 18/18 [00:00<00:00, 18113.60it/s]
# Torch is using cpu
# I0425 17:00:49.431000 10034 site-packages/torch/distributed/nn/jit/instantiator.py:22] Created a temporary directory at /tmp/tmpoky0yy_p
# I0425 17:00:49.435000 10034 site-packages/torch/distributed/nn/jit/instantiator.py:73] Writing /tmp/tmpoky0yy_p/_remote_module_non_scriptable.py
#   0%|          | 0/54 [00:00<?, ?it/s]
#   2%|▏         | 1/54 [00:00<00:07,  7.16it/s]
#   4%|▎         | 2/54 [00:00<00:09,  5.21it/s]
#   6%|▌         | 3/54 [00:00<00:10,  4.82it/s]
#   7%|▋         | 4/54 [00:00<00:10,  4.70it/s]
#   9%|▉         | 5/54 [00:01<00:10,  4.50it/s]
#  11%|█         | 6/54 [00:01<00:11,  4.27it/s]
#  13%|█▎        | 7/54 [00:01<00:11,  4.18it/s]
#  15%|█▍        | 8/54 [00:01<00:10,  4.21it/s]
#  17%|█▋        | 9/54 [00:02<00:10,  4.33it/s]
#  19%|█▊        | 10/54 [00:02<00:10,  4.40it/s]
#  20%|██        | 11/54 [00:02<00:09,  4.37it/s]
#  22%|██▏       | 12/54 [00:02<00:09,  4.39it/s]
#  24%|██▍       | 13/54 [00:02<00:09,  4.41it/s]
#  26%|██▌       | 14/54 [00:03<00:09,  4.43it/s]
#  28%|██▊       | 15/54 [00:03<00:08,  4.50it/s]
#  30%|██▉       | 16/54 [00:03<00:08,  4.62it/s]
#  31%|███▏      | 17/54 [00:03<00:08,  4.40it/s]
#  33%|███▎      | 18/54 [00:04<00:08,  4.43it/s]
#  35%|███▌      | 19/54 [00:04<00:07,  4.55it/s]
#  37%|███▋      | 20/54 [00:04<00:07,  4.49it/s]
#  39%|███▉      | 21/54 [00:04<00:07,  4.46it/s]
#  41%|████      | 22/54 [00:04<00:07,  4.46it/s]
#  43%|████▎     | 23/54 [00:05<00:06,  4.44it/s]
#  44%|████▍     | 24/54 [00:05<00:06,  4.41it/s]
#  46%|████▋     | 25/54 [00:05<00:06,  4.46it/s]
#  48%|████▊     | 26/54 [00:05<00:06,  4.39it/s]
#  50%|█████     | 27/54 [00:06<00:06,  4.47it/s]
#  52%|█████▏    | 28/54 [00:06<00:05,  4.49it/s]
#  54%|█████▎    | 29/54 [00:06<00:05,  4.53it/s]
#  56%|█████▌    | 30/54 [00:06<00:05,  4.50it/s]
#  57%|█████▋    | 31/54 [00:06<00:05,  4.50it/s]
#  59%|█████▉    | 32/54 [00:07<00:04,  4.60it/s]
#  61%|██████    | 33/54 [00:07<00:04,  4.58it/s]
#  63%|██████▎   | 34/54 [00:07<00:04,  4.58it/s]
#  65%|██████▍   | 35/54 [00:07<00:04,  4.58it/s]
#  67%|██████▋   | 36/54 [00:08<00:03,  4.52it/s]
#  69%|██████▊   | 37/54 [00:08<00:03,  4.53it/s]
#  70%|███████   | 38/54 [00:08<00:03,  4.48it/s]
#  72%|███████▏  | 39/54 [00:08<00:03,  4.44it/s]
#  74%|███████▍  | 40/54 [00:08<00:03,  4.42it/s]
#  76%|███████▌  | 41/54 [00:09<00:02,  4.42it/s]
#  78%|███████▊  | 42/54 [00:09<00:02,  4.39it/s]
#  80%|███████▉  | 43/54 [00:09<00:02,  4.43it/s]
#  81%|████████▏ | 44/54 [00:09<00:02,  4.42it/s]
#  83%|████████▎ | 45/54 [00:10<00:02,  4.40it/s]
#  85%|████████▌ | 46/54 [00:10<00:01,  4.37it/s]
#  87%|████████▋ | 47/54 [00:10<00:01,  4.43it/s]
#  89%|████████▉ | 48/54 [00:10<00:01,  4.38it/s]
#  91%|█████████ | 49/54 [00:10<00:01,  4.48it/s]
#  93%|█████████▎| 50/54 [00:11<00:00,  4.34it/s]
#  94%|█████████▍| 51/54 [00:11<00:00,  4.38it/s]
#  96%|█████████▋| 52/54 [00:11<00:00,  4.37it/s]
#  98%|█████████▊| 53/54 [00:11<00:00,  4.31it/s]
# 100%|██████████| 54/54 [00:12<00:00,  4.27it/s]
# 100%|██████████| 54/54 [00:12<00:00,  4.45it/s]
# Epoch 10 Loss 1.0289 time 15.49 lr 1.0
# NestedNarx(
#   (dl_model): Narx(
#     (linearIn): Linear(in_features=2, out_features=64, bias=True)
#     (narx): RNNCell(64, 64)
#     (linearOut): Linear(in_features=64, out_features=1, bias=True)
#   )
# )
# Epoch 10 Valid Loss 1.1688 Valid Metric {'NSE of streamflow': [-680.5365600585938, nan, -62.341819763183594, -2.5589511394500732, nan, nan, -23.632856369018555, -18.84280014038086, nan, nan, 0.45687466859817505, -19.639020919799805, nan, -1.2749407291412354, nan, nan, -6.532687187194824, -10.594403266906738], 'RMSE of streamflow': [1.1216270923614502, nan, 0.8607173562049866, 0.290521502494812, nan, nan, 1.0005472898483276, 1.0005472898483276, nan, nan, 0.290521502494812, 0.8607173562049866, nan, 1.1216270923614502, nan, nan, 0.5103662610054016, 0.5103662610054016], 'R2 of streamflow': [-680.5365600585938, nan, -62.341819763183594, -2.5589511394500732, nan, nan, -23.632856369018555, -18.84280014038086, nan, nan, 0.45687466859817505, -19.639020919799805, nan, -1.2749407291412354, nan, nan, -6.532687187194824, -10.594403266906738], 'KGE of streamflow': [-15.356195128031409, nan, 0.05966293141395751, -2.7649090034765273, nan, nan, -7.3874577742245116, 0.10067593634430694, nan, nan, 0.014847865404440608, -0.3666219577027954, nan, -4.117791945306762, nan, nan, 0.15946721685588872, -3.49628034978332], 'FHV of streamflow': [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan], 'FLV of streamflow': [-23.521709442138672, nan, -47.714080810546875, 72.24737548828125, nan, nan, 318.91888427734375, -76.1290283203125, nan, nan, -41.94396209716797, 91.2560806274414, nan, 30.756059646606445, nan, nan, -67.43423461914062, 207.0708770751953]}
# Weights sucessfully loaded
# All processes are finished!


#   0%|          | 0/54 [00:00<?, ?it/s]
#   2%|▏         | 1/54 [00:00<00:08,  6.18it/s]
#   4%|▎         | 2/54 [00:00<00:09,  5.20it/s]
#   6%|▌         | 3/54 [00:00<00:10,  4.76it/s]
#   7%|▋         | 4/54 [00:00<00:10,  4.63it/s]
#   9%|▉         | 5/54 [00:01<00:10,  4.53it/s]
#  11%|█         | 6/54 [00:01<00:10,  4.39it/s]
#  13%|█▎        | 7/54 [00:01<00:10,  4.30it/s]
#  15%|█▍        | 8/54 [00:01<00:10,  4.25it/s]
#  17%|█▋        | 9/54 [00:02<00:10,  4.15it/s]
#  19%|█▊        | 10/54 [00:02<00:10,  4.19it/s]
#  20%|██        | 11/54 [00:02<00:10,  4.21it/s]
#  22%|██▏       | 12/54 [00:02<00:10,  4.18it/s]
#  24%|██▍       | 13/54 [00:02<00:09,  4.19it/s]
#  26%|██▌       | 14/54 [00:03<00:09,  4.18it/s]
#  28%|██▊       | 15/54 [00:03<00:09,  4.24it/s]
#  30%|██▉       | 16/54 [00:03<00:09,  4.18it/s]
#  31%|███▏      | 17/54 [00:03<00:08,  4.12it/s]
#  33%|███▎      | 18/54 [00:04<00:08,  4.07it/s]
#  35%|███▌      | 19/54 [00:04<00:08,  3.94it/s]
#  37%|███▋      | 20/54 [00:04<00:08,  4.03it/s]
#  39%|███▉      | 21/54 [00:04<00:08,  4.08it/s]
#  41%|████      | 22/54 [00:05<00:07,  4.12it/s]
#  43%|████▎     | 23/54 [00:05<00:07,  4.17it/s]
#  44%|████▍     | 24/54 [00:05<00:07,  4.21it/s]
#  46%|████▋     | 25/54 [00:05<00:06,  4.25it/s]
#  48%|████▊     | 26/54 [00:06<00:06,  4.24it/s]
#  50%|█████     | 27/54 [00:06<00:06,  4.31it/s]
#  52%|█████▏    | 28/54 [00:06<00:06,  4.26it/s]
#  54%|█████▎    | 29/54 [00:06<00:05,  4.22it/s]
#  56%|█████▌    | 30/54 [00:07<00:05,  4.30it/s]
#  57%|█████▋    | 31/54 [00:07<00:05,  4.38it/s]
#  59%|█████▉    | 32/54 [00:07<00:04,  4.40it/s]
#  61%|██████    | 33/54 [00:07<00:04,  4.31it/s]
#  63%|██████▎   | 34/54 [00:07<00:04,  4.27it/s]
#  65%|██████▍   | 35/54 [00:08<00:04,  4.26it/s]
#  67%|██████▋   | 36/54 [00:08<00:04,  4.31it/s]
#  69%|██████▊   | 37/54 [00:08<00:03,  4.26it/s]
#  70%|███████   | 38/54 [00:08<00:03,  4.34it/s]
#  72%|███████▏  | 39/54 [00:09<00:03,  4.27it/s]
#  74%|███████▍  | 40/54 [00:09<00:03,  4.35it/s]
#  76%|███████▌  | 41/54 [00:09<00:03,  4.29it/s]
#  78%|███████▊  | 42/54 [00:09<00:02,  4.22it/s]
#  80%|███████▉  | 43/54 [00:10<00:02,  4.19it/s]
#  81%|████████▏ | 44/54 [00:10<00:02,  4.19it/s]
#  83%|████████▎ | 45/54 [00:10<00:02,  4.19it/s]
#  85%|████████▌ | 46/54 [00:10<00:01,  4.23it/s]
#  87%|████████▋ | 47/54 [00:11<00:01,  4.23it/s]
#  89%|████████▉ | 48/54 [00:11<00:01,  4.23it/s]
#  91%|█████████ | 49/54 [00:11<00:01,  4.23it/s]
#  93%|█████████▎| 50/54 [00:11<00:00,  4.16it/s]
#  94%|█████████▍| 51/54 [00:12<00:00,  4.12it/s]
#  96%|█████████▋| 52/54 [00:12<00:00,  4.08it/s]
#  98%|█████████▊| 53/54 [00:12<00:00,  4.20it/s]
# 100%|██████████| 54/54 [00:12<00:00,  4.18it/s]
# 100%|██████████| 54/54 [00:12<00:00,  4.24it/s]
# Epoch 2 Loss 1.0270 time 32.54 lr 1.0
# NestedNarx(
#   (dl_model): Narx(
#     (linearIn): Linear(in_features=2, out_features=64, bias=True)
#     (narx): RNNCell(64, 64)
#     (linearOut): Linear(in_features=64, out_features=1, bias=True)
#   )
# )
# Epoch 2 Valid Loss 1.1688 Valid Metric {'NSE of streamflow': [-680.5365600585938, nan, -62.341819763183594, -2.5589511394500732, nan, nan, -23.632856369018555, -18.84280014038086, nan, nan, 0.45687466859817505, -19.639020919799805, nan, -1.2749407291412354, nan, nan, -6.532687187194824, -10.594403266906738], 'RMSE of streamflow': [1.1216270923614502, nan, 0.8607173562049866, 0.290521502494812, nan, nan, 1.0005472898483276, 1.0005472898483276, nan, nan, 0.290521502494812, 0.8607173562049866, nan, 1.1216270923614502, nan, nan, 0.5103662610054016, 0.5103662610054016], 'R2 of streamflow': [-680.5365600585938, nan, -62.341819763183594, -2.5589511394500732, nan, nan, -23.632856369018555, -18.84280014038086, nan, nan, 0.45687466859817505, -19.639020919799805, nan, -1.2749407291412354, nan, nan, -6.532687187194824, -10.594403266906738], 'KGE of streamflow': [-15.356195128031409, nan, 0.05966293141395751, -2.7649090034765273, nan, nan, -7.3874577742245116, 0.10067593634430694, nan, nan, 0.014847865404440608, -0.3666219577027954, nan, -4.117791945306762, nan, nan, 0.15946721685588872, -3.49628034978332], 'FHV of streamflow': [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan], 'FLV of streamflow': [-23.521709442138672, nan, -47.714080810546875, 72.24737548828125, nan, nan, 318.91888427734375, -76.1290283203125, nan, nan, -41.94396209716797, 91.2560806274414, nan, 30.756059646606445, nan, nan, -67.43423461914062, 207.0708770751953]}
# Weights sucessfully loaded
# All processes are finished!