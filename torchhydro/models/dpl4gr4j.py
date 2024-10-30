"""
Simulates streamflow over time using the model logic from GR4J as implemented in PyTorch.
This function can be used to offer up the functionality of GR4J with added gradient information.
"""
from typing import Tuple, Optional, Union

import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F

from models.simple_lstm import SimpleLSTM
from torchhydro.models.ann import SimpleAnn
from torchhydro.models.dpl4xaj import ann_pbm, lstm_pbm
from torchhydro.models.kernel_conv import uh_conv


def calculate_precip_store(s, precip_net, x1):
    """Calculates the amount of rainfall which enters the storage reservoir."""
    n = x1 * (1.0 - (s / x1) ** 2) * torch.tanh(precip_net / x1)
    d = 1.0 + (s / x1) * torch.tanh(precip_net / x1)
    return n / d


# Determines the evaporation loss from the production store
def calculate_evap_store(s, evap_net, x1):
    """Calculates the amount of evaporation out of the storage reservoir."""
    n = s * (2.0 - s / x1) * torch.tanh(evap_net / x1)
    d = 1.0 + (1.0 - s / x1) * torch.tanh(evap_net / x1)
    return n / d


# Determines how much water percolates out of the production store to streamflow
def calculate_perc(current_store, x1):
    """Calculates the percolation from the storage reservoir into streamflow."""
    return current_store * (
        1.0 - (1.0 + (4.0 / 9.0 * current_store / x1) ** 4) ** -0.25
    )


def production(
    p_and_e: Tensor, x1: Tensor, s_level: Optional[Tensor] = None
) -> Tuple[Tensor, Tensor]:
    """
    an one-step calculation for production store in GR4J
    the dimension of the cell: [batch, feature]

    Parameters
    ----------
    p_and_e
        P is pe[:, 0] and E is pe[:, 1]; similar with the "input" in the RNNCell
    x1:
        Storage reservoir parameter;
    s_level
        s_level means S in the GR4J Model; similar with the "hx" in the RNNCell
        Initial value of storage in the storage reservoir.

    Returns
    -------
    tuple
        contains the Pr and updated S
    """
    gr4j_device = p_and_e.device
    # Calculate net precipitation and evapotranspiration
    precip_difference = p_and_e[:, 0] - p_and_e[:, 1]
    precip_net = torch.maximum(precip_difference, Tensor([0.0]).to(gr4j_device))
    evap_net = torch.maximum(-precip_difference, Tensor([0.0]).to(gr4j_device))

    if s_level is None:
        s_level = 0.6 * (x1.detach())

    # s_level should not be larger than x1
    s_level = torch.clamp(s_level, torch.full(s_level.shape, 0.0).to(gr4j_device), x1)

    # Calculate the fraction of net precipitation that is stored
    precip_store = calculate_precip_store(s_level, precip_net, x1)

    # Calculate the amount of evaporation from storage
    evap_store = calculate_evap_store(s_level, evap_net, x1)

    # Update the storage by adding effective precipitation and
    # removing evaporation
    s_update = s_level - evap_store + precip_store
    # s_level should not be larger than self.x1
    s_update = torch.clamp(
        s_update, torch.full(s_update.shape, 0.0).to(gr4j_device), x1
    )

    # Update the storage again to reflect percolation out of the store
    perc = calculate_perc(s_update, x1)
    s_update = s_update - perc
    # perc is always lower than S because of the calculation itself, so we don't need clamp here anymore.

    # The precip. for routing is the sum of the rainfall which
    # did not make it to storage and the percolation from the store
    current_runoff = perc + (precip_net - precip_store)
    return current_runoff, s_update


def uh_gr4j(x4):
    """
    Generate the convolution kernel for the convolution operation in routing module of GR4J

    Parameters
    ----------
    x4
        the dim of x4 is [batch]

    Returns
    -------
    list
        UH1s and UH2s for all basins
    """
    gr4j_device = x4.device
    uh1_ordinates = []
    uh2_ordinates = []
    for i in range(len(x4)):
        # for SH1, the pieces are: 0, 0<t<x4, t>=x4
        uh1_ordinates_t1 = torch.arange(
            0.0, torch.ceil(x4[i]).detach().cpu().numpy().item()
        ).to(gr4j_device)
        uh1_ordinates_t = torch.arange(
            1.0, torch.ceil(x4[i] + 1.0).detach().cpu().numpy().item()
        ).to(gr4j_device)
        # for SH2, the pieces are: 0, 0<t<=x4, x4<t<2x4, t>=2x4
        uh2_ords_t1_seq_x4 = torch.arange(
            0.0, torch.floor(x4[i] + 1).detach().cpu().numpy().item()
        ).to(gr4j_device)
        uh2_ords_t1_larger_x4 = torch.arange(
            torch.floor(x4[i] + 1).detach().cpu().numpy().item(),
            torch.ceil(2 * x4[i]).detach().cpu().numpy().item(),
        ).to(gr4j_device)
        uh2_ords_t_seq_x4 = torch.arange(
            1.0, torch.floor(x4[i] + 1).detach().cpu().numpy().item()
        ).to(gr4j_device)
        uh2_ords_t_larger_x4 = torch.arange(
            torch.floor(x4[i] + 1).detach().cpu().numpy().item(),
            torch.ceil(2 * x4[i] + 1.0).detach().cpu().numpy().item(),
        ).to(gr4j_device)
        s_curve1t1 = (uh1_ordinates_t1 / x4[i]) ** 2.5
        s_curve21t1 = 0.5 * (uh2_ords_t1_seq_x4 / x4[i]) ** 2.5
        s_curve22t1 = 1.0 - 0.5 * (2 - uh2_ords_t1_larger_x4 / x4[i]) ** 2.5
        s_curve2t1 = torch.cat([s_curve21t1, s_curve22t1])
        # t1 cannot be larger than x4, but t can, so we should set (uh1_ordinates_t / x4[i]) <=1
        # we don't use torch.clamp, because it seems we have to use mask, or we will get nan for grad. More details
        # could be seen here: https://github.com/waterDLut/hydro-dl-basic/tree/dev/3-more-knowledge/5-grad-problem.ipynb
        uh1_x4 = uh1_ordinates_t / x4[i]
        limit_uh1_x4 = 1 - F.relu(1 - uh1_x4)
        limit_uh2_smaller_x4 = uh2_ords_t_seq_x4 / x4[i]
        uh2_larger_x4 = 2 - uh2_ords_t_larger_x4 / x4[i]
        limit_uh2_larger_x4 = F.relu(uh2_larger_x4)
        s_curve1t = limit_uh1_x4 ** 2.5
        s_curve21t = 0.5 * limit_uh2_smaller_x4 ** 2.5
        s_curve22t = 1.0 - 0.5 * limit_uh2_larger_x4 ** 2.5
        s_curve2t = torch.cat([s_curve21t, s_curve22t])
        uh1_ordinate = s_curve1t - s_curve1t1
        uh2_ordinate = s_curve2t - s_curve2t1
        uh1_ordinates.append(uh1_ordinate)
        uh2_ordinates.append(uh2_ordinate)

    return uh1_ordinates, uh2_ordinates


def routing(q9: Tensor, q1: Tensor, x2, x3, r_level: Optional[Tensor] = None):
    """
    the GR4J routing-module unit cell for time-sequence loop

    Parameters
    ----------
    q9

    q1

    x2
        Catchment water exchange parameter
    x3
        Routing reservoir parameters
    r_level
        Beginning value of storage in the routing reservoir.

    Returns
    -------

    """
    gr4j_device = q9.device
    if r_level is None:
        r_level = 0.7 * (x3.detach())
    # r_level should not be larger than self.x3
    r_level = torch.clamp(r_level, torch.full(r_level.shape, 0.0).to(gr4j_device), x3)
    groundwater_ex = x2 * (r_level / x3) ** 3.5
    r_updated = torch.maximum(
        torch.full(r_level.shape, 0.0).to(gr4j_device), r_level + q9 + groundwater_ex
    )

    qr = r_updated * (1.0 - (1.0 + (r_updated / x3) ** 4) ** -0.25)
    r_updated = r_updated - qr

    qd = torch.maximum(
        torch.full(groundwater_ex.shape, 0.0).to(gr4j_device), q1 + groundwater_ex
    )
    q = qr + qd
    return q, r_updated


class Gr4j4Dpl(nn.Module):
    """
    the nn.Module style GR4J model
    """

    def __init__(self, warmup_length: int):
        """
        Parameters
        ----------
        warmup_length
            length of warmup period
        """
        super(Gr4j4Dpl, self).__init__()
        self.params_names = ["X1", "X2", "X3", "X4"]
        self.x1_scale = [100.0, 1200.0]
        self.x2_sacle = [-5.0, 3.0]
        self.x3_scale = [20.0, 300.0]
        self.x4_scale = [1.1, 2.9]
        self.warmup_length = warmup_length
        self.feature_size = 2

    def forward(self, p_and_e, parameters, return_state=False):
        gr4j_device = p_and_e.device
        x1 = self.x1_scale[0] + parameters[:, 0] * (self.x1_scale[1] - self.x1_scale[0])
        x2 = self.x2_sacle[0] + parameters[:, 1] * (self.x2_sacle[1] - self.x2_sacle[0])
        x3 = self.x3_scale[0] + parameters[:, 2] * (self.x3_scale[1] - self.x3_scale[0])
        x4 = self.x4_scale[0] + parameters[:, 3] * (self.x4_scale[1] - self.x4_scale[0])

        warmup_length = self.warmup_length
        if warmup_length > 0:
            # set no_grad for warmup periods
            with torch.no_grad():
                p_and_e_warmup = p_and_e[0:warmup_length, :, :]
                cal_init = Gr4j4Dpl(0)
                if cal_init.warmup_length > 0:
                    raise RuntimeError("Please set init model's warmup length to 0!!!")
                _, s0, r0 = cal_init(p_and_e_warmup, parameters, return_state=True)
        else:
            # use detach func to make wu0 no_grad as it is an initial value
            s0 = 0.5 * x1.detach()
            r0 = 0.5 * x3.detach()
        inputs = p_and_e[warmup_length:, :, :]
        streamflow_ = torch.full(inputs.shape[:2], 0.0).to(gr4j_device)
        prs = torch.full(inputs.shape[:2], 0.0).to(gr4j_device)
        for i in range(inputs.shape[0]):
            if i == 0:
                pr, s = production(inputs[i, :, :], x1, s0)
            else:
                pr, s = production(inputs[i, :, :], x1, s)
            prs[i, :] = pr
        prs_x = torch.unsqueeze(prs, dim=2)
        conv_q9, conv_q1 = uh_gr4j(x4)
        q9 = torch.full([inputs.shape[0], inputs.shape[1], 1], 0.0).to(gr4j_device)
        q1 = torch.full([inputs.shape[0], inputs.shape[1], 1], 0.0).to(gr4j_device)
        for j in range(inputs.shape[1]):
            q9[:, j : j + 1, :] = uh_conv(
                prs_x[:, j : j + 1, :], conv_q9[j].reshape(-1, 1, 1)
            )
            q1[:, j : j + 1, :] = uh_conv(
                prs_x[:, j : j + 1, :], conv_q1[j].reshape(-1, 1, 1)
            )
        for i in range(inputs.shape[0]):
            if i == 0:
                q, r = routing(q9[i, :, 0], q1[i, :, 0], x2, x3, r0)
            else:
                q, r = routing(q9[i, :, 0], q1[i, :, 0], x2, x3, r)
            streamflow_[i, :] = q
        streamflow = torch.unsqueeze(streamflow_, dim=2)
        if return_state:
            return streamflow, s, r
        return streamflow


class DplLstmGr4j(nn.Module):
    def __init__(
        self,
        n_input_features,
        n_output_features,
        n_hidden_states,
        warmup_length,
        param_limit_func="sigmoid",
        param_test_way="final",
    ):
        """
        Differential Parameter learning model: LSTM -> Param -> Gr4j

        The principle can be seen here: https://doi.org/10.1038/s41467-021-26107-z

        Parameters
        ----------
        n_input_features
            the number of input features of LSTM
        n_output_features
            the number of output features of LSTM, and it should be equal to the number of learning parameters in XAJ
        n_hidden_states
            the number of hidden features of LSTM
        warmup_length
            the length of warmup periods;
            hydrologic models need a warmup period to generate reasonable initial state values
        param_limit_func
            function used to limit the range of params; now it is sigmoid or clamp function
        param_test_way
            how we use parameters from dl model when testing;
            now we have three ways:
            1. "final" -- use the final period's parameter for each period
            2. "mean_time" -- Mean values of all periods' parameters is used
            3. "mean_basin" -- Mean values of all basins' final periods' parameters is used
        """
        super(DplLstmGr4j, self).__init__()
        self.dl_model = SimpleLSTM(
            n_input_features, n_output_features, n_hidden_states
        )
        self.pb_model = Gr4j4Dpl(warmup_length)
        self.param_func = param_limit_func
        self.param_test_way = param_test_way

    def forward(self, x, z):
        """
        Differential parameter learning

        z (normalized input) -> lstm -> param -> + x (not normalized) -> gr4j -> q
        Parameters will be denormalized in gr4j model

        Parameters
        ----------
        x
            not normalized data used for physical model; a sequence-first 3-dim tensor. [sequence, batch, feature]
        z
            normalized data used for DL model; a sequence-first 3-dim tensor. [sequence, batch, feature]

        Returns
        -------
        torch.Tensor
            one time forward result
        """
        q = lstm_pbm(self.dl_model, self.pb_model, self.param_func, x, z)
        return q


class DplAnnGr4j(nn.Module):
    def __init__(
        self,
        n_input_features: int,
        n_output_features: int,
        n_hidden_states: Union[int, tuple, list],
        warmup_length: int,
        param_limit_func="sigmoid",
        param_test_way="final",
    ):
        """
        Differential Parameter learning model only with attributes as DL model's input: ANN -> Param -> Gr4j

        The principle can be seen here: https://doi.org/10.1038/s41467-021-26107-z

        Parameters
        ----------
        n_input_features
            the number of input features of ANN
        n_output_features
            the number of output features of ANN, and it should be equal to the number of learning parameters in XAJ
        n_hidden_states
            the number of hidden features of ANN; it could be Union[int, tuple, list]
        warmup_length
            the length of warmup periods;
            hydrologic models need a warmup period to generate reasonable initial state values
        param_limit_func
            function used to limit the range of params; now it is sigmoid or clamp function
        param_test_way
            how we use parameters from dl model when testing;
            now we have three ways:
            1. "final" -- use the final period's parameter for each period
            2. "mean_time" -- Mean values of all periods' parameters is used
            3. "mean_basin" -- Mean values of all basins' final periods' parameters is used
        """
        super(DplAnnGr4j, self).__init__()
        self.dl_model = SimpleAnn(n_input_features, n_output_features, n_hidden_states)
        self.pb_model = Gr4j4Dpl(warmup_length)
        self.param_func = param_limit_func
        self.param_test_way = param_test_way

    def forward(self, x, z):
        """
        Differential parameter learning

        z (normalized input) -> ANN -> param -> + x (not normalized) -> gr4j -> q
        Parameters will be denormalized in gr4j model

        Parameters
        ----------
        x
            not normalized data used for physical model; a sequence-first 3-dim tensor. [sequence, batch, feature]
        z
            normalized data used for DL model; a 2-dim tensor. [batch, feature]

        Returns
        -------
        torch.Tensor
            one time forward result
        """
        q = ann_pbm(self.dl_model, self.pb_model, self.param_func, x, z)
        return q
