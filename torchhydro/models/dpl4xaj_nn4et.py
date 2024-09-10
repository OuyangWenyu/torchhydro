"""
The method is similar with dpl4xaj.py.
The difference between dpl4xaj and dpl4xaj_nn4et is:
in the former, the parameter of PBM is only one output of a DL model,
while in the latter, time series output of a DL model can be as parameter of PBM 
and some modules could be replaced with neural networks
"""

from typing import Union
import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F

from torchhydro.configs.model_config import MODEL_PARAM_DICT, MODEL_PARAM_TEST_WAY
from torchhydro.models.ann import SimpleAnn
from torchhydro.models.kernel_conv import KernelConv
from torchhydro.models.simple_lstm import SimpleLSTM
from torchhydro.models.dpl4xaj import (
    calculate_prcp_runoff,
    linear_reservoir,
    xaj_sources,
    xaj_sources5mm,
)

PRECISION = 1e-5


class NnModule4Hydro(nn.Module):
    """A NN module for Hydrological model.
    Generally, the difference between it and normal NN is:
    we need constrain its output to some specific value range

    Parameters
    ----------
    nn : _type_
        _description_
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        hidden_size: Union[int, tuple, list] = None,
        dr: Union[float, tuple, list] = 0.0,
    ):
        """
        A simple multi-layer NN model with final linear layer

        Parameters
        ----------
        nx
            number of input neurons
        ny
            number of output neurons
        hidden_size
            a list/tuple which contains number of neurons in each hidden layer;
            if int, only one hidden layer except for hidden_size=0
        dr
            dropout rate of layers, default is 0.0 which means no dropout;
            here we set number of dropout layers to (number of nn layers - 1)
        """
        super(NnModule4Hydro, self).__init__()
        self.ann = SimpleAnn(nx, ny, hidden_size, dr)

    def forward(self, x, w0, prcp, pet, k):
        """the forward function of the NN ET module

        Parameters
        ----------
        x : _type_
            _description_
        w0 : _type_
            water storage
        p : _type_
            precipitation
        pet: tensor
            potential evapotranspiration, used to be part of upper limit of ET
        k: tensor
            coefficient of PET in XAJ model, used to be part of upper limit of ET

        Returns
        -------
        _type_
            _description_
        """
        zeros = torch.full_like(w0, 0.0, device=x.device)
        et = torch.full_like(w0, 0.0, device=x.device)
        w_mask = w0 + prcp > PRECISION
        y = self.ann(x)
        z = y.flatten()
        et[w_mask] = torch.clamp(
            z[w_mask],
            min=zeros[w_mask],
            # torch.minimum computes the element-wise minimum: https://pytorch.org/docs/stable/generated/torch.minimum.html
            # k * pet is real pet in XAJ model
            max=torch.minimum(w0[w_mask] + prcp[w_mask], k[w_mask] * pet[w_mask]),
        )
        return et


def calculate_1layer_w_storage(um, lm, dm, w0, pe, r):
    """
    Update the soil moisture value.

    According to the runoff-generation equation 2.60 in the book "SHUIWENYUBAO", dW = dPE - dR

    Parameters
    ----------
    um
        average soil moisture storage capacity of the upper layer (mm)
    lm
        average soil moisture storage capacity of the lower layer (mm)
    dm
        average soil moisture storage capacity of the deep layer (mm)
    w0
        initial values of soil moisture
    pe
        net precipitation; it is able to be negative value in this function
    r
        runoff

    Returns
    -------
    torch.Tensor
        w -- soil moisture
    """
    xaj_device = pe.device
    tensor_zeros = torch.full_like(w0, 0.0, device=xaj_device)
    # water balance (equation 2.2 in Page 13, also shown in Page 23)
    w = w0 + pe - r
    return torch.clamp(w, min=tensor_zeros, max=(um + lm + dm) - PRECISION)


class Xaj4DplWithNnModule(nn.Module):
    """
    XAJ model for Differential Parameter learning with neural network as submodule
    """

    def __init__(
        self,
        kernel_size: int,
        warmup_length: int,
        nn_module=None,
        param_var_index=None,
        source_book="HF",
        source_type="sources",
        et_output=1,
        nn_hidden_size: Union[int, tuple, list] = None,
        nn_dropout=0.2,
        param_test_way=MODEL_PARAM_TEST_WAY["time_varying"],
    ):
        """
        Parameters
        ----------
        kernel_size
            the time length of unit hydrograph
        warmup_length
            the length of warmup periods;
            XAJ needs a warmup period to generate reasonable initial state values
        nn_module
            We initialize the module when we firstly initialize Xaj4DplWithNnModule.
            Then we will iterately call Xaj4DplWithNnModule module for warmup.
            Hence, in warmup period, we don't need to initialize it again
        param_var_index
            the index of parameters which will be time-varying
            NOTE: at the most, we support k, b, and c to be time-varying
        et_output
            we only support one-layer et now, because its water balance is not easy to handle with
        """
        if param_var_index is None:
            param_var_index = [0, 6]
        if nn_hidden_size is None:
            nn_hidden_size = [16, 8]
        super(Xaj4DplWithNnModule, self).__init__()
        self.params_names = MODEL_PARAM_DICT["xaj_mz"]["param_name"]
        param_range = MODEL_PARAM_DICT["xaj_mz"]["param_range"]
        self.k_scale = param_range["K"]
        self.b_scale = param_range["B"]
        self.im_sacle = param_range["IM"]
        self.um_scale = param_range["UM"]
        self.lm_scale = param_range["LM"]
        self.dm_scale = param_range["DM"]
        self.c_scale = param_range["C"]
        self.sm_scale = param_range["SM"]
        self.ex_scale = param_range["EX"]
        self.ki_scale = param_range["KI"]
        self.kg_scale = param_range["KG"]
        self.a_scale = param_range["A"]
        self.theta_scale = param_range["THETA"]
        self.ci_scale = param_range["CI"]
        self.cg_scale = param_range["CG"]
        self.kernel_size = kernel_size
        self.warmup_length = warmup_length
        # there are 2 input variables in XAJ: P and PET
        self.feature_size = 2
        if nn_module is None:
            # 7: k, um, lm, dm, c, prcp, p_and_e[:, 1] + 1/3: w0 or wu0, wl0, wd0
            self.evap_nn_module = NnModule4Hydro(
                7 + et_output, et_output, nn_hidden_size, nn_dropout
            )
        else:
            self.evap_nn_module = nn_module
        self.source_book = source_book
        self.source_type = source_type
        self.et_output = et_output
        self.param_var_index = param_var_index
        self.nn_hidden_size = nn_hidden_size
        self.nn_dropout = nn_dropout
        self.param_test_way = param_test_way

    def xaj_generation_with_new_module(
        self,
        p_and_e: Tensor,
        k,
        b,
        im,
        um,
        lm,
        dm,
        c,
        *args,
        # wu0: Tensor = None,
        # wl0: Tensor = None,
        # wd0: Tensor = None,
    ) -> tuple:
        # make sure physical variables' value ranges are correct
        prcp = torch.clamp(p_and_e[:, 0], min=0.0)
        pet = torch.clamp(p_and_e[:, 1], min=0.0)
        # wm
        wm = um + lm + dm
        if self.et_output != 1:
            raise NotImplementedError("We only support one-layer evaporation now")
        w0_ = args[0]
        if w0_ is None:
            w0_ = 0.6 * (um.detach() + lm.detach() + dm.detach())
        w0 = torch.clamp(w0_, max=wm - PRECISION)
        concat_input = torch.stack([k, um, lm, dm, c, w0, prcp, pet], dim=1)
        e = self.evap_nn_module(concat_input, w0, prcp, pet, k)
        # Calculate the runoff generated by net precipitation
        prcp_difference = prcp - e
        pe = torch.clamp(prcp_difference, min=0.0)
        r, rim = calculate_prcp_runoff(b, im, wm, w0, pe)
        if self.et_output == 1:
            w = calculate_1layer_w_storage(
                um,
                lm,
                dm,
                w0,
                prcp_difference,
                r,
            )
            return (r, rim, e, pe), (w,)
        else:
            raise ValueError("et_output should be 1")

    def forward(self, p_and_e, parameters_ts, return_state=False):
        """
        run XAJ model

        Parameters
        ----------
        p_and_e
            precipitation and potential evapotranspiration
        parameters_ts
            time series parameters of XAJ model;
            some parameters may be time-varying specified by param_var_index
        return_state
            if True, return state values, mainly for warmup periods

        Returns
        -------
        torch.Tensor
            streamflow got by XAJ
        """
        xaj_device = p_and_e.device
        if self.param_test_way == MODEL_PARAM_TEST_WAY["time_varying"]:
            parameters = parameters_ts[-1, :, :]
        else:
            # parameters_ts must be a 2-d tensor: (basin, param)
            parameters = parameters_ts
        # denormalize the parameters to general range
        # TODO: now the specific parameters are hard coded; 0 is k, 1 is b, 6 is c, same as in model_config.py
        if 0 not in self.param_var_index or self.param_var_index is None:
            ks = self.k_scale[0] + parameters[:, 0] * (
                self.k_scale[1] - self.k_scale[0]
            )
        else:
            ks = self.k_scale[0] + parameters_ts[:, :, 0] * (
                self.k_scale[1] - self.k_scale[0]
            )
        if 1 not in self.param_var_index or self.param_var_index is None:
            bs = self.b_scale[0] + parameters[:, 1] * (
                self.b_scale[1] - self.b_scale[0]
            )
        else:
            bs = self.b_scale[0] + parameters_ts[:, :, 1] * (
                self.b_scale[1] - self.b_scale[0]
            )
        im = self.im_sacle[0] + parameters[:, 2] * (self.im_sacle[1] - self.im_sacle[0])
        um = self.um_scale[0] + parameters[:, 3] * (self.um_scale[1] - self.um_scale[0])
        lm = self.lm_scale[0] + parameters[:, 4] * (self.lm_scale[1] - self.lm_scale[0])
        dm = self.dm_scale[0] + parameters[:, 5] * (self.dm_scale[1] - self.dm_scale[0])
        if 6 not in self.param_var_index or self.param_var_index is None:
            cs = self.c_scale[0] + parameters[:, 6] * (
                self.c_scale[1] - self.c_scale[0]
            )
        else:
            cs = self.c_scale[0] + parameters_ts[:, :, 6] * (
                self.c_scale[1] - self.c_scale[0]
            )
        sm = self.sm_scale[0] + parameters[:, 7] * (self.sm_scale[1] - self.sm_scale[0])
        ex = self.ex_scale[0] + parameters[:, 8] * (self.ex_scale[1] - self.ex_scale[0])
        ki_ = self.ki_scale[0] + parameters[:, 9] * (
            self.ki_scale[1] - self.ki_scale[0]
        )
        kg_ = self.kg_scale[0] + parameters[:, 10] * (
            self.kg_scale[1] - self.kg_scale[0]
        )
        # ki+kg should be smaller than 1; if not, we scale them
        ki = torch.where(
            ki_ + kg_ < 1.0,
            ki_,
            (1 - PRECISION) / (ki_ + kg_) * ki_,
        )
        kg = torch.where(
            ki_ + kg_ < 1.0,
            kg_,
            (1 - PRECISION) / (ki_ + kg_) * kg_,
        )
        a = self.a_scale[0] + parameters[:, 11] * (self.a_scale[1] - self.a_scale[0])
        theta = self.theta_scale[0] + parameters[:, 12] * (
            self.theta_scale[1] - self.theta_scale[0]
        )
        ci = self.ci_scale[0] + parameters[:, 13] * (
            self.ci_scale[1] - self.ci_scale[0]
        )
        cg = self.cg_scale[0] + parameters[:, 14] * (
            self.cg_scale[1] - self.cg_scale[0]
        )

        # initialize state values
        warmup_length = self.warmup_length
        if warmup_length > 0:
            # set no_grad for warmup periods
            with torch.no_grad():
                p_and_e_warmup = p_and_e[0:warmup_length, :, :]
                if self.param_test_way == MODEL_PARAM_TEST_WAY["time_varying"]:
                    parameters_ts_warmup = parameters_ts[0:warmup_length, :, :]
                else:
                    parameters_ts_warmup = parameters_ts
                cal_init_xaj4dpl = Xaj4DplWithNnModule(
                    kernel_size=self.kernel_size,
                    # warmup_length must be 0 here
                    warmup_length=0,
                    nn_module=self.evap_nn_module,
                    param_var_index=self.param_var_index,
                    source_book=self.source_book,
                    source_type=self.source_type,
                    et_output=self.et_output,
                    nn_hidden_size=self.nn_hidden_size,
                    nn_dropout=self.nn_dropout,
                    param_test_way=self.param_test_way,
                )
                if cal_init_xaj4dpl.warmup_length > 0:
                    raise RuntimeError("Please set init model's warmup length to 0!!!")
                _, _, *w0, s0, fr0, qi0, qg0 = cal_init_xaj4dpl(
                    p_and_e_warmup, parameters_ts_warmup, return_state=True
                )
        else:
            # use detach func to make wu0 no_grad as it is an initial value
            if self.et_output == 1:
                # () and , must be added, otherwise, w0 will be a tensor, not a tuple
                w0 = (0.5 * (um.detach() + lm.detach() + dm.detach()),)
            else:
                raise ValueError("et_output should be 1 or 3")
            s0 = 0.5 * (sm.detach())
            fr0 = torch.full(ci.size(), 0.1).to(xaj_device)
            qi0 = torch.full(ci.size(), 0.1).to(xaj_device)
            qg0 = torch.full(cg.size(), 0.1).to(xaj_device)

        inputs = p_and_e[warmup_length:, :, :]
        runoff_ims_ = torch.full(inputs.shape[:2], 0.0).to(xaj_device)
        rss_ = torch.full(inputs.shape[:2], 0.0).to(xaj_device)
        ris_ = torch.full(inputs.shape[:2], 0.0).to(xaj_device)
        rgs_ = torch.full(inputs.shape[:2], 0.0).to(xaj_device)
        es_ = torch.full(inputs.shape[:2], 0.0).to(xaj_device)
        for i in range(inputs.shape[0]):
            if 0 in self.param_var_index or self.param_var_index is None:
                k = ks[i]
            else:
                k = ks
            if 1 in self.param_var_index or self.param_var_index is None:
                b = bs[i]
            else:
                b = bs
            if 6 in self.param_var_index or self.param_var_index is None:
                c = cs[i]
            else:
                c = cs
            if i == 0:
                (r, rim, e, pe), w = self.xaj_generation_with_new_module(
                    inputs[i, :, :], k, b, im, um, lm, dm, c, *w0
                )
                if self.source_type == "sources":
                    (rs, ri, rg), (s, fr) = xaj_sources(
                        pe, r, sm, ex, ki, kg, s0, fr0, book=self.source_book
                    )
                elif self.source_type == "sources5mm":
                    (rs, ri, rg), (s, fr) = xaj_sources5mm(
                        pe, r, sm, ex, ki, kg, s0, fr0, book=self.source_book
                    )
                else:
                    raise NotImplementedError("No such divide-sources method")
            else:
                (r, rim, e, pe), w = self.xaj_generation_with_new_module(
                    inputs[i, :, :], k, b, im, um, lm, dm, c, *w
                )
                if self.source_type == "sources":
                    (rs, ri, rg), (s, fr) = xaj_sources(
                        pe, r, sm, ex, ki, kg, s, fr, book=self.source_book
                    )
                elif self.source_type == "sources5mm":
                    (rs, ri, rg), (s, fr) = xaj_sources5mm(
                        pe, r, sm, ex, ki, kg, s, fr, book=self.source_book
                    )
                else:
                    raise NotImplementedError("No such divide-sources method")
            # impevious part is pe * im
            runoff_ims_[i, :] = rim
            # so for non-imprvious part, the result should be corrected
            rss_[i, :] = rs * (1 - im)
            ris_[i, :] = ri * (1 - im)
            rgs_[i, :] = rg * (1 - im)
            es_[i, :] = e
        # seq, batch, feature
        runoff_im = torch.unsqueeze(runoff_ims_, dim=2)
        rss = torch.unsqueeze(rss_, dim=2)
        es = torch.unsqueeze(es_, dim=2)

        conv_uh = KernelConv(a, theta, self.kernel_size)
        qs_ = conv_uh(runoff_im + rss)

        qs = torch.full(inputs.shape[:2], 0.0).to(xaj_device)
        for i in range(inputs.shape[0]):
            if i == 0:
                qi = linear_reservoir(ris_[i], ci, qi0)
                qg = linear_reservoir(rgs_[i], cg, qg0)
            else:
                qi = linear_reservoir(ris_[i], ci, qi)
                qg = linear_reservoir(rgs_[i], cg, qg)
            qs[i, :] = qs_[i, :, 0] + qi + qg
        # seq, batch, feature
        q_sim = torch.unsqueeze(qs, dim=2)
        if return_state:
            return q_sim, es, *w, s, fr, qi, qg
        return q_sim, es


class DplLstmNnModuleXaj(nn.Module):
    def __init__(
        self,
        n_input_features,
        n_output_features,
        n_hidden_states,
        kernel_size,
        warmup_length,
        param_limit_func="clamp",
        param_test_way="final",
        param_var_index=None,
        source_book="HF",
        source_type="sources",
        nn_hidden_size=None,
        nn_dropout=0.2,
        et_output=3,
    ):
        """
        Differential Parameter learning model: LSTM -> Param -> XAJ

        The principle can be seen here: https://doi.org/10.1038/s41467-021-26107-z

        Parameters
        ----------
        n_input_features
            the number of input features of LSTM
        n_output_features
            the number of output features of LSTM, and it should be equal to the number of learning parameters in XAJ
        n_hidden_states
            the number of hidden features of LSTM
        kernel_size
            the time length of unit hydrograph
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
            but remember these ways are only for non-variable parameters
        param_var_index
            variable parameters' indices in all parameters
        """
        if param_var_index is None:
            param_var_index = [0, 1, 6]
        if nn_hidden_size is None:
            nn_hidden_size = [16, 8]
        super(DplLstmNnModuleXaj, self).__init__()
        self.dl_model = SimpleLSTM(n_input_features, n_output_features, n_hidden_states)
        self.pb_model = Xaj4DplWithNnModule(
            kernel_size,
            warmup_length,
            source_book=source_book,
            source_type=source_type,
            nn_hidden_size=nn_hidden_size,
            nn_dropout=nn_dropout,
            et_output=et_output,
            param_var_index=param_var_index,
            param_test_way=param_test_way,
        )
        self.param_func = param_limit_func
        self.param_test_way = param_test_way
        self.param_var_index = param_var_index

    def forward(self, x, z):
        """
        Differential parameter learning

        z (normalized input) -> lstm -> param -> + x (not normalized) -> xaj -> q
        Parameters will be denormalized in xaj model

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
        gen = self.dl_model(z)
        if torch.isnan(gen).any():
            raise ValueError("Error: NaN values detected. Check your data firstly!!!")
        # we set all params' values in [0, 1] and will scale them when forwarding
        if self.param_func == "sigmoid":
            params = F.sigmoid(gen)
        elif self.param_func == "clamp":
            params = torch.clamp(gen, min=0.0, max=1.0)
        else:
            raise NotImplementedError(
                "We don't provide this way to limit parameters' range!! Please choose sigmoid or clamp"
            )
        # just get one-period values, here we use the final period's values,
        # when the MODEL_PARAM_TEST_WAY is not time_varing, we use the last period's values.
        if self.param_test_way != MODEL_PARAM_TEST_WAY["time_varying"]:
            params = params[-1, :, :]
        # Please put p in the first location and pet in the second
        q, e = self.pb_model(x[:, :, : self.pb_model.feature_size], params)
        return torch.cat([q, e], dim=-1)
