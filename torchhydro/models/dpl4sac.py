import numpy as np
from typing import Union
import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from torchhydro.configs.model_config import MODEL_PARAM_DICT, MODEL_PARAM_TEST_WAY
from torchhydro.models.simple_lstm import SimpleLSTM

PRECISION = 1e-5

# todo: ascertain the time step of the model

def calculate_w_storage( # todo:
    prcp,
    uztwm, lztwm,
    auztw, alztw, lztw,
    ae1, ae3, e3,
    um, lm, dm, wu0, wl0, wd0, eu, el, ed, pe, r
) -> tuple[np.array, np.array, np.array]:
    """
    Update the soil moisture values of the three layers.

    According to the equation (5-72, 5-75) in the book《水文预报》

    Parameters
    ----------
    um
        average soil moisture storage capacity of the upper layer
    lm
        average soil moisture storage capacity of the lower layer
    dm
        average soil moisture storage capacity of the deep layer
    wu0
        initial values of soil moisture in upper layer
    wl0
        initial values of soil moisture in lower layer
    wd0
        initial values of soil moisture in deep layer
    eu
        evaporation of the upper layer; it isn't used in this function
    el
        evaporation of the lower layer
    ed
        evaporation of the deep layer
    pe
        net precipitation; it is able to be negative value in this function
    r
        runoff

    Returns
    -------
    tuple[np.array,np.array,np.array]
        wu,wl,wd -- soil moisture in upper, lower and deep layer
    """
    # pe>0: the upper soil moisture was added firstly, then lower layer, and the final is deep layer
    # pe<=0: no additional water, just remove evapotranspiration,
    # but note the case: e >= p > 0
    # (1) if wu0 + p > e, then e = eu (2) else, wu must be zero

    pav = max(0.0, prcp - (uztwm - (auztw - ae1)))
    adsur = pav * ((alztw - ae3) / lztwm)
    ars = max(0.0, ((pav - adsur) + (alztw -ae3)) - lztwm)
    auztw = min(uztwm, (auztw - ae1) + prcp)
    alztw = min(lztwm, (pav - adsur) + (alztw - ae3))
    lt = lztw - e3

    return auztw, alztw,

def calculate_route(
    hydrodt, area, rivernumber,
    pctim, adimp, ci, cgs, cgp, ke, xe,
    qs0, qi0, qgs0, qgp0,
    rs, ri, rgs, rgp,
    q_sim_0):
    """
    calcualte the route, the simulated flow.
    Parameters
    ----------
    hydrodt
        the time step
    area
        basin area, km^2.
    rivernumber
        the river sections number
    rs
        surface runoff, mm.
    ri
        interflow runoff, mm.
    rgs
        speedy groundwater runoff, mm.
    rgp
        slow groundwater runoff, mm.
    q_sim_0
    Returns
    -------
    q_sim
        the simulated flow, Q(m^3/s).
    """
    u = (1 - pctim - adimp) * area * 1000  # todo: / hydrodt     u, the unit converting coefficient.
    parea = 1 - pctim - adimp

    # slope routing, use the linear reservoir method
    qs = rs * area * 1000.0  # todo: /hydrodt
    qi = ci * qi0 + (1 - ci) * ri * u
    qgs = cgs * qgs0 + (1 - cgs) * rgs * u
    qgp = cgp * qgp0 + (1 - cgp) * rgp * u
    q_sim = (qs + qi + qgs + qgp)

    # river routing, use the Muskingum routing method
    q_sim_0 = qs0 + qi0 + qgs0 + qgp0
    if rivernumber > 0:
        dt = hydrodt / 3600.0 / 2.0  # todo: hydrodt, the timestep of data
        xo = xe
        ko = ke / rivernumber
        c1 = ko * (1.0 - xo) + dt
        c2 = (-ko * xo + dt) / c1
        c3 = (ko * (1.0 - xo) - dt) / c1
        c1 = (ko * xo + dt) / c1
        i1 = q_sim_0
        i2 = q_sim
        for i in range(rivernumber):
            q_sim_0 = qq[i]  # todo:
            i2 = c1 * i1 + c2 * i2 + c3 * q_sim_0
            qq[i] = i2
            i1 = q_sim_0
    q_sim = i2
    return q_sim

class Sac4DplWithNnModule(nn.Module):
    """
    Sacramento model for differential parameter learning with neural network as submodule
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
        param_test_way=MODEL_PARAM_TEST_WAY["time_varying"],  # todo:
    ):
        """
        Initiate a Sacramento model instance
        Parameters
        ----------
        kernel_size
            the time length of unit hydrograph 单位线
        warmup_length
            the length of warmup periods; 预热期
            sac needs a warmup period to generate reasonable initial state values 需要预热，形成初始条件
        nn_module
            We initialize the module when we firstly initialize Sac4DplWithNnModule.
            Then we will iterately call Sac4DplWithNnModule module for warmup. 迭代调用模型单元
            Hence, in warmup period, we don't need to initialize it again.
        param_var_index
            the index of parameters which will be time-varying 随时间变化 时变
            NOTE: at the most, we support k, b, and c to be time-varying  至多支持k、b、c时变
        et_output  蒸发输出 只支持一层蒸发 #
            we only support one-layer et now, because its water balance is not easy to handle with 蒸发的水量平衡不容易处理？
        nn_hidden_size
            the hidden layer size of neural network
        nn_dropout
            the dropout rate of neural network  神经网络的暂退率，(0.3, 0.5)
        param_test_way
            the way to test the model parameters, time-varying. 模型参数测试方式，取时变式。
        """
        super(Sac4DplWithNnModule, self).__init__()
        self.name = "Sacramento"
        self.params_names = MODEL_PARAM_DICT["sac"]["param_name"]
        param_range = MODEL_PARAM_DICT["sac"]["param_range"]
        self.kc_scale = param_range["KC"]
        self.pctim_scale = param_range["PCTIM"]
        self.adimp_scale = param_range["ADIMP"]
        self.uztwm_scale = param_range["UZTWM"]
        self.uzfwm_scale = param_range["UZFWM"]
        self.lztwm_scale = param_range["LZTWM"]
        self.lzfsm_scale = param_range["LZFSM"]
        self.lzfpm_scale = param_range["LZFPM"]
        self.rserv_scale = param_range["RSERV"]
        self.pfree_scale = param_range["PFREE"]
        self.riva_scale = param_range["RIVA"]
        self.zperc_scale = param_range["ZPERC"]
        self.rexp_scale = param_range["REXP"]
        self.uzk_scale = param_range["UZK"]
        self.lzsk_scale = param_range["LZSK"]
        self.lzpk_scale = param_range["LZPK"]
        self.ci_scale = param_range["CI"]
        self.cgs_scale = param_range["CGS"]
        self.cgp_scale = param_range["CGP"]
        self.ke_scale = param_range["KE"]
        self.xe_scale = param_range["XE"]

        self.kernel_size = kernel_size  # the unit-line kernel size #
        self.warmup_length = warmup_length
        # there are 2 input variables in Sac: P and PET
        self.feature_size = 2
        self.source_book = source_book
        self.source_type = source_type
        self.et_output = et_output
        self.param_var_index = param_var_index
        self.nn_hidden_size = nn_hidden_size
        self.nn_dropout = nn_dropout
        self.param_test_way = param_test_way

    def forward(self, p_and_e, parameters, return_state=False):
        """
        run sac model
        forward transmission,

        Parameters
        ----------
        p_and_e
            precipitation and potential evapotranspiration, mm/d.
        parameters
        --model parameters--
        kc
            coefficient of potential evapotranspiration to reference crop evaporation generally
        pctim
            ratio of the permanent impervious area to total area of the basin
        uztwm
            tension water capacity in the upper layer, mm.
        lztwm
            tension water capacity in the lower layer, mm.
        riva
            ratio of river net, lakes and hydrophyte area to total area of the basin
        --middle variables--
        auztw
            the upper layer tension water accumulation on the alterable impervious area, mm.
        alztw
            the lower layer tension water accumulation on the alterable impervious area, mm.
        uztw
            tension water accumulation in the upper layer, mm.
        uzfw
            free water accumulation in the upper layer, mm.
        lztw
            tension water accumulation in the lower layer, mm.
        --hydrodata--
        prcp
            basin mean precipitation, mm.
        pet
            potential evapotranspiration, mm.
        return_state
            if True, return state values, mainly for warmup periods

        Returns
        -------
        torch.Tensor
            streamflow got by XAJ
        q_sim
            the simulate flow, Q(m^3/s).
        es
            the simulate evaporation, E(mm/d)
        """
        sac_device = p_and_e.device
        prcp = torch.clamp(p_and_e[:, 0], min=0.0)
        pet = torch.clamp(p_and_e[:, 1], min=0.0)
        # parameters
        kc = self.kc_scale[0] + parameters[:, 0] * (self.kc_scale[1] - self.kc_scale[0])
        pctim = self.pctiscale[1] + parameters[:, 1] * (self.pctiscale[1] - self.pctiscale[0])
        adimp = self.adimscale[1] + parameters[:, 1] * (self.adimscale[1] - self.adimscale[0])
        uztwm = self.uztwscale[2] + parameters[:, 2] * (self.uztwscale[2] - self.uztwscale[0])
        uzfwm = self.uzfwscale[3] + parameters[:, 3] * (self.uzfwscale[3] - self.uzfwscale[0])
        lztwm = self.lztwscale[4] + parameters[:, 4] * (self.lztwscale[4] - self.lztwscale[0])
        lzfsm = self.lzfsscale[5] + parameters[:, 5] * (self.lzfsscale[5] - self.lzfsscale[0])
        lzfpm = self.lzfpscale[6] + parameters[:, 6] * (self.lzfpscale[6] - self.lzfpscale[0])
        rserv = self.rserv_scale[7] + parameters[:, 7] * (self.rserv_scale[7] - self.rserv_scale[0])
        pfree = self.pfree_scale[8] + parameters[:, 8] * (self.pfree_scale[8] - self.pfree_scale[0])
        riva = self.riva_scale[9] + parameters[:, 9] * (self.riva_scale[9] - self.riva_scale[0])
        zperc = self.zperc_scale[10] + parameters[:, 10] * (self.zperc_scale[10] - self.zperc_scale[0])
        rexp = self.rexscale[11] + parameters[:, 11] * (self.rexscale[11] - self.rexscale[0])
        uzk = self.uzk_scale[12] + parameters[:, 12] * (self.uzk_scale[12] - self.uzk_scale[0])
        lzsk = self.lzsk_scale[13] + parameters[:, 13] * (self.lzsk_scale[13] - self.lzsk_scale[0])
        lzpk = self.lzpk_scale[14] + parameters[:, 14] * (self.lzpk_scale[14] - self.lzpk_scale[0])
        # middle variable  # todo: use warmup_length to handling the initial condition
        auztw = 0
        alztw = 0
        uztw = 0
        uzfw = 0
        lztw = 0
        lzfs = 0
        lzfp = 0

        # evaporation
        # average evaporation of basin
        ep = kc * pet
        # runoff of the permanent impervious area
        roimp = prcp * pctim
        # evaporation of the permanent impervious area
        ae2 = pctim * ep
        # evaporation of the alterable impervious area
        ae1 = torch.min((auztw, ep * (auztw / uztwm)))  # upper layer
        ae3 = (ep - ae1) * (alztw / (uztwm + lztwm)) # lower layer
        pav = torch.max((0.0, prcp - (uztwm - (auztw - ae1))))
        adsur = pav * ((alztw - ae3) / lztwm)
        ars = torch.max((0.0, ((pav - adsur) + (alztw -ae3)) - lztwm))
        auztw = torch.min((uztwm, (auztw - ae1) + prcp))
        alztw = torch.min((lztwm, (pav - adsur) + (alztw - ae3)))
        # evaporation of the permeable area
        e1 = torch.min((uztw, ep * (uztw / uztwm)))  # upper layer
        e2 = torch.min((uzfw, ep - e1))  # lower layer
        e3 = (ep - e1 - e2) * (lztw / (uztwm + lztwm))  # deeper layer
        lt = lztw - e3
        e4 = riva * ep  # river net, lakes and hydrophyte
        # total evaporation
        e0 = ae2 + ae1 + ae3 + e1 + e2 + e3 + e4  # the total evaporation

        # generate runoff
        # runoff of the alterable impervious area
        adsur = adsur * adimp
        ars = ars * adimp  # todo:
        # runoff of the permeable area
        parea = 1 - pctim - adimp
        rs = torch.max((prcp + (uztw + uzfw - e1 - e2) - (uztwm + uzfwm), 0.0)) * parea
        ut = torch.min((uztwm, uztw - e1 + prcp))
        uf = torch.min((uzfwm, prcp + (uztw + uzfw - e1 - e2) - ut))
        ri = uf * uzk  # interflow
        uf = uf - ri
        # the infiltrated water
        pbase = lzfsm * lzsk + lzfpm * lzpk
        defr = 1 - (lzfs + lzfp + lt) / (lzfsm + lzfpm + lztwm)
        perc = pbase * (1 + zperc * pow(defr, rexp)) * uf / uzfwm
        rate = torch.min((perc, (lzfsm + lzfpm + lztwm) - (lzfs + lzfp + lt)))
        # assign the infiltrate water
        fx = torch.min((lzfsm + lzfpm - (lzfs + lzfp), torch.max((rate - (lztwm - lt), rate * pfree))))
        perct = rate - fx
        coef = (lzfpm / (lzfsm + lzfpm)) * (2 * (1 - lzfp / lzfpm) / ((1 - lzfp / lzfpm) + (1 - lzfsm / lzfsm)))
        coef = torch.min((coef, 1))
        percp = torch.min((lzfpm - lzfp, torch.max((fx - (lzfsm - lzfs), coef * fx))))
        percs = fx - percp
        # update the soil moisture accumulation
        lt = lt + perct
        ls = lzfs + percs
        lp = lzfp + percp
        # generate groundwater runoff
        rgs = ls * lzsk
        rgp = lp * lzpk
        ls = ls - rgs  # update
        lp = lp - rgp
        # water balance check
        if ut / uztwm < uf / uzfwm:
            uztw = uztwm * (ut + uf) / (uztwm + uzfwm)
            uzfw = uzfwm * (ut + uf) / (uztwm + uzfwm)
        else:
            uztw = ut
            uzfw = uf
        saved = rserv * (lzfsm + lzfpm)
        ratio = (ls + lp - saved + lt) / (lzfsm + lzfpm - saved + lztwm)
        ratio = torch.max((ratio, 0))
        if lt / lztwm < ratio:
            lztw = lztwm * ratio
            del_ = lztw - lt
            lzfs = torch.max((0.0, ls - del_))
            lzfp = lp - torch.max((0.0, del_ - ls))
        else:
            lztw = lt
            lzfs = ls
            lzfp = lp

        rs = roimp + (adsur + ars) + rs
        ri = ri
        rg = rgs + rgp

        q_sim = calculate_route()
        return q_sim


class DplLstmNnModuleSac(nn.Module):
    """
    Sacramento differential parameter learning - Long short-term memory neural network model
    """
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
        Differential Parameter learning model: LSTM -> Param -> SAC

        The principle can be seen here: https://doi.org/10.1038/s41467-021-26107-z

        Parameters
        ----------
        n_input_features
            the number of input features of LSTM
        n_output_features
            the number of output features of LSTM, and it should be equal to the number of learning parameters in SAC
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
        super(DplLstmNnModuleSac, self).__init__()
        self.dl_model = SimpleLSTM(n_input_features, n_output_features, n_hidden_states)
        self.pb_model = Sac4DplWithNnModule(
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
        Differential parameter learning 微分参数学习

        z (normalized input) -> lstm -> param -> + x (not normalized) -> sac -> q
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
