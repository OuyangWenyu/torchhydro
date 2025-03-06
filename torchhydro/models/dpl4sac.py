import numpy as np
from typing import Union
import torch
from torch import nn
from torch import Tensor
from torchhydro.configs.model_config import MODEL_PARAM_DICT, MODEL_PARAM_TEST_WAY
from torchhydro.models.dpl4xaj_nn4et import NnModule4Hydro

PRECISION = 1e-5

# todo: ascertain the time step of the model

def calculate_evap(
    kc, pctim, uztwm, lztwm, riva,
    auztw, alztw, uztw, uzfw, lztw,
    prcp, pet
    ) -> tuple[np.array, np.array, np.array, np.array, np.array, np.array, np.array]:

    """
    The three-layers evaporation model is described in Page 169;
    The method is same with that in Page 169-170 in "Hydrologic Forecasting (4-th version)" written by Prof. Weimin Bao.
    This book's Chinese name is 《水文预报》

    Parameters
    ----------
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

    Returns
    -------
    tuple[np.array,np.array,np.array,np.array,np.array,np.array]
        ae2/ae1/ae3/e1/e2/e3/e4 are evaporation from upper/lower/deeper layer, respectively, mm.
    """
    # average evaporation of basin
    ep = torch.clamp(kc * pet, min = 0)
    # evaporation of the permanent impervious area
    ae2 = torch.clamp(pctim * ep, min = 0)
    # evaporation of the alterable impervious area
    ae1 = torch.min((auztw, ep*(auztw/uztwm)))  # upper layer
    ae3 = torch.clamp((ep - ae1) * (alztw / (uztwm + lztwm)), min = 0)  # lower layer
    # pav = max(0.0, prcp - (uztwm - (auztw - ae1)))
    # adsur = pav * ((alztw - ae3) / lztwm)
    # ars = max(0.0, ((pav - adsur) + (alztw -ae3)) - lztwm)
    # auztw = min(uztwm, (auztw - ae1) + prcp)
    # alztw = min(lztwm, (pav - adsur) + (alztw - ae3))
    # evaporation of the permeable area
    e1 = torch.min((uztw, ep * (uztw / uztwm)))  # upper layer
    e2 = torch.min((uzfw, ep - e1))  # lower layer
    e3 = torch.clamp((ep - e1 - e2) * (lztw / (uztwm + lztwm)), min = 0)  # deeper layer
    # lt = lztw - e3
    e4 = torch.clamp(riva * ep, min = 0)  # river net, lakes and hydrophyte
    # total evaporation
    e0 = torch.clamp(ae2 + ae1 + ae3 + e1 + e2 + e3 + e4, min = 0)  # the total evaporation

    return ae2, ae1, ae3, e1, e2, e3, e4, e0

def calculate_prcp_runoff(pctim, uztwm, uzfwm, lztwm, lzfsm, lzfpm, uzk, lzsk, lzpk, zperc, rexp, pfree, rserv, adimp,
                          uztw, uzfw, lztw, lzfs, lzfp,
                          pp, pe,
                          pav, alztw,
                          ae3, e1, e2, e3, ):
    """
    Calculates the amount of runoff generated from rainfall after entering the underlying surface

    Same in "Hydrologic Forecasting (4-th version)"

    Parameters # todo:
    ----------
    uztw
        B exponent coefficient
    uzfw
        IMP imperiousness coefficient
    uztwm
        average soil moisture storage capacity
    uzfwm
        initial soil moisture
    "uzk",  # daily outflow coefficient of the upper layer free water 上土层自由水日出流系数
    pe
        net precipitation

    Returns
    -------
    torch.Tensor
        r -- runoff; r_im -- runoff of impervious part
    """
    # generate runoff


    # runoff of the permanent impervious area
    roimp = pp * pctim
    # runoff of the alterable impervious area
    adsur = pav * ((alztw - ae3) / lztwm)
    ars = max(0.0, ((pav - adsur) + (alztw -ae3)) - lztwm) * adimp
    adsur = adsur * adimp  # todo:
    # runoff of the permeable area
    parea = 1 - pctim - adimp
    rs = max(0.0, pp + (uztw + uzfw - e1 - e2) - (uztwm + uzfwm)) * parea
    ut = min(uztwm, uztw - e1 + pp)
    uf = min(uzfwm, pp + (uztw + uzfw - e1 - e2) - ut)

    ri = uf * uzk  # interflow
    uf = uf - ri

    # the infiltrated water
    lt = lztw - e3  #
    pbase = lzfsm * lzsk + lzfpm * lzpk
    defr = 1 - (lzfs + lzfp + lt) / (lzfsm + lzfpm + lztwm)
    perc = pbase * ( 1 + zperc * pow(defr, rexp)) * uf / uzfwm
    rate = min(perc, (lzfsm + lzfpm + lztwm) - (lzfs + lzfp + lt))
    # assign the infiltrate water
    fx = min(lzfsm + lzfpm - (lzfs + lzfp), max(rate - (lztwm - lt), rate * pfree))
    perct = rate - fx
    coef = (lzfpm / (lzfsm + lzfpm)) * (2 * (1 - lzfp / lzfpm) / ((1 - lzfp / lzfpm) + (1 - lzfsm / lzfsm)))
    if coef > 1:
        coef = 1
    percp = min(lzfpm - lzfp, max(fx - (lzfsm - lzfs), coef * fx))
    percs = fx - percp

    # update the soil moisture accumulation
    lt = lt + perct
    ls = lzfs + percs
    lp = lzfp + percp

    # groundwater
    rgs = ls * lzsk  #
    rgp = lp * lzpk  #

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
    if ratio < 0:
        ratio = 0
    if lt / lztwm < ratio:
        lztw = lztwm * ratio
        del_ = lztw - lt
        lzfs = max(0.0, ls - del_)
        lzfp = lp - max(0.0, del_ - ls)
    else:
        lztw = lt
        lzfs = ls
        lzfp = lp

    rs = roimp + (adsur + ars) + rs
    rg = rgs + rgp

    return rs, ri, rg

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




def calculate_1layer_w_storage(uztwm, uzfwm, lztwm, lzfsm, lzfpm, w0, pe, rs, ri, rgs, rgp):
    """
    Update the soil moisture value.

    According to the runoff-generation equation 5.2.2 in the book "SHUIWENYUBAO", dW = dPE - dR

    Parameters
    ----------
    uztwm
        tension soil moisture storage capacity of the upper layer (mm)
    uzfwm
        free soil moisture storage capacity of the upper layer (mm)
    lztwm
        tension soil moisture storage capacity of the lower layer (mm)
    lzfsm
        speedy free soil moisture storage capacity of the lower layer (mm)
    lzfpm
        slow free soil moisture storage capacity of the lower layer (mm)
    pe
        net precipitation (mm), it is able to be negative value in this function.
    rs
        runoff of
    ri
        runoff of interflow
    rgs
        runoff of speedy groundwater
    rgp
        runoff of slow groundwater
    Returns
    -------
    torch.Tensor
        w -- soil moisture
    """
    sac_device = pe.device  #
    tensor_zeros = torch.full_like(w0, 0.0, device=sac_device)
    # water balance
    w = w0 + pe - rs  # todo:
    return torch.clamp(w, min=tensor_zeros, max=(uztwm + uzfwm + lztwm + lzfsm + lzfpm) - PRECISION)   # minus a minimum #


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
        param_test_way=MODEL_PARAM_TEST_WAY["time_varying"],
    ):
        """
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
        if param_var_index is None:
            param_var_index = [0, 6]
        if nn_hidden_size is None:
            nn_hidden_size = [16, 8]
        super(Sac4DplWithNnModule, self).__init__()
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
        if nn_module is None: # todo:
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

    def sac_generation_with_new_module(
        self,
        p_and_e: Tensor,   # input, precipitation and evaporation
        kc,
        pctim,
        adimp,
        uztwm,
        uzfwm,
        lztwm,
        lzfpm,
        rserv,
        pfree,
        *args,
    ) -> tuple:
        """
        产生新的模块？ 蒸散发和产流
        Parameters
        ----------
        p_and_e
            input, precipitation and evaporation.
        kc
        pctim
        adimp
        uztwm
        uzfwm
        lztwm
        lzfpm
        rserv
        pfree

        Returns
        -------

        """
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
        r, rim = calculate_prcp_runoff(b, im, wm, w0, pe)  # todo:
        if self.et_output == 1:
            w = calculate_1layer_w_storage(   # todo:
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
        forward transmission,
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
        q_sim
            the simulate flow, Q(m^3/s).
        es
            the simulate evaporation, E(mm/d)
        """
        xaj_device = p_and_e.device
        if self.param_test_way == MODEL_PARAM_TEST_WAY["time_varying"]:
            parameters = parameters_ts[-1, :, :]
        else:
            # parameters_ts must be a 2-d tensor: (basin, param)
            parameters = parameters_ts
        # denormalize the parameters to general range

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
        im = self.im_scale[0] + parameters[:, 2] * (self.im_scale[1] - self.im_scale[0])
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
