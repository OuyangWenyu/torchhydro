import numpy as np
from typing import Union
import torch
from torch import nn
from torch import Tensor
from torchhydro.configs.model_config import MODEL_PARAM_DICT, MODEL_PARAM_TEST_WAY
from torchhydro.models.dpl4xaj_nn4et import NnModule4Hydro

PRECISION = 1e-5

def calculate_evap(kc, uztwm, lztwm, riva, auztw, alztw, uztw, uzfw, lztw, prcp, pet) -> tuple[np.array, np.array, np.array, np.array, np.array, np.array]:

    """
    The three-layers evaporation model is described in Page 76;
    The method is same with that in Page 22-23 in "Hydrologic Forecasting (4-th version)" written by Prof. Weimin Bao.
    This book's Chinese name is 《水文预报》

    Parameters
    ----------
    kc
        coefficient of potential evapotranspiration to reference crop evaporation generally
    uztwm
        tension water capacity in the upper layer
    lztwm
        tension water capacity in the lower layer
    riva
        ratio of river net, lakes and hydrophyte area to total area of the basin
    auztw
        the upper layer tension water accumulation on the alterable impervious area
    alztw
        the lower layer tension water accumulation on the alterable impervious area
    uztw
        tension water accumulation in the upper layer
    uzfw
        free water accumulation in the upper layer
    lztw
        tension water accumulation in the lower layer
    prcp
        basin mean precipitation
    pet
        potential evapotranspiration

    Returns
    -------
    tuple[np.array,np.array,np.array,np.array,np.array,np.array]
        ae1ae3/e1/e2/e3/e4 are evaporation from upper/lower/deeper layer, respectively
    """
    # evaporation of basin
    ep = kc * pet
    # evaporation of the alterable impervious area
    ae1 = min(auztw, ep*(auztw/uztwm))  #
    ae3 = (ep - ae1) * (alztw / (uztwm + lztwm))  #
    pav = max(0.0, prcp - (uztwm - (auztw - ae1)))
    adsur = pav * ((alztw - ae3) / lztwm)
    ars = max(0.0, ((pav - adsur) + (alztw -ae3)) - lztwm)
    auztw = min(uztwm, (auztw - ae1) + prcp)
    alztw = min(lztwm, (pav - adsur) + (alztw - ae3))
    # evaporation of the permeable area
    e1 = min(uztw, ep * (uztw / uztwm))  # upper layer
    e2 = min(uzfw, ep - e1)  # lower layer
    e3 = (ep - e1 - e2) * (lztw / (uztwm + lztwm))  # deeper layer
    lt = lztw - e3
    e4 = riva * ep  # river net, lakes and hydrophyte
    e0 = ae1 + ae3 + e1 + e2 + e3 + e4  # total evaporation

    return ae1, ae3, e1, e2, e3, e4

def c(uztw, uzfw, uztwm, uzfwm, pe):
    """
    Calculates the amount of runoff generated from rainfall after entering the underlying surface

    Same in "Hydrologic Forecasting (4-th version)"

    Parameters
    ----------
    uztw
        B exponent coefficient
    uzfw
        IMP imperiousness coefficient
    uztwm
        average soil moisture storage capacity
    uzfwm
        initial soil moisture
    pe
        net precipitation

    Returns
    -------
    torch.Tensor
        r -- runoff; r_im -- runoff of impervious part
    """
    ROIMP = pe * pctim
    wmm = wm * (1 + b)
    a = wmm * (1 - (1 - w0 / wm) ** (1 / (1 + b)))
    if any(torch.isnan(a)):
        raise ValueError(
            "Error: NaN values detected. Try set clamp function or check your data!!!"
        )
    r_cal = torch.where(
        pe > 0.0,
        torch.where(
            pe + a < wmm,
            # torch.clamp is used for gradient not to be NaN, see more in xaj_sources function
            pe - (wm - w0) + wm * (1 - torch.clamp(a + pe, max=wmm) / wmm) ** (1 + b),
            pe - (wm - w0),
        ),
        torch.full(pe.size(), 0.0).to(pe.device),
    )
    if any(torch.isnan(r_cal)):
        raise ValueError(
            "Error: NaN values detected. Try set clamp function or check your data!!!"
        )
    r = torch.clamp(r_cal, min=0.0)
    r_im_cal = pe * im
    r_im = torch.clamp(r_im_cal, min=0.0)
    return r, r_im

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

        self.kernel_size = kernel_size
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
