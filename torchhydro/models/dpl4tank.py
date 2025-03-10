import numpy as np
from typing import Union
import torch
from torch import nn
from torch import Tensor
from torchhydro.configs.model_config import MODEL_PARAM_DICT, MODEL_PARAM_TEST_WAY

class SingleStepTank(nn.Module):
    """
    single step Tank model
    """
    def __init__(
        self,
        device: Union[str, torch.device],
        hydrodt: int = 1,
        # area: float = None,
        para: Tensor = None,
        intervar: Tensor = None,
    ):
        """
        Initial a single-step Tank model

        The model is same with that in chapter 5 in "Hydrological Forecasting (4-th version)" written by Prof. Weimin Bao.
        This book's Chinese name is 《水文预报》
        Same in "Hydrological Forecasting (4-th version)"

        Parameters
        -----------
        device: Union[str, torch.device]
            cpu or gpu device
        hydrodt:
            the time step of hydrodata, default to one day.
        para: Tensor
            parameters of tank model, 17.
        intervar: Tensor
            the inter variables in model, 8.
            generate runoff
            xf, the upper layer tension water accumulation on the alterable impervious area, mm.
            xp, the lower layer tension water accumulation on the alterable impervious area, mm.
            x2, tension water accumulation in the upper layer, mm.
            xs, free water accumulation in the upper layer, mm.
            x3, tension water accumulation in the lower layer, mm.
            x4, speed free water accumulation in the lower layer, mm.
            routing
            x5, the flow of surface at the start of timestep.
            qs0, the flow of interflow at the start of timestep.
        """
        super(SingleStepTank, self).__init__()
        self.name = 'SingleStepTank'
        self.device = device
        # self.area = area
        self.hydrodt = hydrodt
        self.para = para
        self.intervar = intervar


    def cal_runoff(
        self,
        prcp: Tensor = None,
        pet: Tensor = None,
    ):
        """

        Parameters
        ----------
        prcp
        pet

        Returns
        -------

        """
        # assign values to the parameters
        kc = self.para[:, 0]
        w1 = self.para[:, 1]
        w2 = self.para[:, 2]
        k1 = self.para[:, 3]
        k2 = self.para[:, 4]
        a0 = self.para[:, 5]
        b0 = self.para[:, 6]
        c0 = self.para[:, 7]
        h1 = self.para[:, 8]
        h2 = self.para[:, 9]
        a1 = self.para[:, 10]
        a2 = self.para[:, 11]
        h3 = self.para[:, 12]
        b1 = self.para[:, 13]
        h4 = self.para[:, 14]
        c1 = self.para[:, 15]
        d1 = self.para[:, 16]
        # middle variables, at the start of timestep.
        xf = self.intervar[:, 0]
        xp = self.intervar[:, 1]
        x2 = self.intervar[:, 2]
        xs = self.intervar[:, 3]
        x3 = self.intervar[:, 4]
        x4 = self.intervar[:, 5]

        # evaporation
        ep = kc * pet  # basin evaporation capacity
        e0 = torch.min(ep, prcp + xp + xf)  # the actual evaporation    only one layer evaporation
        pe = torch.clamp(prcp - e0, min=0.0)  # net precipitation
        # soil moisture
        x = xf
        xf = x - torch.clamp(ep - prcp, min=0.0)  # update the first layer remain free water
        xp = xp - torch.clamp(ep - prcp - x, min=0.0)  # update the first layer remain tension water
        # update soil moisture
        t1 = k1 * torch.min(x2, w1 - xp)
        xp = xp + t1  # update the first layer tension water
        x2 = x2 - t1  # update the second layer free water

        t2 = k2 * (xs * w1 - xp * w2) / (w1 + w2)  # if t2>0,

        xp = xp + t2  # update the first layer tension water
        xs = xs - t2  # update the second layer tension water

        xf = xf + torch.clamp(xp + pe - w1)  # update the first layer free water
        xp = torch.min(w1, xp + pe)  # update the first layer tension water    the next timestep

        # the infiltrated water
        f1 = a0 * xf

        # surface runoff    the first layer generate the surface runoff
        if xf > h2:
            rs = (xf - h1) * a1 + (xf - h2) * a2
        elif xf > h1:
            rs = (xf - h1) * a1
        else:
            rs = 0.0
        # update soil moisture
        x2 = x2 + f1  # update the second layer free water
        xf = xf - (rs + f1)  # the first layer free water
        f2 = b0 * x2  # infiltrated water of the second layer

        # interflow    the second layer generate interflow
        if x2 > h3:
            ri = (x2 - h3) * b1
        else:
            ri = 0
        # update soil moisture
        x2 = x2 - (f2 + ri)  # update the second layer free water
        x3 = x3 + f2  # shallow groundwater accumulation
        f3 = c0 * x3  # shallow infiltrated water of  groundwater

        et = 0
        rs = 0
        ri = 0
        rgs = 0
        rgd = 0
        return et, rs, ri, rgs, rgd, self.intervar[:, :6]



class Tank4Dpl(nn.Module):
    """
    Tank model for differential Parameter learning module
    """
    def __init__(
        self,
        warmup_length: int,
        source_book="HF",
    ):
        """
        Initiate a Tank model instance.
        Parameters
        ----------
        warmup_length: int
            the length of warmup periods 预热期
            tank needs a warmup period to generate reasonable initial state values 需要预热，形成初始条件
        source_book
            "Hydrological Forecasting (4-th version)" written by Prof. Weimin Bao.
        """
        super(Tank4Dpl, self).__init__()
        self.name = "Sacramento"
        self.params_names = MODEL_PARAM_DICT["tank"]["param_name"]
        param_range = MODEL_PARAM_DICT["tank"]["param_range"]
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

        self.warmup_length = warmup_length
        self.feature_size = 2       # there are 2 input variables in Sac, P and PET. P and Pet are two feature in nn model.
        self.source_book = source_book
