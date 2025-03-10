import numpy as np
from typing import Union
import torch
from torch import nn
from torch import Tensor

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
        ep = kc * pet
        e0 = torch.min(ep, prcp + xp +xf)
        pe = torch.clamp(prcp - e0, min=0.0)
        # soil moisture
        x = xf
        xf = x - torch.clamp(ep - prcp, min=0.0)
        xp = xp - torch.clamp(ep - prcp - x, min=0.0)

        # update soil moisture
        t1 = k1 * torch.min(x2, w1 - xp)
        xp = xp + t1
        x2 = x2 - t1

        t2 = k2 * (xs * w1 - xp * w2) / (w1 + w2)  #

        xp = xp + t2
        xs = xs - t2

        xf = xf + torch.clamp(xp + pe - w1)  # update the first layer free water
        xp = torch.min(w1, xp + pe)
        # the infiltrated water
        f1 = a0 * xf


        et = 0
        rs = 0
        ri = 0
        rgs = 0
        rgd = 0
        return et, rs, ri, rgs, rgd, self.intervar[:, :6]
