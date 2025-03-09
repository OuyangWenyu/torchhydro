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
        et = 0
        rs = 0
        ri = 0
        rgs = 0
        rgd = 0
        return et, rs, ri, rgs, rgd, self.intervar[:, :6]
