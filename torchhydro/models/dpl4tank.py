import numpy as np
from typing import Union
import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from torchhydro.configs.model_config import MODEL_PARAM_DICT, MODEL_PARAM_TEST_WAY
from torchhydro.models.simple_lstm import SimpleLSTM
from torchhydro.models.ann import SimpleAnn

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
        prcp: Tensor
            precipitation, mm/d.
        pet: Tensor
            evaporation, mm/d.
        Returns
        -------
        et,
            the total evaporation, mm.
        rs, ri, rgs, rgd,
            the runoff of various water source, mm.
        self.intervar[:, :5]
            the inter variables, 6.
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
        et = torch.min(ep, prcp + xp + xf)  # the actual evaporation    only one layer evaporation
        pe = torch.clamp(prcp - et, min=0.0)  # net precipitation
        # soil moisture
        x = xf
        xf_ = x - torch.clamp(ep - prcp, min=0.0)  # update the first layer remain free water
        xp_ = xp - torch.clamp(ep - prcp - x, min=0.0)  # update the first layer remain tension water
        # update soil moisture
        t1 = k1 * torch.min(x2, w1 - xp_)
        xp = xp_ + t1  # update the first layer tension water
        x2_ = x2 - t1  # update the second layer free water

        t2 = k2 * (xs * w1 - xp * w2) / (w1 + w2)  # if t2>0,

        xp = xp + t2  # update the first layer tension water
        xs = xs - t2  # update the second layer tension water

        xf = xf_ + torch.clamp(xp + pe - w1)  # update the first layer free water
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

        # shallow groundwater runoff
        if x3 > h4:
            rgs = (x3 - h4) * c1
        else:
            rgs = 0
        x3 = x3 - (f3 + rgs)  # update the shallow groundwater

        # deep groundwater    x4 generate the deep layer groundwater runoff
        x4 = x4 + f3
        rgd = d1 * x4
        x4 = x4 - rgd

        # middle variables, at the end of timestep.
        self.intervar[:, 0] = xf
        self.intervar[:, 1] = xp
        self.intervar[:, 2] = x2
        self.intervar[:, 3] = xs
        self.intervar[:, 4] = x3
        self.intervar[:, 5] = x4

        return et, rs, ri, rgs, rgd, self.intervar[:, :5]

    def routing(
        self,
        rs: Tensor = None,
        ri: Tensor = None,
        rgs: Tensor = None,
        rgd: Tensor = None,
    ):
        """
        single step routing

        Parameters
        ----------
        rs: Tensor
            surface runoff, mm.
        ri: Tensor
            runoff of interflow, mm.
        rgs: Tensor
            runoff of speed groundwater, mm.
        rgd: Tensor
            runoff of slow groundwater, mm.
        Returns
        -------
        q_sim: Tensor
            the outflow at the end of timestep, m^3/s.
        self.intervar[:, 6:]
            the inter variables of routing part, m^3/s, 2.
        """
        # parameters
        e1 = self.para[:, 17]
        e2 = self.para[:, 18]
        h = self.para[:, 19]
        # middle variables, at the start of timestep.
        x5 = self.intervar[:, 6]
        qs0 = self.intervar[:, 7]
        # unit conversion
        u = self.hydrodt / 1000.0 / self.area  # todo: basin area
        qs = qs0
        if x5 >= h:
            k0 = 1 / (e1 + e2)
            c0 = e2 * h / (e1 + e2)
        else:
            k0 = 1 / e1
            c0 = 0
        k1 = 1 / e1
        c1 = 0
        dt = 0.5 * u
        i = rs + ri + rgs + rgd
        q1 = (k0 - dt) / (k1 + dt) * qs0 + (i - c1 + c0) / (k1 + dt)
        x5 = k1 * q1 + c1
        k1 = torch.where(x5 > h, 1 / (e1 + e2), k1)
        c1 = torch.where(x5 > h, e2 * h / (e1 + e2), c1)
        q1 = (k0 - dt) / (k1 + dt) * qs0 + (i - c1 + c0) / (k1 + dt)  # update
        x5 = k1 * q1 + c1
        qs0 = q1  # todo: q_sim

        # middle variables, at the start of timestep.
        self.intervar[:, 6] = x5
        self.intervar[:, 7] = qs0

        return q1, self.intervar[:,6:]


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
        self.w1_scale = param_range["W1"]
        self.w2_scale = param_range["W2"]
        self.k1_scale = param_range["K1"]
        self.k2_scale = param_range["K2"]
        self.a0_scale = param_range["a0"]
        self.b0_scale = param_range["b0"]
        self.c0_scale = param_range["c0"]
        self.h1_scale = param_range["h1"]
        self.h2_scale = param_range["h2"]
        self.a1_scale = param_range["a1"]
        self.a2_scale = param_range["a2"]
        self.h3_scale = param_range["h3"]
        self.b1_scale = param_range["b1"]
        self.h4_scale = param_range["h4"]
        self.c1_scale = param_range["c1"]
        self.d1_scale = param_range["d1"]
        self.e1_scale = param_range["e1"]
        self.e2_scale = param_range["e2"]
        self.h_scale = param_range["h"]

        self.warmup_length = warmup_length
        self.feature_size = 2       # there are 2 input variables in Sac, P and PET. P and Pet are two feature in nn model.
        self.source_book = source_book

    def forward(
        self,
        p_and_e: Tensor,
        parameters: Tensor,
        return_state: bool = False,
    ):
        """
        tank model
        forward transmission

        Parameters
        ----------
        p_and_e: Tensor
            time|basin|p_and_e
        prcp
            basin mean precipitation, mm/d.
        pet
            potential evapotranspiration, mm/d.
        parameters: Tensor
            model parameters, 21.
        return_state: bool
            whether to return model state or not.

        Returns
        -------
        q_sim : torch.Tensor
        the simulated flow, Q(m^3/s).
        e_sim : torch.Tensor
            the simulated evaporation, E(mm/d).
        """
        tank_device = p_and_e.device
        n_basin, n_para = parameters.size()

        intervar = torch.full((n_basin,11),0.0).detach()  # basin|inter_variables
        rsnpb = 1  # river sections number per basin
        rivernumber = np.full(n_basin, 1)  # set only one river section.   basin|river_section
        mq = torch.full((n_basin,rsnpb),0.0).detach()  # Muskingum routing space   basin|rivernumber   note: the column number of mp must equle to the column number of rivernumber   todo: ke, river section number

        if self.warmup_length > 0:  # if warmup_length>0, use warmup to calculate initial state.
            # set no_grad for warmup periods
            with torch.no_grad():
                p_and_e_warmup = p_and_e[0:self.warmup_length, :, :]  # time|basin|p_and_e
                cal_init_tank4dpl = Tank4Dpl(
                    # warmup_length must be 0 here
                    warmup_length=0,
                )
                if cal_init_tank4dpl.warmup_length > 0:
                    raise RuntimeError("Please set init model's warmup length to 0!!!")
                _, _, intervar = cal_init_tank4dpl(
                    p_and_e_warmup, parameters, return_state=True
                )
        else:  # if no, set a small value directly.
            intervar = torch.full((n_basin, 11), 0.1).detach()
            mq = torch.full((n_basin, rsnpb), 0.01).detach()

        # parameters
        para = torch.full((n_basin, n_para), 0.0)
        para[:, 0] = self.kc_scale[0] + parameters[:, 0] * (self.kc_scale[1] - self.kc_scale[0])    # parameters[:, 0]是个二维张量， 流域|参数  kc是个一维张量，不同流域的参数。  basin first
        para[:, 1] = self.w1_scale[0] + parameters[:, 1] * (self.w1_scale[1] - self.w1_scale[0])
        para[:, 2] = self.w2_scale[0] + parameters[:, 2] * (self.w2_scale[1] - self.w2_scale[0])
        para[:, 3] = self.k1_scale[0] + parameters[:, 3] * (self.k1_scale[1] - self.k1_scale[0])
        para[:, 4] = self.k2_scale[0] + parameters[:, 4] * (self.k2_scale[1] - self.k2_scale[0])
        para[:, 5] = self.a0_scale[0] + parameters[:, 5] * (self.a0_scale[1] - self.a0_scale[0])
        para[:, 6] = self.b0_scale[0] + parameters[:, 6] * (self.b0_scale[1] - self.b0_scale[0])
        para[:, 7] = self.c0_scale[0] + parameters[:, 7] * (self.c0_scale[1] - self.c0_scale[0])
        para[:, 8] = self.h1_scale[0] + parameters[:, 8] * (self.h1_scale[1] - self.h1_scale[0])
        para[:, 9] = self.h2_scale[0] + parameters[:, 9] * (self.h2_scale[1] - self.h2_scale[0])
        para[:, 10] = self.a1_scale[0] + parameters[:, 10] * (self.a1_scale[1] - self.a1_scale[0])
        para[:, 11] = self.a2_scale[0] + parameters[:, 11] * (self.a2_scale[1] - self.a2_scale[0])
        para[:, 12] = self.h3_scale[0] + parameters[:, 12] * (self.h3_scale[1] - self.h3_scale[0])
        para[:, 13] = self.b1_scale[0] + parameters[:, 13] * (self.b1_scale[1] - self.b1_scale[0])
        para[:, 14] = self.h4_scale[0] + parameters[:, 14] * (self.h4_scale[1] - self.h4_scale[0])
        para[:, 15] = self.c1_scale[0] + parameters[:, 15] * (self.c1_scale[1] - self.c1_scale[0])
        para[:, 16] = self.d1_scale[0] + parameters[:, 16] * (self.d1_scale[1] - self.d1_scale[0])
        para[:, 17] = self.e1_scale[0] + parameters[:, 17] * (self.e1_scale[1] - self.e1_scale[0])
        para[:, 18] = self.e2_scale[0] + parameters[:, 18] * (self.e2_scale[1] - self.e2_scale[0])
        para[:, 19] = self.h_scale[0] + parameters[:, 19] * (self.h_scale[1] - self.h_scale[0])

        prcp = torch.clamp(p_and_e[self.warmup_length:, :, 0], min=0.0)  # time|basin
        pet = torch.clamp(p_and_e[self.warmup_length:, :, 1], min=0.0)  # time|basin
        n_step, n_basin = prcp.size()
        singletank = SingleStepTank(tank_device, 1, para, intervar, rivernumber, mq)
        e_sim_ = torch.full((n_step, n_basin), 0.0).to(tank_device)
        q_sim_ = torch.full((n_step, n_basin), 0.0).to(tank_device)
        for i in range(n_step):
            p = prcp[i, :]
            e = pet[i, :]
            et, roimp, adsur, ars, rs, ri, rgs, rgp, intervar[:, :7] = singletank.cal_runoff(p, e)
            q_sim_[i], intervar[:, 7:] = singletank.cal_routing(roimp, adsur, ars, rs, ri, rgs, rgp)
            e_sim_[i] = et

        # seq, batch, feature
        e_sim = torch.unsqueeze(e_sim_, dim=-1)  # add a dimension
        q_sim = torch.unsqueeze(q_sim_, dim=-1)
        if return_state:
            return q_sim, e_sim, intervar
        return q_sim, e_sim

class DplAnnTank(nn.Module):
    """
    Tank differential parameter learning - neural network model
    """
    def __init__(
        self,
        n_input_features,
        n_output_features,
        n_hidden_states,
        warmup_length,
        param_limit_func="clamp",   # 参数限制函数 限制在[0,1]
        param_test_way="final",
        source_book="HF",
    ):
        """
        Differential Parameter learning model only with attributes as DL model's input: ANN -> Param -> TANK

        The principle can be seen here: https://doi.org/10.1038/s41467-021-26107-z

        Parameters
        ----------
        n_input_features
            the number of input features of ANN
        n_output_features
            the number of output features of ANN, and it should be equal to the number of learning parameters in TANK
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
        self.dl_model = SimpleAnn(n_input_features, n_output_features, n_hidden_states)
        self.pb_model = Tank4Dpl(
            warmup_length,
            source_book=source_book,
        )
        self.param_func = param_limit_func
        self.param_test_way = param_test_way

    def forward(self, x, z):
        """
        Differential parameter learning

        z (normalized input) -> ANN -> param -> + x (not normalized) -> tank -> q

        Parameters
        ----------
        x
            not normalized data used for physical model, a sequence-first 3-dim tensor. [sequence, batch, feature]
            normalized data used for DL model, a 2-dim tensor. [batch, feature]

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


class DplLstmTank(nn.Module):
    """
    Tank differential parameter learning - Long short-term memory neural network model
    """
    def __init__(
        self,
        n_input_features,
        n_output_features,
        n_hidden_states,
        warmup_length,
        param_limit_func="clamp",
        param_test_way="final",
        source_book="HF",
    ):
        """
        Differential Parameter learning model: LSTM -> Param -> TANK

        The principle can be seen here: https://doi.org/10.1038/s41467-021-26107-z

        Parameters
        ----------
        n_input_features
            the number of input features of LSTM
        n_output_features
            the number of output features of LSTM, and it should be equal to the number of learning parameters in TANK
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
            but remember these ways are only for non-variable parameters
        """
        super(DplLstmTank, self).__init__()
        self.dl_model = SimpleLSTM(n_input_features, n_output_features, n_hidden_states)
        self.pb_model = Tank4Dpl(
            warmup_length,
            source_book=source_book,
        )
        self.param_func = param_limit_func
        self.param_test_way = param_test_way
    def forward(self, x, z):
        """
        Differential parameter learning

        z (normalized input) -> lstm -> param -> + x (not normalized) -> tank -> q
        Parameters will be denormalized in tank model

        Parameters
        ----------
        x
            not normalized data used for physical model; a sequence-first 3-dim tensor. [sequence, batch, feature]
        z
            normalized data used for DL model; a sequence-first 3-dim tensor. [sequence, batch, feature]
            21 parameters of sac model, normalized.

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
        # return torch.cat([q, e], dim=-1)
        return q
