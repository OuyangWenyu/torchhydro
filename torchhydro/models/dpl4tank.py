import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from torchhydro.configs.model_config import MODEL_PARAM_DICT, MODEL_PARAM_TEST_WAY
from torchhydro.models.simple_lstm import SimpleLSTM
from torchhydro.models.ann import SimpleAnn


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
            the length of warmup periods
            tank needs a warmup period to generate reasonable initial state values
        source_book
            "Hydrological Forecasting (4-th version)" written by Prof. Weimin Bao.
        """
        super(Tank4Dpl, self).__init__()
        self.name = "Tank"
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
        self.feature_size = 2  # there are 2 input variables in Sac, P and PET. P and Pet are two feature in nn model.
        self.hydrodt = 1
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
            model parameters, 19.
        return_state: bool
            whether to return model state or not.
        rs: Tensor
            surface runoff, mm.
        ri: Tensor
            runoff of interflow, mm.
        rgs: Tensor
            runoff of speed groundwater, mm.
        rgd: Tensor
        runoff of slow groundwater, mm.
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
        Returns
        -------
        q_sim : torch.Tensor
        the simulated flow, Q(m^3/s).
        e_sim : torch.Tensor
            the simulated evaporation, E(mm/d).
        """
        tank_device = p_and_e.device
        n_basin, n_para = parameters.size()

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
                _, _, xf, xp, x2, xs, x3, x4, x5, qs = cal_init_tank4dpl(
                    p_and_e_warmup, parameters, return_state=True
                )
        else:  # if no, set a small value directly.
            xf = (torch.zeros(n_basin, dtype=torch.float32) + 0.01).to(tank_device)
            xp = (torch.zeros(n_basin, dtype=torch.float32) + 0.01).to(tank_device)
            x2 = (torch.zeros(n_basin, dtype=torch.float32) + 0.01).to(tank_device)
            xs = (torch.zeros(n_basin, dtype=torch.float32) + 0.01).to(tank_device)
            x3 = (torch.zeros(n_basin, dtype=torch.float32) + 0.01).to(tank_device)
            x4 = (torch.zeros(n_basin, dtype=torch.float32) + 0.01).to(tank_device)
            x5 = (torch.zeros(n_basin, dtype=torch.float32) + 0.01).to(tank_device)
            qs = (torch.zeros(n_basin, dtype=torch.float32) + 0.01).to(tank_device)

        # parameters
        kc = self.kc_scale[0] + parameters[:, 0] * (self.kc_scale[1] - self.kc_scale[0])
        w1 = self.w1_scale[0] + parameters[:, 1] * (self.w1_scale[1] - self.w1_scale[0])
        w2 = self.w2_scale[0] + parameters[:, 2] * (self.w2_scale[1] - self.w2_scale[0])
        k1 = self.k1_scale[0] + parameters[:, 3] * (self.k1_scale[1] - self.k1_scale[0])
        k2 = self.k2_scale[0] + parameters[:, 4] * (self.k2_scale[1] - self.k2_scale[0])
        a0 = self.a0_scale[0] + parameters[:, 5] * (self.a0_scale[1] - self.a0_scale[0])
        b0 = self.b0_scale[0] + parameters[:, 6] * (self.b0_scale[1] - self.b0_scale[0])
        c0 = self.c0_scale[0] + parameters[:, 7] * (self.c0_scale[1] - self.c0_scale[0])
        h1 = self.h1_scale[0] + parameters[:, 8] * (self.h1_scale[1] - self.h1_scale[0])
        h2 = self.h2_scale[0] + parameters[:, 9] * (self.h2_scale[1] - self.h2_scale[0])
        a1 = self.a1_scale[0] + parameters[:, 10] * (self.a1_scale[1] - self.a1_scale[0])
        a2 = self.a2_scale[0] + parameters[:, 11] * (self.a2_scale[1] - self.a2_scale[0])
        h3 = self.h3_scale[0] + parameters[:, 12] * (self.h3_scale[1] - self.h3_scale[0])
        b1 = self.b1_scale[0] + parameters[:, 13] * (self.b1_scale[1] - self.b1_scale[0])
        h4 = self.h4_scale[0] + parameters[:, 14] * (self.h4_scale[1] - self.h4_scale[0])
        c1 = self.c1_scale[0] + parameters[:, 15] * (self.c1_scale[1] - self.c1_scale[0])
        d1 = self.d1_scale[0] + parameters[:, 16] * (self.d1_scale[1] - self.d1_scale[0])
        e1 = self.e1_scale[0] + parameters[:, 17] * (self.e1_scale[1] - self.e1_scale[0])
        e2 = self.e2_scale[0] + parameters[:, 18] * (self.e2_scale[1] - self.e2_scale[0])
        h = self.h_scale[0] + parameters[:, 19] * (self.h_scale[1] - self.h_scale[0])

        prcp = torch.clamp(p_and_e[self.warmup_length:, :, 0], min=0.0)  # time|basin
        pet = torch.clamp(p_and_e[self.warmup_length:, :, 1], min=0.0)  # time|basin
        n_step, n_basin = prcp.size()
        e_sim_ = torch.full((n_step, n_basin), 0.0).to(tank_device)
        q_sim_ = torch.full((n_step, n_basin), 0.0).to(tank_device)
        rs_ = torch.full((n_step, n_basin), 0.0).to(tank_device)
        ri_ = torch.full((n_step, n_basin), 0.0).to(tank_device)
        rgs_ = torch.full((n_step, n_basin), 0.0).to(tank_device)
        rgd_ = torch.full((n_step, n_basin), 0.0).to(tank_device)
        for i in range(n_step):
            p = prcp[i, :]
            e = pet[i, :]
            # evaporation
            ep = kc * e  # basin evaporation capacity
            et = torch.min(ep, p + xp + xf)  # the actual evaporation    only one layer evaporation
            pe = torch.clamp(p - et, min=0.0)  # net precipitation
            # soil moisture
            x = xf
            xf = x - torch.clamp(ep - p, min=0.0)  # update the first layer remain free water
            xf = torch.clamp(xf, min=0.0)
            xp = xp - torch.clamp(ep - p - x, min=0.0)  # update the first layer remain tension water
            xp = torch.clamp(xp, min=0.0)
            # update soil moisture
            t1 = k1 * torch.min(x2, w1 - xp)
            t1 = torch.clamp(t1, min=0.0)
            xp = xp + t1  # update the first layer tension water
            x2 = x2 - t1  # update the second layer free water
            x2 = torch.clamp(x2, min=0.0)

            t2 = k2 * (xs * w1 - xp * w2) / (w1 + w2)  # if t2>0,
            t2 = torch.clamp(t2, min=0.0)

            xp = xp + t2  # update the first layer tension water
            xs = xs - t2  # update the second layer tension water
            xs = torch.clamp(xs, min=0.0)

            xf = xf + torch.clamp(xp + pe - w1, min=0.0)  # update the first layer free water
            xp = torch.min(w1, xp + pe)  # update the first layer tension water    the next timestep

            # the infiltrated water
            f1 = a0 * xf

            # surface runoff    the first layer generate the surface runoff
            rs = torch.where(
                xf > h2,
                (xf - h1) * a1 + (xf - h2) * a2,
                torch.where(xf > h1, (xf - h1) * a1, 0.0)
            )
            # update soil moisture
            x2 = x2 + f1  # update the second layer free water
            xf = xf - (rs + f1)  # the first layer free water
            xf = torch.clamp(xf, min=0.0)
            f2 = b0 * x2  # infiltrated water of the second layer

            # interflow    the second layer generate interflow
            ri = torch.where(x2 > h3, (x2 - h3) * b1, 0.0)
            # update soil moisture
            x2 = x2 - (f2 + ri)  # update the second layer free water
            x2 = torch.clamp(x2, min=0.0)
            x3 = x3 + f2  # shallow groundwater accumulation
            f3 = c0 * x3  # shallow infiltrated water of  groundwater

            # shallow groundwater runoff
            rgs = torch.where(x3 > h4, (x3 - h4) * c1, 0.0)
            x3 = x3 - (f3 + rgs)  # update the shallow groundwater
            x3 = torch.clamp(x3, min=0.0)

            # deep groundwater    x4 generate the deep layer groundwater runoff
            x4 = x4 + f3
            rgd = d1 * x4
            x4 = x4 - rgd
            x4 = torch.clamp(x4, min=0.0)

            # save
            e_sim_[i] = et
            rs_[i] = rs
            ri_[i] = ri
            rgs_[i] = rgs
            rgd_[i] = rgd

        # routing
        u = self.hydrodt / 1000.0  # unit conversion
        for i in range(n_step):
            k0 = torch.where(x5 >= h, 1 / (e1 + e2), 1 / e1)
            c0 = torch.where(x5 >= h, e2 * h / (e1 + e2), 0)
            k1 = 1 / e1
            c1 = 0
            dt = 0.5 * u
            ii = rs_[i] + ri_[i] + rgs_[i] + rgd_[i]
            q1 = (k0 - dt) / (k1 + dt) * qs + (ii - c1 + c0) / (k1 + dt)
            q1 = torch.clamp(q1, min=0.0)
            x5 = k1 * q1 + c1
            k1 = torch.where(x5 > h, 1 / (e1 + e2), k1)
            c1 = torch.where(x5 > h, e2 * h / (e1 + e2), c1)
            q1 = (k0 - dt) / (k1 + dt) * qs + (ii - c1 + c0) / (k1 + dt)  # update
            q1 = torch.clamp(q1, min=0.0)
            x5 = k1 * q1 + c1
            qs = q1
            q_sim_[i] = q1

        # seq, batch, feature
        e_sim = torch.unsqueeze(e_sim_, dim=-1)  # add a dimension
        q_sim = torch.unsqueeze(q_sim_, dim=-1)
        if return_state:
            return q_sim, e_sim, xf, xp, x2, xs, x3, x4, x5, qs
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
        param_limit_func="clamp",
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
            not normalized data used for physical model, a sequence-first 3-dim tensor.
            normalized data used for DL model, a 2-dim tensor.
        z
            normalized data used for DL model; a sequence-first 3-dim tensor.
            19 parameters of tank model, normalized.

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
            not normalized data used for physical model; a sequence-first 3-dim tensor.
        z
            normalized data used for DL model; a sequence-first 3-dim tensor.
            19 parameters of tank model, normalized.

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
