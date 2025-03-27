import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from torchhydro.configs.model_config import MODEL_PARAM_DICT, MODEL_PARAM_TEST_WAY
from torchhydro.models.simple_lstm import SimpleLSTM
from torchhydro.models.ann import SimpleAnn


class Sac4Dpl(nn.Module):
    """
    Sacramento model for differential Parameter learning module
    """
    def __init__(
        self,
        warmup_length: int,
        source_book="HF",
    ):
        """
        Initiate a Sacramento model instance.
        Parameters
        ----------
        warmup_length
            the length of warmup periods
            sac needs a warmup period to generate reasonable initial state values
        """
        super(Sac4Dpl, self).__init__()
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

        self.warmup_length = warmup_length
        self.feature_size = 2  # there are 2 input variables in Sac, P and PET.
        self.hydrodt = 1  # one day
        self.source_book = source_book

    def forward(
        self,
        p_and_e: Tensor,
        parameters: Tensor,
        return_state: bool = False,
    ):
        """
        sac model, forward transmission.

        Parameters
        ----------
        p_and_e: Tensor
            precipitation and evaporation, (time,basin,p_and_e)
        parameters: Tensor
            model parameters, 21.
        return_state: bool
            whether to return model state or not.
        --inter variables--
        auztw, the upper layer tension water accumulation on the alterable impervious area, mm.
        alztw, the lower layer tension water accumulation on the alterable impervious area, mm.
        uztw, tension water accumulation in the upper layer, mm.
        uzfw, free water accumulation in the upper layer, mm.
        lztw, sleep video water accumulation in the lower layer, mm.
        lzfs, speed free water accumulation in the lower layer, mm.
        lzfp, slow free water accumulation in the lower layer, mm.
        roimp, runoff of the permanent impervious area, mm.
        adsur, runoff of the alterable impervious area, mm.
        ars, runoff of the alterable impervious area, mm.
        rs, surface runoff of the permeable area, mm.
        ri, runoff of interflow, mm.
        rgs, runoff of speed groundwater, mm.
        rgp, runoff of slow groundwater, mm.
        Returns
        -------
        q_sim : torch.Tensor
            the simulated flow, Q(m^3/s).
        e_sim : torch.Tensor
            the simulated evaporation, E(mm/d).
        auztw, alztw, uztw, uzfw, lztw, lzfs, lzfp, qs, qi, qgs, qgp, mq : torch.Tensor
            the state variables.
        """
        sac_device = p_and_e.device

        n_basin, n_para = parameters.size()

        rsnpb = 1  # river sections number per basin
        rivernumber = np.full(n_basin, rsnpb)  # set only one river section.   basin|river_section
        mq = torch.full((n_basin, rsnpb),0.0).detach()  # Muskingum routing space   basin|rivernumber   note: the column number of mp must equle to the column number of rivernumber
        if self.warmup_length > 0:  # if warmup_length>0, use warmup to calculate initial state.
            # set no_grad for warmup periods
            with torch.no_grad():
                p_and_e_warmup = p_and_e[0:self.warmup_length, :, :]  # time|basin|p_and_e
                cal_init_sac4dpl = Sac4Dpl(
                    # warmup_length must be 0 here
                    warmup_length=0,
                )
                if cal_init_sac4dpl.warmup_length > 0:
                    raise RuntimeError("Please set init model's warmup length to 0!!!")
                _, _, auztw, alztw, uztw, uzfw, lztw, lzfs, lzfp, qs, qi, qgs, qgp, mq = cal_init_sac4dpl(  # note: parameter should be the parameters before de-normalizing.
                    p_and_e_warmup, parameters, return_state=True
                )
        else:  # if no, set a small value directly.
            auztw = (torch.zeros(n_basin, dtype=torch.float32) + 0.01).to(sac_device)
            alztw = (torch.zeros(n_basin, dtype=torch.float32) + 0.01).to(sac_device)
            uztw = (torch.zeros(n_basin, dtype=torch.float32) + 0.01).to(sac_device)
            uzfw = (torch.zeros(n_basin, dtype=torch.float32) + 0.01).to(sac_device)
            lztw = (torch.zeros(n_basin, dtype=torch.float32) + 0.01).to(
                sac_device
            )
            lzfs = (torch.zeros(n_basin, dtype=torch.float32) + 0.01).to(
                sac_device
            )
            lzfp = (torch.zeros(n_basin, dtype=torch.float32) + 0.01).to(sac_device)
            qs = (torch.zeros(n_basin, dtype=torch.float32) + 0.01).to(sac_device)
            qi = (torch.zeros(n_basin, dtype=torch.float32) + 0.01).to(sac_device)
            qgs = (torch.zeros(n_basin, dtype=torch.float32) + 0.01).to(sac_device)
            qgp = (torch.zeros(n_basin, dtype=torch.float32) + 0.01).to(sac_device)
            mq = torch.full((n_basin, rsnpb), 0.01).detach()

        # parameters
        kc = self.kc_scale[0] + parameters[:, 0] * (self.kc_scale[1] - self.kc_scale[0])
        pctim = self.pctim_scale[0] + parameters[:, 1] * (self.pctim_scale[1] - self.pctim_scale[0])
        adimp = self.adimp_scale[0] + parameters[:, 2] * (self.adimp_scale[1] - self.adimp_scale[0])
        uztwm = self.uztwm_scale[0] + parameters[:, 3] * (self.uztwm_scale[1] - self.uztwm_scale[0])
        uzfwm = self.uzfwm_scale[0] + parameters[:, 4] * (self.uzfwm_scale[1] - self.uzfwm_scale[0])
        lztwm = self.lztwm_scale[0] + parameters[:, 5] * (self.lztwm_scale[1] - self.lztwm_scale[0])
        lzfsm = self.lzfsm_scale[0] + parameters[:, 6] * (self.lzfsm_scale[1] - self.lzfsm_scale[0])
        lzfpm = self.lzfpm_scale[0] + parameters[:, 7] * (self.lzfpm_scale[1] - self.lzfpm_scale[0])
        rserv = self.rserv_scale[0] + parameters[:, 8] * (self.rserv_scale[1] - self.rserv_scale[0])
        pfree = self.pfree_scale[0] + parameters[:, 9] * (self.pfree_scale[1] - self.pfree_scale[0])
        riva = self.riva_scale[0] + parameters[:, 10] * (self.riva_scale[1] - self.riva_scale[0])
        zperc = self.zperc_scale[0] + parameters[:, 11] * (self.zperc_scale[1] - self.zperc_scale[0])
        rexp = self.rexp_scale[0] + parameters[:, 12] * (self.rexp_scale[1] - self.rexp_scale[0])
        uzk = self.uzk_scale[0] + parameters[:, 13] * (self.uzk_scale[1] - self.uzk_scale[0])
        lzsk = self.lzsk_scale[0] + parameters[:, 14] * (self.lzsk_scale[1] - self.lzsk_scale[0])
        lzpk = self.lzpk_scale[0] + parameters[:, 15] * (self.lzpk_scale[1] - self.lzpk_scale[0])
        ci = self.ci_scale[0] + parameters[:, 16] * (self.ci_scale[1] - self.ci_scale[0])
        cgs = self.cgs_scale[0] + parameters[:, 17] * (self.cgs_scale[1] - self.cgs_scale[0])
        cgp = self.cgp_scale[0] + parameters[:, 18] * (self.cgp_scale[1] - self.cgp_scale[0])
        ke = self.ke_scale[0] + parameters[:, 19] * (self.ke_scale[1] - self.ke_scale[0])
        xe = self.xe_scale[0] + parameters[:, 20] * (self.xe_scale[1] - self.xe_scale[0])

        prcp = p_and_e[self.warmup_length:, :, 0]  # time|basin
        pet = p_and_e[self.warmup_length:, :, 1]  # time|basin
        n_step, n_basin = prcp.size()
        e_sim_ = torch.full((n_step, n_basin), 0.0).to(sac_device)
        q_sim_ = torch.full((n_step, n_basin), 0.0).to(sac_device)
        roimp_ = torch.full((n_step, n_basin), 0.0).to(sac_device)
        adsur_ = torch.full((n_step, n_basin), 0.0).to(sac_device)
        ars_ = torch.full((n_step, n_basin), 0.0).to(sac_device)
        rs_ = torch.full((n_step, n_basin), 0.0).to(sac_device)
        ri_ = torch.full((n_step, n_basin), 0.0).to(sac_device)
        rgs_ = torch.full((n_step, n_basin), 0.0).to(sac_device)
        rgp_ = torch.full((n_step, n_basin), 0.0).to(sac_device)
        # generate runoff
        for i in range(n_step):  # https://zhuanlan.zhihu.com/p/490501696
            p = torch.clamp(prcp[i, :], min=0.0)
            e = torch.nan_to_num(pet[i, :], nan=0.0, posinf=0.0, neginf=0.0)
            e = torch.clamp(e, min=0.0)
            # evaporation
            ep = kc * e  # average evaporation of basin
            roimp = pctim * p  # runoff of the permanent impervious area
            ae2 = pctim * ep  # evaporation of the permanent impervious area
            # evaporation of the alterable impervious area
            ae1 = torch.min(auztw, ep * (auztw / uztwm))  # upper layer
            ae3 = (ep - ae1) * (alztw / (uztwm + lztwm))  # lower layer
            ae3 = torch.clamp(ae3, min=0.0)
            pav = torch.clamp(p - (uztwm - (auztw - ae1)), min=0.0)
            adsur = pav * ((alztw - ae3) / lztwm)
            adsur = torch.clamp(adsur, min=0.0)
            ars = torch.clamp(((pav - adsur) + (alztw - ae3)) - lztwm, min=0.0)
            auztw = torch.min(uztwm, (auztw - ae1) + p)
            auztw = torch.clamp(auztw, min=0.0)
            alztw = torch.min(lztwm, (pav - adsur) + (alztw - ae3))
            alztw = torch.clamp(alztw, min=0.0)
            # evaporation of the permeable area
            e1 = torch.min(uztw, ep * (uztw / uztwm))  # upper layer
            e2 = torch.min(uzfw, ep - e1)  # lower layer
            e2 = torch.clamp(e2, min=0.0)
            e3 = (ep - e1 - e2) * (lztw / (uztwm + lztwm))  # deeper layer
            e3 = torch.clamp(e3, min=0.0)
            lt1 = lztw - e3
            lt1 = torch.clamp(lt1, min=0.0)
            e4 = riva * ep  # river net, lakes and hydrophyte
            # total evaporation
            et = ae2 + ae1 + ae3 + e1 + e2 + e3 + e4  # the total evaporation

            # generate runoff
            # runoff of the alterable impervious area
            adsur_adimp = adsur * adimp
            ars_adimp = ars * adimp
            # runoff of the permeable area
            parea = 1 - pctim - adimp
            rs = torch.clamp(p + (uztw + uzfw - e1 - e2) - (uztwm + uzfwm), min=0.0) * parea
            ut = torch.min(uztwm, uztw - e1 + p)
            ut = torch.clamp(ut, min=0.0)
            uf = torch.min(uzfwm, p + (uztw + uzfw - e1 - e2) - ut)
            uf = torch.clamp(uf, min=0.0)
            ri = uf * uzk  # interflow
            uf = uf - ri
            uf = torch.clamp(uf, min=0.0)
            # the infiltrated water
            pbase = lzfsm * lzsk + lzfpm * lzpk
            defr = 1 - (lzfs + lzfp + lt1) / (lzfsm + lzfpm + lztwm)
            defr = torch.clamp(defr, min=0.0)
            perc = pbase * (1 + zperc * pow(defr, rexp)) * uf / uzfwm
            rate = torch.min(perc, (lzfsm + lzfpm + lztwm) - (lzfs + lzfp + lt1))
            rate = torch.clamp(rate, min=0.0)
            # assign the infiltrate water
            fx = torch.min(lzfsm + lzfpm - (lzfs + lzfp), torch.max(rate - (lztwm - lt1), rate * pfree))
            fx = torch.clamp(fx, min=0.0)
            perct = rate - fx
            perct = torch.clamp(perct, min=0.0)
            coef = (lzfpm / (lzfsm + lzfpm)) * (2 * (1 - lzfp / lzfpm) / ((1 - lzfp / lzfpm) + (1 - lzfsm / lzfsm)))
            coef = torch.clamp(coef, min=0.0, max=1.0)
            percp = torch.min(lzfpm - lzfp, torch.max(fx - (lzfsm - lzfs), coef * fx))
            percp = torch.clamp(percp, min=0.0)
            percs = fx - percp
            percs = torch.clamp(percs, min=0.0)
            # update the soil moisture accumulation
            lt2 = lt1 + perct
            ls = lzfs + percs
            lp = lzfp + percp
            # generate groundwater runoff
            rgs = ls * lzsk
            ls2 = ls - rgs
            ls2 = torch.clamp(ls2, min=0.0)
            rgp = lp * lzpk
            lp2 = lp - rgp
            lp2 = torch.clamp(lp2, min=0.0)
            # water balance check
            utr = ut / uztwm
            ufr = uf / uzfwm
            utfr = (ut + uf) / (uztwm + uzfwm)
            uztw = torch.where(utr < ufr, uztwm * utfr, ut)
            uzfw = torch.where(utr < ufr, uzfwm * utfr, uf)
            saved = rserv * (lzfsm + lzfpm)
            ratio = (ls + lp - saved + lt2) / (lzfsm + lzfpm - saved + lztwm)
            ratio = torch.clamp(ratio, min=0.0)
            ltr = lt2 / lztwm
            lztw = torch.where(ltr < ratio, lztwm * ratio, lt2)
            lzfs = torch.where(ltr < ratio, torch.clamp(ls - (lztw - lt2), min=0.0), ls2)
            lzfp = torch.where(ltr < ratio, lp - torch.clamp((lztw - lt2) - ls, min=0.0), lp2)

            # save
            e_sim_[i] = et
            roimp_[i] = roimp
            adsur_[i] = adsur_adimp
            ars_[i] = ars_adimp
            rs_[i] = rs
            ri_[i] = ri
            rgs_[i] = rgs
            rgp_[i] = rgp

        # routing
        u = parea * 1000  # daily coefficient, no need conversion.
        for i in range(n_step):
            q_sim_0 = (qs + qi + qgs + qgp).detach()
            # routing
            parea = 1 - pctim - adimp
            parea = torch.clamp(parea, min=0.0)
            # slope routing, use the linear reservoir method
            qs = (roimp_[i] + (adsur_[i] + ars_[i]) * adimp + rs_[i] * parea) * 1000.0
            qi = ci * qi + (1 - ci) * ri_[i] * u
            qi = torch.clamp(qi, min=0.0)
            qgs = cgs * qgs + (1 - cgs) * rgs_[i] * u
            qgs = torch.clamp(qgs, min=0.0)
            qgp = cgp * qgp + (1 - cgp) * rgp_[i] * u
            qgp = torch.clamp(qgp, min=0.0)
            q_sim_[i] = (qs + qi + qgs + qgp)  # time|basin

            # river routing, use the Muskingum routing method
            ke_ = torch.full(ke.size(), 0.0).detach()
            for j in range(n_basin):
                if rivernumber[j] > 0:
                    dt = self.hydrodt * 24.0 / 2.0
                    xo = xe[j].detach()
                    ke_[j] = (ke[j] * 24.0).detach()  # KE is hourly coefficient, need convert to daily.
                    ko = (ke_[j] / rivernumber[j]).detach()
                    c1 = (max(ko * (1.0 - xo) + dt, 0.0))
                    c2 = (max((-ko * xo + dt) / c1, 0.0))
                    c3 = max((ko * (1.0 - xo) - dt) / c1, 0.0)
                    c1_ = ((ko * xo + dt) / c1).detach()
                    i1 = q_sim_0[j].detach()  # flow at the start of timestep, inflow.
                    i2 = q_sim_[i][j].detach()  # flow at the end of timestep, outflow.
                    for k in range(rivernumber[j]):
                        q_sim_0[j] = mq[j, k]  # basin|rivernumber
                        i2_ = (c1_ * i1 + c2 * i2 + c3 * q_sim_0[j]).detach()
                        mq[j, k] = i2_
                        i1 = q_sim_0[j].detach()
                        i2 = i2_.detach()
                q_sim_[i][j] = i2

        # seq, batch, feature
        e_sim = torch.unsqueeze(e_sim_, dim=-1)  # add a dimension
        q_sim = torch.unsqueeze(q_sim_, dim=-1)
        if return_state:
            return q_sim, e_sim, auztw, alztw, uztw, uzfw, lztw, lzfs, lzfp, qs, qi, qgs, qgp, mq
        return q_sim, e_sim

class DplAnnSac(nn.Module):
    """
    Sacramento differential parameter learning - neural network model
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
        Differential Parameter learning model only with attributes as DL model's input: ANN -> Param -> SAC

        The principle can be seen here: https://doi.org/10.1038/s41467-021-26107-z

        Parameters
        ----------
        n_input_features
            the number of input features of ANN
        n_output_features
            the number of output features of ANN, and it should be equal to the number of learning parameters in SAC
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
        super(DplAnnSac, self).__init__()
        self.dl_model = SimpleAnn(n_input_features, n_output_features, n_hidden_states)
        self.pb_model = Sac4Dpl(
            warmup_length,
            source_book=source_book,
        )
        self.param_func = param_limit_func
        self.param_test_way = param_test_way

    def forward(self, x, z):
        """
        Differential parameter learning

        z (normalized input) -> ANN -> param -> + x (not normalized) -> sac -> q

        Parameters
        ----------
        x
            not normalized data used for physical model, a sequence-first 3-dim tensor. [sequence, batch, feature]
        z
            normalized data used for DL model, a 2-dim tensor.

        Returns
        -------
        torch.Tensor
            one time forward result
        """
        gen = self.dl_model(z)
        # if torch.isnan(gen).any():
        #     raise ValueError("Error: NaN values detected. Check your data firstly!!!")
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


class DplLstmSac(nn.Module):
    """
    Sacramento differential parameter learning - Long short-term memory neural network model
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
        super(DplLstmSac, self).__init__()
        self.dl_model = SimpleLSTM(n_input_features, n_output_features, n_hidden_states)
        self.pb_model = Sac4Dpl(
            warmup_length,
            source_book=source_book,
        )
        self.param_func = param_limit_func
        self.param_test_way = param_test_way

    def forward(self, x, z):
        """
        Differential parameter learning

        z (normalized input) -> lstm -> param -> + x (not normalized) -> sac -> q
        Parameters will be denormalized in sac model

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
        gen = self.dl_model(z)  # todo: nan values when lack of evaporation.
        # if torch.isnan(gen).any():
        #     raise ValueError("Error: NaN values detected. Check your data firstly!!!")
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
