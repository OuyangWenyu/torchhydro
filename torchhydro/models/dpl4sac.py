import numpy as np
from typing import Union
import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from torchhydro.configs.model_config import MODEL_PARAM_DICT, MODEL_PARAM_TEST_WAY
from torchhydro.models.simple_lstm import SimpleLSTM
from torchhydro.models.ann import SimpleAnn

PRECISION = 1e-5


class SingleStepSacramento(nn.Module):
    """
    single step sacramento model
    """
    def __init__(
        self,
        device: Union[str, torch.device],
        hydrodt: int = 1,
        # area: float = None,
        rivernumber: Tensor = None,
        para: Tensor = None,
        intervar: Tensor = None,
        mq: Tensor = None,
    ):
        """
        Initial a single-step sacramento model
        """
        super(SingleStepSacramento, self).__init__()
        self.name = 'SingleStepSacramento'
        self.device = device
        # self.area = area
        self.rivernumber = rivernumber
        self.hydrodt = hydrodt
        self.para = para
        self.intervar = intervar
        self.mq = mq  # Muskingum routing space

    def cal_runoff(
        self,
        prcp: Tensor = None,
        pet: Tensor = None,
    ):
        """
        single step evaporation and generating runoff
        Parameters
        -----------
        hydrodt : int
            time step
        prcp : float
            precipitation, mm/d.
        pet : float
            evaporation, mm/d.
        para : Union[tuple, list]
            model parameters.
        intervar: Union[tuple, list]
            inter variables in model.
            auztw, the upper layer tension water accumulation on the alterable impervious area, mm.
            alztw, the lower layer tension water accumulation on the alterable impervious area, mm.
            uztw, tension water accumulation in the upper layer, mm.
            uzfw, free water accumulation in the upper layer, mm.
            lztw, tension water accumulation in the lower layer, mm.
            lzfs, speed free water accumulation in the lower layer, mm.
            lzfp, slow free water accumulation in the lower layer, mm.
        Returns
        -------

        """
        # parameters
        kc = self.para[0]
        pctim = self.para[1]
        adimp = self.para[2]
        uztwm = self.para[3]
        uzfwm = self.para[4]
        lztwm = self.para[5]
        lzfsm = self.para[6]
        lzfpm = self.para[7]
        rserv = self.para[8]
        pfree = self.para[9]
        riva = self.para[10]
        zperc = self.para[11]
        rexp = self.para[12]
        uzk = self.para[13]
        lzsk = self.para[14]
        lzpk = self.para[15]
        # middle variables, at the start of timestep.
        auztw = self.intervar[0]
        alztw = self.intervar[1]
        uztw = self.intervar[2]
        uzfw = self.intervar[3]
        lztw = self.intervar[4]
        lzfs = self.intervar[5]
        lzfp = self.intervar[6]

        # evaporation
        ep = kc * pet  # average evaporation of basin
        roimp = prcp * pctim  # runoff of the permanent impervious area
        ae2 = pctim * ep  # evaporation of the permanent impervious area
        # evaporation of the alterable impervious area
        ae1 = torch.min((auztw, ep * (auztw / uztwm)))  # upper layer
        ae3 = (ep - ae1) * (alztw / (uztwm + lztwm))  # lower layer
        pav = torch.max((0.0, prcp - (uztwm - (auztw - ae1))))
        adsur = pav * ((alztw - ae3) / lztwm)
        ars = torch.max((0.0, ((pav - adsur) + (alztw - ae3)) - lztwm))
        auztw = torch.min((uztwm, (auztw - ae1) + prcp))
        alztw = torch.min((lztwm, (pav - adsur) + (alztw - ae3)))
        # evaporation of the permeable area
        e1 = torch.min((uztw, ep * (uztw / uztwm)))  # upper layer
        e2 = torch.min((uzfw, ep - e1))  # lower layer
        e3 = (ep - e1 - e2) * (lztw / (uztwm + lztwm))  # deeper layer
        lt = lztw - e3
        e4 = riva * ep  # river net, lakes and hydrophyte
        # total evaporation
        et = ae2 + ae1 + ae3 + e1 + e2 + e3 + e4  # the total evaporation

        # generate runoff
        # runoff of the alterable impervious area
        adsur = adsur * adimp
        ars = ars * adimp
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
        ls = ls - rgs
        rgp = lp * lzpk
        lp = lp - rgp
        # water balance check
        if ut / uztwm < uf / uzfwm:
            uztw = uztwm * (ut + uf) / (uztwm + uzfwm)
            uzfw = uzfwm * (ut + uf) / (uztwm + uzfwm)
        else:
            uztw = ut
            uzfw = uf
        saved = rserv * (lzfsm + lzfpm)
        ratio_ = (ls + lp - saved + lt) / (lzfsm + lzfpm - saved + lztwm)
        ratio = torch.max((ratio_, 0.0))
        if lt / lztwm < ratio:
            lztw = lztwm * ratio
            del_ = lztw - lt
            lzfs = torch.max((0.0, ls - del_))
            lzfp = lp - torch.max((0.0, del_ - ls))
        else:
            lztw = lt
            lzfs = ls
            lzfp = lp

        # rs = roimp + (adsur + ars) + rs  # the total surface runoff

        # middle variables, at the end of timestep.
        self.intervar[0] = auztw
        self.intervar[1] = alztw
        self.intervar[2] = uztw
        self.intervar[3] = uzfw
        self.intervar[4] = lztw
        self.intervar[5] = lzfs
        self.intervar[6] = lzfp

        return et, roimp, adsur, ars, rs, ri, rgs, rgp, self.intervar[:6]  # todo:

    def cal_routing(
        self,
        roimp, adsur, ars, rs, ri, rgs, rgp,
    ):
        """
        single step routing
        Parameters
        ----------
        hydrodt
        roimp
        adsur
        ars
        rs
        ri
        rgs
        rgp
        intervar

        Returns
        -------
        q_sim
            the outflow at the end of timestep, m^3/s.
        """
        # parameters
        pctim = self.para[1]
        adimp = self.para[2]
        ci = self.para[16]
        cgs = self.para[17]
        cgp = self.para[18]
        ke = self.para[19]
        xe = self.para[20]
        # middle variables, at the start of timestep.
        qs0 = torch.full(self.intervar[7,:].size(),self.intervar[:7]).to(self.device)
        qi0 = torch.full(self.intervar[8,:].size(),self.intervar[:8]).to(self.device)
        qgs0 = self.intervar[9]
        qgp0 = self.intervar[10]

        # routing
        u = (1 - pctim - adimp) * 1000  # * self.area   # daily coefficient, no need conversion.  # todo: area
        parea = 1 - pctim - adimp
        # slope routing, use the linear reservoir method
        qs = (roimp + (adsur + ars) * adimp + rs * parea) * 1000.0  # * self.area  # todo: area
        qi = ci * qi0 + (1 - ci) * ri * u
        qgs = cgs * qgs0 + (1 - cgs) * rgs * u
        qgp = cgp * qgp0 + (1 - cgp) * rgp * u
        q_sim_ = (qs + qi + qgs + qgp)
        # middle variable, at the end of timestep.
        self.intervar[7] = qs
        self.intervar[8] = qi
        self.intervar[9] = qgs
        self.intervar[10] = qgp

        # river routing, use the Muskingum routing method
        q_sim_0 = qs0 + qi0 + qgs0 + qgp0
        if self.rivernumber > 0:
            dt = self.hydrodt * 24.0 / 2.0
            xo = xe
            ke = ke * 24.0  # KE is hourly coefficient, need convert to daily.
            ko = ke / self.rivernumber
            c1 = ko * (1.0 - xo) + dt
            c2 = (-ko * xo + dt) / c1
            c3 = (ko * (1.0 - xo) - dt) / c1
            c1 = (ko * xo + dt) / c1
            i1 = q_sim_0
            i2 = q_sim_
            for i in range(self.rivernumber):
                q_sim_0 = self.mq[i,:]  # rivernumber|basin
                i2 = c1 * i1 + c2 * i2 + c3 * q_sim_0
                self.mq[i,:] = i2
                i1 = q_sim_0
        q_sim_ = i2
        return q_sim_, self.intervar[7:]


class Sacramento():
    """
    sacramento model
    """

    def __init__(
        self,
        hydrodt: int = 1,
        p_and_e: np.ndarray = None,
        para: Union[tuple, list] = None,
        # area: float = None,
        warmup_length: int = None,

    ):
        self.para = para
        # self.area = area
        self.warmup_length = warmup_length
        # hydrodata
        self.hydrodt = hydrodt
        self.sequence_length = p_and_e.shape[0]
        self.prcp = p_and_e[:, 0]
        self.evap = p_and_e[:, 1]

    def sacmodel(self):
        """

        Returns
        -------

        """




class Sac4Dpl(nn.Module):
    """
    Sacramento model for Differential Parameter learning as a submodule
    """
    def __init__(
        self,
        warmup_length: int,
        source_book="HF",
        nn_hidden_size: Union[int, tuple, list] = None,
        nn_dropout=0.2,
        param_test_way=MODEL_PARAM_TEST_WAY["final"],
    ):
        """
        Initiate a Sacramento model instance.
        Parameters
        ----------
        warmup_length
            the length of warmup periods 预热期
            sac needs a warmup period to generate reasonable initial state values 需要预热，形成初始条件
        nn_hidden_size
            the hidden layer size of neural network
        nn_dropout
            the dropout rate of neural network 神经网络中间层神经元的暂退率，(0.3, 0.5)，提高模型的稳健性。暂时退出、暂时不参与模拟计算的概率。什么时候重回计算呢？动态暂退？
        param_test_way
            the way to test the model parameters, final. 模型参数测试方式，取final式。
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
        self.feature_size = 2       # there are 2 input variables in Sac, P and PET. P and Pet are two feature in nn model.
        self.source_book = source_book
        self.nn_hidden_size = nn_hidden_size
        self.nn_dropout = nn_dropout
        self.param_test_way = param_test_way

    def forward(self, p_and_e, parameters, return_state=False):
        """
        sac model
        forward transmission

        Parameters
        ----------
        p_and_e
        prcp
            basin mean precipitation, mm/d.
        pet
            potential evapotranspiration, mm/d.
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
        q_sim : torch.Tensor
        the simulated flow, Q(m^3/s).
        e_sim : torch.Tensor
            the simulated evaporation, E(mm/d).
        """
        sac_device = p_and_e.device
        prcp = torch.clamp(p_and_e[:, 0], min=0.0)
        pet = torch.clamp(p_and_e[:, 1], min=0.0)
        # parameters
        # kc = self.kc_scale[0] + parameters[:, 0] * (self.kc_scale[1] - self.kc_scale[0])    # parameters[:, 0]是个二维张量， 流域|参数  kc是个一维张量，不同流域的参数。  basin first
        # pctim = self.pctim_scale[0] + parameters[:, 1] * (self.pctiscale[1] - self.pctim_scale[0])
        # adimp = self.adimp_scale[0] + parameters[:, 2] * (self.adimp_scale[1] - self.adimp_scale[0])
        # uztwm = self.uztwm_scale[0] + parameters[:, 3] * (self.uztwscale[1] - self.uztwm_scale[0])
        # uzfwm = self.uzfwm_scale[0] + parameters[:, 4] * (self.uzfwm_scale[1] - self.uzfwm_scale[0])
        # lztwm = self.lztwm_scale[0] + parameters[:, 5] * (self.lztwm_scale[1] - self.lztwm_scale[0])
        # lzfsm = self.lzfsm_scale[0] + parameters[:, 6] * (self.lzfsm_scale[1] - self.lzfsm_scale[0])
        # lzfpm = self.lzfpm_scale[0] + parameters[:, 7] * (self.lzfpm_scale[1] - self.lzfpm_scale[0])
        # rserv = self.rserv_scale[0] + parameters[:, 8] * (self.rserv_scale[1] - self.rserv_scale[0])
        # pfree = self.pfree_scale[0] + parameters[:, 9] * (self.pfree_scale[1] - self.pfree_scale[0])
        # riva = self.riva_scale[0] + parameters[:, 10] * (self.riva_scale[1] - self.riva_scale[0])
        # zperc = self.zperc_scale[0] + parameters[:, 11] * (self.zperc_scale[1] - self.zperc_scale[0])
        # rexp = self.rexp_scale[0] + parameters[:, 12] * (self.rexp_scale[1] - self.rexp_scale[0])
        # uzk = self.uzk_scale[0] + parameters[:, 13] * (self.uzk_scale[1] - self.uzk_scale[0])
        # lzsk = self.lzsk_scale[0] + parameters[:, 14] * (self.lzsk_scale[1] - self.lzsk_scale[0])
        # lzpk = self.lzpk_scale[0] + parameters[:, 15] * (self.lzpk_scale[1] - self.lzpk_scale[0])
        # ci = self.ci_scale[0] + parameters[:, 16] * (self.ci_scale[1] - self.ci_scale[0])
        # cgs = self.cgs_scale[0] + parameters[:, 17] * (self.cgs_scale[1] - self.cgs_scale[0])
        # cgp = self.cgp_scale[0] + parameters[:, 18] * (self.cgp_scale[1] - self.cgp_scale[0])
        # ke = self.ke_scale[0] + parameters[:, 19] * (self.ke_scale[1] - self.ke_scale[0])
        # xe = self.xe_scale[0] + parameters[:, 20] * (self.xe_scale[1] - self.xe_scale[0])
        para = torch.full(parameters.shap(), 0.0)
        para[:, 0] = self.kc_scale[0] + parameters[:, 0] * (self.kc_scale[1] - self.kc_scale[0])    # parameters[:, 0]是个二维张量， 流域|参数  kc是个一维张量，不同流域的参数。  basin first
        para[:, 1] = self.pctim_scale[0] + parameters[:, 1] * (self.pctiscale[1] - self.pctim_scale[0])
        para[:, 2] = self.adimp_scale[0] + parameters[:, 2] * (self.adimp_scale[1] - self.adimp_scale[0])
        para[:, 3] = self.uztwm_scale[0] + parameters[:, 3] * (self.uztwscale[1] - self.uztwm_scale[0])
        para[:, 4] = self.uzfwm_scale[0] + parameters[:, 4] * (self.uzfwm_scale[1] - self.uzfwm_scale[0])
        para[:, 5] = self.lztwm_scale[0] + parameters[:, 5] * (self.lztwm_scale[1] - self.lztwm_scale[0])
        para[:, 6] = self.lzfsm_scale[0] + parameters[:, 6] * (self.lzfsm_scale[1] - self.lzfsm_scale[0])
        para[:, 7] = self.lzfpm_scale[0] + parameters[:, 7] * (self.lzfpm_scale[1] - self.lzfpm_scale[0])
        para[:, 8] = self.rserv_scale[0] + parameters[:, 8] * (self.rserv_scale[1] - self.rserv_scale[0])
        para[:, 9] = self.pfree_scale[0] + parameters[:, 9] * (self.pfree_scale[1] - self.pfree_scale[0])
        para[:, 10] = self.riva_scale[0] + parameters[:, 10] * (self.riva_scale[1] - self.riva_scale[0])
        para[:, 11] = self.zperc_scale[0] + parameters[:, 11] * (self.zperc_scale[1] - self.zperc_scale[0])
        para[:, 12] = self.rexp_scale[0] + parameters[:, 12] * (self.rexp_scale[1] - self.rexp_scale[0])
        para[:, 13] = self.uzk_scale[0] + parameters[:, 13] * (self.uzk_scale[1] - self.uzk_scale[0])
        para[:, 14] = self.lzsk_scale[0] + parameters[:, 14] * (self.lzsk_scale[1] - self.lzsk_scale[0])
        para[:, 15] = self.lzpk_scale[0] + parameters[:, 15] * (self.lzpk_scale[1] - self.lzpk_scale[0])
        para[:, 16] = self.ci_scale[0] + parameters[:, 16] * (self.ci_scale[1] - self.ci_scale[0])
        para[:, 17] = self.cgs_scale[0] + parameters[:, 17] * (self.cgs_scale[1] - self.cgs_scale[0])
        para[:, 18] = self.cgp_scale[0] + parameters[:, 18] * (self.cgp_scale[1] - self.cgp_scale[0])
        para[:, 19] = self.ke_scale[0] + parameters[:, 19] * (self.ke_scale[1] - self.ke_scale[0])
        para[:, 20] = self.xe_scale[0] + parameters[:, 20] * (self.xe_scale[1] - self.xe_scale[0])
        # para = torch.full((kc.size(),parameters.shap(1)),[kc, pctim, adimp, uztwm, uzfwm, lztwm, lzfsm, lzfpm, rserv, pfree, riva, zperc, rexp, uzk, lzsk, lzpk, ci, cgs, cgp, ke, xe],)

        # area = [2252.7, 573.6, 3676.17, 769.05, 909.1, 383.82, 180.98, 250.64, 190.92, 31.3]     # dpl4sac_args camelsus [gage_id.area]  # todo: 线性水库汇流需要用到流域面积将产流runoff的mm乘以面积的m^2将平面径流转化为流量的m^3/s
        rivernumber = torch.full(para[:, 0].size(),1)  # set only one river section.
        intervar = torch.full((para[:, 0].size(),11),0.0)  # basin|inter_variables
        mq = torch.full((para[:, 0].size(),rivernumber[0]),0.0)  # Muskingum routing space
        singlesac = SingleStepSacramento(sac_device, 1, rivernumber, para, intervar, mq)

        if self.warmup_length > 0:
            # set no_grad for warmup periods
            with torch.no_grad():
                p_and_e_warmup = p_and_e[0:self.warmup_length, :, :]
                if self.param_test_way == MODEL_PARAM_TEST_WAY["time_varying"]:
                    parameters_ts_warmup = para[0:self.warmup_length, :, :]
                else:
                    parameters_ts_warmup = para
                cal_init_sac4dpl = singlesac(
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
                if cal_init_sac4dpl.warmup_length > 0:
                    raise RuntimeError("Please set init model's warmup length to 0!!!")
                _, _, *w0, s0, fr0, qi0, qg0 = cal_init_sac4dpl(
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
            fr0 = torch.full(ci.size(), 0.1).to(sac_device)
            qi0 = torch.full(ci.size(), 0.1).to(sac_device)
            qg0 = torch.full(cg.size(), 0.1).to(sac_device)

        e_sim_ = torch.full(p_and_e.shape[:2], 0.0).to(sac_device)
        q_sim_ = torch.full(p_and_e.shape[:2], 0.0).to(sac_device)
        for i in range(p_and_e.shape[0]):
            et, roimp, adsur, ars, rs, ri, rgs, rgp, intervar[:6] = singlesac.cal_runoff(prcp[i], pet[i])
            q_sim_[i], intervar[7:] = singlesac.cal_routing(roimp, adsur, ars, rs, ri, rgs, rgp)
            e_sim_[i] = et


        # slope routing, use the linear reservoir method
        # qi0 = torch.full(ci.size(),0.0).to(sac_device)
        # qgs0 = torch.full(cgs.size(),0.0).to(sac_device)
        # qgp0 = torch.full(cgp.size(),0.0).to(sac_device)
        # qs0 = 0.8 * 10 * area
        # qs0 = torch.full(imputs.shape[:2],0.0).to(sac_device)

        # seq, batch, feature
        q_sim = torch.unsqueeze(q_sim_, dim=2)
        e_sim = torch.unsqueeze(e_sim_, dim=2)

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
        param_limit_func="clamp",   # 参数限制函数 限制在[0,1]
        param_test_way="final",
        source_book="HF",
        nn_dropout=0.2,
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
            nn_dropout=nn_dropout,
            param_test_way=param_test_way,
        )
        self.param_func = param_limit_func
        self.param_test_way = param_test_way

    def forward(self, x, z):
        """
        Differential parameter learning

        z (normalized input) -> ANN -> param -> + x (not normalized) -> sac -> q
        Parameters will be denormalized in sac model  在sac模型中参数会去正则化，即展开。

        Parameters
        ----------
        x
            not normalized data used for physical model, a sequence-first 3-dim tensor. [sequence, batch, feature]  非正则数据，序列优先的三维张量，用于物理模型。 [序列，批次，特征]  [时间|批次划分|特征（降雨、蒸发）]  流域？
            # todo: 批次划分作为一个维度？ 那是如何划分的？ 划分的过程在哪？
        z
            normalized data used for DL model, a 2-dim tensor. [batch, feature]  正则化后的数据，二维张量，用于DL模型。 [批次，特征]  [批次划分|参数]？

        Returns
        -------
        torch.Tensor
            one time forward result 单步前向传播结果？
        """
        gen = self.dl_model(z)  # SimpleAnn   使用 nn 进化参数  计算各参数对目标值的梯度？   传出来的是参数，经过激活函数后传到物理模型中，去正则化，进行洪水演算，模拟径流。
        if torch.isnan(gen).any():
            raise ValueError("Error: NaN values detected. Check your data firstly!!!")
        # we set all params' values in [0, 1] and will scale them when forwarding
        if self.param_func == "sigmoid":  # activation function
            params = F.sigmoid(gen)   # 参数是通过nn模型算出来的数据，然后传进下面的物理模型中，算出 降雨 和 蒸发。 # todo: ?
        elif self.param_func == "clamp":
            params = torch.clamp(gen, min=0.0, max=1.0)  #将输入input张量每个元素的范围限制到区间[min，max], 返回一个新张量。
        else:
            raise NotImplementedError(
                "We don't provide this way to limit parameters' range!! Please choose sigmoid or clamp"
            )
        # just get one-period values, here we use the final period's values,
        # when the MODEL_PARAM_TEST_WAY is not time_varing, we use the last period's values.
        if self.param_test_way != MODEL_PARAM_TEST_WAY["time_varying"]:
            params = params[-1, :, :]   # todo: added a dimension?    basin|parameters
        # Please put p in the first location and pet in the second
        q, e = self.pb_model(x[:, :, : self.pb_model.feature_size], params)   # 再将参数代入物理模型计算径流，然后使用实测数据比对、计算目标值。反复迭代优化，计算目标值损失量。    第三维是数据项/属性项，降雨和蒸发数据   todo:流域和时间哪个是第一维？
        return torch.cat([q, e], dim=-1)  # catenate, 拼接 q 和 e，按列。 [0,1]


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
        nn_hidden_size=None,
        nn_dropout=0.2,
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
            nn_hidden_size=nn_hidden_size,
            nn_dropout=nn_dropout,
            param_test_way=param_test_way,
        )
        self.param_func = param_limit_func
        self.param_test_way = param_test_way

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
