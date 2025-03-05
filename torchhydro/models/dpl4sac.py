from typing import Union
import torch
from torch import nn
from torch import Tensor
from torchhydro.configs.model_config import MODEL_PARAM_DICT, MODEL_PARAM_TEST_WAY
from torchhydro.models.dpl4xaj_nn4et import NnModule4Hydro

PRECISION = 1e-5


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
            the time length of unit hydrograph
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
        et_output  蒸发输出？
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
        p_and_e: Tensor,
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
        kc
        pctim,
        adimp,
        uztwm,
        uzfwm,
        lztwm,
        lzfpm,
        rserv,
        pfree,

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
