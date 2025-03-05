from typing import Union

from torch import nn

from torchhydro.configs.model_config import MODEL_PARAM_DICT, MODEL_PARAM_TEST_WAY
from torchhydro.models.dpl4xaj_nn4et import NnModule4Hydro

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
        er_output=1,
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
            the length of warmup periods;
            sac needs a warmup period to generate reasonable initial state values
        nn_module
            We initialize the module when we firstly initialize Sac4DplWithNnModule.
            Then we will iterately call Sac4DplWithNnModule module for warmup.
            Hence, in warmup period, we don't need to initialize it again.
        param_var_index
            the index of parameters which will be time-varying
            NOTE: at the most, we support k, b, and c to be time-varying
        et_output
            we only support one-layer et now, because its water balance is not easy to handle with

        """
        if param_var_index is None:
            param_var_index = [0, 6]
        if nn_hidden_size is None:
            nn_hidden_size = [16, 8]
        super(Sac4DplWithNnModule, self).__init__()
        self.params_names = MODEL_PARAM_DICT["sac"]["param_name"]
        param_range = MODEL_PARAM_DICT["sac"]["param_range"]
        self.k_scale = param_range["KC"]
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
