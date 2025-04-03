import torch
from torch import nn
from torch.nn import functional as F
from torchhydro.configs.model_config import MODEL_PARAM_TEST_WAY
from torchhydro.models.dpl4sac import Sac4Dpl
from torchhydro.models.simple_narx import SimpleNarx


class DplNarxSac(nn.Module):
    """
    Sacramento differential parameter learning - narx neural network model
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
        Differential Parameter learning model: narx -> Param -> SAC

        The principle can be seen here: https://doi.org/10.1038/s41467-021-26107-z

        Parameters
        ----------
        n_input_features
            the number of input features of narx
        n_output_features
            the number of output features of narx, and it should be equal to the number of learning parameters in SAC
        n_hidden_states
            the number of hidden features of narx
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
        super(DplNarxSac, self).__init__()
        self.dl_model = SimpleNarx(n_input_features, n_output_features, n_hidden_states)
        self.pb_model = Sac4Dpl(
            warmup_length,
            source_book=source_book,
        )
        self.param_func = param_limit_func
        self.param_test_way = param_test_way

    def forward(self, x, z):
        """
        Differential parameter learning

        z (normalized input) -> narx -> param -> + x (not normalized) -> sac -> q
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
