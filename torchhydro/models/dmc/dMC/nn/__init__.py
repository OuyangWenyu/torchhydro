import logging
import math

from omegaconf import DictConfig
import torch.nn as nn

log = logging.getLogger(__name__)


class Initialization:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

    def kaiming_normal_initializer(self, x) -> None:
        nn.init.kaiming_normal_(x, mode=self.cfg.mlp.fan, nonlinearity="sigmoid")

    def xavier_normal_initializer(self, x) -> None:
        log.info(f"Gain: {self.cfg.mlp.gain}")
        nn.init.xavier_normal_(x, gain=self.cfg.mlp.gain)

    @staticmethod
    def sparse_initializer(x) -> None:
        nn.init.sparse_(x, sparsity=0.5)

    @staticmethod
    def uniform_initializer(x) -> None:
        # hardcoding hidden size for now
        stdv = 1.0 / math.sqrt(6)
        nn.init.uniform_(x, a=-stdv, b=stdv)

    def get(self):
        init = self.cfg.mlp.initialization.lower()
        log.info(f"Using {init} initialization")
        if init == "kaiming_normal":
            func = self.kaiming_normal_initializer
        elif init == "kaiming_uniform":
            func = nn.init.kaiming_uniform_
        elif init == "orthogonal":
            func = nn.init.orthogonal_
        elif init == "sparse":
            func = self.sparse_initializer
        elif init == "trunc_normal":
            func = nn.init.trunc_normal_
        elif init == "xavier_normal":
            func = self.xavier_normal_initializer
        elif init == "xavier_uniform":
            func = nn.init.xavier_uniform_
        else:
            log.info(f"Defaulting to a uniform initialization")
            func = self.uniform_initializer
        return func
