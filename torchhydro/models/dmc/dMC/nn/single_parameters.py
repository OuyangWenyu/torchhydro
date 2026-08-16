import logging

from omegaconf import DictConfig
import torch
import torch.nn as nn

log = logging.getLogger(__name__)


class SingleParameters(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(SingleParameters, self).__init__()
        self.cfg = cfg
        self.n = nn.Parameter(torch.tensor(self.cfg.variables.n))
        self.q = nn.Parameter(torch.tensor(self.cfg.variables.q))

    def forward(self, inputs):
        pass
