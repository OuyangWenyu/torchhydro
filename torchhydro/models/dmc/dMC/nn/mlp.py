from functools import partial
import logging

from omegaconf import DictConfig
from omegaconf.errors import MissingMandatoryValue
import torch
import torch.nn as nn

from dMC.nn import Initialization

log = logging.getLogger(__name__)


class MLP(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(MLP, self).__init__()
        self.cfg = cfg
        self.Initialization = Initialization(self.cfg)
        self.layers = nn.Sequential(
            nn.Linear(self.cfg.mlp.input_size, self.cfg.mlp.hidden_size),
            nn.Linear(self.cfg.mlp.hidden_size, self.cfg.mlp.hidden_size),
            nn.Linear(self.cfg.mlp.hidden_size, self.cfg.mlp.hidden_size),
            nn.Linear(self.cfg.mlp.hidden_size, self.cfg.mlp.output_size),
            nn.Sigmoid(),
        )
        self.n = None
        try:
            if self.cfg.train_q:
                log.info("Using an nn.Param for q")
                self.q = nn.Parameter(torch.tensor(self.cfg.variables.q))
            else:
                log.info("Using a fixed value for q")
                self.q = torch.tensor(self.cfg.variables.q)
            self.mlp_q = False
        except MissingMandatoryValue:
            log.info("Using many q values with a gradient")
            self.q = None
            self.mlp_q = True

    def initialize_weights(self):
        func = self.Initialization.get()
        init_func = partial(self._initialize_weights, func=func)
        self.apply(init_func)

    def _denormalize(self, name: str, param: torch.Tensor) -> torch.Tensor:
        value_range = self.cfg.transformations[name]
        output = (param * (value_range[1] - value_range[0])) + value_range[0]
        return output

    @staticmethod
    def _initialize_weights(m, func):
        if isinstance(m, nn.Linear):
            func(m.weight)

    def forward(self, inputs: torch.Tensor) -> None:
        x = self.layers(inputs)
        x_transpose = x.transpose(0, 1)
        self.n = self._denormalize("n", x_transpose[0])
        if self.mlp_q:
            self.q = self._denormalize("q_spatial", x_transpose[1])
