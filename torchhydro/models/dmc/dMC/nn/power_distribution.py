import logging

import numpy as np
from omegaconf import DictConfig
from omegaconf.errors import MissingMandatoryValue
import torch
import torch.nn as nn

log = logging.getLogger(__name__)


class Power(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(Power, self).__init__()
        self.cfg = cfg
        self.areas = np.load(self.cfg.save_paths.areas)
        try:
            if self.cfg.is_base:
                # Lower base model (from paper submission 1)
                n_bounds = [0.0915, 0.131]
            else:
                # Higher base model (a new distribution)
                n_bounds = [0.0915, 0.05]
            self.n = torch.tensor(self._power(n_bounds))
            self.q = torch.tensor(self.cfg.variables.q)
        except MissingMandatoryValue:
            # There is no q value defined. We should use a distribution
            q_bounds = [2.1, 0.357]
            self.q = torch.tensor(self._power(q_bounds))

    def _power(self, bounds):
        return bounds[0] / (self.areas ** bounds[1])

    def _plot_distribution(self):
        import matplotlib.pyplot as plt

        median = np.median(self.n.detach().numpy())
        plt.scatter(
            self.areas,
            self.n.detach().numpy(),
            c="tab:blue",
            label=f"n median: {median}",
        )

        plt.legend(loc="upper right")
        plt.xlabel("Drainage Area (KM^2)")
        plt.ylabel("Manning's n (unitless)")
        plt.title(f"Prescribed n distribution", fontsize=16)
        # plt.savefig(f"{self.path}n_param_{self.type}_prescribed.png")
        plt.show()
        plt.clf()

        median = np.median(self.q.detach().numpy())
        plt.scatter(
            self.areas,
            self.q.detach().numpy(),
            c="tab:red",
            label=f"Spatial Q median: {median}",
        )

        plt.legend(loc="upper right")
        plt.xlabel("Area")
        plt.ylabel("q (unitless)")
        plt.title(f"Prescribed q distribution", fontsize=16)
        # plt.savefig(f"{self.path}q_spatial_param_{self.type}_prescribed.png")
        plt.show()
        plt.clf()

    def forward(self, inputs):
        pass
