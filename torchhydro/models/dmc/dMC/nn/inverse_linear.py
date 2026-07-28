import logging

import numpy as np
from omegaconf import DictConfig
import torch
import torch.nn as nn

log = logging.getLogger(__name__)


class InverseLinear(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(InverseLinear, self).__init__()
        self.cfg = cfg
        self.areas = np.load(self.cfg.save_paths.areas)
        # n_bounds = [0.06, 2.67e-5]
        n_bounds = [0.06, 8e-6]
        # q_bounds = [2, 3.33e-4]
        # q_bounds = [2, 0.00018]
        q_bounds = [1, 0.00018]
        self.n = torch.tensor(self._inverse_linear(n_bounds))
        self.q = torch.tensor(self._inverse_linear(q_bounds))
        # mask = self.areas > 1500
        # self.n[mask] = 0.02
        # self.q[mask] = 1.5
        # self.q = nn.Parameter(torch.tensor(self.cfg.variables.q))

    def _inverse_linear(self, bounds):
        return bounds[0] - (bounds[1] * self.areas)

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
        plt.show()
        plt.clf()

    def forward(self, inputs):
        pass
