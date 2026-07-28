import logging

import numpy as np
from omegaconf import DictConfig
import torch
import torch.nn as nn

log = logging.getLogger(__name__)


class ParameterList(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(ParameterList, self).__init__()
        self.cfg = cfg
        self.areas = np.load(self.cfg.save_paths.areas)
        self.n = self._generate_noise(self.cfg.variables.n)
        self.q = self._generate_noise(self.cfg.variables.q)
        # self.n = torch.full(
        #     [len(self.areas)], self.cfg.variables.n, dtype=torch.float64
        # )
        # self.q = torch.full(
        #     [len(self.areas)], self.cfg.variables.q, dtype=torch.float64
        # )

    def _generate_noise(self, value):
        log.info("Generating Noise for Parameter List")
        min_val = value - self.cfg.noise
        max_val = value + self.cfg.noise
        lower_bound = min_val - value  # -0.01
        upper_bound = max_val - value  # 0.01
        noise = torch.zeros(len(self.areas), dtype=torch.float64).uniform_(
            lower_bound, upper_bound
        )
        return torch.full([len(self.areas)], value, dtype=torch.float64) + noise

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
