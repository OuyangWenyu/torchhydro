import logging
from typing import List

from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


class MonotonicLoss(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        super(MonotonicLoss, self).__init__()
        self.cfg = cfg
        self.lb = torch.tensor(self.cfg.lb)
        self.ub = torch.tensor(self.cfg.ub)
        self.areas_lb = self.cfg.areas.lb
        self.areas_ub = self.cfg.areas.ub
        self.alpha = torch.tensor(self.cfg.alpha)
        log.info(f"upper bound: {self.areas_ub}")
        log.info(f"Alpha: {self.alpha}")

    def forward(self, n: torch.Tensor, areas: List[float]) -> torch.Tensor:
        areas_tensor = torch.tensor(areas)
        idx = (areas_tensor >= self.areas_lb) & (areas_tensor <= self.areas_ub)
        areas_sub = areas_tensor[idx]
        sorted_areas, sorted_indices = torch.sort(areas_sub)
        n_sub = n[idx]
        n_low_area = n_sub[sorted_indices]
        alpha = self.cfg.alpha
        d_n = n_low_area[:-1] - n_low_area[1:]
        monotonic_penalty = alpha * F.relu(d_n).mean()
        loss = monotonic_penalty
        return loss
