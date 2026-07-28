import logging

from omegaconf import DictConfig
import torch
import torch.nn as nn

log = logging.getLogger(__name__)


class MeanRangeBoundLoss(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        """
        A loss function that ensures the learned paramters exist within logical physics bounds
        :param cfg:
        """
        super(MeanRangeBoundLoss, self).__init__()
        log.info("Initializing MeanRangeBoundLoss")
        self.cfg = cfg
        self.lb = torch.tensor(self.cfg.lb)
        self.ub = torch.tensor(self.cfg.ub)
        self.factor = torch.tensor(self.cfg.factor)
        log.info(f"Factor: {self.factor}")
        log.info(f"lb: {self.cfg.lb}")
        log.info(f"ub: {self.cfg.ub}")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        The loss function. This determines if there are parameters outside of the
        upper and lower bounds set in the cfg file. The loss is then averaged, and
        reported back
        :param inputs: parameter values
        :return:
        """
        loss = 0
        for i in range(len(inputs)):
            lb = self.lb[i]
            ub = self.ub[i]
            upper_bound_loss = torch.relu(inputs[i] - ub)
            lower_bound_loss = torch.relu(lb - inputs[i])
            mean_loss = self.factor * (upper_bound_loss + lower_bound_loss).mean() / 2.0
            loss = loss + mean_loss
        return loss
