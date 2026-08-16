import logging

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig

from dMC.criterion.mean_range_bound_loss import MeanRangeBoundLoss
from dMC.experiments import Experiment
from dMC.experiments.writer import Writer

log = logging.getLogger(__name__)


class TrainModel(Experiment):
    def __init__(self, cfg: DictConfig, writer: Writer):
        super(TrainModel, self).__init__(cfg=cfg, writer=writer)

    def run(self, data_loader: torch.utils.data.DataLoader, physics_model: nn.Module):
        warmup = self.cfg.warmup
        prediction_ = None
        observation_ = None
        criterion = nn.MSELoss()
        range_bound_loss = MeanRangeBoundLoss(self.cfg)
        optimizer = torch.optim.Adam(
            physics_model.parameters(), lr=self.cfg.learning_rate
        )
        # writing the first epoch
        self.writer.write_parameters(
            ["n", "q_spatial"], [physics_model.n, physics_model.q_spatial], 0, None,
        )
        for epoch in range(1, self.cfg.epochs + 1):
            physics_model.epoch = epoch
            for data in data_loader:
                hydrofabric, observation_ = data
                prediction_ = physics_model(data)
            prediction = prediction_[..., warmup:].squeeze()
            observation = observation_[..., warmup:].squeeze()
            l1 = criterion(prediction, observation)
            l2 = range_bound_loss([physics_model.n, physics_model.q_spatial],)
            loss = l1 + l2
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            self.writer.write_parameters(
                ["n", "q_spatial"],
                [physics_model.n, physics_model.q_spatial],
                epoch,
                hydrofabric.attributes[..., 4].numpy(),
            )
            self.writer.write_metrics(prediction, observation, loss, epoch)
            if epoch % 5 == 0:
                prediction_np = prediction_.detach().numpy()
                np.save(
                    self.save_path / f"{self.cfg.name}_predictions_epoch_{epoch}.npy",
                    prediction_np,
                )
                torch.save(
                    physics_model,
                    self.save_path / f"{self.cfg.name}_epoch_{epoch}_model.pth",
                )
