import logging

import numpy as np
from omegaconf import DictConfig
import torch
import torch.nn as nn

from dMC.experiments import Experiment
from dMC.criterion.mean_range_bound_loss import MeanRangeBoundLoss
from dMC.criterion.monotonic_loss import MonotonicLoss
from dMC.experiments.writer import Writer

log = logging.getLogger(__name__)


class TrainModel(Experiment):
    def __init__(self, cfg: DictConfig, writer: Writer):
        super().__init__(cfg=cfg, writer=writer)

    def run(
        self, data_loader: torch.utils.data.DataLoader, physics_model: nn.Module
    ) -> None:
        warmup = self.cfg.warmup
        prediction_ = None
        observation_ = None
        criterion = nn.MSELoss()
        range_bound_loss = MeanRangeBoundLoss(self.cfg)
        monotonic_loss = MonotonicLoss(self.cfg)
        optimizer = torch.optim.Adam(
            physics_model.parameters(), lr=self.cfg.learning_rate
        )
        physics_model.neural_network.initialize_weights()
        for epoch in range(1, self.cfg.epochs + 1):
            physics_model.epoch = epoch
            for data in data_loader:
                hydrofabric, observation_ = data
                normalized_attributes = hydrofabric.normalized_attributes[
                    :, [0, 1, 2, 3, 4, 6, 7, 8]
                ]
                physics_model.neural_network(normalized_attributes)
                log.info(f"{self.cfg.save_path}: median n: {physics_model.n.mean().item()}")
                prediction_ = physics_model(data)
            prediction = prediction_[..., warmup:]
            observation = observation_[..., warmup:].to(self.cfg.device)
            areas = hydrofabric.attributes[..., 4].cpu().numpy()
            self.writer.integrated_grad(prediction[-1, -1], physics_model.neural_network.n, "n_integrated_grad",  epoch)
            self.writer.integrated_grad(prediction[-1, -1], physics_model.neural_network.q, "q_integrated_grad",  epoch)
            l1 = 0
            for i in range(observation_.shape[0]):
                l1 = l1 + criterion(prediction[i], observation[i])
            l2 = range_bound_loss(
                [physics_model.n, physics_model.q_spatial],
            )
            l3 = monotonic_loss(physics_model.n, areas)
            loss = l1 + l2 + l3
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            self.writer.write_parameters(
                ["n", "q_spatial"],
                [physics_model.n, physics_model.q_spatial],
                epoch,
                areas,
            )
            self.writer.write_metrics(
                prediction.squeeze(), observation.squeeze(), loss, epoch
            )
            if epoch % 5 == 0:
                prediction_np = prediction_.cpu().detach().numpy()
                np.save(
                    self.save_path / f"{self.cfg.name}_predictions_epoch_{epoch}.npy",
                    prediction_np,
                )
                torch.save(
                    physics_model,
                    self.save_path / f"{self.cfg.name}_epoch_{epoch}_model.pth",
                )
