import logging
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig
import torch
import torch.nn as nn

from dMC.experiments import Experiment
from dMC.experiments.writer import Writer

log = logging.getLogger(__name__)


class GenerateSynthetic(Experiment):
    def __init__(self, cfg: DictConfig, writer: Writer) -> None:
        super(GenerateSynthetic, self).__init__(cfg=cfg, writer=writer)

    def run(self, data_loader: torch.utils.data.DataLoader, physics_model: nn.Module):
        physics_model.eval()
        log.info(f"Running synthetic {self.cfg.name} forward")
        for data in data_loader:
            with torch.no_grad():
                output = physics_model(data)
        # Note the output here is sorted from lowest to highest
        output_df = pd.DataFrame(
            output.cpu().squeeze().transpose(0, 1), columns=self.cfg.output_cols
        )
        output_df.to_csv(self.save_path / f"{self.cfg.name}.csv")
        log.info(f"output median @ 4809: {output_df[4809].median()}")
        torch.save(physics_model, self.save_path / f"{self.cfg.name}_model.pth")
