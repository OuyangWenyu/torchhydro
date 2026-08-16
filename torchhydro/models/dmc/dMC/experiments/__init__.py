from abc import ABC, abstractmethod
import logging
from pathlib import Path

from omegaconf import DictConfig
import torch
import torch.nn as nn

from dMC.experiments.writer import Writer

log = logging.getLogger(__name__)


class Experiment(ABC):

    def __init__(self, cfg: DictConfig, writer: Writer) -> None:
        self.cfg = cfg
        self.writer = writer
        log.info(f"Training model: {self.cfg.name}")
        self.save_path = Path(self.cfg.save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def run(self, data_loader: torch.utils.data.DataLoader, model: nn.Module) -> None:
        """a method that runs your experiment"""
        raise NotImplementedError
