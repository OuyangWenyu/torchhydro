import logging
import time
import random
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

import hydra
from injector import inject, Injector
import numpy as np
from omegaconf import DictConfig
import torch.utils.data
import torch.nn as nn

from dMC import configure
from dMC.factory import Factory
from dMC.experiments import Experiment

log = logging.getLogger(__name__)


class ExperimentHandler:
    """
    Handles the execution of an experiment.

    Attributes:
        experiment (Experiment): The experiment to run.
        data_loader (torch.utils.data.DataLoader): Data loader for the experiment.
        physics_model (nn.Module): Physics model for the experiment.

    Methods:
        run: Executes the experiment.
    """

    @inject
    def __init__(
        self,
        data_loader: torch.utils.data.DataLoader,
        experiment: Experiment,
        physics_model: nn.Module,
    ):
        """
        Initializes the ExperimentHandler.

        :param data_loader: Data loader for the experiment.
        :type data_loader: torch.utils.data.DataLoader
        :param experiment: The experiment to run.
        :type experiment: Experiment
        :param physics_model: Physics model for the experiment.
        :type physics_model: nn.Module
        """
        self.experiment = experiment
        self.data_loader = data_loader
        self.physics_model = physics_model

    def run(self):
        """
        Executes the experiment.
        """
        self.experiment.run(self.data_loader, self.physics_model)


@hydra.main(
    version_base=None,
    config_path="conf/",
    config_name="global_settings",
)
def main(cfg: DictConfig) -> None:
    """
    Main function for running experiments.

    :param cfg: Configuration object.
    :type cfg: DictConfig
    :return: None
    """
    _set_seed(cfg)
    start = time.perf_counter()
    injector = Injector([configure(cfg), Factory])
    handler = injector.get(ExperimentHandler)
    handler.run()
    end = time.perf_counter()
    log.info(f"Experiment: {cfg.name} took {(end - start):.6f} seconds")


def _set_seed(cfg):
    torch.manual_seed(cfg.config.model.seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.set_device(cfg.device_num)
        torch.cuda.manual_seed(cfg.config.model.seed)
        torch.cuda.manual_seed_all(cfg.config.model.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(1)
    random.seed(0)


if __name__ == "__main__":
    main()
