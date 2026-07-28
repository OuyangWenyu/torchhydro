import logging

from injector import Module, provider
import torch
import torch.nn as nn

from dMC.configuration import Configuration
from dMC.data import DataLoader
from dMC.data.dates import Dates
from dMC.data.normalize.min_max import MinMax
from dMC.experiments import Experiment
from dMC.experiments.writer import Writer

log = logging.getLogger(__name__)


class Factory(Module):
    """
    Factory class for creating differentiable components.

    Methods:
        provide_dataset: Provides a DataLoader object.
        provide_experiment: Provides an Experiment object.
        provide_model: Provides a Physics Model object.
    """

    @provider
    def provide_dataset(
        self, configuration: Configuration
    ) -> torch.utils.data.DataLoader:
        """
        Provides a DataLoader object based on the configuration.

        :param configuration: Configuration object.
        :type configuration: Configuration
        :return: DataLoader object.
        :rtype: torch.utils.data.DataLoader
        """
        log.info("Creating Dataset")
        cfg_data = configuration.cfg.data
        dates = Dates(cfg_data)
        normalize = MinMax(cfg=cfg_data)
        data = configuration.import_(
            "data", cfg=cfg_data, dates=dates, normalize=normalize
        )
        observations = configuration.import_(
            "observations", cfg=cfg_data, dates=dates, normalize=normalize
        )
        data_loader = DataLoader(data, observations)
        return data_loader(cfg_data)

    @provider
    def provide_experiment(self, configuration: Configuration) -> Experiment:
        """
        Provides an Experiment object based on the configuration.

        :param configuration: Configuration object.
        :type configuration: Configuration
        :return: Experiment object.
        :rtype: Experiment
        """
        log.info("Creating Experiment")
        cfg_experiment = configuration.cfg.experiment
        writer = Writer(cfg_experiment)
        experiment = configuration.import_(
            "experiment", cfg=cfg_experiment, writer=writer
        )
        return experiment

    @provider
    def provide_model(self, configuration: Configuration) -> nn.Module:
        """
        Provides a Physics Model object based on the configuration.

        :param configuration: Configuration object.
        :type configuration: Configuration
        :return: Physics Model object.
        :rtype: nn.Module
        """
        cfg_model = configuration.cfg.model

        neural_network = configuration.import_("neural_network", cfg=cfg_model).to(cfg_model.device)
        log.info(f"Initialized {configuration.cfg.service_locator.neural_network}")
        physics_model = configuration.import_(
            "physics", cfg=cfg_model, neural_network=neural_network
        )
        log.info(f"Initialized {configuration.cfg.service_locator.physics}")
        return physics_model
