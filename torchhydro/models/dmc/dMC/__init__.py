import logging

from injector import singleton
from omegaconf import DictConfig
import torch.nn as nn

from dMC.configuration import Configuration
from dMC.experiments import Experiment
from dMC.data.datasets import Dataset
from dMC.data.observations import Observations

log = logging.getLogger(__name__)


def configure(cfg: DictConfig):
    """
    Configuration function for dependency injection.

    :param cfg: Configuration object.
    :type cfg: DictConfig
    :return: Binder function.
    :rtype: function
    """
    package_paths = {
        "experiment": "dMC.experiments",
        "data": "dMC.data.datasets",
        "observations": "dMC.data.observations",
        "neural_network": "dMC.nn",
        "physics": "dMC.physics",
    }

    def _bind(binder):
        """
        Binds the Configuration to a singleton scope.

        :param binder: Binder object.
        """
        configuration = Configuration(cfg, package_paths)
        binder.bind(Configuration, to=configuration, scope=singleton)

    return _bind
