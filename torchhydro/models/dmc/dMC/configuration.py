import importlib
import logging
from typing import Dict

from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


class Configuration:
    """
    Configuration class for the application.

    Attributes:
        package_paths (Dict): Mapping of keys to package paths.
        cfg (DictConfig): Configuration object.
        service_locator (Dict): Dictionary of instantiated classes.

    Methods:
        _import_and_instantiate: Imports and instantiates classes based on the configuration.
    """

    def __init__(self, cfg: DictConfig, package_paths: Dict):
        """
        Initializes the Configuration.

        :param cfg: Configuration object.
        :type cfg: DictConfig
        """
        _set_device(cfg)
        self.cfg = cfg.config
        self.package_paths = package_paths
        self.service_locator = self._import_and_instantiate()

    def _import_and_instantiate(self):
        """
        Imports and instantiates classes based on the configuration.

        :return: Dictionary of instantiated classes.
        """
        instances = {}
        for key, value in self.cfg.service_locator.items():
            package_path = self.package_paths.get(key)
            if not package_path:
                log.warning(f"No package path found for key: {key}")
                continue
            module_str, class_str = value.rsplit(".", 1)
            full_module_str = f"{package_path}.{module_str}"
            try:
                module = importlib.import_module(full_module_str)
                class_ = getattr(module, class_str)
                instances[key] = class_
            except ImportError as e:
                log.error(f"Could not import {full_module_str}: {e}")
                instances[key] = None
            except AttributeError as e:
                log.error(
                    f"Could not find class {class_str} in module {full_module_str}: {e}"
                )
                instances[key] = None
        return instances

    def import_(self, class_ref, **kwargs):
        """
        Instantiates a class with given kwargs.

        :param class_ref: The class reference to instantiate.
        :param kwargs: Keyword arguments to pass to the class during instantiation.
        :return: An instance of the class or None if instantiation fails.
        """
        try:
            log.info(f"Class ref: {self.cfg.service_locator[class_ref]}")
            return self.service_locator[class_ref](**kwargs)
        except KeyError:
            log.error(f"No reference to {class_ref}, using an empty Protocol")
            raise KeyError
        except TypeError:
            log.error(f"No class found in the service locator. Check cfg file or dictionary pathing")
            raise TypeError


def _set_device(cfg):
    """
    Sets our device
    :return:
    """
    OmegaConf.set_struct(cfg, False)
    cfg.config.model.device = cfg.device
    cfg.config.experiment.device = cfg.device
    cfg.config.data.device = cfg.device
    OmegaConf.set_struct(cfg, True)
