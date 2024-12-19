from typing import List

import hydra
from lightning import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config.

    :param callbacks_cfg: A DictConfig object containing callback configurations.
    :return: A list of instantiated callbacks.
    """
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    """Instantiates loggers from config.

    :param logger_cfg: A DictConfig object containing logger configurations.
    :return: A list of instantiated loggers.
    """
    logger: List[Logger] = []

    if not logger_cfg:
        log.warning("No logger configs found! Skipping...")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger


def instantiate_plugins(plugins_cfg: DictConfig) -> List:
    """Instantiates plugins from config.

    :param plugins_cfg: A DictConfig object containing plugin configurations.
    :return: A list of instantiated plugins.
    """
    plugins: List = []

    if not plugins_cfg:
        log.warning("No plugin configs found! Skipping...")
        return plugins

    if not isinstance(plugins_cfg, DictConfig):
        raise TypeError("Plugins config must be a DictConfig!")

    for _, pl_conf in plugins_cfg.items():
        if isinstance(pl_conf, DictConfig) and "_target_" in pl_conf:
            log.info(f"Instantiating plugin <{pl_conf._target_}>")
            plugins.append(hydra.utils.instantiate(pl_conf))

    return plugins[0] if len(plugins) == 1 else plugins
