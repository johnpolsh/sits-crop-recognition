#

import hydra
import lightning as L
import rootutils
import torch
from lightning import (
    Callback,
    Trainer
)
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from pathlib import Path
from typing import (
    Any,
    Dict,
    Optional
)

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.base_datamodule import BaseDataModule
from src.models.base_module import BaseModule
from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    instantiate_plugins,
    log_hyperparameters,
    task_wrapper,
)


_logger = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def train(cfg: DictConfig) -> tuple[Dict[str, Any], Dict[str, Any]]:
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    _logger.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: BaseDataModule = hydra.utils.instantiate(cfg.data)

    _logger.info(f"Instantiating model <{cfg.model._target_}>")
    model: BaseModule = hydra.utils.instantiate(cfg.model)

    _logger.info("Instantiating callbacks...")
    callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    _logger.info("Instantiating loggers...")
    logger: list[Logger] = instantiate_loggers(cfg.get("logger"))

    _logger.info("Instantiating plugins...")
    plugins: list = instantiate_plugins(cfg.get("plugins"))

    _logger.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
        plugins=plugins,
        )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        _logger.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    _logger.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    _logger.info("Saving model...")
    save_path = Path(cfg.paths.output_dir) / "model.pth"
    torch.save(model.net.state_dict(), save_path)

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train_mae.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    extras(cfg)
    metric_dict, _ = train(cfg)
    metric_value = get_metric_value(
        metric_dict=metric_dict,
        metric_name=cfg.get("optimized_metric")
        )

    return metric_value


if __name__ == "__main__":
    main()
