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
from typing import Optional
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
    task_wrapper,
)


_logger = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def train(cfg: DictConfig):
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    _logger.info("Instantiating callbacks...")
    callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    _logger.info("Instantiating plugins...")
    plugins: list = instantiate_plugins(cfg.get("plugins"))

    for k_shot in cfg.k_shots:
        _logger.info(f"Starting few-shot training with {k_shot} shots")

        _logger.info("Instantiating loggers...")
        logger: list[Logger] = instantiate_loggers(cfg.get("logger"))

        _logger.info(f"Instantiating model <{cfg.model._target_}>")
        model: BaseModule = hydra.utils.instantiate(cfg.model)

        _logger.info(f"Instantiating trainer <{cfg.trainer._target_}>")
        trainer: Trainer = hydra.utils.instantiate(
            cfg.trainer,
            callbacks=callbacks,
            logger=logger,
            plugins=plugins,
            )

        _logger.info(f"Instantiating datamodule <{cfg.data._target_}>")
        datamodule: BaseDataModule = hydra.utils.instantiate(cfg.data, k_shot=k_shot)

        if cfg.get("pretrained_model"):
            _logger.info(f"Loading pretrained model from {cfg.pretrained_model}")
            checkpoint = torch.load(cfg.pretrained_model, weights_only=True)
            encoder_state_dict = {
                k: v for k, v in checkpoint.items() if k.startswith("encoder.") and model.net.state_dict()[k].shape == v.shape
            }
            missing_keys, unexpected_keys = model.net.load_state_dict(encoder_state_dict, strict=False)
            _logger.debug(f"Encoder weights loaded. Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}")

        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    return None, None


@hydra.main(version_base="1.3", config_path="../configs", config_name="train_few_shot.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    extras(cfg)
    metric_dict = train(cfg)
    metric_value = get_metric_value(
        metric_dict=metric_dict,
        metric_name=cfg.get("optimized_metric")
        )

    return metric_value


if __name__ == "__main__":
    main()
