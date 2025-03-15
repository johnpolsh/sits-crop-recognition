#

import mlflow
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.loggers.mlflow import MLFlowLogger
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from mlflow.models.model import ModelInfo
from pathlib import Path
from torch import Tensor, load
from typing import Any, Callable, Literal, Optional, Union
from .loggers import MLFlowLoggerCustom
from ..utils.pylogger import RankedLogger


_logger = RankedLogger(__name__, rank_zero_only=True)


def _parse_input_example_array(pl_module: LightningModule):
    example_input_array = pl_module.example_input_array

    def parse_data(data):
        if isinstance(data, Tensor):
            return data.numpy()
        elif isinstance(data, (list, tuple)):
            return (parse_data(t) for t in data)
        elif isinstance(data, dict):
            return { k: parse_data(v) for k, v in data.items() }
        else:
            return data
    
    return parse_data(example_input_array)


class MLFlowLogModel(Callback):
    def __init__(
            self,
            artifact_path: str = "model",
            flavor: Literal["pytorch", "onnx"] = "onnx",
            log_model: Literal["best_ckpt", "last"] = "best_ckpt",
            with_input_example: bool = False
            ):
        assert flavor in ["pytorch", "onnx"],\
            f"Flavor must be one of ['pytorch', 'onnx'], got {flavor}"
        assert log_model in ["best_ckpt", "last"],\
            f"Log model must be one of ['best_ckpt', 'last'], got {log_model}"
        self.artifact_path = artifact_path
        self.flavor = flavor
        self.log_model = log_model
        self.with_input_example = with_input_example
        self._mlflow_logger = None

    def _log_pytorch_model(
            self,
            model: LightningModule
            ) -> ModelInfo:
        info = mlflow.pytorch.log_model(
            model,
            artifact_path=self.artifact_path,
            input_example=_parse_input_example_array(model) if self.with_input_example else None
            )
            
        return info

    def _log_onnx_model(
            self,
            model: LightningModule
            ) -> ModelInfo:
        import onnx # type: ignore
        output_path = Path(model.trainer.default_root_dir)
        model.save_onnx(output_path)
        onnx_model = onnx.load(output_path / "model.onnx")
        onnx.checker.check_model(onnx_model)

        info = mlflow.onnx.log_model(
            onnx_model,
            artifact_path=self.artifact_path,
            input_example=_parse_input_example_array(model) if self.with_input_example else None,
            save_as_external_data=False # Keep weights in model file
            )
        
        return info

    def _load_best_ckpt(
            self,
            trainer: Trainer,
            pl_module: LightningModule
            ):
        best_ckpt: str = trainer.checkpoint_callback.best_model_path # type: ignore
        if best_ckpt:
            model_ckpt = load(best_ckpt)
            pl_module.load_state_dict(model_ckpt["state_dict"], strict=False)
        else:
            _logger.info("No best ckpt found, using current model weights.")

    def setup(self, trainer, pl_module, stage):
        assert isinstance(trainer.logger, MLFlowLoggerCustom),\
            "Requires MLFlowLogger to be used."
        if self.with_input_example:
            assert hasattr(pl_module.net, "example_input_array"),\
                "Model must have an `example_input_array` attribute."
        assert hasattr(pl_module, "save_onnx"),\
            "Model must have a method `save_onn`."

    def on_fit_end(self, trainer, pl_module):
        mlflow_logger: MLFlowLoggerCustom = trainer.logger # type: ignore
        _logger.info(f"Logging {self.log_model} model to MLFlow with flavor '{self.flavor}'")

        with mlflow.start_run(run_id=mlflow_logger.run_id):
            try:
                if self.log_model == "best_ckpt":
                    self._load_best_ckpt(trainer, pl_module)

                if self.flavor == "pytorch":
                    info = self._log_pytorch_model(pl_module)
                elif self.flavor == "onnx":
                    info = self._log_onnx_model(pl_module)

                mlflow_logger._check_run_and_terminate("success")
                _logger.info(f"Model logged to uri: {info.model_uri}") # type: ignore
            except Exception as e:
                _logger.error(f"Failed to log model to MLFlow: {e}")
                raise e


def _error_fig_generator(pl_module, batch, outputs):
    raise NotImplementedError("No figure generator provided.")


class LogBatchImages(Callback):
    def __init__(
            self,
            mode: Literal["figure", "image"] = "figure",
            num_samples: int = 3,
            handler: dict[Literal["train", "val", "test"], list[Callable]] = {}
            ):
        self.mode = mode
        self.num_samples = num_samples
        self.handler = handler
        self._stage = None

    def _mlflow_log(
            self,
            logger: MLFlowLogger,
            data: Any,
            name: Optional[str] = None,
            step: int = 0,
            global_step: str = ""
            ):
        if self.mode == "image":
            if isinstance(data, Tensor):
                data = data.numpy()
            logger.experiment.log_image(
                logger.run_id,
                image=data,
                artifact_file=name if name else None,
                key=f"{self._stage}-plot" if not name else None,
                step=step if not name else None
                )
        elif self.mode == "figure":
            logger.experiment.log_figure(
                logger.run_id,
                figure=data,
                artifact_file="figures/" + (name or f"{self._stage}-{step}-{global_step}.png")
                )

    def _tensorboard_log(
            self,
            logger: TensorBoardLogger,
            data: Any,
            name: Optional[str] = None,
            step: int = 0,
            global_step: Optional[int] = None
            ):
        if self.mode == "image":
            logger.experiment.add_image(
                f"{self._stage}/{name or ('image-' + str(step))}",
                data,
                global_step=global_step
                )
        elif self.mode == "figure":
            logger.experiment.add_figure(
                f"{self._stage}/{name or ('figure-' + str(step))}",
                data,
                global_step=global_step
                )

    def _log_samples(
            self,
            trainer: Trainer,
            pl_module: LightningModule,
            batch,
            outputs,
            batch_idx,
            fig_generator: Callable
            ):
        data: Union[Any, tuple[Any, str]] = fig_generator(
            pl_module,
            batch,
            outputs,
            batch_idx
            )
        if not isinstance(data, tuple):
            name = None
        else:
            data, name = data

        trainer_logger = trainer.logger
        if isinstance(trainer_logger, MLFlowLogger):
            self._mlflow_log(
                trainer_logger,
                data,
                name,
                batch_idx,
                str(pl_module.current_epoch)
                )
        elif isinstance(trainer_logger, TensorBoardLogger):
            self._tensorboard_log(
                trainer_logger,
                data,
                name,
                batch_idx,
                pl_module.current_epoch
                )
    
    def setup(self, trainer, pl_module, stage):
        if not isinstance(trainer.logger, (MLFlowLogger, TensorBoardLogger)):
            _logger.warning("No supported logger found. Won't log images!")

    def on_train_batch_end(
            self,
            trainer,
            pl_module,
            outputs,
            batch,
            batch_idx,
            dataloader_idx = 0
            ):
        generators = self.handler.get("train")
        if batch_idx < self.num_samples and generators:
            self._stage = "train"
            for generator in generators:
                self._log_samples(trainer, pl_module, batch, outputs, batch_idx, generator)

    def on_test_batch_end(
            self,
            trainer,
            pl_module,
            outputs,
            batch,
            batch_idx,
            dataloader_idx = 0
            ):
        generators = self.handler.get("test")
        if batch_idx < self.num_samples and generators:
            self._stage = "test"
            for generator in generators:
                self._log_samples(trainer, pl_module, batch, outputs, batch_idx, generator)

    def on_validation_batch_end(
            self,
            trainer,
            pl_module,
            outputs,
            batch,
            batch_idx,
            dataloader_idx = 0
            ):
        if trainer.sanity_checking:
            return

        generators = self.handler.get("val")
        if batch_idx < self.num_samples and generators:
            self._stage = "val"
            for generator in generators:
                self._log_samples(trainer, pl_module, batch, outputs, batch_idx, generator)
            