#

import os
import tempfile
import yaml
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.loggers.mlflow import MLFlowLogger
from lightning.fabric.loggers.logger import rank_zero_experiment
from lightning.fabric.utilities.rank_zero import rank_zero_only
from lightning.pytorch.loggers.utilities import _scan_checkpoints
from mlflow.tracking import MlflowClient
from pathlib import Path
from torch import Tensor
from typing import Any, Literal, Optional
from typing_extensions import override
from ..utils.pylogger import RankedLogger


_logger = RankedLogger(__name__, rank_zero_only=True)

# TODO: add parameters to set experiment description and run tags, description, etc.
class MLFlowLoggerCustom(MLFlowLogger):
    def __init__(
            self,
            experiment_name: str = "lightning_logs",
            run_name: Optional[str] = None,
            tracking_uri: Optional[str] = os.getenv("MLFLOW_TRACKING_URI"),
            tags: Optional[dict[str, Any]] = None,
            experiment_description: Optional[str] = None,
            save_dir: Optional[str] = "./mlruns",
            log_model: Literal[True, False, "all"] = False,
            prefix: str = "",
            artifact_location: Optional[str] = None,
            run_id: Optional[str] = None,
            synchronous: Optional[bool] = None,
            close_after_fit: bool = True
            ):
        super().__init__(
            experiment_name=experiment_name,
            run_name=run_name,
            tracking_uri=tracking_uri,
            tags=tags,
            save_dir=save_dir,
            log_model=log_model,
            prefix=prefix,
            artifact_location=artifact_location,
            run_id=run_id,
            synchronous=synchronous
            )
        self._close_after_fit = close_after_fit

        self.experiment_tags = {
            "mlflow.note.content": experiment_description,
        } if experiment_description else {}

        if self.tags is not None and self.tags.get("description"):
            self.tags["mlflow.note.content"] = self.tags.pop("description")
        self.tags = { k: v for k, v in self.tags.items() if v is not None }

    def _make_api_status(self, status: str) -> str:
        if status == "success":
            return "FINISHED"
        elif status == "failed":
            return "FAILED"
        elif status == "aborted":
            return "KILLED"
        else:
            return "SCHEDULED"

    def _terminate_run(self, status: str):
        status = self._make_api_status(status)
        self.experiment.set_terminated(self.run_id, status)

    def _check_run_and_terminate(self, status: str):
        if self.experiment.get_run(self.run_id):
            self._terminate_run(status)

    def _scan_and_log_checkpoints(self, checkpoint_callback: ModelCheckpoint) -> None:
        checkpoints = _scan_checkpoints(checkpoint_callback, self._logged_model_time)
        for t, p, s, tag in checkpoints:
            metadata = {
                "score": s.item() if isinstance(s, Tensor) else s,
                "original_filename": Path(p).name,
                "Checkpoint": {
                    k: getattr(checkpoint_callback, k)
                    for k in [
                        "monitor",
                        "mode",
                        "save_last",
                        "save_top_k",
                        "save_weights_only",
                        "_every_n_train_steps",
                        "_every_n_val_epochs",
                    ]
                    if hasattr(checkpoint_callback, k)
                },
            }
            aliases = ["latest", "best"] if p == checkpoint_callback.best_model_path else ["latest"]

            artifact_path = f"checkpoints/{Path(p).stem}"
            self.experiment.log_artifact(self._run_id, p, artifact_path)

            with tempfile.TemporaryDirectory(prefix="test", suffix="test", dir=os.getcwd()) as tmp_dir:
                with open(f"{tmp_dir}/metadata.yaml", "w") as tmp_file_metadata:
                    yaml.dump(metadata, tmp_file_metadata, default_flow_style=False)

                with open(f"{tmp_dir}/aliases.txt", "w") as tmp_file_aliases:
                    tmp_file_aliases.write(str(aliases))

                self.experiment.log_artifacts(self._run_id, tmp_dir, artifact_path)

            self._logged_model_time[p] = t

    @property
    @rank_zero_experiment
    def experiment(self) -> MlflowClient:
        if not self._initialized:
            experiment = super().experiment
            
            for key, value in self.experiment_tags.items():
                experiment.set_experiment_tag(self._experiment_id, key, value)

            for key, value in self.tags.items():
                experiment.set_tag(self._run_id, key, value)

        return self._mlflow_client

    @override
    @rank_zero_only
    def finalize(self, status: str = "success"):
        if not self._initialized:
            return
        
        # log ckpts as artifacts
        if self._checkpoint_callback:
            self._scan_and_log_checkpoints(self._checkpoint_callback)

        if self._close_after_fit:
            self._check_run_and_terminate(status)
