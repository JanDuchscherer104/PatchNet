from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple
from warnings import warn

import mlflow
import torch
from optuna.trial import Trial
from optuna_integration import PyTorchLightningPruningCallback
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks import (
    BatchSizeFinder,
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.types import STEP_OUTPUT

from .config import Config
from .hparams import HParams
from .lit_datamodule import LitJigsawDatamodule
from .lit_module import LitJigsawModule


class TrainerFactory:
    @classmethod
    def create_trainer(
        cls,
        config: Config,
        hparams: HParams,
        trial: Optional[Trial],
        **trainer_kwargs,
    ) -> Trainer:
        """Create and initialize Callback instances."""
        assert isinstance(trial, Trial) == config.is_optuna

        torch.set_float32_matmul_precision(config.matmul_precision)

        factory_instance = cls(config, hparams, trial)

        callbacks = factory_instance._assemble_callbacks()
        tb_logger = factory_instance._assemble_loggers()

        if config.is_optuna:
            config.active_callbacks["OptunaPruning"] = True

        if config.is_debug:
            config.is_gpu = False
            config.is_fast_dev_run = True
            config.num_workers = 0
            config.is_multiproc = False
            config.verbose = True
            config.is_mlflow = False
            hparams.optimizer.batch_size = 8
            torch.autograd.set_detect_anomaly(True)

            config.active_callbacks["ModelCheckpoint"] = False
        # elif config.is_mlflow:
        #     mlflow.pytorch.autolog(
        #         log_every_n_epoch=1,
        #         log_every_n_step=None,
        #         log_models=True,
        #         log_datasets=False,
        #         disable=False,
        #         exclusive=False,
        #         disable_for_unsupported_versions=False,
        #         silent=False,
        #         registered_model_name=None,
        #         extra_tags=None,
        #         checkpoint=True,
        #         checkpoint_monitor="val_loss",
        #         checkpoint_mode="min",
        #         checkpoint_save_best_only=True,
        #         checkpoint_save_weights_only=False,
        #         checkpoint_save_freq="epoch",
        #     )

        # Create Trainer
        return Trainer(
            accelerator="auto" if config.is_gpu else "cpu",
            logger=tb_logger,
            callbacks=callbacks,
            max_epochs=config.max_epochs,
            default_root_dir=config.paths.root,
            fast_dev_run=config.is_fast_dev_run,
            log_every_n_steps=config.log_every_n_steps,
            enable_model_summary=not config.active_callbacks["ModelSummary"],
            **trainer_kwargs,
        )

    @classmethod
    def create_all(
        cls,
        config: Config,
        hparams: HParams,
        setup: List[Literal["fit", "validate", "test"]] = ["fit", "validate"],
        trial: Optional[Trial] = None,
        **trainer_kwargs,
    ) -> Tuple[Trainer, LitJigsawModule, LitJigsawDatamodule]:
        """Create and initialize Callback instances."""
        trainer = cls.create_trainer(config, hparams, trial, **trainer_kwargs)
        if isinstance(config.from_ckpt, Path):
            print(f"Loading model from checkpoint: {config.from_ckpt}")
            lit_module = LitJigsawModule.load_from_checkpoint(
                config.from_ckpt, config=config, hparams=hparams
            )
        else:
            lit_module = LitJigsawModule(config, hparams)
        lit_datamodule = LitJigsawDatamodule(config, hparams)
        if len(setup) > 0:
            for stage in setup:
                lit_datamodule.setup(stage)
                lit_module.setup(stage)
        return (
            trainer,
            lit_module,
            lit_datamodule,
        )

    def _get_callback_map(self) -> Dict[str, Callable]:
        return {
            "ModelCheckpoint": self._create_model_checkpoint,
            "TQDMProgressBar": self._create_tqdm_progress_bar,
            "EarlyStopping": self._create_early_stopping,
            "BatchSizeFinder": self._create_batch_size_finder,
            "LearningRateMonitor": self._create_lr_monitor,
            "ModelSummary": self._create_model_summary,
            "OptunaPruning": self._create_optuna_pruning,
        }

    def __init__(
        self,
        config: Config,
        hparams: HParams,
        trial: Optional[Trial] = None,
    ):
        """Private constructor to set config and hyper_params."""
        self.config = config
        self.hparams = hparams
        self.trial = trial

    def _create_model_checkpoint(self):
        return ModelCheckpoint(
            dirpath=self.config.paths.checkpoints,
            filename=f"{self.config.mlflow_config.run_name}-{{epoch:02d}}-{{val_loss:.2f}}",
            monitor="val_loss",
            verbose=self.config.verbose,
        )

    def _create_tqdm_progress_bar(self):
        return CustomTQDMProgressBar()

    def _create_lr_monitor(self):
        return LearningRateMonitor(logging_interval="step", log_momentum=True)

    def _create_early_stopping(self):
        return EarlyStopping(
            monitor="val_loss",
            patience=self.config.early_stopping_patience,
            verbose=self.config.verbose,
            mode="min",
        )

    def _create_optuna_pruning(self):
        if self.trial is not None:
            return PyTorchLightningPruningCallback(
                trial=self.trial,
                monitor=self.config.optuna_config.monitor,
            )

    def _create_model_summary(self):
        return ModelSummary(max_depth=4)

    def _create_batch_size_finder(self):
        return BatchSizeFinder(
            mode="binsearch",
            steps_per_trial=3,
            init_val=self.hparams.optimizer.batch_size,
            max_trials=25,
            batch_arg_name="batch_size",
        )

    def _assemble_loggers(self):
        return [
            TensorBoardLogger(
                save_dir=self.config.paths.tb_logs,
                name=self.config.mlflow_config.run_name,
            ),
        ]

    def _assemble_callbacks(self) -> List[Callback]:
        callbacks = []
        callback_map = self._get_callback_map()
        for key, is_active in self.config.active_callbacks.items():
            if is_active:
                create_callback = callback_map.get(key)
                if create_callback:
                    callback = create_callback()
                    if callback:
                        callbacks.append(callback)
                    else:
                        warn(f"Callback {key} could not be created.")
                else:
                    warn(f"No method found in TrainerFactory for key {key}.")

        return callbacks


class CustomTQDMProgressBar(TQDMProgressBar):
    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        description = f"train_loss: {trainer.callback_metrics.get('train_loss', 0):.2f}"
        self.train_progress_bar.set_postfix_str(description, refresh=True)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        super().on_validation_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )
        description = f"val_loss: {trainer.callback_metrics.get('val_loss', 0):.2f}"
        self.val_progress_bar.set_postfix_str(description, refresh=True)
