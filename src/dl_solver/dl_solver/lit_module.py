from typing import Any, Dict, Literal, Optional, Tuple, Union
from warnings import warn

import mlflow
import numpy as np
import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .config import Config, HyperParameters
from .jigsaw_criteria import JigsawCriteria, RoundedPreds
from .patchnet import PatchNet


class LitJigsawModule(pl.LightningModule):
    config: Config
    hparams: HyperParameters
    model: PatchNet
    criteria: JigsawCriteria

    def __init__(self, config: Config, hparams: HyperParameters):
        super().__init__()

        self.config = config
        self.save_hyperparameters(hparams.model_dump())

        self.model = PatchNet(hparams).to(self.device)
        print(f"Moved {self.model.__class__.__name__} to {self.device}.")

        self.criteria = JigsawCriteria(config=config, hparams=hparams)

        self.cached_sample: Optional[Tuple[Tensor, Tensor]] = None

        torch.set_float32_matmul_precision(self.config.matmul_precision)

    def forward(self, x: Tensor, y: Optional[Tensor] = None, *args, **kwargs) -> Tensor:
        return self.model(x, y if self.training else None)

    def on_train_start(self) -> None:
        if self.config.is_mlflow:
            mlflow.start_run(
                run_name=self.config.mlflow_config.run_name,
                experiment_id=self.config.mlflow_config.experiment_id,
                tags={
                    "stage": "train",
                    "model": self.model.__class__.__name__,
                    "is_debug": self.config.is_debug,
                },
            )
        if self.config.is_gpu:
            if (model_device := self.model.parameters().__next__().device) != (
                device := self.device
            ) or model_device != torch.device("cuda"):
                warn(
                    f"Model device: {model_device} is different from LightningModule device: {device}."
                )

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, y = batch
        y_pred = self(x, y)

        criteria = self.criteria.forward(y_pred, y, self.global_step, "fit")
        self.log(
            "train_loss",
            criteria.losses["total_loss"],
            prog_bar=True,
            on_epoch=True,
            on_step=True,
        )
        self.log_dict(
            {
                f"train-loss/{k}": v
                for k, v in criteria.losses.items()
                if k != "total_loss"
            },
            on_epoch=True,
            on_step=True,
        )

        self.log(
            "tain_accuracy",
            criteria.accuracies["total_accuracy"],
            prog_bar=True,
            on_epoch=True,
            on_step=True,
        )
        self.log_dict(
            {
                f"train-accuracy/{k}": v
                for k, v in criteria.accuracies.items()
                if k != "total_accuracy"
            },
            on_epoch=True,
            on_step=True,
        )
        self.log_pred_histograms(criteria.rounded_preds, "fit")

        return criteria.losses["total_loss"]

    def on_train_end(self) -> None:
        if self.config.is_mlflow:
            mlflow.end_run()

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        x, y = batch
        y_pred = self(x, None)

        criteria = self.criteria.forward(y_pred, y, self.global_step, "validate")
        self.log(
            "val_loss", criteria.losses["total_loss"], prog_bar=True, on_epoch=True
        )
        self.log_dict(
            {
                f"val-loss/{k}": v
                for k, v in criteria.losses.items()
                if k != "total_loss"
            },
            on_epoch=True,
            on_step=False,
        )

        if (
            self.cached_sample is None
        ):  # and torch.rand(1) < 0.05:  # ~5% chance to cache
            sample_idx = np.random.randint(0, x.size(0))
            self.cached_sample = (
                x[sample_idx, ...].clone().detach().cpu(),
                y_pred[0][sample_idx, ...]
                .clone()
                .detach()
                .cpu()
                .round()
                .to(torch.int64),
            )

        self.log(
            "val_accuracy",
            criteria.accuracies["total_accuracy"],
            prog_bar=True,
            on_epoch=True,
            on_step=False,
        )
        self.log_dict(
            {
                f"val-accuracy/{k}": v
                for k, v in criteria.accuracies.items()
                if k != "total_accuracy"
            },
            on_step=False,
            on_epoch=True,
        )

        self.log_pred_histograms(criteria.rounded_preds, "validate")

    def log_pred_histograms(
        self, rounded_preds: RoundedPreds, stage: Literal["fit", "validate"]
    ) -> None:
        stage_prefix = "train" if stage == "fit" else "val"
        num_rows, num_cols = self.hparams.puzzle_shape
        self.logger.experiment.add_histogram(
            f"{stage_prefix}-pred-hist/row",
            rounded_preds["row_preds"],
            self.current_epoch,
            bins=num_rows,
        )
        self.logger.experiment.add_histogram(
            f"{stage_prefix}-pred-hist/col",
            rounded_preds["col_preds"],
            self.current_epoch,
            bins=num_cols,
        )
        self.logger.experiment.add_histogram(
            f"{stage_prefix}-pred-hist/rot",
            rounded_preds["rot_preds"],
            self.current_epoch,
            bins=4,
        )

    def on_validation_epoch_end(self):
        if self.cached_sample is not None:
            fig = self.trainer.datamodule.jigsaw_val.plot_sample(
                pieces_and_labels=(self.cached_sample), is_plot=False
            )
            self.logger.experiment.add_figure("val_sample", fig, self.current_epoch)
            fig.savefig(
                self.config.paths.tb_logs
                / self.config.mlflow_config.run_name
                / f"val_sample{self.current_epoch}.png"
            )
            plt.show()
            plt.close(fig)
            self.cached_sample = None
        return super().on_validation_epoch_end()

    def configure_optimizers(
        self,
    ) -> Dict[str, Union[torch.optim.Optimizer, Dict[str, Any]]]:
        # Sets up an optimizer and learning rate scheduler for the model.
        # This allows finer control over how each part of the model is updated during training. Specifically:
        # - Uses a lower learning rate for the pre-trained backbone CNN.
        # - Uses a standard learning rate for the transformer and newly added classifier layers on the CNN.
        # - Employs the ReduceLROnPlateau scheduler, adjusting the learning rate based on the validation loss.

        # Returns:
        #     A tuple containing the optimizer and a lr_scheduler config.

        # Optimizer setup with different learning rates for different model components
        classifier_params = set(self.model.backbone.backbone.classifier.parameters())
        backbone_params = [
            param
            for param in self.model.backbone.backbone.parameters()
            if param not in classifier_params
        ]

        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": self.hparams.lr_backbone},
                {
                    "params": self.model.backbone.backbone.classifier.parameters(),
                    "lr": self.hparams.lr_classifier,
                },
                {
                    "params": self.model.transformer.parameters(),
                    "lr": self.hparams.lr_transformer,
                },
                {
                    "params": self.model.classifier.parameters(),
                    "lr": self.hparams.lr_classifier,
                },
            ],
            weight_decay=self.hparams.weight_decay,
        )

        # Scheduler setup
        lr_scheduler_config = {
            "scheduler": ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, patience=3, verbose=True
            ),
            "interval": "epoch",
            "monitor": "val_loss",
            "frequency": 1,
        }

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_config,
        }

    def make_graph(self, xy: Tuple[Tensor, Tensor]) -> None:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter()
        writer.add_graph(
            self.model, input_to_model=xy[0], verbose=True, use_strict_trace=False
        )
        writer.close()

    def torchviz_model(self, xy: Tuple[Tensor, Tensor]) -> None:
        from torchviz import make_dot

        make_dot(self.model(xy[0]), params=dict(self.model.named_parameters())).render(
            self.model.__class__.__name__,
            directory=self.config.paths.model_viz_dir,
            format="svg",
            cleanup=True,
        )
