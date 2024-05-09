from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from pytorch_lightning.loggers import TensorBoardLogger
from torch import Tensor, nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .config import Config, HyperParameters
from .patchnet import PatchNet


class LitJigsawModule(pl.LightningModule):
    config: Config
    hparams: HyperParameters

    model: PatchNet
    mse_loss: nn.Module
    ce_loss: nn.Module

    def __init__(self, config: Config, hparams: HyperParameters):
        super().__init__()

        self.config = config
        self.save_hyperparameters(hparams.model_dump())

        self.model = PatchNet(hparams).to(self.device)
        print(f"Moved {self.model.__class__.__name__} to {self.device}.")

        self.mse_loss = nn.MSELoss(reduction="mean")
        self.ce_loss = nn.CrossEntropyLoss()

        self.cached_sample: Optional[Tuple[Tensor, Tensor]] = None

        torch.set_float32_matmul_precision(self.config.matmul_precision)

    def forward(self, x: Tensor, y: Optional[Tensor] = None, *args, **kwargs) -> Tensor:
        return self.model(x, y if self.training else None)

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, y = batch
        y_pred = self(x, y)

        losses = self.loss_function(y_pred, y)
        self.log(
            "train_loss",
            losses["total_loss"],
            prog_bar=True,
            on_epoch=True,
            on_step=True,
        )
        self.log_dict(
            {f"train-loss/{k}": v for k, v in losses.items() if k != "total_loss"},
            on_epoch=True,
            on_step=True,
        )

        accuracies = self.compute_accuracy(y_pred[0], y)
        self.log(
            "tain_accuracy",
            accuracies["total_accuracy"],
            prog_bar=True,
            on_epoch=True,
            on_step=True,
        )
        self.log_dict(
            {
                f"train-accuracy/{k}": v
                for k, v in accuracies.items()
                if k != "total_accuracy"
            },
            on_epoch=True,
            on_step=True,
        )

        return losses["total_loss"]

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        x, y = batch
        y_pred = self(x, None)
        losses = self.loss_function(y_pred, y)
        self.log("val_loss", losses["total_loss"], prog_bar=True, on_epoch=True)
        self.log_dict(
            {f"val-loss/{k}": v for k, v in losses.items() if k != "total_loss"},
            on_epoch=True,
            on_step=False,
        )

        if (
            self.cached_sample is None
        ):  # and torch.rand(1) < 0.05:  # ~5% chance to cache
            sample_idx = np.random.randint(0, x.size(0))
            self.cached_sample = (
                x[sample_idx, ...].detach().cpu(),
                y_pred[0][sample_idx, ...].detach().cpu(),
            )

        accuracies = self.compute_accuracy(y_pred[0], y)
        self.log(
            "val_accuracy",
            accuracies["total_accuracy"],
            prog_bar=True,
            on_epoch=True,
            on_step=False,
        )
        self.log_dict(
            {
                f"val-accuracy/{k}": v
                for k, v in accuracies.items()
                if k != "total_accuracy"
            },
            on_step=False,
            on_epoch=True,
        )

        # Log position and rotation predictions as histograms
        num_rows, num_cols = self.hparams.puzzle_shape
        self.logger.experiment.add_histogram(
            "pos-hist/row", y_pred[0][:, 0], self.current_epoch, bins=num_rows
        )
        self.logger.experiment.add_histogram(
            "pos-hist/col", y_pred[0][:, 1], self.current_epoch, bins=num_cols
        )
        self.logger.experiment.add_histogram(
            "pos-hist/rot", y_pred[0][:, 2], self.current_epoch, bins=4
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

    def loss_function(
        self,
        y_pred: Tuple[Tensor, Tensor, Tuple[Tensor, Tensor, Tensor]],
        y: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Compute the combined loss of MSE for positions and CrossEntropy for classifications.

        Args:
            y_pred: a tuple containing
                pos_seq: Tensor[torch.float32] - (B, num_pieces, 3) [row_idx, col_idx, rotation]
                unique_indices: Tensor[torch.bool] - (B, num_pieces)
                logits: Tuple[Tensor[torch.float32], Tensor[torch.float32], Tensor[torch.float32]]
                    row_logits: Tensor[torch.float32] - (B, num_pieces, max_rows)
                    col_logits: Tensor[torch.float32] - (B, num_pieces, max_cols)
                    rotation_logits: Tensor[torch.float32] - (B, num_pieces, 3)
            y: Tensor of true labels of shape (B, 12, 3) [row~_idx, col_idx, rotation]

        Returns:
            Total loss as a scalar tensor.

        >>> total_loss
        tensor(5.2782, grad_fn=<AddBackward0>)
        >>> ce_loss_cols
        tensor(1.5307, grad_fn=<NllLoss2DBackward0>)
        >>> ce_loss_rot
        tensor(1.4689, grad_fn=<NllLoss2DBackward0>)
        >>> ce_loss_rows
        tensor(1.1953, grad_fn=<NllLoss2DBackward0>)
        >>> mse_loss_position
        tensor(1.0833)
        >>> unique_loss
        tensor(2.)
        """
        pos_seq, unique_indices, (row_logits, col_logits, rot_logits) = y_pred

        # Unpack true values, calculate MSE loss for row / col indices
        y_rows, y_cols, y_rot = y[:, :, 0], y[:, :, 1], y[:, :, 2]
        mse_loss_position = self.mse_loss(
            pos_seq[:, :, :2].float(), y[:, :, :2].float()
        )

        # Uniqeness Loss
        unique_loss = torch.mean(unique_indices.logical_not().float())

        # nn.CrossEntropyLoss expects y_pred.shape = (B, C, D) and y.shape = (B, D)
        ce_loss_rows = self.ce_loss(row_logits.permute(0, 2, 1), y_rows)
        ce_loss_cols = self.ce_loss(col_logits.permute(0, 2, 1), y_cols)
        ce_loss_rot = self.ce_loss(rot_logits.permute(0, 2, 1), y_rot)

        total_loss = (
            (ce_loss_rows + ce_loss_cols) * self.hparams.w_ce_pos_loss
            + ce_loss_rot * self.hparams.w_ce_rot_loss
            + mse_loss_position * self.hparams.w_mse_loss
            + unique_loss * self.hparams.w_unique_loss
        )

        return {
            "total_loss": total_loss,
            "ce_loss_rows": ce_loss_rows,
            "ce_loss_cols": ce_loss_cols,
            "ce_loss_rot": ce_loss_rot,
            "mse_loss_position": mse_loss_position,
            "unique_loss": unique_loss,
        }

    def compute_accuracy(self, y_pred: Tensor, y: Tensor) -> Dict[str, float]:
        row_preds, col_preds, rot_preds = (
            y_pred[:, :, 0],
            y_pred[:, :, 1],
            y_pred[:, :, 2],
        )
        correct_rows = row_preds == y[:, :, 0]
        correct_cols = col_preds == y[:, :, 1]
        correct_rots = rot_preds == y[:, :, 2]

        # Position accuracy: both row and column are correct
        correct_positions = correct_rows & correct_cols
        pos_acc = correct_positions.float().mean()

        # Total accuracy: row, column, and rotation all are correct
        correct_total = correct_positions & correct_rots
        total_acc = correct_total.float().mean()

        return {
            "row_accuracy": correct_rows.float().mean(),
            "col_accuracy": correct_cols.float().mean(),
            "rot_accuracy": correct_rots.float().mean(),
            "pos_accuracy": pos_acc,
            "total_accuracy": total_acc,
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
