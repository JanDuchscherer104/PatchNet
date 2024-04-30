from typing import Any, Dict, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn

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

        self.mse_loss = nn.MSELoss(reduction="mean")
        self.ce_loss = nn.CrossEntropyLoss()

        self.cached_sample: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        torch.set_float32_matmul_precision(self.config.matmul_precision)

    def forward(
        self, x: torch.Tensor, y: Optional[torch.Tensor] = None, *args, **kwargs
    ) -> torch.Tensor:
        return self.model(x, y if self.training else None)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
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
            {f"train/loss/{k}": v for k, v in losses.items() if k != "total_loss"},
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
                f"train/accuracy/{k}": v
                for k, v in accuracies.items()
                if k != "total_accuracy"
            },
            on_epoch=True,
            on_step=True,
        )

        return losses["total_loss"]

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        y_pred = self(x, None)
        y = y
        losses = self.loss_function(y_pred, y)
        self.log("val_loss", losses["total_loss"], prog_bar=True, on_epoch=True)
        self.log_dict(
            {f"val/loss/{k}": v for k, v in losses.items() if k != "total_loss"},
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
                f"val/accuracy/{k}": v
                for k, v in accuracies.items()
                if k != "total_accuracy"
            },
            on_step=False,
            on_epoch=True,
        )

    def on_validation_epoch_end(self):
        if self.cached_sample is not None:
            tb_logger = self.logger.experiment
            if isinstance(tb_logger, TensorBoardLogger):
                fig = self.trainer.datamodule.jigsaw_val.plot_sample(
                    pieces_and_labels=(self.cached_sample), is_plot=False
                )
                tb_logger.experiment.add_figure("val_sample", fig, self.current_epoch)
                fig.savefig(
                    self.config.paths.tb_logs
                    / self.config.mlflow_config.run_name
                    / f"val_sample{self.current_epoch}.png"
                )
                plt.close(fig)
            self.cached_sample = None
        return super().on_validation_epoch_end()

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        optimizers = {
            "optimizer": optimizer,
        }
        if self.config.is_lr_scheduler:
            optimizers["lr_scheduler"] = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=0.1, patience=3
                ),
                "interval": "epoch",
                "monitor": "val_loss",
                "frequency": 1,
            }
        return optimizers

    def loss_function(
        self,
        y_pred: Tuple[
            torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ],
        y: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the combined loss of MSE for positions and CrossEntropy for classifications.

        Args:
            y_pred: a tuple containing
                pos_seq: torch.Tensor[torch.float32] - (B, num_pieces, 3) [row_idx, col_idx, rotation]
                unique_indices: torch.Tensor[torch.bool] - (B, num_pieces)
                logits: Tuple[torch.Tensor[torch.float32], torch.Tensor[torch.float32], torch.Tensor[torch.float32]]
                    row_logits: torch.Tensor[torch.float32] - (B, num_pieces, max_rows)
                    col_logits: torch.Tensor[torch.float32] - (B, num_pieces, max_cols)
                    rotation_logits: torch.Tensor[torch.float32] - (B, num_pieces, 3)
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
        unique_loss = 2 * torch.mean(unique_indices.logical_not().float())

        # nn.CrossEntropyLoss expects y_pred.shape = (B, C, D) and y.shape = (B, D)
        ce_loss_rows = self.ce_loss(row_logits.permute(0, 2, 1), y_rows)
        ce_loss_cols = self.ce_loss(col_logits.permute(0, 2, 1), y_cols)
        ce_loss_rot = self.ce_loss(rot_logits.permute(0, 2, 1), y_rot)

        total_loss = (
            ce_loss_rows + ce_loss_cols + ce_loss_rot + mse_loss_position + unique_loss
        )

        return {
            "total_loss": total_loss,
            "ce_loss_rows": ce_loss_rows,
            "ce_loss_cols": ce_loss_cols,
            "ce_loss_rot": ce_loss_rot,
            "mse_loss_position": mse_loss_position,
            "unique_loss": unique_loss,
        }

    def compute_accuracy(
        self, y_pred: torch.Tensor, y: torch.Tensor
    ) -> Dict[str, float]:
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

    def make_graph(self, xy: Tuple[torch.Tensor, torch.Tensor]) -> None:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter()
        writer.add_graph(
            self.model, input_to_model=xy[0], verbose=True, use_strict_trace=False
        )
        writer.close()

    def torchviz_model(self, xy: Tuple[torch.Tensor, torch.Tensor]) -> None:
        from torchviz import make_dot

        make_dot(self.model(xy[0]), params=dict(self.model.named_parameters())).render(
            self.model.__class__.__name__,
            directory=self.config.paths.model_viz_dir,
            format="svg",
            cleanup=True,
        )
