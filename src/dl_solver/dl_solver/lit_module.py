from typing import Any, Dict, Tuple

import pytorch_lightning as pl
import torch
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

        self.model = PatchNet(hparams)

        self.mse_loss = nn.MSELoss(reduction="mean")
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.model(x)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        loss = self.loss_function(self(x), y)
        self.log("train_loss", loss)
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        x, y = batch
        loss = self.loss_function(self(x), y)
        self.log("val_loss", loss)

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
        y_pred: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the combined loss of MSE for positions and CrossEntropy for classifications.

        Args:
            y_pred: a tuple containing
                    - pos_seq: Tensor of predicted positions, shape (B, num_pieces, 3)
                    - (row_logits, col_logits, rot_logits): each a Tensor of shape (B, num_pieces, num_classes)
            y: Tensor of true labels of shape (B, 12, 3)

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
        pos_seq = y_pred[..., :3]
        unique_indices = y_pred[..., 3].squeeze()
        max_rows, max_cols = self.hparams.puzzle_shape
        row_logits = y_pred[..., 4 : 4 + max_rows]
        col_logits = y_pred[..., 4 + max_rows : 4 + max_rows + max_cols]
        rot_logits = y_pred[..., 4 + max_rows + max_cols :]

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

        return total_loss

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
