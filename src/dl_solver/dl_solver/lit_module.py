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
    cross_entropy_loss: nn.Module

    def __init__(self, config: Config, hparams: HyperParameters):
        super().__init__()

        self.config = config
        self.save_hyperparameters(hparams.model_dump())

        self.model = PatchNet(self.hparams)

        self.mse_loss = nn.MSELoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        return {
            "optimizer": optimizer,
            "lr_scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer),
        }

    def loss_function(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Split into position and rotation parts
        y_position, y_rotation = y[:, :2], y[:, 2]
        y_pred_position, y_pred_rotation = y_pred[:, :2], y_pred[:, 2]

        # Combine the losses
        total_loss = nn.functional.mse_loss(
            y_pred_position, y_position
        ) + self.hparams.rotation_loss_weight * nn.functional.cross_entropy(
            y_pred_rotation, y_rotation
        )

        return total_loss
