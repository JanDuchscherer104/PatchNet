from typing import Any, Dict

import pytorch_lightning as pl
import torch
from torch import nn

from .config import Config, HyperParameters


class LitModule(pl.LightningModule):
    config: Config
    hparams: HyperParameters

    def __init__(self, config: Config, hparams: HyperParameters):
        super().__init__()

        self.config = config
        self.save_hyperparameters(hparams.model_dump())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("val_loss", loss)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        return {
            "optimizer": optimizer,
            "lr_scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer),
        }
