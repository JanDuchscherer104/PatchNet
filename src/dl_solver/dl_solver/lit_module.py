from typing import Any, Dict, Literal, Optional, Tuple, Union
from warnings import warn

import mlflow
import numpy as np
import optuna
import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from .config import Config
from .hparams import HParams
from .jigsaw_criteria import JigsawCriteria, RoundedPreds
from .patchnet import PatchNet


class LitJigsawModule(pl.LightningModule):
    config: Config
    hparams: HParams
    model: PatchNet
    criteria: JigsawCriteria

    def __init__(self, config: Config, hparams: HParams):
        super().__init__()

        self.config = config
        self.save_hyperparameters(hparams.model_dump())

        self.model = PatchNet(hparams).to(self.device)
        print(f"Moved {self.model.__class__.__name__} to {self.device}.")

        self.criteria = JigsawCriteria(config=config, hparams=hparams)

    def forward(self, x: Tensor, y: Optional[Tensor] = None, *args, **kwargs) -> Tensor:
        return self.model(x, y if self.training else None)

    def on_train_start(self) -> None:
        self.criteria.max_num_steps = (
            self.get_num_steps_per_epoch() * self.trainer.max_epochs
        )
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
            ) or model_device != torch.device("cuda:0"):
                warn(
                    f"Model device: {model_device} is different from LightningModule device: {device}."
                )

    def on_train_end(self) -> None:
        if self.config.is_mlflow:
            mlflow.end_run()

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
            "train_accuracy",
            criteria.accuracies["total_accuracy"],
            prog_bar=True,
            on_epoch=False,
            on_step=True,
        )
        self.log_dict(
            {
                f"train-accuracy/{k}": v
                for k, v in criteria.accuracies.items()
                if k != "total_accuracy"
            },
            on_epoch=False,
            on_step=True,
        )

        self.log_pred_histograms(criteria.rounded_preds, "fit")

        self.log("sigma_unique", self.criteria.unique_cost_sigma, on_step=True)

        return criteria.losses["total_loss"]

    def on_train_end(self) -> None:
        if self.config.is_mlflow:
            mlflow.end_run()

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        x, y = batch
        y_pred = self(x, None)

        criteria = self.criteria.forward(y_pred, y, self.global_step, "validate")
        self.log_metrics(criteria, "val")

        if batch_idx == 0:
            sample_idx = np.random.randint(0, x.size(0))
            val_sample = (
                x[sample_idx, ...].clone().detach().cpu(),
                y_pred[1][sample_idx, ...]
                .clone()
                .detach()
                .cpu()
                .round()
                .to(torch.int64),
            )
            self.log_val_sample(val_sample)

    def log_metrics(
        self, criteria: JigsawCriteria.Criteria, stage: Literal["train", "val"]
    ) -> None:
        self.log(
            f"{stage}_loss",
            criteria.losses["total_loss"],
            prog_bar=True,
            on_epoch=True,
            on_step=stage == "train",
        )
        self.log_dict(
            {
                f"{stage}-loss/{k}": v
                for k, v in criteria.losses.items()
                if k != "total_loss"
            },
            on_epoch=True,
            on_step=stage == "train",
        )

        self.log(
            f"{stage}_accuracy",
            criteria.accuracies["total_accuracy"],
            prog_bar=True,
            on_epoch=True,
            on_step=stage == "train",
        )
        self.log_dict(
            {
                f"{stage}-accuracy/{k}": v
                for k, v in criteria.accuracies.items()
                if k != "total_accuracy"
            },
            on_epoch=True,
            on_step=stage == "train",
        )

        self.log_pred_histograms(criteria.rounded_preds, stage)

        if self.config.is_mlflow:
            # Log losses
            mlflow.log_metric(
                f"{stage}_loss", criteria.losses["total_loss"], step=self.current_epoch
            )
            for k, v in criteria.losses.items():
                if k != "total_loss":
                    mlflow.log_metric(f"{stage}-loss/{k}", v, step=self.current_epoch)

            # Log accuracies
            mlflow.log_metric(
                f"{stage}_accuracy",
                criteria.accuracies["total_accuracy"],
                step=self.current_epoch,
            )
            for k, v in criteria.accuracies.items():
                if k != "total_accuracy":
                    mlflow.log_metric(
                        f"{stage}-accuracy/{k}", v, step=self.current_epoch
                    )

    def log_val_sample(self, val_sample: Tuple[Tensor, Tensor]):
        fig = self.trainer.datamodule.jigsaw_val.plot_sample(
            pieces_and_labels=(val_sample), is_plot=False
        )
        self.logger.experiment.add_figure("val_sample", fig, self.current_epoch)
        fig.savefig(
            self.config.paths.tb_logs
            / self.config.mlflow_config.run_name
            / f"val_sample{self.current_epoch}.png"
        )
        plt.show()
        plt.close(fig)

    def log_pred_histograms(
        self, rounded_preds: RoundedPreds, stage: Literal["train", "val"]
    ) -> None:
        num_rows, num_cols = self.hparams.puzzle_shape
        self.logger.experiment.add_histogram(
            f"{stage}-pred-hist/row",
            rounded_preds["row_preds"],
            self.current_epoch,
            bins=num_rows,
        )
        self.logger.experiment.add_histogram(
            f"{stage}-pred-hist/col",
            rounded_preds["col_preds"],
            self.current_epoch,
            bins=num_cols,
        )
        self.logger.experiment.add_histogram(
            f"{stage}-pred-hist/rot",
            rounded_preds["rot_preds"],
            self.current_epoch,
            bins=4,
        )

    def configure_optimizers(
        self,
    ) -> Dict[str, Union[torch.optim.Optimizer, Dict[str, Any]]]:

        # backbone_embedding_params = list(
        #     self.model.backbone.backbone.classifier.parameters()
        # )
        # param_groups = [
        #         {
        #             "params": backbone_embedding_params,
        #             "lr": self.hparams.optimizer["lr_classifier"],

        #         },
        #         {
        #             "params": self.model.transformer.parameters(),
        #             "lr": self.hparams.optimizer["lr_transformer"],

        #         },
        #         {
        #             "params": self.model.classifier.parameters(),
        #             "lr": self.hparams.optimizer["lr_classifier"],
        #         },
        # ]

        # if self.hparams.backbone["is_trainable"]:
        #     param_groups.append(
        #         {"params":
        #             [param
        #             for param in self.model.backbone.backbone.parameters()
        #             if param not in set(backbone_embedding_params)],
        #         "lr": self.hparams.optimizer["lr_backbone"],
        #         "weight_decay": self.hparams.optimizer["weight_decay"] * 1e-2
        #         }
        #     )

        optimizer = AdamW(
            params=self.model.parameters(),
            weight_decay=self.hparams.optimizer["weight_decay"],
            lr=self.hparams.optimizer["lr_transformer"],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": OneCycleLR(
                    optimizer,
                    max_lr=self.hparams.optimizer["lr_max"],
                    epochs=self.trainer.max_epochs,
                    steps_per_epoch=self.get_num_steps_per_epoch(),
                    pct_start=0.3,  # 30% of the cycle for increasing the learning rate
                    anneal_strategy="cos",  # Cosine annealing
                    cycle_momentum=True,  # Cycles momentum inversely to learning rate
                    base_momentum=0.85,  # Lower boundary of momentum
                    max_momentum=0.95,  # Upper boundary of momentum
                    div_factor=25.0,  # Initial learning rate = max_lr/div_factor
                    final_div_factor=1e4,  # Minimum learning rate = initial_lr/final_div_factor
                ),
                "interval": "step",
                "frequency": 1,
                "monitor": "train_loss",
            },
        }

    def get_num_steps_per_epoch(self) -> int:
        if not hasattr(self, "__num_steps_per_epoch"):
            self.__num_steps_per_epoch = len(self.trainer.datamodule.train_dataloader())
        return self.__num_steps_per_epoch

    @staticmethod
    def optuna_sweep(config: Config, hparams: HParams):

        config.optuna_config.study.optimize(
            func=lambda trial: LitJigsawModule.optuna_objective(
                trial=trial, config=config.model_copy(), hparams=hparams.model_copy()
            ),
            n_trials=config.optuna_config.n_trials,
            show_progress_bar=True,
            n_jobs=1,
        )

    @staticmethod
    def optuna_objective(trial: optuna.Trial, config: Config, hparams: HParams):
        from .lit_trainer_factory import TrainerFactory

        hparams.suggest_hparams(trial)
        trainer, module, datamodule = TrainerFactory.create_all(config, hparams)
        trainer.fit(model=module, datamodule=datamodule)

        loss = (
            loss.item()
            if (loss := trainer.callback_metrics.get(config.optuna_config.monitor))
            is not None
            else float("inf")
        )
        return loss

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
