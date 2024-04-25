from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .album_transforms import AlbumTransforms
from .config import Config, HyperParameters
from .jigsaw_dataset import JigsawDataset


class LitJigsawDatamodule(pl.LightningDataModule):
    config: Config
    hparams: HyperParameters

    transforms: AlbumTransforms

    jigsaw_train: JigsawDataset
    jigsaw_val: JigsawDataset
    jigsaw_test: JigsawDataset

    def __init__(self, config: Config, hparams: HyperParameters):
        super().__init__()
        self.config = config
        self.save_hyperparameters(hparams.model_dump())

        self.transforms = AlbumTransforms(resize=self.hparams.segment_shape)

    def setup(self, stage: Optional[str] = None):
        match stage:
            case "fit":
                self.jigsaw_train = JigsawDataset(
                    self.config.paths.jigsaw_dir,
                    split="train",
                    puzzle_shape=self.hparams.puzzle_shape,
                    transforms=self.transforms,
                )
            case "validate":
                self.jigsaw_val = JigsawDataset(
                    self.config.paths.jigsaw_dir,
                    split="val",
                    puzzle_shape=self.hparams.puzzle_shape,
                    transforms=self.transforms,
                )
            case "test":
                self.jigsaw_test = JigsawDataset(
                    self.config.paths.jigsaw_dir,
                    split="test",
                    puzzle_shape=self.hparams.puzzle_shape,
                    transforms=self.transforms,
                )
            case _:
                raise ValueError(f"Unknown stage: {stage}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.jigsaw_train,
            batch_size=self.hparams.batch_size,
            pin_memory=self.config.pin_memory,
            num_workers=self.config.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.jigsaw_val,
            batch_size=self.hparams.batch_size,
            pin_memory=self.config.pin_memory,
            num_workers=self.config.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.jigsaw_test,
            batch_size=self.hparams.batch_size,
            pin_memory=self.config.pin_memory,
            num_workers=self.config.num_workers,
        )

    def prepare_data(self) -> None:
        for stage in ("fit", "validate", "test"):
            try:
                self.setup(stage)
                split = (
                    "train"
                    if stage == "fit"
                    else "val" if stage == "validate" else stage
                )
                ds: JigsawDataset = getattr(self, f"jigsaw_{split}")
                ds._refurb_df(is_save_df=False)

            except Exception as e:
                print(f"Error setting up {stage} dataset: {e}")
                continue
