from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from pandarallel import pandarallel
from PIL import Image
from torch.utils.data import Dataset

from .album_transforms import AlbumTransforms

"""
TODO: mv utility methods to derived class in a separate file
TODO: check: are the IMAGE_NET_STATS RGB or BGR?
"""


class JigsawDataset(Dataset):
    IMAGENET_STATS = {
        "mean": np.array([0.485, 0.456, 0.406]),
        "std": np.array([0.229, 0.224, 0.225]),
    }

    dataset_dir: Path
    split: Literal["train", "val", "test"]
    is_train: bool
    puzzle_shape: Tuple[int, int]
    transforms: AlbumTransforms

    df: pd.DataFrame
    filtered_df: pd.DataFrame  # filtered by shape

    def __init__(
        self,
        dataset_dir: Path,
        split: Literal["train", "val", "test"],
        puzzle_shape: Tuple[int, int],
        transforms: AlbumTransforms,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.split = split
        self.is_train = split == "train"
        self.puzzle_shape = puzzle_shape
        self.transforms = transforms

        self.__df: Optional[pd.DataFrame] = None
        self.__filtered_df: Optional[pd.DataFrame] = None

    @property
    def pickle_file_path(self) -> Path:
        return self.dataset_dir / f"{self.split}_jigsaw.pkl"

    @property
    def df(self) -> pd.DataFrame:
        if self.__df is None:
            self.__df = pd.read_pickle(self.pickle_file_path)
        return self.__df

    @df.setter
    def df(self, value: Optional[pd.DataFrame]) -> None:
        assert value is None or isinstance(value, pd.DataFrame)
        self.__df = value

    @property
    def filtered_df(self) -> pd.DataFrame:
        if self.__filtered_df is None:
            rows, cols = self.puzzle_shape
            self.__filtered_df = self.df.query("rows == @rows and cols == @cols")
            self.df = None
        return self.__filtered_df

    def get_max_segment_shape(self) -> Tuple[int, int]:
        return self.df["max_width"].max(), self.df["max_height"].max()

    def get_min_segment_shape(self) -> Tuple[int, int]:
        return self.df["min_width"].min(), self.df["min_height"].min()

    def __len__(self) -> int:
        return len(self.filtered_df)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
            Tuple[
                X: torch.Tensor[torch.float32] - (num_pieces, 3, H, W)
                y: torch.Tensor[torch.int64] - (num_pieces, 3) [row_idx, col_idx, rotation]
                    row in {0, 1, ..., rows - 1}
                    col in {0, 1, ..., cols - 1}
                    rotation in {0, 1, 2, 3}
                    num_pieces = rows * cols
            ]
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.filtered_df.iloc[idx]
        class_dir = self.dataset_dir / "images" / self.split / row["class_id"]
        if self.split == "train":
            sample_dir = class_dir / str(row["num_sample"])
        else:
            sample_dir = class_dir / str(row["num_sample"]).zfill(8)

        # Load labels
        labels = np.load(sample_dir / "labels.npy")

        # TODO: load the puzzle pieces according to the id in labels!!
        # Load puzzle pieces
        puzzle_pieces = {
            f"piece_{i}": np.array(Image.open(sample_dir / f"piece_{i}.png"))
            for i in range(len(labels))
        }

        return self.transforms(puzzle_pieces, labels, self.is_train)

    def plot_sample(
        self,
        idx: Optional[int] = None,
        pieces_and_labels: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> None:

        if pieces_and_labels is not None:
            puzzle_pieces, labels = pieces_and_labels
        else:
            idx = idx or np.random.randint(0, len(self) - 1)
            puzzle_pieces, labels = self[idx]

        rows, cols = self.puzzle_shape

        _, axs = plt.subplots(
            rows,
            cols,
            figsize=(cols * 2, rows * 2),
        )

        # Plot each piece
        for label, puzzle_piece in zip(labels, puzzle_pieces):
            row, col, rotation = label
            img = puzzle_piece.numpy().transpose((1, 2, 0))

            # Undo normalization
            img = self.IMAGENET_STATS["std"] * img + self.IMAGENET_STATS["mean"]
            img = np.clip(img, 0, 1)
            img = Image.fromarray((img * 255).astype(np.uint8))

            img = img.rotate(-90 * rotation)  # undo the rotation
            img = np.array(img)
            axs[row, col].imshow(img)
            axs[row, col].axis("off")

        plt.subplots_adjust(wspace=0.0005, hspace=0.0005)
        plt.show()

    def _update_min_dimensions(self) -> None:
        import pandarallel

        pandarallel.pandarallel.initialize(progress_bar=True, nb_workers=16)

        def update_row(row: pd.Series) -> pd.Series:
            hdf5_filepath = (
                self.dataset_dir
                / "images"
                / row["class_id"]
                / f"{int(row['num_sample'])}.hdf5"
            )
            min_width = row["max_width"]
            min_height = row["max_height"]
            try:
                with h5py.File(hdf5_filepath, "r") as f:
                    for dataset_name, dataset in f.items():
                        if dataset_name.startswith("piece_"):
                            img = np.array(dataset)
                            row["min_width"] = min(min_width, img.shape[1])
                            row["min_height"] = min(min_height, img.shape[0])
            except Exception as e:
                # mark the row to be dropped
                row["drop"] = True
            return row

        self.df["drop"] = False
        df = self.df.parallel_apply(update_row, axis=1)

        print(f"Dropping {df['drop'].sum()} rows")
        self.df = df.query("drop == False").drop(columns=["drop"])
        self.df = self.df.reset_index(drop=True).astype(
            {
                "min_width": "int16",
                "min_height": "int16",
                "max_width": "int16",
                "max_height": "int16",
                "width": "int16",
                "height": "int16",
                "rows": "uint8",
                "cols": "uint8",
                "image_id": str,
                "class_id": str,
                "num_sample": "uint32",
            }
        )

    def _refurb_df(self, is_save_df: bool = False) -> None:
        # check if each file exists
        self.df = self.df.reset_index(drop=True).astype(
            {
                "min_width": "int16",
                "min_height": "int16",
                "max_width": "int16",
                "max_height": "int16",
                "width": "int16",
                "height": "int16",
                "rows": "uint8",
                "cols": "uint8",
                "image_id": str,
                "class_id": str,
                "num_sample": "uint32",
                "stochastic_nub": "bool",
            }
        )

        if is_save_df:
            self.save_df()

    def save_df(self) -> None:
        self.df.to_pickle(self.pickle_file_path)

    def transform_storage_format(self):
        pandarallel.initialize(progress_bar=True, nb_workers=16)
        self.df.parallel_apply(self.transform_sample, axis=1)

    def transform_sample(self, row):
        # Define new directory path for the sample
        sample_dir = (
            self.dataset_dir / "images" / row["class_id"] / str(row["num_sample"])
        )
        if sample_dir.exists():
            return
        sample_dir.mkdir(parents=True, exist_ok=True)

        hdf5_filepath = (
            self.dataset_dir / "images" / row["class_id"] / f"{row['num_sample']}.hdf5"
        )
        with h5py.File(hdf5_filepath, "r") as hdf:
            for dataset_name in hdf.keys():
                if dataset_name.startswith("piece_"):
                    image_data = np.array(hdf[dataset_name])
                    img = Image.fromarray(image_data)
                    img.save(sample_dir / f"{dataset_name}.png")

            # Save labels
            labels = np.array(hdf["id_row_col"])
            np.save(sample_dir / "labels.npy", labels)

        hdf5_filepath.unlink()
        print(sample_dir)
