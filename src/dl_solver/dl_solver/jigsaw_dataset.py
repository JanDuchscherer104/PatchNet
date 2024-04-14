from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import Dataset

from .album_transforms import AlbumTransforms


class JigsawDataset(Dataset):
    IMGNET_STATS = {
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
    def csv_file_path(self) -> Path:
        return self.dataset_dir / f"{self.split}_jigsaw.csv"

    @property
    def df(self) -> pd.DataFrame:
        if self.__df is None:
            self.__df = pd.read_csv(self.csv_file_path)
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
        hdf5_filepath = (
            self.dataset_dir / "images" / row["class_id"] / f"{row['num_sample']}.hdf5"
        )

        with h5py.File(hdf5_filepath, "r") as f:
            # TODO: make this a list!
            puzzle_pieces = {
                dataset_name: np.array(dataset)
                for dataset_name, dataset in f.items()
                if dataset_name.startswith("piece_")
            }
            labels = np.array(f["id_row_col"])

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
            img = self.IMGNET_STATS["std"] * img + self.IMGNET_STATS["mean"]
            img = np.clip(img, 0, 1)
            img = Image.fromarray((img * 255).astype(np.uint8))

            img = img.rotate(-90 * rotation)  # undo the rotation
            img = np.array(img)
            axs[row, col].imshow(img)
            axs[row, col].axis("off")

        plt.subplots_adjust(wspace=0.0005, hspace=0.0005)
        plt.show()

    def update_min_dimensions(self) -> None:
        min_widths = []
        min_heights = []
        rows_to_drop = []

        for index, row in self.df.iterrows():
            hdf5_filepath = (
                self.dataset_dir
                / "images"
                / row["class_id"]
                / f"{int(row['num_sample'])}.hdf5"
            )
            min_width = min_height = float("inf")
            try:
                with h5py.File(hdf5_filepath, "r") as f:
                    for dataset_name, dataset in f.items():
                        if dataset_name.startswith("piece_"):
                            img = np.array(dataset)
                            min_width = min(min_width, img.shape[1])
                            min_height = min(min_height, img.shape[0])
            except Exception as e:
                # remove the row if the file is not found
                rows_to_drop.append(index)
                continue
            else:
                min_widths.append(min_width)
                min_heights.append(min_height)

        print(f"Dropping {len(rows_to_drop)} rows")
        self.df = (
            self.df.drop(rows_to_drop)
            .reset_index(drop=True)
            .assign(min_width=min_widths, min_height=min_heights)
            .astype(
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
            .drop(columns=["Unnamed: 0", "Unnamed: 0.1"])
        )

    def refurb_df(self, is_save_df: bool = False) -> None:
        # check if each file exists
        rows_to_drop = []
        for index, row in self.df.iterrows():
            hdf5_filepath = (
                self.dataset_dir
                / "images"
                / row["class_id"]
                / f"{int(row['num_sample'])}.hdf5"
            )
            if not hdf5_filepath.exists():
                rows_to_drop.append(index)

        self.df = (
            self.df.drop(rows_to_drop)
            .reset_index(drop=True)
            .astype(
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
        )

        if is_save_df:
            self.save_df()

    def save_df(self) -> None:
        self.df.to_csv(self.csv_file_path, index=False)
