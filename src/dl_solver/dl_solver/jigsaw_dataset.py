from pathlib import Path
from typing import Dict, Literal, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset


class JigsawDataset(Dataset):
    dataset_dir: Path
    split: Literal["train", "val", "test"]

    def __init__(
        self, dataset_dir: Path, split: Literal["train", "val", "test"]
    ) -> None:
        self.dataset_dir = dataset_dir
        self.split = split

        self.df = pd.read_csv(self.csv_file_path)

    @property
    def csv_file_path(self) -> Path:
        return self.dataset_dir / f"{self.split}_jigsaw.csv"

    def filter_by_shape(self, rows: int, cols: int) -> None:
        self.df = self.df.query("rows == @rows and cols == @cols")

    def get_max_segment_shape(self) -> Tuple[int, int]:
        return self.df["max_width"].max(), self.df["max_height"].max()

    def get_min_segment_shape(self) -> Tuple[int, int]:
        return self.df["min_width"].min(), self.df["min_height"].min()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx) -> Tuple[Dict, torch.Tensor]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.df.iloc[idx]
        hdf5_filepath = (
            self.dataset_dir / "images" / row["class_id"] / f"{row['num_sample']}.hdf5"
        )

        with h5py.File(hdf5_filepath, "r") as f:
            puzzle_pieces = {
                dataset_name: torch.from_numpy(np.array(dataset))
                for dataset_name, dataset in f.items()
                if dataset_name.startswith("piece_")
            }
            labels = torch.from_numpy(np.array(f["id_row_col"]))

        return (
            puzzle_pieces,
            labels,
        )

    def plot_sample(self, idx: int) -> None:
        puzzle_pieces, labels = self[idx]
        rows = self.df.iloc[idx]["rows"]
        cols = self.df.iloc[idx]["cols"]

        fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))

        # Plot each piece
        for piece in labels:
            id, row, col = piece
            axs[row, col].imshow(puzzle_pieces[f"piece_{id}"].numpy().astype(int))
            axs[row, col].axis("off")

        # Remove empty subplots
        for row in range(rows):
            for col in range(cols):
                if not axs[row, col].has_data():
                    fig.delaxes(axs[row, col])

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

    def refurb_df(self) -> None:
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
