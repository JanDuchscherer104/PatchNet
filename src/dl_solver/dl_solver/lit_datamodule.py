from pathlib import Path
from typing import Literal, Tuple

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
        self.dataset_dir / f"{self.split}_jigsaw.csv"

    def filter_by_shape(self, rows: int, cols: int) -> None:
        self.df = self.df.query("rows == @rows and cols == @cols")

    def get_max_segment_shape(self) -> Tuple[int, int]:
        return self.df["max_width"].max(), self.df["max_height"].max()

    def get_min_segment_shape(self) -> Tuple[int, int]:
        return self.df["min_width"].min(), self.df["min_height"].min()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.dataframe.iloc[idx]
        hdf5_filepath = self.root_dir / row["class_id"] / f"{row['num_sample']}.hdf5"

        with h5py.File(hdf5_filepath, "r") as f:
            puzzle_pieces = {
                dataset_name: torch.from_numpy(np.array(dataset))
                for dataset_name, dataset in f.items()
                if dataset_name.startswith("piece_")
            }
            labels = torch.from_numpy(np.array(f["id_row_col"]))

        sample = {
            "puzzle_pieces": puzzle_pieces,
            "labels": labels,
        }

        return sample

    def plot_sample(self, idx: int) -> None:
        sample = self.__getitem__(idx)
        puzzle_pieces = sample["puzzle_pieces"]
        rows = sample["rows"]
        cols = sample["cols"]

        fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))

        for piece in puzzle_pieces["id_row_col"]:
            id, row, col = piece
            axs[row, col].imshow(puzzle_pieces[f"piece_{id}"].numpy().astype(int))
            axs[row, col].axis("off")

        for row in range(rows):
            for col in range(cols):
                if not axs[row, col].has_data():
                    fig.delaxes(axs[row, col])

        plt.subplots_adjust(wspace=0.0005, hspace=0.0005)
        plt.show()
