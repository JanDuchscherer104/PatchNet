from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Literal, Optional

import h5py
import numpy as np
import pandas as pd
import psutil
from pandarallel import pandarallel
from PIL import Image

from dl_solver import Config

from .smol_piecemaker import SmolPiecemaker


class ImageNetParser:
    config: Config
    piecemaker: SmolPiecemaker

    def __init__(self, config: Config, split: Literal["train", "val", "test"]) -> None:
        self.config = config
        self.split = split

        self.imagenet_dir = self.config.paths.imagenet_dir
        self.synset_mapping_file: Path = self.imagenet_dir / "LOC_synset_mapping.txt"

        self.piecemaker = SmolPiecemaker(
            self.config.piecemaker_config, split=self.split
        )
        self.__existing_df: Optional[pd.DataFrame] = None

        if self.config.is_multiproc:
            pandarallel.initialize(
                progress_bar=True, nb_workers=psutil.cpu_count(logical=True)
            )

    def read_synset_mappings(self) -> pd.DataFrame:
        with open(self.synset_mapping_file, "r") as f:
            data = [
                [parts[0], " ".join(parts[1:]).split(",")[0].replace(" ", "_")]
                for parts in (line.split(" ") for line in f)
            ]
        df = pd.DataFrame(
            {"class_id": [d[0] for d in data], "class_name": [d[1] for d in data]}
        ).assign(
            class_name=lambda x: x.class_name.str.replace("\n", "").str.replace(
                "-", "_"
            )
        )
        return df

    def read_solution_csv(self) -> pd.DataFrame:
        solution_file: Path = self.imagenet_dir / f"LOC_{self.split}_solution.csv"
        df = pd.read_csv(solution_file).rename(columns={"ImageId": "image_id"})
        df[["class_id", "num_sample"]] = df["image_id"].str.rsplit(
            "_", n=1, expand=True
        )

        if self.split != "test":
            df = df.assign(class_id=df["PredictionString"].str.extract(r"(n\d{8})")[0])
        return df

    def to_jigsaw(self, df: pd.DataFrame) -> pd.DataFrame:
        images_dir = self.imagenet_dir / f"ILSVRC/Data/CLS-LOC/{self.split}"
        assert images_dir.exists()

        df.assign(
            height=0,
            width=0,
            rows=0,
            cols=0,
            max_width=0,
            max_height=0,
            min_width=0,
            min_height=0,
            stochastic_nub=False,
        ).astype(
            {
                "height": "uint16",
                "width": "uint16",
                "rows": "uint8",
                "cols": "uint8",
                "max_width": "uint16",
                "max_height": "uint16",
                "min_width": "uint16",
                "min_height": "uint16",
                "stochastic_nub": "bool",
            }
        )

        def make_jigsaw(row: pd.Series) -> pd.Series:
            match self.split:
                case "train":
                    img_pth = (
                        images_dir / str(row["class_id"]) / str(row["image_id"])
                    ).with_suffix(".JPEG")
                case "val" | "test":
                    img_pth = (images_dir / str(row["image_id"])).with_suffix(".JPEG")
            try:
                assert img_pth.exists(), f"Image {img_pth} does not exist."
                with Image.open(img_pth) as img:
                    width, height = img.size
                    row["height"], row["width"] = height, width

                (
                    row["rows"],
                    row["cols"],
                    row["max_width"],
                    row["max_height"],
                    row["min_width"],
                    row["min_height"],
                    row["stochastic_nub"],
                ) = self.piecemaker.conv_to_jigsaw(
                    img_pth,
                    str(row["class_id"]),
                    str(row["num_sample"]),
                    width=width,
                    height=height,
                )
            except Exception as e:
                print(f"Error: {e}")
                (
                    row["height"],
                    row["width"],
                    row["rows"],
                    row["cols"],
                    row["max_width"],
                    row["max_height"],
                    row["min_width"],
                    row["min_height"],
                    row["stochastic_nub"],
                ) = [None] * 9
            return row

        existing_df = self.existing_df()
        if not existing_df.empty:
            df = df[~df["image_id"].isin(existing_df["image_id"])]

        try:
            df = (
                df.parallel_apply(make_jigsaw, axis=1)
                if self.config.is_multiproc
                else df.apply(make_jigsaw, axis=1)
            )
        except Exception as e:
            print(f"Error: {e}")

        pkl_path = self.config.paths.jigsaw_dir / f"{self.split}_jigsaw.pkl"
        if pkl_path.exists():
            df = (
                self.existing_df()
                .set_index("image_id")
                .combine_first(df.set_index("image_id"))
                .reset_index()
            )

        df.to_pickle(self.config.paths.jigsaw_dir / f"{self.split}_jigsaw.pkl")

        return df

    def existing_df(self) -> pd.DataFrame:
        if self.__existing_df is None:
            try:
                self.__existing_df = pd.read_pickle(
                    self.config.paths.jigsaw_dir / f"{self.split}_jigsaw.pkl"
                )
            except FileNotFoundError:
                return pd.DataFrame()
        return self.__existing_df

    def add_missing_info(self) -> pd.DataFrame:
        raise NotImplementedError()
        files = list(Path(self.config.paths.jigsaw_dir / "images").glob("**/*.hdf5"))
        if self.config.is_multiproc:
            with ProcessPoolExecutor(max_workers=self.config.num_workers) as executor:
                data = list(executor.map(self.process_file, files))
        else:
            data = [self.process_file(file) for file in files]

        data = [d for d in data if d is not None]  # Filter out None values
        new_df = pd.DataFrame(data).set_index("image_id")
        existing_df = self.existing_df("train").set_index("image_id")
        df = existing_df.combine_first(new_df).reset_index()

        return df

    def process_file(self, file: Path) -> dict:
        raise NotImplementedError()
        try:
            with h5py.File(file, "r") as f:
                attrs = f.attrs
                img_pth = (
                    self.imagenet_dir
                    / f"ILSVRC/Data/CLS-LOC/train"
                    / file.parent.name
                    / f"{file.parent.name}_{file.stem}"
                ).with_suffix(".JPEG")
                with Image.open(img_pth) as img:
                    width, height = img.size
                data = {
                    "image_id": f"{file.parent.name}_{file.stem}",
                    "class_id": file.parent.name,
                    "num_sample": int(file.stem),
                    "piece_count": int(attrs["piece_count"]),
                    "cols": int(attrs["cols"]),
                    "rows": int(attrs["rows"]),
                    "max_width": int(attrs["max_width"]),
                    "max_height": int(attrs["max_height"]),
                    "min_width": int(attrs["min_width"]),
                    "min_height": int(attrs["min_height"]),
                    "width": int(width),
                    "height": int(height),
                    "stochastic_nub": None,
                }
            return data
        except Exception as e:
            print(f"Error processing file {file}: {e}, removing file.")
            file.unlink()
            return None
