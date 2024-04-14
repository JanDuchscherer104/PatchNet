from pathlib import Path
from typing import Literal

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

    def __init__(self, config: Config) -> None:
        self.config = config
        self.imagenet_dir = self.config.paths.imagenet_dir
        self.synset_mapping_file: Path = self.imagenet_dir / "LOC_synset_mapping.txt"

        self.piecemaker = SmolPiecemaker(self.config.piecemaker_config)

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

    def read_solution_csv(self, split: str) -> pd.DataFrame:
        solution_file: Path = self.imagenet_dir / f"LOC_{split}_solution.csv"
        df = pd.read_csv(solution_file).rename(columns={"ImageId": "image_id"})
        df[["class_id", "num_sample"]] = df["image_id"].str.rsplit(
            "_", n=1, expand=True
        )
        return df

    def to_jigsaw(
        self, df: pd.DataFrame, split: Literal["train", "val", "test"]
    ) -> pd.DataFrame:
        images_dir = self.imagenet_dir / f"ILSVRC/Data/CLS-LOC/{split}"
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
            img_pth = (
                images_dir
                / Path(row["class_id"])
                / Path(row["image_id"]).with_suffix(".JPEG")
            )
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
                ) = self.piecemaker.conv_to_jigsaw(img_pth, width=width, height=height)
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

        df = (
            df.parallel_apply(make_jigsaw, axis=1)
            if self.config.is_multiproc
            else df.apply(make_jigsaw, axis=1)
        )

        csv_path = self.config.paths.jigsaw_dir / f"{split}_jigsaw.csv"
        if csv_path.exists():
            df = (
                pd.read_csv(csv_path)
                .set_index("image_id")
                .combine_first(df.set_index("image_id"))
                .reset_index()
            )

        df.to_csv(self.config.paths.jigsaw_dir / f"{split}_jigsaw.csv")

        return df
