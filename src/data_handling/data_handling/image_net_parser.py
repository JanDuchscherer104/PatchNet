from pathlib import Path
from typing import Literal

import cv2
import pandas as pd
import swifter


class ImageNetParser:
    def __init__(self, data_dir: Path) -> None:
        assert data_dir.exists(), f"Data directory {data_dir} does not exist."
        self.data_dir = data_dir
        self.synset_mapping_file: Path = self.data_dir / "LOC_synset_mapping.txt"

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
        solution_file: Path = self.data_dir / f"LOC_{split}_solution.csv"
        df = pd.read_csv(solution_file).rename(columns={"ImageId": "image_id"})
        df[["class_id", "num_sample"]] = df["image_id"].str.rsplit(
            "_", n=1, expand=True
        )
        return df

    def read_images(
        self, df: pd.DataFrame, split: Literal["train", "val", "test"]
    ) -> pd.DataFrame:
        images_dir = self.data_dir / f"ILSVRC/Data/CLS-LOC/{split}"
        assert images_dir.exists()

        df.assign(image=None, height=0, width=0).astype(
            {"height": "uint16", "width": "uint16"}
        )

        def read_image(row: pd.Series) -> pd.Series:
            img_pth = (
                images_dir
                / Path(row["class_id"])
                / Path(row["image_id"]).with_suffix(".JPEG")
            )
            assert img_pth.exists(), f"Image {img_pth} does not exist."
            img = cv2.imread(str(img_pth))
            row["height"], row["width"] = img.shape[:2]
            row["image"] = img
            return row

        df = df.swifter.apply(read_image, axis=1)

        return df
