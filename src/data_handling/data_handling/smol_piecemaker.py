import json
import random
import tempfile
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import piecemaker
import piecemaker.base
import piecemaker.distribution
import piecemaker.lines_svg
import piecemaker.reduce
from PIL import Image

from dl_solver import PiecemakerConfig


class SmolPiecemaker:
    config: PiecemakerConfig

    def __init__(self, config: PiecemakerConfig) -> None:
        self.config = config

    def conv_to_jigsaw(
        self, img_path: Path, height: int, width: int, is_new_mask: bool = True
    ) -> Tuple[int, int, int, int, int, int, bool]:

        scaled_sizes = set(self.config.scaled_sizes)
        minimum_scale = min(scaled_sizes)
        # overlap_threshold = int(self.config.minimum_piece_size)

        svg_file = Path(
            tempfile.NamedTemporaryFile(
                suffix=".svg", delete=False, delete_on_close=False
            ).name
        )

        choice = random.choices(
            ["interlockingnubs", "stochasticnubs"],
            [
                1 - self.config.stochastic_nubs_probability,
                self.config.stochastic_nubs_probability,
            ],
            k=1,
        )[0]

        with tempfile.TemporaryDirectory() as tmp_dir:
            (img_path, jpc, rows, cols) = piecemaker.lines_svg.create_lines_svg(
                output_dir=tmp_dir,
                minimum_piece_size=self.config.minimum_piece_size,
                maximum_piece_size=self.config.maximum_piece_size,
                width=width,
                height=height,
                number_of_pieces=self.config.number_of_pieces,
                imagefile=img_path,
                variant=choice,
                svg_file=svg_file,
            )
        storchastic_nub: bool = choice[0] == "s"

        width = jpc.width
        height = jpc.height

        max_piece_side = max(jpc._piece_width, jpc._piece_height)
        # min_piece_side = min(jpc._piece_width, jpc._piece_height)
        minimum_pixels = jpc.pieces * self.config.minimum_piece_size**2
        minimum_side = np.sqrt(minimum_pixels)
        side_count = np.sqrt(jpc.pieces)
        new_minimum_piece_size = np.ceil(minimum_side / side_count)
        if minimum_scale < 100:
            minimum_scale = min(
                100, np.ceil((new_minimum_piece_size / max_piece_side) * 100.0)
            )

        scaled_sizes.add(minimum_scale)
        sizes = sorted(list(scaled_sizes))

        scale_for_size_100 = 100 if self.config.use_max_size else minimum_scale

        with tempfile.TemporaryDirectory() as tmp_dir:
            identifier = img_path.stem
            # tmp_dir = self.config.out_dir /
            # tmp_dir.mkdir(parents=True, exist_ok=True)

            full_size_dir = Path(tmp_dir) / f"size-{scale_for_size_100}"
            full_size_dir.mkdir(parents=True, exist_ok=True)

            pieces = piecemaker.base.Pieces(
                svg_file,
                img_path,
                full_size_dir,
                scale=scale_for_size_100,
                max_pixels=(width * height),
                include_border_pixels=self.config.gap,
            )
            svg_file.unlink()

            img_path = pieces._scaled_image
            pieces.cut()
            pieces.generate_resources()

            piece_count = len(pieces.pieces)

            for size in sizes:
                if size == scale_for_size_100:
                    continue
                piecemaker.reduce.reduce_size(
                    scale=size,
                    minimum_scale=scale_for_size_100,
                    output_dir=tmp_dir,
                )

            with Image.open(img_path) as img:
                (width, height) = img.size

            with (full_size_dir / "pieces.json").open("r") as pieces_json:
                piece_bboxes = json.load(pieces_json)

            # Convert data to a structured NumPy array
            ids = np.array(list(piece_bboxes.keys())).astype(int)
            boxes = np.array(list(piece_bboxes.values()))

            # Calculate midpoints of boxes
            x_mid = (boxes[:, 0] + boxes[:, 2]) / 2
            y_mid = (boxes[:, 1] + boxes[:, 3]) / 2

            # Normalize midpoints to range [0, 1]
            x_norm = x_mid / boxes[:, 2].max()
            y_norm = y_mid / boxes[:, 3].max()

            col_indices = np.floor(x_norm * cols).astype(int)
            row_indices = np.floor(y_norm * rows).astype(int)

            # Combine ids with their row and column indices
            id_row_col = np.vstack((ids, row_indices, col_indices)).T

            segments_dir = full_size_dir / "raster_with_padding"
            class_id, sample_number = identifier.split("_")
            cls_dir = self.config.jigsaw_dir / "images" / class_id
            cls_dir.mkdir(parents=True, exist_ok=True)

            max_width = 0
            max_height = 0

            with h5py.File((cls_dir / sample_number).with_suffix(".hdf5"), "w") as f:
                for png_file in segments_dir.glob("*.jpg"):
                    img = np.array(Image.open(png_file))
                    f.create_dataset(f"piece_{png_file.stem}", data=img)

                    max_width = max(max_width, img.shape[1])
                    max_height = max(max_height, img.shape[0])
                    min_width = min(max_width, img.shape[1])
                    min_height = min(max_height, img.shape[0])

                f.attrs["piece_count"] = piece_count
                f.attrs["rows"] = rows
                f.attrs["cols"] = cols
                f.attrs["max_width"] = max_width
                f.attrs["max_height"] = max_height
                f.create_dataset("id_row_col", data=id_row_col)

            return (
                rows,
                cols,
                max_width,
                max_height,
                min_width,
                min_height,
                storchastic_nub,
            )
