# Data-Piecemaker

## Piecemaker
- [GitHub::piecemaker](https://github.com/jkenlooper/piecemaker/tree/main?tab=readme-ov-file)
- [location](lib/piecemaker) of the submodule

### Installation
```bash
sudo apt-get -y install libspatialindex6
sudo apt-get -y install optipng
sudo apt-get -y install potrace
% activate conda / python environment
cd lib/piecemaker
pip install --upgrade --upgrade-strategy eager -e .
```

### Folder Structure

- `index.json`: This file contains metadata about the puzzle, such as the total number of pieces, the puzzle's height and width, and the size of each piece.
    - `piece_cut_variant`: The style of the puzzle piece cuts.
    - `full_size`: The full size of the puzzle.
    - `sizes`: An array of sizes for the puzzle.
    - `sides`: An array of sides for the puzzle.
    - `piece_count`: The total number of pieces in the puzzle.
    - `image_author`: The author of the image used to generate the puzzle.
    - `image_link`: The link to the image used to generate the puzzle.
    - `image_title`: The title of the image used to generate the puzzle.
    - `image_description`: The description of the image used to generate the puzzle.
    - `image_width`: The width of the image used to generate the puzzle.
    - `image_height`: The height of the image used to generate the puzzle.
    - `outline_bbox`: The bounding box of the puzzle outline.
    - `puzzle_author`: The author of the puzzle.
    - `puzzle_link`: The link to the puzzle.
    - `table_width`: The width of the table on which the puzzle is displayed.
    - `table_height`: The height of the table on which the puzzle is displayed.
    - `piece_properties`: An array of objects, each representing a puzzle piece. Each object has the following properties:
        - `id`: The ID of the piece.
        - `x` and `y`: The x and y coordinates of the piece on the table.
        - `ox` and `oy`: The original x and y coordinates of the piece in the image.
        - `r`: The rotation of the piece.
        - `s`: The scale of the piece.
        - `w` and `h`: The width and height of the piece.
        - `rotate`: The rotation of the piece.
        - `g`: The group of the piece.

- `adjacent.json`: This file contains information about which pieces are adjacent to each other in the original image.

- `size-100/`: This directory contains data for the puzzle pieces when the puzzle size is set to 100.

    - `pieces.json`: This file contains information about each piece, including its row and column index in the original image, its rotation (in degrees), and its height and width.

    - `sprite_with_padding_layout.json` and `sprite_without_padding_layout.json`: These files contain information about how the pieces are laid out in the sprite image.

    - `masks.json` and `piece_id_to_mask.json`: These files contain information about the masks used to extract each piece from the original image.

    - `data_uri/`: This directory contains Base64 encoded images of each piece.

    - `mask`: Contains the unique masks as `bmp` files.

    - `raster`: Contains the N segemented pieces of the Jigsaw puzzle as `jpg` files.

    - `raster_with_padding`: Contains the N segemented pieces of the Jigsaw puzzle as `jpg` files with padding to rectangular shape.

---

## Original ImageNet Dataset
can be found [here](https://www.kaggle.com/c/imagenet-object-localization-challenge/data)

### Folder Structure
1. **Class Labels**: The class labels are located in the `LOC_synset_mapping.txt` file. Each line in this file likely contains a class ID and its corresponding label. Example lines:
    ```
    n01440764 tench, Tinca tinca
    n01443537 goldfish, Carassius auratus
    n01484850 great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias
    ```
2. **Splits**: The `train`, `val` & `test` set images are located in `.data/imagenet/ILSVRC/Data/CLS-LOC/<split>`.
    - `train-folder`: Contains the images in subfolders named after the class labels. For example, the first directory is `n01440764`, and contains images of tench fish. The labels are also provided via `.data/imagenet/LOC_train_solution.csv` - same format as the validation set.
    - `val-folder`: Contains the images directly as `ILSVRC2012_val_<id>.JPEG`. The labels are located in `.data/imagenet/LOC_val_solution.csv` and look like this:
        ```csv
        ImageID                , PredictionString
        ILSVRC2012_val_00008726,n02119789 255 142 454 329 n02119789 44 21 322 295
                                <class id><BBX          > <class id><BBX        >
        ```
        This example contains two predictions of the same class.
    - `test-folder`: Containes the images directly as `ILSVRC2012_test_<id>.JPEG`

**Consideration**:
- We should adhere to the dataset order as per the `Loc_<split>_solution.csv` files. Meaning: first line --> first sample
- We should transfer the following information into our Jigsaw-Imagenet dataset:
    - ImageID
    - Class ID (s)
    - Number of bbox predictions
- We might onyl use the cropped bbox images for the Jigsaw puzzles. And maybe only those that have a certain minimum size.
- In the Jigsaw-Imagenet dataset, we should store the separate pieces in a single file to avoid loading multiple files for each sample. consider formats: `HDF5` or `Parquet`.
- We might even save multiple samples in a single file to simplify the data loading process.

---
---