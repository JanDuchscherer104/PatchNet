# Data-Piecemaker

## Jigsaw Dataset
### Generation of samples
```py
from dl_solver import Config, JigsawDataset
from data_handling import ImageNetParser

from data_handling import ImageNetParser
from dl_solver import Config
config = Config(is_multiproc=True)
imagenet_parser = ImageNetParser(config, split="train")
df = imagenet_parser.read_solution_csv().pipe(imagenet_parser.to_jigsaw)

# Run Data Cleaning (if something went wrong!)
dataset = JigsawDataset(dataset_dir=config.paths.jigsaw_dir, split='train', puzzle_shape=(None, None), transforms=None)
ds._refurb_df(is_save_df=True)
```

>
```py
>>> ds.df.head()
```
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>image_id</th>
      <th>class_id</th>
      <th>cols</th>
      <th>height</th>
      <th>max_height</th>
      <th>max_width</th>
      <th>min_height</th>
      <th>min_width</th>
      <th>num_sample</th>
      <th>piece_count</th>
      <th>rows</th>
      <th>stochastic_nub</th>
      <th>width</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>n01440764_10040</td>
      <td>n01440764</td>
      <td>4</td>
      <td>375</td>
      <td>169</td>
      <td>178</td>
      <td>169</td>
      <td>161</td>
      <td>10040</td>
      <td>12</td>
      <td>3</td>
      <td>True</td>
      <td>500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>n01440764_10048</td>
      <td>n01440764</td>
      <td>4</td>
      <td>300</td>
      <td>158</td>
      <td>166</td>
      <td>104</td>
      <td>106</td>
      <td>10048</td>
      <td>12</td>
      <td>3</td>
      <td>False</td>
      <td>400</td>
    </tr>
    <tr>
      <th>2</th>
      <td>n01440764_1009</td>
      <td>n01440764</td>
      <td>4</td>
      <td>375</td>
      <td>175</td>
      <td>198</td>
      <td>175</td>
      <td>137</td>
      <td>1009</td>
      <td>12</td>
      <td>3</td>
      <td>True</td>
      <td>500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>n01440764_10293</td>
      <td>n01440764</td>
      <td>4</td>
      <td>375</td>
      <td>200</td>
      <td>202</td>
      <td>169</td>
      <td>135</td>
      <td>10293</td>
      <td>12</td>
      <td>3</td>
      <td>True</td>
      <td>500</td>
    </tr>
    <tr>
      <th>4</th>
      <td>n01440764_10342</td>
      <td>n01440764</td>
      <td>6</td>
      <td>234</td>
      <td>114</td>
      <td>122</td>
      <td>97</td>
      <td>87</td>
      <td>10342</td>
      <td>18</td>
      <td>3</td>
      <td>True</td>
      <td>500</td>
    </tr>
  </tbody>
</table>
</div>

```py
>>> ds.plot_sample()
```
![Augmented Sample from Training Set](.doc-assets/train_aug.png)

```
jigsaw/
├── imagenet_semantic_label_map.txt  # Mapping from class IDs to semantic labels
├── images/                          # Directory containing all images
│   ├── train/                       # Training images
│   │   ├── n01440764/               # Directory for each class ID
│   │   │   ├── 10040/               # Directory for each sample
│   │   │   │   ├── labels.npy       # Numpy file containing labels for the sample
│   │   │   │   ├── piece_0.png      # Puzzle piece images for the sample
│   │   │   │   ├── ...
│   │   │   │   ├── piece_11.png
│   │   │   │   ├── ...
│   │   │   │   └── piece_9.png
│   │   │   ├── 10048/               # Another sample
│   │   │   │   ├── ...
│   ├── val/                         # Validation images
│   │   ├── n01440764/               # Directory for each class ID
│   │   │   ├── 00000293/            # Directory for each sample
│   │   │   ├── 00003014/            # Another sample
│   │   │   ├── ...
├── test_jigsaw.csv                  # CSV file containing test data
├── train_jigsaw.csv                 # CSV file containing training data
└── val_jigsaw.csv                   # CSV file containing validation data
```

**Distribution of Rows & Cols in the Dataset**:

![Distribution of #Rows & #Cols in the Dataset](.doc-assets/row_col_hist_n12_min48_max256.png)

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