# TODOs

## Data Generation

- [ ] Explore [piecemaker]([piecemaker](https://github.com/jkenlooper/piecemaker/tree/main?tab=readme-ov-file)) parameters to generate puzzles resembling real-world scenarios.
- [x] Implement parallel processing in `piecemaker`.
- [x] Generate a dataset of 1,000 images from [ImgNet](https://www.kaggle.com/c/imagenet-object-localization-challenge/data).
    - Generated `len(jigsaw_dataset) = 231653`
- [x] Choose a suitable format / data structure for the labels.
- [x] Store the dataset in a more efficient format (e.g. HDF5), to enable batched loading.
- [ ] Generate `val` and `test` datasets.
- [ ] Revise `Jigsaw-Dataset` to **not** store all images belonging to a sample as `np.ndarray` in `hdf5` files, but as `png` or `jpg` in a shared directory.
- [ ] Find optimal `width` and `height` for the puzzle pieces.
- [ ] Improve metadata handling

### Imagenet ~ **DONE**
- We should adhere to the dataset order as per the `Loc_<split>_solution.csv` files. Meaning: first line --> first sample
- We should transfer the following information into our Jigsaw-Imagenet dataset:
  - ImageID
  - Class ID (s)
  - Number of bbox predictions
- We might onyl use the cropped bbox images for the Jigsaw puzzles. And maybe only those that have a certain minimum size.
- In the Jigsaw-Imagenet dataset, we should store the separate pieces in a single file to avoid loading multiple files for each sample. consider formats: `HDF5` or `Parquet`.
- We might even save multiple samples in a single file to simplify the data loading process.

---

## Model Research

### CNN

- [x] Choose an appropriate pre-trained CNN as the backbone.
  - Candidates:
    - ~~``ResNet``~~
    - ``EfficientNetV2``
        - [PyTorch::EfficientNetV2S](https://pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_v2_s.html#torchvision.models.efficientnet_v2_s)
        - [TIMM](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/efficientnet.py)


### Transformer & Positional Embeddings

- [ ] Determine the input / output format for the Transformer Decoder.
- [ ] Should we employ beam-search for the Transformer Decoder?
- [ ] Explore handling a variable number of puzzle pieces.
- [ ] Employ Beam Search?

#### Positional Embeddings

- [ ] Decide between concatenating or adding positional embeddings.
- [x] Find code for [Learnable Fourier Features for Spatially Encoded Transformers](https://arxiv.org/pdf/2106.02795v1): [GitHub](https://github.com/JHLew/Learnable-Fourier-Features)
- [ ] Implement 2D/3D Learnable Fourier Features for spatial encoding in the Transformer Decoder.
- The Transformer Decoders Vocabulary should be the set of all possible piece positions and rotations.

### Dimensionality Reduction

- [ ] Investigate (pre-trained ?) auto-encoders (or PCA) for efficient dimensionality of the CNN's latent space. Is this necessary?
---

## Model Implementation

- [x] Develop an augmentation pipeline using [Albumentations](https://albumentations.ai/).
- [x] [PyTorch Lightning](https://www.pytorchlightning.ai/) Framework consisting of
  - [ ] [Lightning Module](../src/dl_solver/dl_solver/lit_module.py)
    - General sturcute done, Cost function done, optimizer done
  - [x] [Lightning Datamodule](../src/dl_solver/dl_solver/lit_datamodule.py)

---
---