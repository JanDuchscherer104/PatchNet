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

- [x] Determine the input / output format for the Transformer Decoder.
    It was choosen to use a LearnableFourierFeatures embedding of the `(row_idx, col_idx, rot_idx)` of shape `[B, L, 3]` tuple.
- [ ] Explore handling a variable number of puzzle pieces.
- [ ] Employ Beam Search?
- [ ] Embedding of `num_rows` and `num_cols` in the Transformer Decoder.
- [ ] Embedding of the puzzle's imagenet class into the Transformer Decoder.
- [ ] Implement a differentiable constraint that ensures that the puzzle pieces are unique.
    - Transform the output probabilities (which follow a categorical distribution) into a differentiable distribution.
    - Use joint probabilities of both the row and column predictions.
    - Sort the probabilites of all tokens for possible coordinate $ (x, y)_i $.
    - Assign the "wanted" coordinate to the token with that higest probabilit $ \Pr\{(x, y)_i\} $.
    - But we should have a look at the joint probabilites of the entire set of puzzle pieces: $ \Pr\{(x, y)_i, (x, y)_j\} $. % TODO rewrite!
    - having separate encoders for both the rows and columns
        - make the temperatures of the gumble softmax learnable parameters. Possible input: `logits` ($ \mathcal{L}^{row}, \mathcal{L}^{col}, \mathcal{L}^{rot}$), `num_rows` and num_cols embeddings.
    - Embbedding of the uniqueness of each piece.
    - Predicting whether a piece is a border, corner or inner piece.
- [ ] Explore the use of [Pointer Networks](https://arxiv.org/abs/1506.03134) instead of Transformer Decoder.

- Optimize the cost function heuristically using Optuna.
- Implement a custom loss function that penalizes overlapping placements more severely.
- Use a permutation-invariant loss function. [Hungarian loss](https://en.wikipedia.org/wiki/Hungarian_algorithm) could possibly be adapted!
- We need to optimize for the joint probability of the entire sequence rather than optimizing all tokens individually.


- Embedding of the original sizes of each piece, since we are cropping and resizing the pieces.

- Dimensionality reduction of the CNN's latent space.
- Don't use a low dimensional dense layer between the CNN and the Transformer. The ouput of the CNN for `(width, heiht) = (48, 48)` is of shape [B, 1280]
```
(avgpool): AdaptiveAvgPool2d(output_size=1)
        (classifier): Sequential(
          (0): Dropout(p=0.2, inplace=True)
          (1): Linear(in_features=1280, out_features=512, bias=True)
        )
```
- What Layer could we use instead of the `nn.Linear` layer?
    - Maybe a `nn.Conv2d` layer with a kernel size of 1x1?
    - Maybe a `nn.Conv1d` layer?
- maybe we don't need to include a Transformer Encoder.
- Use GrouptNorm instead of LayerNorm.

#### Positional Embeddings

- [ ] Decide between concatenating or adding positional embeddings.
- [x] Find code for [Learnable Fourier Features for Spatially Encoded Transformers](https://arxiv.org/pdf/2106.02795v1): [GitHub](https://github.com/JHLew/Learnable-Fourier-Features)
- [ ] Implement 2D/3D Learnable Fourier Features for spatial encoding in the Transformer Decoder.
- The Transformer Decoders Vocabulary should be the set of all possible piece positions and rotations.
- The auxiliary network's cost function could the to optimize for decreasing $ \frac{\mathcal{L}}/{ds} $.
### Dimensionality Reduction

- [ ] Investigate (pre-trained ?) auto-encoders (or PCA) for efficient dimensionality of the CNN's latent space. Is this necessary?
---

## Pipeline
- save all logs and checkpoints via mlflow or wandb
- add option to limit the number of samples in the dataset
- create an auxiliary network as cost function - should be adverserial in some way.

## Model Implementation

- [x] Develop an augmentation pipeline using [Albumentations](https://albumentations.ai/).
- [x] [PyTorch Lightning](https://www.pytorchlightning.ai/) Framework consisting of
  - [ ] [Lightning Module](../src/dl_solver/dl_solver/lit_module.py)
    - General sturcute done, Cost function done, optimizer done
  - [x] [Lightning Datamodule](../src/dl_solver/dl_solver/lit_datamodule.py)

---
---