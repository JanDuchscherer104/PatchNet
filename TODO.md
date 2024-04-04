# TODOs

## Data Generation

- [ ] Explore [piecemaker]([piecemaker](https://github.com/jkenlooper/piecemaker/tree/main?tab=readme-ov-file)) parameters to generate puzzles resembling real-world scenarios.
- [ ] Select parameters for puzzles that exhibit more common Jigsaw puzzle properties / characteristics.
- [ ] Implement parallel processing in `piecemaker`.
- [ ] Generate a dataset of 1,000 images from [ImgNet](https://www.kaggle.com/c/imagenet-object-localization-challenge/data).
- [ ] Choose a suitable format / data structure for the labels.
- [ ] Store the dataset in a more efficient format (e.g. HDF5), to enable batched loading.

$$
(x, y,\varphi), \quad \text{where } x \in \{0, \ldots, M\},\: y \in \{0, \ldots, M\},\: \varphi \in \{k\cdot90|k \in{0,\ldots,3}\}
$$

---

## Model Research

### CNN

- [ ] Choose an appropriate pre-trained CNN as the backbone.
  - Candidates:
  - ``ResNet``
  - ``EfficientNet``

### Dimensionality Reduction

- [ ] Investigate (pre-trained ?) auto-encoders (or PCA) for efficient dimensionality of the CNN's latent space. Is this necessary?

### Transformer & Positional Embeddings

- [ ] Determine the input / output format for the Transformer Decoder.
- [ ] Should we employ beam-search for the Transformer Decoder?
- [ ] Explore handling a variable number of puzzle pieces.
- [ ] Employ Beam Search?

#### Positional Embeddings

- [ ] Decide between concatenating or adding positional embeddings.
- The Transformer Decoders Vocabulary should be the set of all possible piece positions and rotations.

---

## Model Implementation

- [ ] Develop an augmentation pipeline using [Albumentations](https://albumentations.ai/).
- [ ] [PyTorch Lightning](https://www.pytorchlightning.ai/) Framework
