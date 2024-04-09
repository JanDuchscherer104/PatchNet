# PATCH-Net: Deep Learning for Jigsaw Puzzle Solving

## Overview

**PATCH-Net** (**P**uzzle **A**ssembly by **T**ransformer and **C**NN **H**ybrid **Net**work) utilizes a pre-trained CNN backbone for feature extraction, an optional method for dimensionality reduction of the CNNs latent space, and a subsequent Transformer architecture with 2D/3D Learnable Fourier Features for spatial encoding. The model predicts the position and orientation of puzzle pieces, with three classification heads.

## Other MD Files
- [TODO.md](TODO.md)
- [DATA-PIECEMAKER.md](DATA-PIECEMAKER.md): Information about the submodule `piecemaker` for jigsaw puzzle generation, and the data structure of the original and puzzle-ized dataset.

## Architecture

![patch-net.svg](./.doc-assets/patch-net.svg)

- **Input**: Images represented as $\mathbf{X} \in \mathbb{R}^{L \times C_{\text{RGB}} \times H \times W}$
- **CNN**: A fine-tuned backbone (ResNet, EfficientNet) for feature extraction.
    - A good starting-point for model selection might be [An evaluation of pre-trained models for feature extraction in image classification
](https://ar5iv.labs.arxiv.org/html/2310.02037), [Paper](https://arxiv.org/abs/2310.02037)
- (optional) **Dimensionality Reduction**: PCA or Autoencoders reduce the feature space dimensionality.
- **Transformer**:
  - Encoder processes features without positional embedding.
  - Decoder uses 2D/3D [Learnable Fourier Features](https://arxiv.org/pdf/2106.02795v1) for spatial encoding.
- **Output**: Predicts position $(x, y)$ and orientation $\varphi$

$$
(x, y,\varphi), \quad \text{where } x \in \{0, \ldots, N-1\}, y \in \{0, \ldots, M-1\}, \varphi \in \{k\cdot90|k \in{0,\ldots,3}\}
$$

- **Classification Heads**: Three heads (`nn.Linear`) for predicting the location and orientation of puzzle pieces. Handling varying numbers of puzzle pieces is a future consideration.
- **Loss Calculation**: Combines $\mathcal{L}_2$ for position and $\mathcal{L}_{\text{CE}}$ for orientation.

$$
\mathcal{L}_{\text{total
}} = \| \mathbf{Y}_{:,:,:2} - \mathbf{\hat{Y}}_{:,:,:2} \|_F^2 + \eta \mathcal{L}_{\text{CE}}(\mathbf{Y}_{:,:,2}, \mathbf{\hat{Y}}_{:,:,2})
$$
where $\eta \in \mathbb{R}^+$ is a hyperparameter.


## Considerations

- **Data Representation**: $\mathbf{y} \in \{ 0, \ldots, N_{\max}\} \times\{ 0, \ldots, M_{\max}\} \times \{0, \ldots 3 \}$
- **Concatenation/Addition in Transformer Decoder**: $\widetilde{\mathbf{X}}^{\langle t\rangle} \gets \widetilde{\mathbf{X}}^{\langle t-1\rangle}_{:,:,:d_{\text{model}}} \oplus \mathbf{E}_{\text{P}}$
- **Dimensionality Reduction**: May occur after the dense layer for enhanced performance.
- **Unique Labels Check**: Ensures no repetition of puzzle piece positions.
- **Beam Search**: To improve the decoder's predictions.
- It might be a more straightforward approach to employ a pre-trained ViT model instead of a pre-trained CNN $\oplus$ custom Transformer architecture.

---

## Third-Party Libraries

### Jigsaw Puzzle Generation
- [GitHub::piecemaker](https://github.com/jkenlooper/piecemaker/tree/main?tab=readme-ov-file)
- [location](lib/piecemaker) of the submodule
- find further information (installation, usage, ...) in [DATA-PIECEMAKER.md](./DATA-PIECEMAKER.md)


- **Data Augmentation**: [Albumentations](https://albumentations.ai/)
- **DL-Framework**: [PyTorch Lightning](https://www.pytorchlightning.ai/)

## Literature & Resources

- **Dataset**: [ImageNet](https://www.kaggle.com/c/imagenet-object-localization-challenge/data)
- [Learnable Fourier Features for Spatially Encoded Transformers](https://arxiv.org/pdf/2106.02795v1)
- [Jigsaw-ViT: Learning jigsaw puzzles in vision transformer](https://www.sciencedirect.com/science/article/pii/S0167865522003920), [GitHub](https://github.com/yingyichen-cyy/JigsawViT/tree/master)
- [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- [Intuition for Embeddings](https://www.youtube.com/watch?v=wjZofJX0v4M)

---
---