from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.functional import F
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s
from transformers import BertModel, BertTokenizer

from . import LearnableFourierFeatures
from .config import HyperParameters


class ImageNetClassEmbedding(nn.Module):
    def __init__(self, model_name="bert-base-uncased", freeze=True):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)
        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, class_labels):
        """
        Args:
            class_labels (List[str]): List of class labels

        Returns:
            torch.Tensor: Tensor of embeddings
        """
        inputs = self.tokenizer(
            class_labels, return_tensors="pt", padding=True, truncation=True
        )
        outputs = self.bert(**inputs)
        # We take the embeddings from the [CLS] token which is a common practice.
        embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings


class EfficientNetV2(nn.Module):
    def __init__(self, num_features_out: int, is_trainable: bool = False, **kwargs):
        """
        TODO: Do not set to non-trainable. Use different learning rates and lr-schedulers for backbone and head.
        Args:
            num_features_out: int - Number of output features from the backbone
            **kwargs:
                inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]],
                dropout: float,
                stochastic_depth_prob: float = 0.2,
                num_classes: int = 1000,
                norm_layer: Optional[Callable[..., nn.Module]] = None,
                last_channel: Optional[int] = None,
        """
        super().__init__()

        self.backbone = efficientnet_v2_s(
            weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1, **kwargs
        )

        # Replace the classification head
        self.backbone.classifier[1] = nn.Linear(
            self.backbone.classifier[1].in_features, num_features_out
        )

        # Set the backbone to be non-trainable
        if not is_trainable:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.backbone.classifier[1].requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: torch.Tensor[torch.float32] - (B, num_pieces, 3, H, W)
        Returns:
            torch.Tensor[torch.float32] - (B, num_pieces, num_features)

        Extracts features of each puzzle piece independently. And returns a flattened
        feature tensor for each piece.
        """

        return torch.stack([self.backbone(x_i) for x_i in x.unbind(dim=1)], dim=1)


class Transformer(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        decoder_dim_feedforward=512,
    ):
        super().__init__()
        # TODO all is args are pretty much arbitrarys: Optimize hyperparameters
        # - d_model
        # - dim_feedforward
        # - nhead
        # - num_encoder_layers
        # - norm
        # TODO: Should we use both Encoder and Decoder? We've only used Decoder so far.
        # TODO: Find papet that discusses different use cases of Encoder and Decoder

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                batch_first=True,
                dim_feedforward=512,
                activation=F.silu,
            ),
            num_layers=num_encoder_layers,
        )

        # TODO: try using a single positional embedding, but with different groups: extend rotational embedding to two dimensions!
        self.spatial_embedding = LearnableFourierFeatures(
            pos_dim=2, f_dim=128, h_dim=128, d_dim=d_model, g_dim=1
        )
        self.rotation_embedding = LearnableFourierFeatures(
            pos_dim=1, f_dim=64, h_dim=64, d_dim=d_model, g_dim=1
        )

        assert d_model % nhead == 0
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                batch_first=True,
                dim_feedforward=decoder_dim_feedforward,
                activation=F.silu,
            ),
            num_layers=num_decoder_layers,
            norm=None,  # TODO: Use GroupNorm instead of LayerNorm
        )

    def forward(
        self,
        src: torch.Tensor,
        pos: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            src: torch.Tensor[torch.float32] - (B, num_pieces, num_features)
            pos: torch.Tensor[torch.float32] - (B, num_pieces, 3) [row_idx, col_idx, rotation]
            memory: torch.Tensor[torch.float32] - (B, num_pieces, num_features)
        Returns:
            torch.Tensor[torch.float32] - (B, num_pieces, num_features)
            torch.Tensor[torch.float32] - (B, num_pieces, num_features)
        """
        spatial_pos = pos[:, :, :2].unsqueeze(-2)
        rotation_pos = pos[:, :, 2:].unsqueeze(-2)

        spatial_encoding = self.spatial_embedding(spatial_pos)
        rotation_encoding = self.rotation_embedding(rotation_pos)

        encoder_memory = self.encoder(src) if memory is None else memory

        pos_encoding = spatial_encoding + rotation_encoding
        decoder_output = self.decoder(pos_encoding, encoder_memory)

        return decoder_output, encoder_memory


class DynamicPuzzleClassifier(nn.Module):
    def __init__(self, input_features, max_rows: int, max_cols: int):
        super().__init__()
        self.max_rows = max_rows
        self.max_cols = max_cols

        self.fc_rows = nn.Linear(input_features, max_rows)
        self.fc_cols = nn.Linear(input_features, max_cols)
        self.fc_rotation = nn.Linear(input_features, 4)  # For rotations

    def forward(
        self, x: torch.Tensor, actual_rows: int, actual_cols: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: torch.Tensor[torch.float32] - (B, num_pieces, num_features)
            actual_rows: int - Number of actual rows in the puzzle
            actual_cols: int - Number of actual columns in the puzzle
        Returns:
            row_logits: torch.Tensor[torch.float32] - (B, num_pieces, max_rows)
            col_logits: torch.Tensor[torch.float32] - (B, num_pieces, max_cols)
            rotation_logits: torch.Tensor[torch.float32] - (B, num_pieces, 4)
        """
        row_logits = self.fc_rows(x)
        col_logits = self.fc_cols(x)
        rotation_logits = self.fc_rotation(x)

        row_logits[:, actual_rows:].fill_(float("-inf"))
        col_logits[:, actual_cols:].fill_(float("-inf"))

        # Concatenate the tensors along a new dimension
        return row_logits, col_logits, rotation_logits


class PatchNet(nn.Module):
    hparams: HyperParameters
    backbone: EfficientNetV2
    transformer: Transformer
    classifier: DynamicPuzzleClassifier

    def __init__(self, hparams, pretrained_embedding_model="bert-base-uncased"):
        super().__init__()
        self.hparams = hparams

        # Network components
        self.backbone = EfficientNetV2(
            num_features_out=hparams.num_features_out,
            is_trainable=hparams.backbone_is_trainable,
        )
        self.transformer = Transformer(
            d_model=hparams.num_features_out,
            nhead=4,
            num_encoder_layers=3,
            num_decoder_layers=3,
        )
        self.classifier = DynamicPuzzleClassifier(
            input_features=hparams.num_features_out,  # Assuming embeddings are handled inside the transformer
            max_rows=hparams.puzzle_shape[0],
            max_cols=hparams.puzzle_shape[1],
        )
        self.class_embeddings = ImageNetClassEmbedding(pretrained_embedding_model)

    def forward(self, x, class_labels, true_pos_seq=None):
        """
        Args:
            x (torch.Tensor): Image tensor.
            class_labels (List[str]): Class labels for each image in the batch.
            true_pos_seq (Optional[torch.Tensor]): True position sequences for training.
        """
        embeddings = self.class_embeddings(
            class_labels
        )  # Get embeddings for each class label
        x = self.backbone(x)

        # Concatenate class embeddings along the feature dimension
        embeddings = embeddings.unsqueeze(1).repeat(
            1, x.size(1), 1
        )  # Ensure correct shape
        x = torch.cat((x, embeddings), dim=-1)

        pos_seq = (
            true_pos_seq.to(torch.float32).to(x.device)
            if self.training and true_pos_seq is not None
            else torch.rand(
                (*x.shape[:-1], 3), device=x.device, dtype=x.dtype, requires_grad=True
            )
        )

        if self.training:
            x, _ = self.transformer(x, pos_seq)
            logits = self.classifier(
                x, self.hparams.puzzle_shape[0], self.hparams.puzzle_shape[1]
            )
        else:
            encoder_memory = None
            for _ in range(self.hparams.num_decoder_iters):
                x, encoder_memory = self.transformer(x, pos_seq, encoder_memory)
                logits = self.classifier(
                    x, self.hparams.puzzle_shape[0], self.hparams.puzzle_shape[1]
                )
                pos_seq = (
                    torch.stack(
                        [torch.argmax(logit, dim=-1) for logit in logits], dim=-1
                    )
                    .to(torch.float32)
                    .to(x.device)
                )

        unique_indices = self.check_unique_indices(pos_seq[:, :, :2])
        return pos_seq, unique_indices, logits

    def check_unique_indices(self, spatial_indices: torch.Tensor) -> torch.Tensor:
        """
        Check uniqueness of spatial indices within each batch.
        Args:
            spatial_indices: torch.Tensor[torch.int64] - (B, num_pieces, 2) [row_idx, col_idx]
        Returns:
            is_unique: torch.Tensor[torch.bool] - (B, num_pieces)
        """
        batch_size, num_pieces = spatial_indices.size(0), spatial_indices.size(1)
        unique_mask = torch.ones(
            (batch_size, num_pieces), dtype=torch.bool, device=spatial_indices.device
        )

        # Check each batch independently
        for i in range(batch_size):
            _, inverse_indices, counts = torch.unique(
                spatial_indices[i], dim=0, return_inverse=True, return_counts=True
            )
            unique_mask[i] = counts[inverse_indices] == 1

        # batch_size, num_pieces = spatial_indices.size(0), spatial_indices.size(1)
        # spatial_indices_flat = spatial_indices.reshape(batch_size, -1)
        # _, inverse_indices, counts = torch.unique(
        #     spatial_indices_flat, dim=1, return_inverse=True, return_counts=True
        # )
        # unique_mask = counts[inverse_indices] == 1
        # unique_mask = unique_mask.clone().view(batch_size, num_pieces)

        return unique_mask
