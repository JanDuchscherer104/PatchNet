from typing import Tuple

import torch
import torch.nn as nn
from torch.functional import F
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s

from . import LearnableFourierFeatures
from .config import HyperParameters


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

    def forward(self, src: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src: torch.Tensor[torch.float32] - (B, num_pieces, num_features)
            pos: torch.Tensor[torch.float32] - (B, num_pieces, 3) [row_idx, col_idx, rotation]
        Returns:
            torch.Tensor[torch.float32] - (B, num_pieces, num_features)
        """
        spatial_pos = pos[:, :, :2].unsqueeze(-2)
        rotation_pos = pos[:, :, 2:].unsqueeze(-2)

        spatial_encoding = self.spatial_embedding(spatial_pos)
        rotation_encoding = self.rotation_embedding(rotation_pos)

        encoder_memory = self.encoder(src)

        pos_encoding = spatial_encoding + rotation_encoding
        decoder_output = self.decoder(pos_encoding, encoder_memory)

        return decoder_output


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
    ) -> torch.Tensor:
        """
        Args:
            x: torch.Tensor[torch.float32] - (B, num_pieces, num_features)
            actual_rows: int - Number of actual rows in the puzzle
            actual_cols: int - Number of actual columns in the puzzle
        Returns:
            logits: (B, num_pieces, max_rows + max_cols + 4)
        """
        row_logits = self.fc_rows(x)
        col_logits = self.fc_cols(x)
        rotation_logits = self.fc_rotation(x)

        row_logits[:, actual_rows:].fill_(float("-inf"))
        col_logits[:, actual_cols:].fill_(float("-inf"))

        # Concatenate the tensors along a new dimension
        return torch.cat([row_logits, col_logits, rotation_logits], dim=-1)


class PatchNet(nn.Module):
    hparams: HyperParameters
    backbone: EfficientNetV2
    transformer: Transformer
    classifier: DynamicPuzzleClassifier

    def __init__(self, hparams: HyperParameters):
        super().__init__()
        self.hparams = hparams

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
            input_features=hparams.num_features_out,
            max_rows=hparams.puzzle_shape[0],  # TODO
            max_cols=hparams.puzzle_shape[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: torch.Tensor[torch.float32] - (B, num_pieces, 3, H, W)
        Returns:
            pos_seq: torch.Tensor[torch.float32] - (B, num_pieces, 3) [row_idx, col_idx, rotation]
            unique_indices: torch.Tensor[torch.bool] - (B, num_pieces)
            logits: Tuple[torch.Tensor[torch.float32], torch.Tensor[torch.float32], torch.Tensor[torch.float32]]
                row_logits: torch.Tensor[torch.float32] - (B, num_pieces, max_rows)
                col_logits: torch.Tensor[torch.float32] - (B, num_pieces, max_cols)
                rotation_logits: torch.Tensor[torch.float32] - (B, num_pieces, 4)
        """
        x = self.backbone(x)
        # TODO: Potential Feature Reduction
        x.requires_grad = True
        # TODO: create initial positional embedding of shape (num_pieces, 3) [row_idx, col_idx, rotation]
        pos_seq = torch.rand(
            (*x.shape[:-1], 3), device=x.device, dtype=x.dtype, requires_grad=True
        )

        x = self.transformer(x, pos_seq)
        logits = self.classifier(
            x,
            actual_rows=self.hparams.puzzle_shape[0],
            actual_cols=self.hparams.puzzle_shape[1],
        )
        max_rows, max_cols = self.hparams.puzzle_shape
        row_logits = logits[..., :max_rows]
        col_logits = logits[..., max_rows : max_rows + max_cols]
        rotation_logits = logits[..., max_rows + max_cols :]
        pos_seq = (
            torch.stack(
                [
                    torch.argmax(row_logits, dim=-1),
                    torch.argmax(col_logits, dim=-1),
                    torch.argmax(rotation_logits, dim=-1),
                ],
                dim=-1,
            )
            .to(torch.float32)
            .to(x.device)
        )
        unique_indices = self.check_unique_indices(pos_seq[:, :, :2])

        # potentially embed unique_indices into x and pass it through the decoder again!s
        # TODO: How to embed unique_indices into x?

        return torch.cat([pos_seq, unique_indices.unsqueeze(-1), logits], dim=-1)

    def check_unique_indices(self, spatial_indices: torch.Tensor) -> torch.Tensor:
        """
        Check uniqueness of spatial indices within each batch.
        Args:
            spatial_indices: torch.Tensor[torch.int64] - (B, num_pieces, 2) [row_idx, col_idx]
        Returns:
            is_unique: torch.Tensor[torch.bool] - (B, num_pieces)
        """
        # batch_size, num_pieces = spatial_indices.size(0), spatial_indices.size(1)
        # unique_mask = torch.ones(
        #     (batch_size, num_pieces), dtype=torch.bool, device=spatial_indices.device
        # )

        # # Check each batch independently
        # for i in range(batch_size):
        #     _, inverse_indices, counts = torch.unique(
        #         spatial_indices[i], dim=0, return_inverse=True, return_counts=True
        #     )
        #     unique_mask[i] = counts[inverse_indices] == 1

        batch_size, num_pieces = spatial_indices.size(0), spatial_indices.size(1)
        spatial_indices_flat = spatial_indices.reshape(batch_size, -1)
        _, inverse_indices, counts = torch.unique(
            spatial_indices_flat, dim=1, return_inverse=True, return_counts=True
        )
        unique_mask = counts[inverse_indices] == 1
        unique_mask = unique_mask.clone().view(batch_size, num_pieces)

        return unique_mask
