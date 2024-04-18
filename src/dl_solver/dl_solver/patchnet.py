import sys
from typing import Tuple

import torch
import torch.nn as nn
from lib.learnable_fourier_features.positional_encoding import LearnableFourierFeatures
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s

from .config import HyperParameters


class EfficientNetV2(nn.Module):
    def __init__(self, num_features_out: int, is_trainable: bool = False, **kwargs):
        """
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

        assert kwargs.keys() in {
            "inverted_residual_setting",
            "dropout",
            "stochastic_depth_prob",
            "num_classes",
            "norm_layer",
            "last_channel",
        }
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: torch.Tensor[torch.float32] - (B, num_pieces, 3, H, W)
        Returns:
            torch.Tensor[torch.float32] - (B, num_pieces, num_features)

        Extracts features of each puzzle piece independently. And returns a flattened
        feature tensor for each piece.
        """

        return torch.stack([self.backbone(x_i) for x_i in x], dim=0)


class Transformer(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
    ):
        super().__init__()
        # TODO all is args are pretty much arbitrarys
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead),
            num_layers=num_encoder_layers,
        )
        # TODO: Current values are arbitrary
        self.embedding = LearnableFourierFeatures(
            pos_dim=2, f_dim=128, h_dim=256, d_dim=64, g_dim=1
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead),
            num_layers=num_decoder_layers,
        )

    def forward(self, src: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src: torch.Tensor[torch.float32] - (B, num_pieces, num_features)
            pos: torch.Tensor[torch.float32] - (B, num_pieces, 3) [row_idx, col_idx, rotation]
        Returns:
            torch.Tensor[torch.float32] - (B, num_pieces, num_features)
        """
        pos_encoding = self.embedding(pos)
        src = src + pos_encoding
        memory = self.encoder(src)
        decoder_output = self.decoder(memory, memory)
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
        self, x, actual_rows, actual_cols
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row_logits = self.fc_rows(x)
        col_logits = self.fc_cols(x)
        rotation_logits = self.fc_rotation(x)

        # Apply masking to logits
        if actual_rows < self.max_rows:
            row_logits[:, actual_rows:] = torch.tensor(
                float("-inf"), dtype=row_logits.dtype, device=col_logits.device
            )
        if actual_cols < self.max_cols:
            col_logits[:, actual_cols:] = torch.tensor(
                float("-inf"), dtype=col_logits.dtype, device=col_logits.device
            )

        return row_logits, col_logits, rotation_logits


class PatchNet(nn.Module):
    hparams: HyperParameters
    backbone: EfficientNetV2
    transformer: Transformer
    classifier: DynamicPuzzleClassifier

    def __init__(self, hparams: HyperParameters):
        super().__init__()

        self.backbone = EfficientNetV2(
            num_features_out=hparams.num_features_out,
            is_trainable=hparams.backbone_is_trainable,
        )

        self.transformer = Transformer(
            d_model=hparams.num_features_out,
            nhead=8,
            num_encoder_layers=3,
            num_decoder_layers=3,
        )

        self.classifier = DynamicPuzzleClassifier(
            input_features=1,  # TODO
            max_rows=hparams.puzzle_shape[0],  # TODO
            max_cols=hparams.puzzle_shape[1],
        )

    def forward(self, x):
        """
        Args:
            x: torch.Tensor[torch.float32] - (B, num_pieces, 3, H, W)
        Returns:
            y: torch.Tensor[torch.int64] - (B, num_pieces, 3) [row_idx, col_idx, rotation]
                row in {0, 1, ..., rows - 1}
                col in {0, 1, ..., cols - 1}
                rotation in {0, 1, 2, 3} for 0, 90, 180, 270 degreess
                num_pieces = rows * cols
        """
        x = self.backbone(x)

        all_unique_indices = False
        # TODO: create initial positional embedding of shape (num_pieces, 3) [row_idx, col_idx, rotation]
        initial_pos_embedding = torch.rand(
            (x.shape[:-1], 3), device=x.device, dtype=x.dtype
        )
        while not all_unique_indices:  # TODO: and some_stopping_criterion:
            x = self.transformer(x, initial_pos_embedding)
            x = self.classifier(x)
            unique_indices = self.check_unique_indices(x[:, :, :2])

            # potentially embed unique_indices into x and pass it through the decoder again!s
            # TODO: How to embed unique_indices into x?
            all_unique_indices = unique_indices.all().item()

        return x

        # Revision from ChatGPT
        # def forward(self, x):
        # # x shape: (B, num_pieces, 3, H, W)
        # batch_size, num_pieces, _, _, _ = x.shape
        # x = self.backbone(x.view(-1, *x.shape[2:])).view(batch_size, num_pieces, -1)  # Reshape after backbone

        # # Initialize positions randomly or based on some logic
        # positions = torch.rand((batch_size, num_pieces, 3), device=x.device, dtype=torch.float32)

        # # Transformer processing
        # x = self.transformer(x, positions)

        # # Classification for each piece
        # row_logits, col_logits, rotation_logits = self.classifier(x, self.hparams.puzzle_shape[0], self.hparams.puzzle_shape[1])

        # if self.hparams.is_check_unique:
        #     # Check for uniqueness of spatial indices (rows and cols)
        #     unique_indices = self.check_unique_indices(torch.stack((torch.argmax(row_logits, dim=2), torch.argmax(col_logits, dim=2)), dim=2))
        #     all_unique_indices = unique_indices.all().item()
        #     if not all_unique_indices:
        #         # Handle non-unique scenario, maybe re-run with updated positions or apply penalties
        #         print("Non-unique indices detected, handling required.")

        # return F.log_softmax(row_logits, dim=2), F.log_softmax(col_logits, dim=2), F.log_softmax(rotation_logits, dim=2)

    def check_unique_indices(self, spatial_indices: torch.Tensor) -> torch.Tensor:
        # Example uniqueness check, you might need more sophisticated logic
        _, counts = torch.unique(spatial_indices, dim=1, return_counts=True)
        return counts == 1

    def check_unique_indices(self, spatial_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spatial_indices: torch.Tensor[torch.int64] - (B, num_pieces, 2) [row_idx, col_idx]
        Returns:
                is_unique: torch.Tensor[torch.bool] - (B, num_pieces, )
        """
        return torch.unique(spatial_indices, return_counts=True)[1] == 1


if __name__ == "__main__":
    ...
