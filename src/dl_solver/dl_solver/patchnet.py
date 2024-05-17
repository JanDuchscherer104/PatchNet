from typing import Annotated, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.functional import F
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s

from .config import HyperParameters
from .nn_utils import nn_utils
from .positional_encoding import LearnableFourierFeatures


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
        # self.classifier = nn.Sequential(
        #     nn.Dropout(p=dropout, inplace=True),
        #     nn.Linear(lastconv_output_channels, num_classes),
        # )
        self.backbone.classifier[1] = nn.Linear(
            self.backbone.classifier[1].in_features, num_features_out
        )

        # Set the backbone to be non-trainable
        if not is_trainable:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.backbone.classifier[1].requires_grad_(True)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor[torch.float32] - (B, num_pieces, 3, H, W)
        Returns:
            Tensor[torch.float32] - (B, num_pieces, num_features)

        Extracts features of each puzzle piece independently. And returns a flattened
        feature tensor for each piece.
        """

        return torch.stack([self.backbone(x_i) for x_i in x.unbind(dim=1)], dim=1)


class PuzzleTypeClassifier(nn.Module):
    """
    For each puzzle piece, predict the type of the piece:
        - Corner (0) ~ (row_idx == 0 || row_idx == num_rows - 1) && (col_idx == 0 || col_idx == num_cols - 1)
        - Edge (1) ~ (row_idx == 0 || row_idx == num_rows - 1) || (col_idx == 0 || col_idx == num_cols - 1)
        - Center (2) ~ else
    """

    def __init__(self, input_features: int) -> None:
        super().__init__()
        self.fc = nn.Linear(input_features, 3)
        self.embedding = nn.Sequential(
            nn.Softmax(dim=-1),
            LearnableFourierFeatures(
                pos_dim=3, d_dim=input_features, f_dim=128, h_dim=256, g_dim=1
            ),
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x (Tensor): [B, num_pieces, num_features]

        Returns:
            Tensor: [B, num_pieces, 3] - [corner, edge, center]
        """
        x = self.fc(x)
        return x, self.embedding(x.unsqueeze(-2))


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        decoder_dim_feedforward: int,
        nhead: int = 8,
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
            norm=nn.LayerNorm(d_model),
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
            norm=nn.LayerNorm(d_model),  # TODO: Use GroupNorm instead of LayerNorm
        )

    def forward(
        self,
        src: Tensor,
        pos_encoding: Tensor,
        memory: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            src: Tensor[torch.float32] - (B, num_pieces, num_features)
            pos_encoding: Tensor[torch.float32] - (B, num_pieces, num_features) [row_idx, col_idx, rotation]
            memory: Tensor[torch.float32] - (B, num_pieces, num_features)
        Returns:
            Tensor[torch.float32] - (B, num_pieces, num_features)
            Tensor[torch.float32] - (B, num_pieces, num_features)
        """
        encoder_memory = self.encoder(src) if memory is None else memory

        if self.training:
            tgt_mask = nn_utils.generate_causal_mask(
                pos_encoding.size(1), pos_encoding.device
            )

        decoder_output = self.decoder(
            pos_encoding,
            encoder_memory,
            tgt_is_causal=self.training,
            tgt_mask=tgt_mask if self.training else None,
        )

        return decoder_output, encoder_memory


class DynamicPuzzleClassifier(nn.Module):
    def __init__(self, input_features, max_rows: int, max_cols: int):
        super().__init__()
        self.max_rows = max_rows
        self.max_cols = max_cols

        self.fc_rows = nn.Linear(input_features, max_rows)
        self.fc_cols = nn.Linear(input_features, max_cols)
        # self.fc_pos = nn.Linear(input_features, max_rows * max_cols) TODO: use
        self.fc_rot = nn.Linear(input_features, 4)  # For rotations

    def forward(
        self, x: Tensor, actual_rows: int, actual_cols: int
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            x: Tensor[torch.float32] - (B, num_pieces, num_features)
            actual_rows: int - Number of actual rows in the puzzle
            actual_cols: int - Number of actual columns in the puzzle
        Returns:
            row_logits: Tensor[torch.float32] - (B, num_pieces, max_rows)
            col_logits: Tensor[torch.float32] - (B, num_pieces, max_cols)
            rotation_logits: Tensor[torch.float32] - (B, num_pieces, 4)
        """
        row_logits = self.fc_rows(x)
        col_logits = self.fc_cols(x)
        rot_logits = self.fc_rot(x)

        row_logits[..., actual_rows:].fill_(float("-inf"))
        col_logits[..., actual_cols:].fill_(float("-inf"))

        # Concatenate the tensors along a new dimension
        return row_logits, col_logits, rot_logits


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
            nhead=8,
            num_encoder_layers=3,
            num_decoder_layers=3,
            decoder_dim_feedforward=512,
        )

        # TODO: try using a single positional embedding, but with different groups: extend rotational embedding to two dimensions!
        self.spatial_embedding = LearnableFourierFeatures(
            pos_dim=2, f_dim=128, h_dim=256, d_dim=hparams.num_features_out, g_dim=1
        )
        self.rotation_embedding = LearnableFourierFeatures(
            pos_dim=1, f_dim=2, h_dim=64, d_dim=hparams.num_features_out, g_dim=1
        )

        # Start and End tokens for the decoder sequence
        self.start_of_seq_token = nn.Parameter(
            torch.randn(1, 1, hparams.num_features_out, dtype=torch.float32),
            requires_grad=True,
        )
        self.end_of_seq_token = nn.Parameter(
            torch.randn(1, 1, hparams.num_features_out, dtype=torch.float32),
            requires_grad=True,
        )

        self.classifier = DynamicPuzzleClassifier(
            input_features=hparams.num_features_out,
            max_rows=hparams.puzzle_shape[0],  # TODO
            max_cols=hparams.puzzle_shape[1],
        )

        self.puzzle_type_classifier = PuzzleTypeClassifier(hparams.num_features_out)

    def forward(
        self, x: Tensor, true_pos_seq: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tuple[Tensor, Tensor, Tensor]]:
        """
        Args:
            x: Tensor[torch.float32] - (B, num_pieces, 3, H, W)
        Returns:
            puzzle_type_logits: Tensor[torch.float32] - (B, num_pieces, 3) [corner, edge, center]
            pos_seq: Tensor[torch.float32] - (B, num_pieces, 3) [row_idx, col_idx, rotation]
            logits: Tuple[Tensor[torch.float32], Tensor[torch.float32], Tensor[torch.float32]]
                row_logits: Tensor[torch.float32] - (B, num_pieces, max_rows)
                col_logits: Tensor[torch.float32] - (B, num_pieces, max_cols)
                rotation_logits: Tensor[torch.float32] - (B, num_pieces, 4)
        """
        x = self.backbone(x)
        puzzle_type_logits, puzzle_type_embedding = self.puzzle_type_classifier.forward(
            x
        )
        x = x + puzzle_type_embedding

        # TODO: Potential Feature Reduction
        # TODO: create initial positional embedding of shape (num_pieces, 3) [row_idx, col_idx, rotation]

        if self.training:
            assert true_pos_seq is not None
            pos_seq = torch.cat(
                [
                    self.start_of_seq_token.expand(x.size(0), 1, -1),
                    self._embedd_pos_seq(true_pos_seq.clone().to(x)),
                    self.end_of_seq_token.expand(x.size(0), 1, -1),
                ],
                dim=1,
            )
            pos_seq, logits = self._soft_forward_step(x, pos_seq)
        else:
            pos_seq, logits = self._autoregressive_decode(x)

        def remove_special_tokens(x):
            if not self.training:
                return x
            if isinstance(x, tuple):
                return tuple(map(remove_special_tokens, x))
            return x[:, 1:-1, ...]

        return (puzzle_type_logits, *remove_special_tokens((pos_seq, logits)))  # type: ignore

    def _soft_forward_step(
        self, x: Tensor, pos_seq: Tensor, encoder_memory: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:

        x, encoder_memory = self.transformer(x, pos_seq, encoder_memory)
        logits = self.classifier(x, *self.hparams.puzzle_shape)

        joint_probs = nn_utils.compute_joint_probabilities(*logits[:2])
        joint_logits = nn_utils.apply_penalties(joint_probs)

        # Stack the indices to form the final position sequence tensor
        pos_seq = torch.cat(
            [
                nn_utils.softargmax2d(joint_logits),
                nn_utils.softargmax1d(logits[2]).unsqueeze(-1),
            ],
            dim=-1,
        )

        return pos_seq, logits

    def _embedd_pos_seq(self, pos: Tensor) -> Tensor:
        spatial_encoding = self.spatial_embedding(pos[:, :, :2].unsqueeze(-2))
        rotation_encoding = self.rotation_embedding(pos[:, :, 2:].unsqueeze(-2))

        return spatial_encoding + rotation_encoding

    def _autoregressive_decode(
        self, x: Tensor
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        B, L, _F = x.shape
        pos_seq = torch.cat([self.start_of_seq_token.expand(B, 1, -1)], dim=1)
        encoder_memory = None

        logits_list: list[tuple[Tensor, Tensor, Tensor]] = []
        for _token in range(L):
            decoder_output, encoder_memory = self.transformer(
                x, pos_seq, encoder_memory
            )
            logits = self.classifier(
                decoder_output[:, -1, :], *self.hparams.puzzle_shape
            )
            row_logits, col_logits, rot_logits = logits
            logits_list.append(logits)

            # Compute joint probabilities and apply penalties
            if row_logits.dim() == 2:
                row_logits = row_logits.unsqueeze(1)
                col_logits = col_logits.unsqueeze(1)
                rot_logits = rot_logits.unsqueeze(1)
            joint_probs = nn_utils.compute_joint_probabilities(row_logits, col_logits)

            # Find the next token using the adjusted joint probabilities
            next_token = torch.cat(
                [
                    nn_utils.argmax2d(joint_probs),
                    torch.argmax(rot_logits, dim=-1, keepdim=True),
                ],
                dim=-1,
            ).to(torch.float32)

            pos_seq = torch.cat([pos_seq, self._embedd_pos_seq(next_token)], dim=1)

        # Stack the logits along the sequence length dimension and compute the final position
        logits = tuple(torch.stack(t, dim=1) for t in zip(*logits_list))
        row_logits, col_logits, _ = logits
        joint_probs = nn_utils.compute_joint_probabilities(row_logits, col_logits)
        joint_probs = nn_utils.apply_penalties(joint_probs)
        pos_seq = torch.cat(
            [
                nn_utils.softargmax2d(joint_probs),
                nn_utils.softargmax1d(logits[2]).unsqueeze(-1),
            ],
            dim=-1,
        )

        return pos_seq, logits
