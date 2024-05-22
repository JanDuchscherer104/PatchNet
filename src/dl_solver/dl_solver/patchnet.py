from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.functional import F
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s

from .hparams import HParams
from .nn_utils import nn_utils
from .positional_encoding import LearnableFourierFeatures


class EfficientNetV2(nn.Module):
    def __init__(self, hparams: HParams.Backbone):
        """
        EfficientNetV2 backbone with optional trainable parameters.
        """
        super().__init__()

        self.backbone = efficientnet_v2_s(
            weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1
        )

        # Replace the classification head
        assert isinstance(linear := self.backbone.classifier[1], nn.Linear)
        self.backbone.classifier[1] = nn.Linear(
            linear.in_features, hparams.num_features_out, bias=hparams.has_bias
        )

        # Set the backbone to be non-trainable if specified
        if not hparams.is_trainable:
            for param in self.backbone.parameters():
                param.requires_grad_(False)
        self.backbone.classifier[1].requires_grad_(True)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for EfficientNetV2.

        Args:
            x (Tensor['B num_pieces 3 H W', float32]): The puzzle pieces

        Returns:
            Tensor['B num_pieces num_features_out', float32]: Features for each puzzle piece
        """
        B, num_pieces, C, H, W = x.shape
        x = x.view(B * num_pieces, C, H, W)
        features = self.backbone(x)
        features = features.view(B, num_pieces, -1)
        return features


class PuzzleTypeClassifier(nn.Module):
    def __init__(self, hparams: HParams.TypeClassifier):
        """
        Classifier for predicting the type of each puzzle piece.

        Args:
            hparams (HyperParameters.Classifier): Hyperparameters for the classifier.
        """
        super().__init__()
        self.fc = nn.Linear(hparams.input_features, 3)
        self.embedding = nn.Sequential(
            nn.Softmax(dim=-1),
            nn.Linear(3, hparams.fourier_embedding.d_dim, bias=False),
            # LearnableFourierFeatures(**hparams.fourier_embedding.model_dump()),
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass for PuzzleTypeClassifier.

        Args:
            x (Tensor['B num_pieces num_features_out', float32]):

        Returns:
            Tuple[Tensor, Tensor]:
                - Logits for each puzzle piece type of shape (Tensor['B num_pieces 3', float32])
                - Embedded tensor of shape (Tensor['B num_pieces num_features_out', float32])
        """
        x = self.fc(x)
        return x, self.embedding(x)


class Transformer(nn.Module):
    def __init__(self, hparams: HParams.Transformer):
        """
        Transformer model for encoding and decoding puzzle piece features.

        Args:
            hparams (HyperParameters.Transformer): Hyperparameters for the transformer.
        """
        super().__init__()

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hparams.d_model,
                nhead=hparams.nhead,
                batch_first=True,
                dim_feedforward=hparams.dim_feedforward,
                activation=F.silu,
            ),
            num_layers=hparams.num_encoder_layers,
            norm=nn.LayerNorm(hparams.d_model),
        )

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hparams.d_model,
                nhead=hparams.nhead,
                batch_first=True,
                dim_feedforward=hparams.dim_feedforward,
                activation=F.silu,
            ),
            num_layers=hparams.num_decoder_layers,
            norm=nn.LayerNorm(hparams.d_model),
        )

    def forward(
        self, src_seq: Tensor, tgt_seq: Tensor, memory: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass for Transformer.

        Args:
            src_seq (Tensor['B num_pieces num_features_out', float32]): The sequence of puzzle piece features
            tgt_seq (Tensor['B num_pieces num_features_out', float32]): The target sequence of position embeddings
            memory (Tensor['B num_pieces num_features_out', float32], optional):

        Returns:
            Tuple[Tensor, Tensor]:
                - Decoder output tensor (Tensor['B num_pieces num_features_out', float32])
                - Encoder memory tensor (Tensor['B num_pieces num_features_out', float32])
        """
        encoder_memory = self.encoder.forward(src_seq) if memory is None else memory

        tgt_mask = None
        if self.training:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                tgt_seq.size(1), device=tgt_seq.device
            )

        decoder_output = self.decoder.forward(
            tgt=tgt_seq,
            memory=encoder_memory,
            tgt_mask=tgt_mask,
            tgt_is_causal=self.training,
        )

        return decoder_output, encoder_memory


class DynamicIdxClassifier(nn.Module):
    def __init__(self, hparams: HParams.IdxClassifier):
        """
        Classifier for predicting row, column, and rotation logits for each puzzle piece.

        Args:
            hparams (HyperParameters.Classifier): Hyperparameters for the classifier.
        """
        super().__init__()
        self.max_rows = hparams.max_rows
        self.max_cols = hparams.max_cols

        self.fc_rows = nn.Linear(hparams.input_features, self.max_rows)
        self.fc_cols = nn.Linear(hparams.input_features, self.max_cols)
        self.fc_rot = nn.Linear(hparams.input_features, 4)  # For rotations

    def forward(
        self, x: Tensor, actual_rows: int, actual_cols: int
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass for DynamicPuzzleClassifier.

        Args:
            x Tensor['B num_pieces num_features_out', float32]:
            actual_rows (int): Number of actual rows in the puzzle
            actual_cols (int): Number of actual columns in the puzzle

        Returns:
            Tuple[Tensor, Tensor, Tensor]:
                - Row logits (Tensor['B num_pieces max_rows', float32])
                - Column logits (Tensor['B num_pieces max_cols', float32])
                - Rotation logits (Tensor['B num_pieces 4', float32])
        """
        row_logits = self.fc_rows(x)
        col_logits = self.fc_cols(x)
        rot_logits = self.fc_rot(x)

        row_logits[..., actual_rows:].fill_(float("-inf"))
        col_logits[..., actual_cols:].fill_(float("-inf"))

        return row_logits, col_logits, rot_logits


class PatchNet(nn.Module):
    def __init__(self, hparams: HParams):
        """
        PATCHNet model for solving jigsaw puzzles using a Transformer and CNN hybrid approach.

        Args:
            hparams (HyperParameters): Hyperparameters for the model.
        """
        super().__init__()
        self.hparams = hparams

        self.backbone = EfficientNetV2(hparams.backbone)
        self.puzzle_type_classifier = PuzzleTypeClassifier(hparams.type_classifier)
        self.transformer = Transformer(hparams.transformer)
        self.classifier = DynamicIdxClassifier(hparams.idx_classifier)

        self.spatial_embedding = LearnableFourierFeatures(
            **hparams.fourier_embedding_pos.model_dump()
        )
        self.rotation_embedding = LearnableFourierFeatures(
            **hparams.fourier_embedding_rot.model_dump()
        )

        self.start_of_seq_token = nn.Parameter(
            torch.randn(1, 1, hparams.transformer.d_model, dtype=torch.float32),
            requires_grad=True,
        )

    def forward(
        self, x: Tensor, true_pos_seq: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tuple[Tensor, Tensor, Tensor]]:
        """
        Forward pass for PATCHNet.

        Args:
            x (Tensor['B num_pieces 3 H W', float32]): The puzzle pieces
            true_pos_seq (Tensor['B num_pieces 3', float32], optional): The tgt position sequence containing {row, col, rot} in dim=-1

        Returns:
            Tuple[Tensor, Tensor, Tuple[Tensor, Tensor, Tensor]]:
                - Puzzle type logits (Tensor['B num_pieces 3', float32])
                - Position sequence tensor (Tensor['B num_pieces 3', float32])
                - Tuple of logits for row, column, and rotation
        """
        x = self.backbone(x)
        puzzle_type_logits, puzzle_type_embedding = self.puzzle_type_classifier.forward(
            x
        )
        x = x + puzzle_type_embedding

        if self.training:
            assert true_pos_seq is not None
            pos_seq = torch.cat(
                [
                    self.start_of_seq_token.expand(x.size(0), 1, -1),
                    self._embedd_pos_seq(true_pos_seq.clone().to(x)),
                ],
                dim=1,
            )
            pos_seq, logits = self._soft_forward_step(x, pos_seq)
        else:
            pos_seq, logits = self._autoregressive_decode(x)

        def remove_sart_token(x):
            if not self.training:
                return x
            if isinstance(x, tuple):
                return tuple(map(remove_sart_token, x))
            return x[:, 1:, ...]

        return (puzzle_type_logits, *remove_sart_token((pos_seq, logits)))  # type: ignore

    def _soft_forward_step(
        self, x: Tensor, pos_seq: Tensor, encoder_memory: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Soft forward step for training.

        Args:
            x (Tensor['B num_pieces num_features_out', float32]):
            pos_seq (Tensor['B num_pieces num_features_out', float32]):
            encoder_memory (Tensor['B num_pieces num_features_out', float32], optional):

        Returns:
            Tuple[Tensor, Tensor]:
                - Tensor['B num_pieces num_features_out', float32]: Position sequence tensor
                - Tensor['B num_pieces num_features_out', float32]: Logits tensor
        """
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
        """
        Autoregressive decoding for inference.

        Args:
            x (Tensor['B num_pieces num_features_out', float32]):

        Returns:
            Tuple[Tensor, Tuple[Tensor, Tensor, Tensor]]:
                - Tensor['B num_pieces num_features_out', float32]: Position sequence tensor
                - Tuple of logits for row, column, and rotation
        """
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
