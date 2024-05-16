from typing import Annotated, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.functional import F
from torchtyping import TensorType
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s

from .config import HyperParameters
from .positional_encoding import LearnableFourierFeatures

# from .hungarian_net import HungarianNet


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
        self.backbone.classifier[1].requires_grad = True

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


class Transformer(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
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
            norm=nn.LayerNorm(d_model),
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
            norm=nn.LayerNorm(d_model),  # TODO: Use GroupNorm instead of LayerNorm
        )

    def generate_square_subsequent_mask(
        self, size: int, device: torch.device
    ) -> Tensor:
        mask = torch.triu(
            torch.ones(size, size, device=device),
            diagonal=1,
        )
        mask[mask == 1] = float("-inf")
        return mask

    def forward(
        self,
        src: TensorType["B, num_pieces, num_features", torch.float32],
        pos: Annotated[Tensor, "B, num_pieces, num_features", torch.float32],
        memory: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            src: Tensor[torch.float32] - (B, num_pieces, num_features)
            pos: Tensor[torch.float32] - (B, num_pieces, 3) [row_idx, col_idx, rotation]
            memory: Tensor[torch.float32] - (B, num_pieces, num_features)
        Returns:
            Tensor[torch.float32] - (B, num_pieces, num_features)
            Tensor[torch.float32] - (B, num_pieces, num_features)
        """
        spatial_pos = pos[:, :, :2].unsqueeze(-2)
        rotation_pos = pos[:, :, 2:].unsqueeze(-2)

        spatial_encoding = self.spatial_embedding(spatial_pos)
        rotation_encoding = self.rotation_embedding(rotation_pos)

        encoder_memory = self.encoder(src) if memory is None else memory

        pos_encoding = spatial_encoding + rotation_encoding

        if self.training:
            tgt_mask = self.generate_square_subsequent_mask(
                pos_encoding.size(1), pos.device
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
        # self.fc_pos = nn.Linear(input_features, max_rows * max_cols)
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

        # pos_logits[..., actual_rows * actual_cols :].fill_(float("-inf"))

        row_logits[..., actual_rows:].fill_(float("-inf"))
        col_logits[..., actual_cols:].fill_(float("-inf"))

        # Concatenate the tensors along a new dimension
        return row_logits, col_logits, rot_logits


class LearnableTemperatures(nn.Module):
    def __init__(self, hparams: HyperParameters):
        # Use hparams.softmax_temperature and hparams.gumbel_temperature
        ...


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
        )

        self.classifier = DynamicPuzzleClassifier(
            input_features=hparams.num_features_out,
            max_rows=hparams.puzzle_shape[0],  # TODO
            max_cols=hparams.puzzle_shape[1],
        )

        # self.hungarian = HungarianNet(max_len=np.prod(hparams.puzzle_shape))
        # self.temperatures = LearnableTemperatures(hparams)

    def forward(
        self, x: Tensor, true_pos_seq: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        """
        Args:
            x: Tensor[torch.float32] - (B, num_pieces, 3, H, W)
        Returns:
            pos_seq: Tensor[torch.float32] - (B, num_pieces, 3) [row_idx, col_idx, rotation]
            logits: Tuple[Tensor[torch.float32], Tensor[torch.float32], Tensor[torch.float32]]
                row_logits: Tensor[torch.float32] - (B, num_pieces, max_rows)
                col_logits: Tensor[torch.float32] - (B, num_pieces, max_cols)
                rotation_logits: Tensor[torch.float32] - (B, num_pieces, 4)
        """
        x = self.backbone(x)
        # TODO: Potential Feature Reduction
        # TODO: create initial positional embedding of shape (num_pieces, 3) [row_idx, col_idx, rotation]
        pos_seq = (
            true_pos_seq.clone().to(x)  # type: ignore
            if self.training
            else torch.rand(
                (*x.shape[:-1], 3),
                device=x.device,
                dtype=torch.float32,
                requires_grad=True,
            )
        )

        if self.training:
            pos_seq, logits, _ = self._soft_forward_step(x, pos_seq)
        else:
            encoder_memory = None
            for _ in range(self.hparams.num_decoder_iters):
                pos_seq, logits, encoder_memory = self._soft_forward_step(
                    x, pos_seq.to(torch.float32), encoder_memory
                )

        # potentially embed unique_indices into x and pass it through the decoder again!

        return pos_seq, logits

    def _forward_step(
        self,
        x: Tensor,
        pos_seq: Tensor,
        encoder_memory: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            x: Tensor[torch.float32] - (B, num_pieces, num_features)
        Returns:
            pos_seq: Tensor[torch.float32] - (B, num_pieces, 3) [row_idx, col_idx, rotation]
            logits: Tuple[Tensor[torch.float32], Tensor[torch.float32], Tensor[torch.float32]]
            encoder_memory: Tensor[torch.float32] - (B, num_pieces, num_features)
        """
        x, encoder_memory = self.transformer(x, pos_seq, encoder_memory)
        logits = self.classifier(x, *self.hparams.puzzle_shape)
        pos_seq = (
            torch.stack([torch.argmax(logit, dim=-1) for logit in logits], dim=-1)
            .to(torch.float32)
            .to(x.device)
        )

        return pos_seq, logits, encoder_memory

    def _soft_forward_step(
        self, x: Tensor, pos_seq: Tensor, encoder_memory: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        x, encoder_memory = self.transformer(x, pos_seq, encoder_memory)
        logits = self.classifier(x, *self.hparams.puzzle_shape)

        joint_probs = self._compute_joint_probabilities(*logits[:2])
        joint_logits = self.apply_penalties(joint_probs)

        # Differentiable Hungarian or similar algorithm to get unique assignments  # Make sure to initialize this correctly
        # assignments, _, _ = self.hungarian(
        #     flat_joint_probs
        # )  # Expected shape: [B, num_pieces]

        # Stack the indices to form the final position sequence tensor
        pos_seq = torch.cat(
            [
                self.softargmax2d(joint_logits),
                self.softargmax1d(logits[2]).unsqueeze(-1),
            ],
            dim=-1,
        )

        return pos_seq, logits, encoder_memory

    @staticmethod
    def softargmax1d(input: Tensor, beta=100) -> Tensor:
        *_, n = input.shape
        input = F.softmax(beta * input, dim=-1)
        indices = torch.linspace(0, 1, n, device=input.device)
        result = torch.sum((n - 1) * input * indices, dim=-1)
        return result

    @staticmethod
    def softargmax2d(input: Tensor, beta=100) -> Tensor:
        *_, h, w = input.shape

        input = input.reshape(*_, h * w)
        input = nn.functional.softmax(beta * input, dim=-1)

        indices_c, indices_r = torch.meshgrid(
            torch.linspace(0, 1, w, device=input.device),
            torch.linspace(0, 1, h, device=input.device),
        )

        indices_r = indices_r.reshape(-1, h * w)
        indices_c = indices_c.reshape(-1, h * w)

        result_r = torch.sum((h - 1) * input * indices_r, dim=-1)
        result_c = torch.sum((w - 1) * input * indices_c, dim=-1)

        result = torch.stack([result_r, result_c], dim=-1)

        return result

    def apply_penalties(self, joint_probs: Tensor) -> Tensor:
        """
        Aims to penalize high probabilities for the same coordinates
        Args:
            joint_probs: Tensor [B, L, num_rows, num_cols] of Joint Probabilities.


        Case distinctions:
            - max_per_class & max_per_token -> assign
            - ~max_per_class & max_per_token -> penalize, subsidize others
            - max_per_class & ~max_per_token -> subsidize
        """
        flat_probs = joint_probs.view(*joint_probs.shape[:2], -1)
        max_probs_per_token, _ = flat_probs.max(dim=1, keepdim=True)
        max_probs_per_class, _ = flat_probs.max(dim=-1, keepdim=True)

        max_per_token = (flat_probs == max_probs_per_token).float()
        max_per_class = (flat_probs == max_probs_per_class).float()

        # Apply a dynamic penalty for non-max elements and a boost for max elements where they aren't the max token-wise
        # Penalizes non-max tokens
        penalties = (
            (1 - max_per_class)
            * max_per_token
            * ((max_probs_per_class - flat_probs) / max_probs_per_class)
        )
        # Boost max-class elements that are not max-token
        incentives = (
            (1 - max_per_token)
            * max_per_class
            * ((max_probs_per_token - flat_probs) / max_probs_per_token)
        )

        adjusted_probs = flat_probs * (1 + incentives - penalties)

        return (
            (adjusted_probs + torch.finfo(torch.float32).eps).log().softmax(-1)
        ).view_as(joint_probs)

    def _compute_joint_probabilities(
        self, row_logits: Tensor, col_logits: Tensor
    ) -> Tensor:
        """
        Args:
            row_logits (Tensor[B, num_pieces, num_rows])
            col_logits (Tensor[B, num_pieces, num_cols])

        Returns:
            Tensor[B, num_pieces, num_rows, num_cols]: Joint probabilities
        """
        # Compute probabilities within each token over all classes
        row_probs = F.softmax(row_logits, dim=-1)
        col_probs = F.softmax(col_logits, dim=-1)

        joint_probs = row_probs[:, :, :, None] * col_probs[:, :, None, :]

        return joint_probs
