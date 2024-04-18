import torch
import torch.nn as nn
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
        super(EfficientNetV2, self).__init__()

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
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(num_features, num_features_out)

        # Set the backbone to be non-trainable
        if not is_trainable:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: torch.Tensor[torch.float32] - (num_pieces, 3, H, W)
        Returns:
            torch.Tensor[torch.float32] - (num_pieces, num_features)

        Extracts features of each puzzle piece independently. And returns a flattened
        feature tensor for each piece.
        """

        return torch.stack([self.backbone(x_i) for x_i in x], dim=0)


class PatchNet(nn.Module):
    hparams: HyperParameters
    backbone: EfficientNetV2

    def __init__(self, hparams: HyperParameters):
        super(PatchNet, self).__init__()

        # Initialize the EfficientNet V2-S as the backbone
        self.backbone = EfficientNetV2(
            num_features_out=hparams.num_features_out,
            is_trainable=hparams.backbone_is_trainable,
        )

    def forward(self, x):
        """
        Args:
            x: torch.Tensor[torch.float32] - (num_pieces, 3, H, W)
        Returns:
            y: torch.Tensor[torch.int64] - (num_pieces, 3) [row_idx, col_idx, rotation]
                row in {0, 1, ..., rows - 1}
                col in {0, 1, ..., cols - 1}
                rotation in {0, 1, 2, 3} for 0, 90, 180, 270 degreess
                num_pieces = rows * cols
        """
        x = self.backbone(x)

        # Encoder
        # x = self.encoder(x)

        # Decoder

        all_unique_indices = False
        while not all_unique_indices:
            # x = self.decoder(x)
            unique_indices = self.check_unique_indices(x[:, :2])

            # potentially embed unique_indices into x and pass it through the decoder again!s
            all_unique_indices = unique_indices.all().item()

        return x

    def check_unique_indices(self, spatial_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spatial_indices: torch.Tensor[torch.int64] - (num_pieces, 2) [row_idx, col_idx]
        Returns:
            is_unique: torch.Tensor[torch.bool] - (num_pieces, )
        """
        return torch.unique(spatial_indices, return_counts=True)[1] == 1
