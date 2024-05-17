import torch
import torch.nn as nn
from torch import Tensor


class HungarianNet(nn.Module):
    """
    Module designed to approximate the Hungarian algorithm with differentiability,
    using a TransformerEncoder for sequence processing and custom output handling.

    Args:
        max_len (int): The maximum length of the sequence, representing the number of tasks/pieces.
        d_model (int): The size of the hidden dimension for the transformer.
        nhead (int): The number of heads in the multi-head attention mechanism.
        dim_feedforward (int): The dimension of the feedforward network in transformer.
        dropout (float): The dropout rate.
    """

    def __init__(
        self,
        max_len: int = 4,
        d_model: int = 128,
        nhead: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Linear(max_len, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=1
        )
        self.fc1 = nn.Linear(d_model, max_len)

    def forward(self, query: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # Embed and process the query
        query = self.embedding(query)
        out = self.transformer_encoder(query)
        out = torch.tanh(self.fc1(out))

        # Flatten output to match original functionality
        out_flat = out.view(out.shape[0], -1)  # Flatten output

        # Get max values across dimensions for additional outputs
        out_max_row, _ = torch.max(out, dim=1)  # Max across rows
        out_max_col, _ = torch.max(out, dim=2)  # Max across columns

        return out_flat, out_max_row, out_max_col
