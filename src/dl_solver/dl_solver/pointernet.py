import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PointerAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super(PointerAttention, self).__init__()
        self.hidden_dim = hidden_dim

        self.W1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.vt = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, decoder_state: Tensor, encoder_outputs: Tensor, mask: Tensor):
        encoder_transform = self.W1(encoder_outputs)
        decoder_transform = self.W2(decoder_state).unsqueeze(1)
        u_i = self.vt(torch.tanh(encoder_transform + decoder_transform)).squeeze(-1)
        log_score = masked_log_softmax(u_i, mask, dim=-1)
        return log_score


class PointerTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ):
        super(PointerTransformer, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Linear(input_dim, embedding_dim, bias=False)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_layers,
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_layers,
        )
        self.attention = PointerAttention(hidden_dim)

    def forward(
        self, input_seq: Tensor, input_lengths: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch_size, max_seq_len = input_seq.shape[:2]
        embedded = self.embedding(input_seq)
        encoder_outputs = self.encoder(embedded)  # (N, S, E)

        decoder_input = encoder_outputs.new_zeros((batch_size, 1, self.hidden_dim))
        mask_tensor = (
            torch.arange(max_seq_len, device=input_lengths.device)
            .unsqueeze(0)
            .expand(batch_size, max_seq_len)
            < input_lengths.unsqueeze(1)
        ).float()

        pointer_log_scores = []
        pointer_argmaxs = []

        for _ in range(max_seq_len):
            tgt_mask = self._generate_square_subsequent_mask(decoder_input.size(1)).to(
                decoder_input.device
            )
            decoder_output = self.decoder(
                decoder_input,
                encoder_outputs,
                tgt_mask=tgt_mask,
            )  # (N, T, E)
            log_pointer_score = self.attention(
                decoder_output[:, -1, :], encoder_outputs, mask_tensor
            )
            pointer_log_scores.append(log_pointer_score)
            _, masked_argmax = masked_max(
                log_pointer_score, mask_tensor, dim=1, keepdim=True
            )
            pointer_argmaxs.append(masked_argmax)
            decoder_input = torch.cat(
                [
                    decoder_input,
                    torch.gather(
                        encoder_outputs,
                        dim=1,
                        index=masked_argmax.unsqueeze(-1).expand(
                            batch_size, 1, self.hidden_dim
                        ),
                    ),
                ],
                dim=1,
            )

        pointer_log_scores = torch.stack(pointer_log_scores, 1)
        pointer_argmaxs = torch.cat(pointer_argmaxs, 1)

        return pointer_log_scores, pointer_argmaxs, mask_tensor

    def _generate_square_subsequent_mask(self, sz: int) -> Tensor:
        mask = torch.triu(torch.ones(sz, sz)) == 1
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask


# Adopted from (https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py)
def masked_log_softmax(vector: Tensor, mask: Tensor, dim: int = -1) -> Tensor:
    """
    Performs a log_softmax on just the non-masked portions of a vector.

    Args:
        vector (Tensor): Input tensor that can have an arbitrary number of dimensions.
        mask (Tensor, optional): Mask tensor that is broadcastable to the vector's shape.
            If mask has fewer dimensions than vector, it will be unsqueezed on dimension 1 until they match.
            If None, a regular log_softmax is performed.

    Returns:
        Tensor: The result of the log_softmax operation. In the case that the input vector is completely masked,
            the return value is arbitrary, but not nan.

    Notes:
        - You should be masking the result of whatever computation comes out of this in the case of a completely
            masked input vector, so the specific values returned shouldn't matter.
        - The way that we deal with a completely masked input vector relies on having single-precision floats;
            mixing half-precision floats with fully-masked vectors will likely give you nans.
        - If your logits are all extremely negative (i.e., the max value in your logit vector is -50 or lower),
            the way we handle masking here could mess you up. But if you've got logit values that extreme,
            you've got bigger problems than this.
    """
    if mask is not None:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        vector = vector + (mask + torch.finfo(mask.dtype).eps).log()
    return torch.nn.functional.log_softmax(vector, dim=dim)


def masked_max(
    vector: Tensor, mask: Tensor, dim: int, keepdim: bool = False, min_val: float = -1e7
) -> tuple[Tensor, Tensor]:
    """
    Calculates max along certain dimensions on masked values.

    Args:
        vector (Tensor): The vector to calculate max, assume unmasked parts are already zeros.
        mask (Tensor): The mask of the vector. It must be broadcastable with vector.
        dim (int): The dimension to calculate max.
        keepdim (bool, optional): Whether to keep dimension. Defaults to False.
        min_val (float, optional): The minimal value for paddings. Defaults to -1e7.

    Returns:
        tuple[Tensor, Tensor]:
            - A Tensor including the maximum values.
            - The indices of the maximum values.
    """
    one_minus_mask = (1.0 - mask).byte()
    replaced_vector = vector.masked_fill(one_minus_mask, min_val)
    max_value, max_index = replaced_vector.max(dim=dim, keepdim=keepdim)
    return max_value, max_index


# Unit tests to check tensor shapes and types
def test_pointer_transformer():
    batch_size = 2
    seq_len = 5
    input_dim = 10
    embedding_dim = 16
    hidden_dim = 16
    num_heads = 2
    num_layers = 2
    dim_feedforward = 64
    dropout = 0.1

    model = PointerTransformer(
        input_dim,
        embedding_dim,
        hidden_dim,
        num_heads,
        num_layers,
        dim_feedforward,
        dropout,
    )
    input_seq = torch.randn(batch_size, seq_len, input_dim)
    input_lengths = torch.tensor([seq_len, seq_len - 1])

    log_scores, argmaxs, mask = model(input_seq, input_lengths)

    assert log_scores.shape == (batch_size, seq_len, seq_len)
    assert argmaxs.shape == (batch_size, seq_len)
    assert mask.shape == (batch_size, seq_len)

    print("PointerTransformer test passed!")


# Run the test
test_pointer_transformer()
