import torch
import torch.nn as nn
from torch import Tensor
from torch.functional import F


def compute_joint_probabilities(row_logits: Tensor, col_logits: Tensor) -> Tensor:
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


def apply_penalties(joint_probs: Tensor) -> Tensor:
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


def softargmax1d(input: Tensor, beta=100) -> Tensor:
    # https://github.com/david-wb/softargmax
    *_, n = input.shape
    input = F.softmax(beta * input, dim=-1)
    indices = torch.linspace(0, 1, n, device=input.device)
    result = torch.sum((n - 1) * input * indices, dim=-1)
    return result


def softargmax2d(input: Tensor, beta=100) -> Tensor:
    # https://github.com/david-wb/softargmax
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


def generate_causal_mask(size: int, device: torch.device) -> Tensor:
    mask = torch.triu(
        torch.ones(size, size, device=device),
        diagonal=1,
    )
    mask[mask == 1] = float("-inf")
    return mask
