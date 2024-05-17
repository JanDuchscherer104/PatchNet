from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, TypedDict

import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor

from .config import Config, HyperParameters


class Losses(TypedDict):
    ce_type: Tensor  # dtype: torch.float32, shape: scalar
    ce_row: Tensor  # dtype: torch.float32, shape: scalar
    ce_col: Tensor  # dtype: torch.float32, shape: scalar
    ce_rot: Tensor  # dtype: torch.float32, shape: scalar
    mse_pos: Tensor  # dtype: torch.float32, shape: scalar
    unique: Tensor  # dtype: torch.float32, shape: scalar
    total_loss: Tensor  # dtype: torch.float32, shape: scalar


class Accuracies(TypedDict):
    type_accuracy: Tensor  # dtype: torch.float32, shape: scalar
    row_accuracy: Tensor  # dtype: torch.float32, shape: scalar
    col_accuracy: Tensor  # dtype: torch.float32, shape: scalar
    rot_accuracy: Tensor  # dtype: torch.float32, shape: scalar
    pos_accuracy: Tensor  # dtype: torch.float32, shape: scalar
    total_accuracy: Tensor  # dtype: torch.float32, shape: scalar


class RoundedPreds(TypedDict):
    row_preds: Tensor  # dtype: torch.int64, shape: (B, num_pieces)
    col_preds: Tensor  # dtype: torch.int64, shape: (B, num_pieces)
    rot_preds: Tensor  # dtype: torch.int64, shape: (B, num_pieces)


@dataclass
class Criteria:
    losses: Losses
    accuracies: Accuracies
    rounded_preds: Optional[RoundedPreds] = None


class JigsawCriteria(nn.Module):
    config: Config
    hparams: HyperParameters
    loss_df: pd.DataFrame

    cached_losses: List[Tuple[int, Losses]]

    mse_loss: nn.Module
    ce_loss: nn.Module

    def __init__(
        self,
        config: Config,
        hparams: HyperParameters,
    ):
        """
        Initialize the dynamic loss scaler with a DataFrame containing initial losses and the number of steps per epoch.
        """
        super(JigsawCriteria, self).__init__()
        self.config = config
        self.hparams = hparams

        self.loss_df = pd.read_csv(self.config.paths.loss_df).astype("float32")
        self.loss_means = self.loss_df.mean()
        self.loss_stds = self.loss_df.std()
        self.cached_losses = []

        self.mse_loss = nn.MSELoss(reduction="mean")
        self.ce_loss = nn.CrossEntropyLoss()

    def compute_loss(
        self,
        y_pred: Tuple[Tensor, Tensor, Tuple[Tensor, Tensor, Tensor]],
        y: Tensor,
    ) -> Losses:
        """
        Compute the combined loss of MSE for positions and CrossEntropy for classifications.

        Args:
            y_pred: a tuple containing
                puzzle_type_logits: Tensor[torch.float32] - (B, num_pieces, 3) [corner, edge, center]
                pos_seq: Tensor[torch.float32] - (B, num_pieces, 3) [row_idx, col_idx, rotation]
                logits: Tuple[Tensor[torch.float32], Tensor[torch.float32], Tensor[torch.float32]]
                    row_logits: Tensor[torch.float32] - (B, num_pieces, max_rows)
                    col_logits: Tensor[torch.float32] - (B, num_pieces, max_cols)
                    rotation_logits: Tensor[torch.float32] - (B, num_pieces, 3)
            y: Tensor of true labels of shape (B, 12, 3) [row~_idx, col_idx, rotation]

        Returns:
            Dict: The differern Losses as scalar tensors

        >>> total_loss
        tensor(5.2782, grad_fn=<AddBackward0>)
        >>> ce_loss_cols
        tensor(1.5307, grad_fn=<NllLoss2DBackward0>)
        >>> ce_loss_rot
        tensor(1.4689, grad_fn=<NllLoss2DBackward0>)
        >>> ce_loss_rows
        tensor(1.1953, grad_fn=<NllLoss2DBackward0>)
        >>> mse_loss_position
        tensor(1.0833)
        >>> unique_loss
        tensor(2.)
        """
        puzzle_type_logits, pos_seq, (row_logits, col_logits, rot_logits) = y_pred

        # Unpack true values
        y_rows, y_cols, y_rot = y[..., 0], y[..., 1], y[..., 2]

        self.y_puzzle_type = self.get_puzzle_type_labels(y_rows, y_cols)
        ce_loss_type = self.ce_loss(
            puzzle_type_logits.permute(0, 2, 1), self.y_puzzle_type
        )

        # Calculate MSE loss for row / col indices
        mse_loss_position = self.mse_loss(pos_seq[..., :2], y[..., :2].float())

        # Uniqeness Loss
        unique_loss = self.soft_unique_penalty(pos_seq[..., :2]).mean()

        # nn.CrossEntropyLoss expects y_pred.shape = (B, C, D) and y.shape = (B, D)
        ce_loss_rows = self.ce_loss(row_logits.permute(0, 2, 1), y_rows)
        ce_loss_cols = self.ce_loss(col_logits.permute(0, 2, 1), y_cols)
        ce_loss_rot = self.ce_loss(rot_logits.permute(0, 2, 1), y_rot)

        total_loss = (
            (ce_loss_rows + ce_loss_cols) * self.hparams.w_ce_pos_loss
            + ce_loss_rot * self.hparams.w_ce_rot_loss
            + mse_loss_position * self.hparams.w_mse_loss
            + unique_loss * self.hparams.w_unique_loss
            + ce_loss_type * self.hparams.w_ce_type_loss
        )

        return Losses(
            ce_type=ce_loss_type,
            ce_row=ce_loss_rows,
            ce_col=ce_loss_cols,
            ce_rot=ce_loss_rot,
            mse_pos=mse_loss_position,
            unique=unique_loss,
            total_loss=total_loss,
        )

    def get_puzzle_type_labels(self, rows: Tensor, cols: Tensor) -> Tensor:
        """
        Generate the puzzle type labels based on the positions.
        """
        num_rows, num_cols = self.hparams.puzzle_shape
        is_corner = (
            (rows == 0) | (rows == num_rows - 1) & (cols == 0) | (cols == num_cols - 1)
        )
        is_edge = (
            (rows == 0) | (rows == num_rows - 1) | (cols == 0) | (cols == num_cols - 1)
        )
        puzzle_type = torch.full_like(rows, 2)  # Initialize with center type
        puzzle_type[is_corner] = 0
        puzzle_type[is_edge & ~is_corner] = 1
        return puzzle_type

    def soft_unique_penalty(self, positions: Tensor) -> Tensor:
        """
        Apply a soft penalty to encourage uniqueness in position assignments.

        Args:
            positions (Tensor): [B, num_pieces, 2] tensor containing x, y coordinates.

        Returns:
            Tensor: [B, num_pieces] tensor of penalties for each position in the batch.
        """
        # Calculate the squared differences between all pairs of positions
        diff = positions.unsqueeze(2) - positions.unsqueeze(
            1
        )  # Shape: [B, num_pieces, num_pieces, 2]
        norm = torch.norm(
            diff, dim=-1, p=2
        )  # Euclidean distance, lishape: [B, num_pieces, num_pieces]

        # Apply a Gaussian-like penalty for close positions
        sigma = self.hparams.unique_cost_sigma  # Width of the Gaussian
        penalties = torch.exp(-(norm**2) / (2 * sigma**2))

        # Sum penalties for each position, excluding self-to-self comparison (diagonal elements)
        eye = torch.eye(positions.size(1), device=positions.device).unsqueeze(0)
        penalties = (1 - eye) * penalties
        penalty = penalties.sum(dim=-1)

        return penalty

    def forward(
        self,
        y_pred: Tuple[Tensor, Tensor, Tuple[Tensor, Tensor, Tensor]],
        y: Tensor,
        step_idx: int,
        stage: Literal["fit", "validate", "test"],
    ) -> Criteria:
        losses = self.compute_loss(y_pred, y)
        accuracies, rounded_preds = self.fetch_rounded_preds_and_compute_accuracy(
            y_pred[:2], y
        )

        if stage == "fit":
            self.cached_losses.append((step_idx, self.detach_losses(losses)))

        return Criteria(
            losses=losses, accuracies=accuracies, rounded_preds=rounded_preds
        )

    def detach_losses(self, losses: Losses) -> Losses:
        return Losses(
            ce_type=losses["ce_type"].clone().detach().cpu().item(),
            ce_row=losses["ce_row"].clone().detach().cpu().item(),
            ce_col=losses["ce_col"].clone().detach().cpu().item(),
            ce_rot=losses["ce_rot"].clone().detach().cpu().item(),
            mse_pos=losses["mse_pos"].clone().detach().cpu().item(),
            unique=losses["unique"].clone().detach().cpu().item(),
            total_loss=losses["total_loss"].clone().detach().cpu().item(),
        )

    def update_scaling_factors(self):
        """
        Update scaling factors based on new observed data.
        """
        current_means = self.loss_df.mean()
        current_stds = self.loss_df.std()
        self.loss_means.update(current_means)
        self.loss_stds.update(current_stds)

    def fetch_rounded_preds_and_compute_accuracy(
        self, y_pred_tup: Tuple[Tensor, Tensor], y: Tensor
    ) -> Tuple[Accuracies, RoundedPreds]:
        y_pred_type_logits, y_pred = [t.clone().detach().round() for t in y_pred_tup]

        puzzle_type_accuracy = self.y_puzzle_type == y_pred_type_logits.argmax(dim=-1)
        self.y_puzzle_type = None

        row_preds, col_preds, rot_preds = (
            y_pred[..., 0],
            y_pred[..., 1],
            y_pred[..., 2],
        )
        correct_rows = row_preds == y[..., 0]
        correct_cols = col_preds == y[..., 1]
        correct_rots = rot_preds == y[..., 2]

        # Position accuracy: both row and column are correct
        correct_positions = correct_rows & correct_cols
        pos_acc = correct_positions.float().mean()

        # Total accuracy: row, column, and rotation all are correct
        correct_total = correct_positions & correct_rots
        total_acc = correct_total.float().mean()

        return (
            Accuracies(
                type_accuracy=puzzle_type_accuracy.float().mean(),
                row_accuracy=correct_rows.float().mean(),
                col_accuracy=correct_cols.float().mean(),
                rot_accuracy=correct_rots.float().mean(),
                pos_accuracy=pos_acc,
                total_accuracy=total_acc,
            ),
            RoundedPreds(
                row_preds=row_preds.to(torch.int64),
                col_preds=col_preds.to(torch.int64),
                rot_preds=rot_preds.to(torch.int64),
            ),
        )

    def save_losses_to_dataframe(self):
        """
        Convert cached losses to a DataFrame and append to the existing DataFrame, then save to a CSV file.
        """
        # Convert cached losses to a list of dictionaries for DataFrame conversion
        new_data = []
        for step_idx, loss in self.cached_losses:
            loss_dict = {
                "Step": step_idx,
                "ce_loss_col": loss["ce_col"],
                "ce_loss_rot": loss["ce_rot"],
                "ce_loss_row": loss["ce_row"],
                "mse_loss_pos": loss["mse_pos"],
                "unique_loss": loss["unique"],
            }
            new_data.append(loss_dict)

        # Create a DataFrame from the new data
        new_df = pd.DataFrame(new_data)

        # Append new data to the existing DataFrame
        self.loss_df = pd.concat([self.loss_df, new_df], ignore_index=True)

        # Save updated DataFrame to CSV
        self.loss_df.to_csv(self.config.paths.loss_df, index=False)

        # Reset cached_losses after saving
        self.cached_losses = []
