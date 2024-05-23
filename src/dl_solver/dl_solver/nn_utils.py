import numpy as np
import torch
from scipy import optimize
from torch import Tensor
from torch.functional import F


class nn_utils:
    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def softargmax1d(input: Tensor, beta=100) -> Tensor:
        # https://github.com/david-wb/softargmax
        *_, n = input.shape
        input = F.softmax(beta * input, dim=-1)
        indices = torch.linspace(0, 1, n, device=input.device)
        result = torch.sum((n - 1) * input * indices, dim=-1)
        return result

    @staticmethod
    def softargmax2d(input: Tensor, beta=100) -> Tensor:
        # https://github.com/david-wb/softargmax
        *_, h, w = input.shape

        input = input.reshape(*_, h * w)
        input = F.softmax(beta * input, dim=-1)

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

    @staticmethod
    def argmax2d(joint_logits: Tensor) -> Tensor:
        """
        Computes the argmax over 2D logits.
        Args:
            joint_logits: Tensor [B, L, num_rows, num_cols]

        Returns:
            Tensor [B, L, 2] - Indices for rows and columns
        """
        if joint_logits.ndim == 3:
            joint_logits = joint_logits.unsqueeze(1)
        B, L, _, num_cols = joint_logits.shape
        flat_indices = torch.argmax(joint_logits.view(B, L, -1), dim=-1)

        row_indices = flat_indices // num_cols
        col_indices = flat_indices % num_cols

        return torch.stack([row_indices, col_indices], dim=-1)

    @staticmethod
    def linear_sum_assignment(joint_probs: Tensor) -> Tensor:
        """
        Computes the linear sum assignment for the joint probabilities.

        Args:
            joint_probs (Tensor['B, L, num_rows, num_cols']) of Joint Probabilities.

        Returns:
            Tensor [B, L, 2] - Indices for rows and columns
        """
        with torch.no_grad():
            B, L, H, W = joint_probs.shape
            flat_probs = joint_probs.view(B, L, H * W)

            cost_matrices = -flat_probs.log().cpu().numpy()

            row_indices = torch.zeros(B, L, dtype=torch.long, device="cpu")
            col_indices = torch.zeros(B, L, dtype=torch.long, device="cpu")

            for b in range(B):
                row_ind, col_ind = optimize.linear_sum_assignment(cost_matrices[b])
                row_indices[b, torch.from_numpy(row_ind)] = (
                    torch.from_numpy(col_ind) // W
                )
                col_indices[b, torch.from_numpy(row_ind)] = (
                    torch.from_numpy(col_ind) % W
                )

            return torch.stack((row_indices, col_indices), dim=-1).to(flat_probs)


if __name__ == "__main__":
    import unittest

    import torch
    from torch import Tensor

    class TestLinearSumAssignment(unittest.TestCase):

        def test_linear_sum_assignment(self):
            """
            Test for the linear_sum_assignment function.
            """
            joint_probs = torch.tensor(
                [
                    [
                        [[0.8, 0.2], [0.3, 0.7]],
                        [[0.6, 0.4], [0.5, 0.5]],
                        [[0.4, 0.6], [0.2, 0.8]],
                        [[0.9, 0.1], [0.1, 0.9]],
                    ],
                    [
                        [[0.1, 0.9], [0.4, 0.6]],
                        [[0.3, 0.7], [0.5, 0.5]],
                        [[0.8, 0.2], [0.6, 0.4]],
                        [[0.9, 0.1], [0.7, 0.3]],
                    ],
                ]
            )  # Example joint probabilities (B, L, H, W)

            indices = nn_utils.linear_sum_assignment(joint_probs)

            # Check the shape of the output
            self.assertEqual(indices.shape, (2, 4, 2))

            # Check if the values are within the expected range
            self.assertTrue((indices >= 0).all())
            self.assertTrue(
                (indices < 2).all()
            )  # Since H and W are 2, valid indices are in the range [0, 1]

            # Check the uniqueness constraint
            for b in range(indices.shape[0]):
                unique_rows = indices[b, :, 0].unique()
                unique_cols = indices[b, :, 1].unique()
                self.assertEqual(len(unique_rows), 2)  # Since H is 2
                self.assertEqual(len(unique_cols), 2)  # Since W is 2

    unittest.main()
