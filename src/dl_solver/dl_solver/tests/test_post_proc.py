import unittest

import torch
from torch.functional import F

from dl_solver import HyperParameters, PatchNet


class TestPatchNet(unittest.TestCase):

    def setUp(self):
        # Create a PatchNet instance with some hypothetical hyperparameters
        hparams = HyperParameters(
            puzzle_shape=(10, 10),
            num_features_out=128,
            backbone_is_trainable=True,
            softmax_temperature=1.0,
            gumbel_temperature=0.5,
            non_unique_penalty=0.5,
            num_post_iters=10,
        )
        self.model = PatchNet(hparams)
        self.num_rows = 3
        self.num_cols = 4
        self.num_pieces = (
            self.num_rows * self.num_cols
        )  # Should equal num_rows * num_cols

        # Mock data
        self.x = torch.rand(
            2, 12, 3, 48, 48
        )  # Example shape: (batch_size, num_pieces, channels, height, width)
        self.pos_seq = torch.rand(
            2, 12, 3
        )  # Example shape: (batch_size, num_pieces, pos_dims)
        self.encoder_memory = None

        # Mock logits for row and column predictions
        self.row_logits = torch.randn(2, 12, 3)  # Shape: (B, L, num_rows)
        self.col_logits = torch.randn(2, 12, 4)  # Shape: (B, L, num_cols)

    def test_soft_max(self):
        # Compute softmax probabilities with temperature scaling
        row_probs = F.softmax(self.row_logits, dim=-1)

        print(row_probs.sum(dim=-1))
        self.assertTrue(
            torch.allclose(row_probs.sum(dim=-1), torch.tensor(1.0).repeat((2, 12)))
        )

    def test_compute_joint_probabilities(self):
        joint_probs = self.model._compute_joint_probabilities(
            self.row_logits, self.col_logits
        )
        self.assertEqual(joint_probs.shape, (2, 12, 3, 4))
        self.assertTrue(torch.all(joint_probs >= 0))
        self.assertTrue(torch.all(joint_probs <= 1))

    def test_check_unique_indices(self):
        # Create a spatial_indices tensor where some indices are intentionally duplicated
        spatial_indices = torch.tensor(
            [[[0, 1], [0, 1], [2, 3]], [[4, 5], [6, 7], [6, 7]]]
        )
        unique_mask = self.model._check_unique_indices(spatial_indices)
        self.assertEqual(unique_mask.shape, (2, 3))
        self.assertTrue(
            torch.all(
                unique_mask
                == torch.tensor([[False, False, True], [True, False, False]])
            )
        )

    def test_no_competition(self):
        """Test scenario where each piece picks a unique class, no penalties needed."""
        # Create a dummy input where each piece uniquely selects a class
        joint_probs = torch.eye(self.num_pieces).reshape(
            1, self.num_pieces, self.num_rows, self.num_cols
        )
        adjusted_probs = self.model.apply_penalties(joint_probs)
        self.assertTrue(torch.allclose(adjusted_probs.argmax(1), joint_probs.argmax(1)))
        self.assertTrue(
            torch.allclose(adjusted_probs.argmax(-1), joint_probs.argmax(-1))
        )

    def test_full_competition(self):
        """Test where all pieces compete for the same class, checking if penalties are applied."""
        joint_probs = torch.full(
            (1, self.num_pieces, self.num_rows, self.num_cols), fill_value=0.1
        )
        joint_probs[:, :, 0, 0] = torch.arange(1, self.num_pieces + 1)
        adjusted_probs = self.model.apply_penalties(joint_probs)
        self.assertTrue(
            torch.any(adjusted_probs[:, :, 0, 0] < joint_probs[:, :, 0, 0]),
            "Penalties should reduce probabilities of the competed class.",
        )

    def test_probability_normalization(self):
        """Ensure that probabilities are normalized correctly after applying penalties."""
        joint_probs = torch.rand(1, self.num_pieces, self.num_rows, self.num_cols)
        adjusted_probs = self.model.apply_penalties(joint_probs)
        sums = adjusted_probs.sum(dim=-1).sum(dim=-1)
        expected_sums = torch.ones_like(sums)
        self.assertTrue(
            torch.allclose(sums, expected_sums),
            "Probabilities should sum to 1 across all classes.",
        )


if __name__ == "__main__":
    unittest.main()
