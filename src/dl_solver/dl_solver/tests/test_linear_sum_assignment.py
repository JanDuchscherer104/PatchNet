import unittest

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import Tensor


def optimize_positions(logits):
    if not torch.is_tensor(logits):
        raise TypeError("Logits must be a PyTorch tensor")
    if logits.dtype not in [torch.float32, torch.float64]:
        raise ValueError("Logits tensor must be of float type")

    probabilities = F.softmax(logits, dim=-1)
    cost_matrix = -torch.log(
        probabilities + 1e-9
    )  # Adding a small epsilon to avoid log(0)
    cost_matrix = cost_matrix.detach().cpu().numpy()
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    return row_indices, col_indices


class TestOptimizePositions(unittest.TestCase):
    def test_single_piece_single_position(self):
        logits = Tensor([[0.0]], dtype=torch.float32)
        row_indices, col_indices = optimize_positions(logits)
        self.assertEqual(list(row_indices), [0])
        self.assertEqual(list(col_indices), [0])

    def test_multiple_pieces_single_best_fit(self):
        logits = Tensor([[10, 1, -10], [1, -10, 10], [-10, 10, 1]], dtype=torch.float32)
        row_indices, col_indices = optimizate_positions(logits)
        self.assertCountEqual(row_indices, [0, 1, 2])
        self.assertCountEqual(col_indices, [0, 2, 1])

    def test_identical_logits(self):
        logits = Tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=torch.float32)
        row_indices, col_indices = optimize_positions(logits)
        self.assertCountEqual(row_indices, [0, 1, 2])
        self.assertCountEqual(col_indices, [0, 1, 2])

    def test_invalid_inputs(self):
        logits = "invalid input type"
        with self.assertRaises(TypeError):
            optimize_positions(logits)

    def test_invalid_dtype(self):
        logits = Tensor([[1, 2], [3, 4]], dtype=torch.int)
        with self.assertRaises(ValueError):
            optimize_positions(logits)


if __name__ == "__main__":
    unittest.main()
