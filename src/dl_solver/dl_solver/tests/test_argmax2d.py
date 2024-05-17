import unittest

import torch

from dl_solver import nn_utils


class TestArgmax2d(unittest.TestCase):
    def test_argmax2d(self):
        # Test case 1: Basic functionality
        joint_logits = torch.tensor(
            [[[[0.1, 0.2], [0.4, 0.3]], [[0.5, 0.6], [0.7, 0.8]]]]
        )  # Shape [1, 2, 2, 2]
        expected_output = torch.tensor([[[1, 0], [1, 1]]])  # Shape [1, 2, 2]
        output = nn_utils.argmax2d(joint_logits)
        self.assertTrue(torch.equal(output, expected_output))

        # Test case 2: Multiple batches
        joint_logits = torch.tensor(
            [
                [[[0.1, 0.8], [0.4, 0.3]], [[0.2, 0.5], [0.7, 0.1]]],
                [[[0.9, 0.2], [0.6, 0.3]], [[0.4, 0.7], [0.8, 0.1]]],
            ]
        )  # Shape [2, 2, 2, 2]
        expected_output = torch.tensor(
            [[[0, 1], [1, 0]], [[0, 0], [1, 0]]]
        )  # Shape [2, 2, 2]
        output = nn_utils.argmax2d(joint_logits)
        self.assertTrue(torch.equal(output, expected_output))

        # Test case 3: Single layer
        joint_logits = torch.tensor([[[0.1, 0.2], [0.3, 0.4]]])  # Shape [1, 2, 2]
        expected_output = torch.tensor([[[1, 1]]])  # Shape [1, 1, 2]
        output = nn_utils.argmax2d(joint_logits)
        self.assertTrue(torch.equal(output, expected_output))


if __name__ == "__main__":
    unittest.main()
