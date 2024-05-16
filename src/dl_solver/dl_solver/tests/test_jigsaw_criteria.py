import unittest
from dataclasses import fields

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from dl_solver import Config, HyperParameters, JigsawCriteria


class TestJigsawCriteria(unittest.TestCase):
    def setUp(self):
        # Mock configuration and hyperparameters
        config = Config(paths={"loss_df": "/mnt/c/Users/jandu/Downloads/losses.csv"})
        hparams = HyperParameters(
            w_ce_pos_loss=1.0,
            w_ce_rot_loss=1.0,
            w_mse_loss=1.0,
            w_unique_loss=1.0,
            unique_cost_sigma=0.5,
        )

        # Initialize JigsawCriteria with a dummy DataFrame
        self.criteria = JigsawCriteria(config, hparams)

        # Example tensors for y_pred and y
        self.y_pred = (
            torch.rand(1, 12, 3, device="cpu", dtype=torch.float32),  # pos_seq
            (
                torch.rand(1, 12, 3, device="cpu", dtype=torch.float32),
                torch.rand(1, 12, 4, device="cpu", dtype=torch.float32),
                torch.rand(1, 12, 3, device="cpu", dtype=torch.float32),
            ),  # logits
        )
        self.y = torch.randint(0, 3, (1, 12, 3), device="cpu", dtype=torch.int64)

    # def test_gradient_flow(self):
    #     self.y_pred[0].requires_grad_(True)
    #     criteria = self.criteria.forward(self.y_pred, self.y, 356, "fit")
    #     loss = criteria.losses.total_loss
    #     self.assertTrue(loss.requires_grad)

    # def test_update_df(self):
    #     self.y_pred[0].requires_grad_(True)
    #     _ = self.criteria.forward(self.y_pred, self.y, 356, "fit")
    #     prev_len = len(self.criteria.loss_df)
    #     self.assertTrue(len(self.criteria.cached_losses) > 0)

    def test_penalty_for_identical_positions(self):
        positions = torch.tensor(
            [[1.0, 1.0], [1.0, 1.0]], dtype=torch.float32
        )  # Identical positions
        penalty = self.criteria.soft_unique_penalty(positions.unsqueeze(0))
        # Check if the penalty is high for identical positions
        self.assertTrue(
            penalty.mean().item() > 0, "Penalty should be high for identical positions."
        )

    def test_penalty_for_unique_positions(self):
        positions = torch.tensor(
            [[0.0, 0.0], [10.0, 10.0]], dtype=torch.float32
        )  # Distinct positions
        penalty = self.criteria.soft_unique_penalty(positions.unsqueeze(0))
        # Penalty should be close to zero for highly distinct positions
        self.assertTrue(torch.allclose(penalty, torch.zeros_like(penalty)))

    def test_penalty_for_increasing_distances(self):
        positions = torch.tensor(
            [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=torch.float32
        )
        penalties = [
            self.criteria.soft_unique_penalty(positions[: i + 1].unsqueeze(0))
            .mean()
            .item()
            for i in range(1, len(positions))
        ]
        self.assertTrue(
            all(p1 > p2 for p1, p2 in zip(penalties, penalties[1:])),
            "Penalty should decrease with increasing distances.",
        )

    def test_plot_penalty_vs_distance(self):
        distances = np.linspace(0, 1, 100)
        penalties = [
            self.criteria.soft_unique_penalty(
                torch.tensor(
                    [[0, 0], [np.sqrt(d), np.sqrt(d)]], dtype=torch.float32
                ).unsqueeze(0)
            )
            .mean()
            .item()
            for d in distances
        ]
        plt.plot(distances, penalties)
        plt.xlabel("Distance")
        plt.ylabel("Penalty")
        plt.title("Penalty vs. Distance Between Two Positions")
        plt.show()


if __name__ == "__main__":
    unittest.main()
