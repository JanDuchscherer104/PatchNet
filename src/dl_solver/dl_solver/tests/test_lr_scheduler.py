import unittest

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import Adam

# Assuming HybridScheduler is defined in hybrid_scheduler.py
from dl_solver import HybridScheduler


class TestHybridScheduler(unittest.TestCase):
    def setUp(self):
        self.model = torch.nn.Linear(10, 1)
        self.optimizer = Adam(self.model.parameters(), lr=1e-3)
        self.total_steps = 1000 * 5
        self.scheduler = HybridScheduler(
            self.optimizer, step_size=self.total_steps, patience=3
        )

    def test_hybrid_scheduler_with_losses(self):
        losses = np.concatenate(
            [
                np.sin(np.linspace(0, 10, 333 * 5)),  # Sinusoidal losses
                np.ones(334 * 5),  # Constant losses
                np.sin(np.linspace(0, 10, 333 * 5)),  # Sinusoidal losses
            ]
        )

        # Track learning rates
        lrs = []

        for epoch, loss in enumerate(losses):
            self.scheduler.step(metrics=loss)
            lrs.append(self.optimizer.param_groups[0]["lr"])

        # Create a figure and a set of subplots
        fig, ax1 = plt.subplots()

        # Plot the learning rates
        ax1.plot(lrs, label="Learning Rate", color="tab:blue")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Learning Rate", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax1.set_yscale("log")

        # Instantiate a second y-axis that shares the same x-axis
        ax2 = ax1.twinx()
        ax2.plot(losses, label="Loss", color="tab:red")
        ax2.set_ylabel("Loss", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")

        # Title and grid
        plt.title("Hybrid Scheduler Learning Rate Schedule")
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.grid(True)
        plt.show()

        # Ensure learning rates have been updated
        self.assertNotEqual(len(set(lrs)), 1, "Learning rates did not update correctly")


if __name__ == "__main__":
    unittest.main()
