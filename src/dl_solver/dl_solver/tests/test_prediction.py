import unittest

import torch

from dl_solver import (
    Config,
    HyperParameters,
    LitJigsawDatamodule,
    LitJigsawModule,
    TrainerFactory,
)


class TestDifferentiablePrediction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Setting up the environment and loading a batch
        cls.hparams = HyperParameters(
            batch_size=2,
            puzzle_shape=(3, 4),
            num_features_out=128,
            backbone_is_trainable=True,
        )
        cls.config = Config(is_debug=True)

        cls.data_module = LitJigsawDatamodule(cls.config, cls.hparams)
        cls.module = LitJigsawModule(cls.config, cls.hparams)

        cls.data_module.setup(stage="fit")

        cls.train_dataloader = cls.data_module.train_dataloader()
        cls.batch = next(iter(cls.train_dataloader))

    def test_differentiable_prediction(self):
        # Simulating the forward pass to get logits
        x, y = self.batch
        logits = self.module(x, y)

        # Assuming logits is a tuple of (row_logits, col_logits, rot_logits)
        row_logits, col_logits, rot_logits = logits

        # Call the method to test
        pos_seq = self.module.differentiable_prediction(
            row_logits, col_logits, rot_logits
        )

        # Assertions to check output shape and differentiability
        self.assertIsInstance(pos_seq, torch.Tensor)
        self.assertEqual(
            pos_seq.shape,
            (
                self.hparams.batch_size,
                self.hparams.puzzle_shape[0] * self.hparams.puzzle_shape[1],
                3,
            ),
        )
        self.assertTrue(pos_seq.requires_grad)  # Checking for differentiability

        # Optional: Check values range, etc.
        rows, cols, rots = pos_seq[..., 0], pos_seq[..., 1], pos_seq[..., 2]
        self.assertTrue(torch.all(rows.ge(0) & rows.lt(self.hparams.puzzle_shape[0])))
        self.assertTrue(torch.all(cols.ge(0) & cols.lt(self.hparams.puzzle_shape[1])))
        self.assertTrue(torch.all(rots.ge(0) & rots.lt(rot_logits.shape[-1])))


if __name__ == "__main__":
    unittest.main()
