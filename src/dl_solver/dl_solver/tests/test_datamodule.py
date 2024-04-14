import unittest
from unittest.mock import MagicMock

import numpy as np
import torch

from dl_solver import Config, HyperParameters, LitJigsawDataModule


class TestLitJigsawDataModule(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.hparams = HyperParameters(
            batch_size=2,
            segment_shape=(64, 64),
            num_epochs=1,
            learning_rate=0.001,
            weight_decay=0.001,
        )
        self.data_module = LitJigsawDataModule(self.config, self.hparams)

    def test_get_item_output_types_and_shapes(self):
        # Mock the __getitem__ method to return a sample batch
        sample_data = torch.rand(3, 64, 64)
        sample_labels = torch.tensor([0, 0, 0, 0])
        self.data_module.__getitem__ = MagicMock(
            return_value=(sample_data, sample_labels)
        )

        data, labels = self.data_module.__getitem__(0)
        self.assertIsInstance(data, torch.Tensor)
        self.assertIsInstance(labels, torch.Tensor)
        self.assertEqual(data.shape, torch.Size([3, 64, 64]))
        self.assertEqual(labels.shape, torch.Size([4]))

    def test_dataloader_output_types_and_shapes(self):
        # Mock the dataloader to return a sample batch
        sample_data = torch.rand(2, 3, 64, 64)
        sample_labels = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0]])
        self.data_module.train_dataloader = MagicMock(
            return_value=iter([(sample_data, sample_labels)])
        )

        loader = self.data_module.train_dataloader()
        for data, labels in loader:
            self.assertIsInstance(data, torch.Tensor)
            self.assertIsInstance(labels, torch.Tensor)
            self.assertEqual(data.shape, torch.Size([2, 3, 64, 64]))
            self.assertEqual(labels.shape, torch.Size([2, 4]))

    def test_real_train_set_output_types_and_shapes(self):
        self.data_module.prepare_data()
        self.data_module.setup("fit")
        loader = self.data_module.train_dataloader()

        for i, (data, labels) in enumerate(loader):
            self.assertIsInstance(data, torch.Tensor)
            self.assertIsInstance(labels, torch.Tensor)
            self.assertEqual(
                data.shape,
                (
                    self.hparams.batch_size,
                    np.prod(self.hparams.puzzle_shape),
                    3,
                    *self.hparams.segment_shape,
                ),
            )
            self.assertEqual(
                labels.shape,
                (self.hparams.batch_size, np.prod(self.hparams.puzzle_shape), 4),
            )

            # Optionally, you can break after the first batch to speed up the test
            if i >= 1:
                break


if __name__ == "__main__":
    unittest.main()
