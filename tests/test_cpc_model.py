
import unittest

import torch

import contrastlib


class TestContrastivePredictiveModel(unittest.TestCase):

    def setUp(self):
        self.model = contrastlib.ContrastivePredictiveModel()

    def test_forward(self):
        x = torch.rand(8, 6, 3, 64, 64)
        c = self.model(x)

        self.assertTupleEqual(c.size(), (8, 6, 10))

    def test_loss_func(self):
        x_p = torch.rand(8, 6, 3, 64, 64)
        x_n = torch.rand(8, 6, 3, 64, 64)
        loss_dict = self.model.loss_func(x_p, x_n)

        self.assertIsInstance(loss_dict, dict)
        self.assertFalse(torch.isnan(loss_dict["loss"]).any())


if __name__ == "__main__":
    unittest.main()
