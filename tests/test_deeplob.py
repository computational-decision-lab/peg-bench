from __future__ import annotations

import unittest

import torch

from src.models.deeplob import DeepLOBConfig, DeepLOBTimeBarModel


class DeepLOBTests(unittest.TestCase):
    def test_forward_shapes(self) -> None:
        model = DeepLOBTimeBarModel(DeepLOBConfig(seq_len=50, input_dim=80))
        batch = torch.randn(4, 50, 4, 10, 2)
        outputs = model(batch)
        self.assertEqual(outputs["return_pred"].shape, (4,))
        self.assertEqual(outputs["direction_logit"].shape, (4,))


if __name__ == "__main__":
    unittest.main()
