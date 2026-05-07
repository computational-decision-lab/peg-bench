from __future__ import annotations

import unittest

import torch

from src.models.baseline import KairosBaselineConfig, KairosBaselineModel
from src.models.policy import GaussianActionPolicy, GaussianPolicyConfig


class PolicyTest(unittest.TestCase):
    def test_policy_outputs_valid_shapes(self) -> None:
        baseline = KairosBaselineModel(KairosBaselineConfig(seq_len=8))
        policy = GaussianActionPolicy(GaussianPolicyConfig(baseline=baseline.config), backbone=baseline)
        tokens = torch.zeros((2, 8), dtype=torch.long)
        risk_ids = torch.tensor([0, 1], dtype=torch.long)
        mean, std = policy(tokens, risk_ids=risk_ids)
        self.assertEqual(mean.shape, torch.Size([2]))
        self.assertEqual(std.shape, torch.Size([2]))
        self.assertTrue(torch.all(std > 0))


if __name__ == "__main__":
    unittest.main()
