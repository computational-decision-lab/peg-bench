from __future__ import annotations

import unittest

from src.simulation.cost_model import CapacityAwareCostModel


class CostModelTest(unittest.TestCase):
    def test_cost_increases_with_trade_size(self) -> None:
        model = CapacityAwareCostModel()
        small = model.total_cost_decimal(0.1, 10.0, 0.001)
        large = model.total_cost_decimal(1.0, 10.0, 0.001)
        self.assertGreater(large, small)


if __name__ == "__main__":
    unittest.main()
