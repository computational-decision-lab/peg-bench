from __future__ import annotations

import unittest

import numpy as np

from src.preference.filtering import AdaptiveConformalController


class ACITest(unittest.TestCase):
    def test_alpha_updates_and_pair_filtering(self) -> None:
        controller = AdaptiveConformalController()
        controller.fit(np.array([0.0, 1.0, 2.0]), np.array([0.0, 0.5, 1.5]))
        new_alpha = controller.update(0.2)
        self.assertGreater(new_alpha, 0.0)
        keep, _, _ = controller.keep_preference(1.0, 0.0)
        self.assertIsInstance(keep, bool)


if __name__ == "__main__":
    unittest.main()
