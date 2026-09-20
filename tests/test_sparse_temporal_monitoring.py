import math
import unittest

import torch

from utils.logging_utils import summarize_temporal_weight_sensitivity


class SparseTemporalMonitoringTest(unittest.TestCase):
    def test_reports_fourfold_late_to_early_decay(self):
        sensitivity = torch.tensor(
            [1.0, 1.0, 2.0, 2.0, 4.0, 4.0],
            dtype=torch.float64,
        ).reshape(6, 1, 1)

        stats = summarize_temporal_weight_sensitivity(sensitivity)

        self.assertEqual(stats["early_abs_mean"], 1.0)
        self.assertEqual(stats["middle_abs_mean"], 2.0)
        self.assertEqual(stats["late_abs_mean"], 4.0)
        self.assertEqual(stats["early_to_late_ratio"], 0.25)
        self.assertEqual(stats["late_to_early_decay_factor"], 4.0)
        self.assertAlmostEqual(
            stats["early_to_late_log10_ratio"],
            math.log10(0.25),
        )
        self.assertEqual(stats["ratio_valid"], 1.0)

    def test_zero_signal_marks_ratio_invalid(self):
        stats = summarize_temporal_weight_sensitivity(torch.zeros(6, 2, 3))

        self.assertEqual(stats["norm"], 0.0)
        self.assertEqual(stats["ratio_valid"], 0.0)

    def test_nonfinite_signal_is_sanitized_and_flagged(self):
        sensitivity = torch.ones(6, 1, 1)
        sensitivity[0] = float("nan")

        stats = summarize_temporal_weight_sensitivity(sensitivity)

        self.assertLess(stats["finite_fraction"], 1.0)
        self.assertEqual(stats["ratio_valid"], 0.0)
        self.assertTrue(math.isfinite(stats["norm"]))


if __name__ == "__main__":
    unittest.main()
