import unittest

import torch

from utils.logging_utils import get_gradient_alignment_stats


class GradientAlignmentDiagnosticsTest(unittest.TestCase):
    def test_parallel_gradients_have_unit_alignment(self):
        stats = get_gradient_alignment_stats(
            (torch.tensor([1.0, -2.0]),),
            (torch.tensor([2.0, -4.0]),),
        )

        self.assertAlmostEqual(stats['cosine'], 1.0)
        self.assertEqual(stats['usable'], 1.0)
        self.assertEqual(stats['paired_elements'], 2.0)

    def test_opposite_gradients_have_negative_unit_alignment(self):
        stats = get_gradient_alignment_stats(
            (torch.tensor([1.0, -2.0]),),
            (torch.tensor([-3.0, 6.0]),),
        )

        self.assertAlmostEqual(stats['cosine'], -1.0)
        self.assertEqual(stats['usable'], 1.0)

    def test_unpaired_and_nonfinite_entries_do_not_corrupt_metric(self):
        stats = get_gradient_alignment_stats(
            (None, torch.tensor([1.0, float('nan'), 2.0])),
            (torch.tensor([3.0]), torch.tensor([1.0, 5.0, 2.0])),
        )

        self.assertAlmostEqual(stats['cosine'], 1.0)
        self.assertEqual(stats['paired_elements'], 2.0)
        self.assertEqual(stats['usable'], 1.0)


if __name__ == '__main__':
    unittest.main()
