import unittest

import torch

from utils.logging_utils import merge_task_priority_gradients


class WorkerGradientBalanceTest(unittest.TestCase):
    def test_conflicting_proxy_component_is_projected_away(self):
        proxy = (torch.tensor([-100.0, 100.0]),)
        task = (torch.tensor([1.0, 0.0]),)

        merged, stats = merge_task_priority_gradients(
            proxy,
            task,
            max_proxy_to_task_ratio=0.5,
        )

        self.assertEqual(stats['conflict_projected'], 1.0)
        self.assertLess(stats['proxy_scale'], 0.01)
        self.assertGreaterEqual(torch.dot(merged[0], task[0]).item(), 1.0)
        self.assertAlmostEqual(merged[0][0].item(), 1.0, places=6)

    def test_aligned_proxy_is_norm_limited_relative_to_task(self):
        proxy = (torch.tensor([100.0, 0.0]),)
        task = (torch.tensor([1.0, 0.0]),)

        merged, stats = merge_task_priority_gradients(
            proxy,
            task,
            max_proxy_to_task_ratio=0.5,
        )

        self.assertEqual(stats['conflict_projected'], 0.0)
        self.assertAlmostEqual(stats['proxy_scale'], 0.005, places=6)
        self.assertTrue(torch.allclose(merged[0], torch.tensor([1.5, 0.0])))

    def test_missing_proxy_gradient_keeps_task_gradient(self):
        merged, stats = merge_task_priority_gradients(
            (None,),
            (torch.tensor([2.0]),),
        )

        self.assertEqual(stats['proxy_scale'], 0.0)
        self.assertTrue(torch.equal(merged[0], torch.tensor([2.0])))


if __name__ == '__main__':
    unittest.main()
