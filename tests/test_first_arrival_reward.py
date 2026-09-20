import unittest

import torch

from utils.tensor_utils import compute_arrival_reward


class FirstArrivalRewardTest(unittest.TestCase):
    @staticmethod
    def _straight_line_distances(distances, requires_grad=False):
        values = torch.tensor(distances, dtype=torch.float64)
        positions = torch.zeros((len(distances), 1, 3), dtype=torch.float64)
        positions[:, 0, 0] = values
        return positions.requires_grad_(requires_grad)

    def test_earlier_first_arrival_has_larger_log_time_reward(self):
        target = torch.zeros((1, 3), dtype=torch.float64)
        early = self._straight_line_distances([2.0, 0.0, 0.0, 0.0, 0.0])
        late = self._straight_line_distances([2.0, 2.0, 2.0, 2.0, 0.0])

        early_reward, early_hit, _ = compute_arrival_reward(
            early, target, radius=0.5, temperature=0.02
        )
        late_reward, late_hit, _ = compute_arrival_reward(
            late, target, radius=0.5, temperature=0.02
        )

        self.assertGreater(early_reward.item(), late_reward.item())
        self.assertEqual(early_hit.item(), 1.0)
        self.assertEqual(late_hit.item(), 1.0)

    def test_never_arriving_has_near_zero_reward(self):
        target = torch.zeros((1, 3), dtype=torch.float64)
        never = self._straight_line_distances([3.0] * 8)

        reward, hit_rate, best_dist = compute_arrival_reward(
            never, target, radius=0.5, temperature=0.05
        )

        self.assertLess(reward.item(), 1e-12)
        self.assertEqual(hit_rate.item(), 0.0)
        self.assertAlmostEqual(best_dist.item(), 3.0, places=6)

    def test_reward_supports_first_and_second_order_gradients(self):
        target = torch.zeros((1, 3), dtype=torch.float64)
        trajectory = self._straight_line_distances(
            [0.9, 0.7, 0.55, 0.48, 0.4], requires_grad=True
        )

        reward, _, _ = compute_arrival_reward(
            trajectory, target, radius=0.5, temperature=0.08
        )
        first_grad = torch.autograd.grad(reward, trajectory, create_graph=True)[0]
        second_grad = torch.autograd.grad(first_grad.square().sum(), trajectory)[0]

        self.assertTrue(torch.isfinite(first_grad).all())
        self.assertTrue(torch.isfinite(second_grad).all())
        self.assertGreater(first_grad.abs().sum().item(), 0.0)
        self.assertGreater(second_grad.abs().sum().item(), 0.0)


if __name__ == '__main__':
    unittest.main()
