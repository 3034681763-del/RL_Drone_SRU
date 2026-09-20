import unittest

import torch

from utils.tensor_utils import (
    compute_action_energy_loss_per_step,
    compute_action_smoothness_losses_per_step,
    compute_goal_progress_preference_loss,
)


class PreferenceLossesTest(unittest.TestCase):
    def test_goal_progress_is_signed_bounded_and_action_aligned(self):
        p_current = torch.zeros(3, 1, 3, dtype=torch.float64)
        p_next = p_current.clone()
        p_next[0, 0, 0] = 1.0   # toward goal
        p_next[1, 0, 0] = -1.0  # away from goal
        target = torch.tensor([[10.0, 0.0, 0.0]], dtype=torch.float64)

        loss = compute_goal_progress_preference_loss(
            p_current,
            p_next,
            target,
            step_scale=1.0,
        )

        self.assertLess(float(loss[0, 0]), 0.0)
        self.assertGreater(float(loss[1, 0]), 0.0)
        self.assertAlmostEqual(float(loss[2, 0]), 0.0, places=10)
        self.assertTrue((loss.abs() <= 1.0).all())

    def test_preference_losses_support_second_order_gradients(self):
        p_current = torch.zeros(2, 1, 3, dtype=torch.float64)
        p_next = torch.tensor(
            [[[0.2, 0.0, 0.0]], [[0.0, 0.1, 0.0]]],
            dtype=torch.float64,
            requires_grad=True,
        )
        target = torch.tensor([[2.0, 0.0, 0.0]], dtype=torch.float64)
        progress = compute_goal_progress_preference_loss(
            p_current, p_next, target, step_scale=0.5
        ).sum()
        first = torch.autograd.grad(progress, p_next, create_graph=True)[0]
        second = torch.autograd.grad(first.sum(), p_next)[0]

        self.assertTrue(torch.isfinite(first).all())
        self.assertTrue(torch.isfinite(second).all())

    def test_smoothness_returns_one_value_per_scored_action(self):
        # Two seed actions followed by three actions.
        act_buffer = torch.zeros(5, 2, 3, dtype=torch.float64)
        act_buffer[2, :, 0] = 1.0
        act_buffer[3, :, 0] = 2.0
        act_buffer[4, :, 0] = 2.0
        gravity = torch.tensor([0.0, 0.0, -9.80665], dtype=torch.float64)

        smooth, jerk, snap = compute_action_smoothness_losses_per_step(
            act_buffer, gravity
        )

        self.assertEqual(smooth.shape, torch.Size([3, 2]))
        self.assertEqual(jerk.shape, torch.Size([3, 2]))
        self.assertEqual(snap.shape, torch.Size([3, 2]))
        self.assertTrue((smooth >= 0.0).all())
        self.assertTrue((jerk >= 0.0).all())
        self.assertTrue((snap >= 0.0).all())

    def test_energy_is_normalized_squared_action_magnitude(self):
        actions = torch.tensor([[[3.0, 4.0, 0.0]]])
        loss = compute_action_energy_loss_per_step(actions, action_scale=5.0)
        self.assertTrue(torch.allclose(loss, torch.ones_like(loss)))


if __name__ == "__main__":
    unittest.main()
