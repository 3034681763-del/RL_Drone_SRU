import unittest

import torch

from WorkNet_sru import SpatiallyEnhancedRecurrentUnit, WorkNet


class SpatiallyEnhancedWorkerTest(unittest.TestCase):
    def test_worker_keeps_fixed_size_spatial_memory(self):
        model = WorkNet(
            dim_obs=64,
            dim_action=7,
            hidden_channels=32,
        ).eval()
        depth = torch.randn(2, 1, 12, 16)
        state = torch.randn(2, 64)
        memory = None

        with torch.no_grad():
            for _ in range(5):
                action, auxiliary, memory = model(depth, state, memory)

        self.assertIsNone(auxiliary)
        self.assertEqual(action.shape, torch.Size([2, 7]))
        self.assertEqual(memory.shape, torch.Size([2, 32, 6, 8]))
        self.assertTrue(torch.isfinite(action).all())
        self.assertTrue(torch.isfinite(memory).all())

    def test_recurrent_memory_changes_the_next_decision(self):
        torch.manual_seed(11)
        model = WorkNet(dim_obs=12, dim_action=7, hidden_channels=32).eval()
        depth_first = torch.randn(2, 1, 12, 16)
        depth_second = torch.randn(2, 1, 12, 16)
        state = torch.randn(2, 12)

        with torch.no_grad():
            _, _, memory = model(depth_first, state, None)
            action_with_memory, _, _ = model(depth_second, state, memory)
            action_without_memory, _, _ = model(depth_second, state, None)

        self.assertFalse(torch.allclose(action_with_memory, action_without_memory))

    def test_gradients_flow_through_depth_state_and_previous_memory(self):
        model = WorkNet(dim_obs=12, dim_action=7, hidden_channels=32).train()
        depth = torch.randn(2, 1, 12, 16, requires_grad=True)
        state = torch.randn(2, 12, requires_grad=True)
        previous = torch.randn(2, 32, 6, 8, requires_grad=True)

        action, _, memory = model(depth, state, previous)
        (action.square().mean() + memory.square().mean()).backward()

        for tensor in (depth, state, previous):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())

    def test_cell_uses_local_and_dilated_spatial_branches(self):
        cell = SpatiallyEnhancedRecurrentUnit(32)
        self.assertEqual(cell.local_candidate.dilation, (1, 1))
        self.assertEqual(cell.context_candidate.dilation, (2, 2))

    def test_invalid_memory_shape_is_rejected(self):
        model = WorkNet(dim_obs=12, dim_action=7, hidden_channels=32)
        depth = torch.randn(2, 1, 12, 16)
        state = torch.randn(2, 12)
        wrong_memory = torch.randn(2, 32, 3, 4)

        with self.assertRaisesRegex(ValueError, "memory shape"):
            model(depth, state, wrong_memory)


if __name__ == "__main__":
    unittest.main()
