import math
import unittest

import torch

from utils.tensor_utils import (
    build_command_velocity,
    compute_velocity_tracking_loss,
    decode_worker_action,
    sample_command_speed,
)


class WorkerOfficialVelocityControlTest(unittest.TestCase):
    def test_v2_decodes_interleaved_acceleration_velocity_and_yaw(self):
        act = torch.tensor([[1.0, 4.0, 2.0, 5.0, 3.0, 6.0, 0.5]], dtype=torch.float32)
        rotation = torch.eye(3, dtype=torch.float32).unsqueeze(0)

        acceleration, velocity, yaw_rate = decode_worker_action(act, rotation, math.pi)

        self.assertTrue(torch.allclose(acceleration, torch.tensor([[1.0, 2.0, 3.0]])))
        self.assertTrue(torch.allclose(velocity, torch.tensor([[4.0, 5.0, 6.0]])))
        self.assertTrue(torch.allclose(yaw_rate, torch.tanh(act[:, 6:7]) * math.pi))

    def test_legacy_decodes_acceleration_without_velocity_or_yaw(self):
        act = torch.tensor([[1.0, 2.0, 0.0, 3.0, 0.0, 4.0]], dtype=torch.float32)
        yaw_90 = torch.tensor(
            [[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]],
            dtype=torch.float32,
        )

        acceleration, velocity, yaw_rate = decode_worker_action(act, yaw_90, math.pi)

        self.assertTrue(torch.allclose(acceleration, torch.tensor([[0.0, 1.0, 0.0]])))
        self.assertTrue(torch.allclose(velocity, torch.tensor([[-3.0, 2.0, 4.0]])))
        self.assertIsNone(yaw_rate)

    def test_grouped_random_commands_share_speed_within_each_group(self):
        torch.manual_seed(7)
        command = sample_command_speed(
            8, 0.75, 3.25, 2.0, torch.device('cpu'), n_drones_per_group=4
        )
        self.assertEqual(tuple(command.shape), (8, 1))
        self.assertTrue(torch.allclose(command[:4], command[:1].expand(4, 1)))
        self.assertTrue(torch.allclose(command[4:], command[4:5].expand(4, 1)))
        self.assertTrue(bool(((command >= 1.5) & (command <= 6.5)).all()))

    def test_goal_velocity_slows_linearly_near_target(self):
        position = torch.tensor([[0.0, 0.0, 0.0], [9.5, 0.0, 0.0]])
        target = torch.tensor([[10.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        command = torch.tensor([[3.0], [3.0]])

        target_v, effective = build_command_velocity(position, target, command, slowdown_time=1.0)

        self.assertTrue(torch.allclose(target_v[:, 0], torch.tensor([3.0, 0.5]), atol=1e-5))
        self.assertTrue(torch.allclose(effective[:, 0], torch.tensor([3.0, 0.5]), atol=1e-5))

    def test_velocity_tracking_loss_is_differentiable(self):
        velocity = torch.zeros((4, 2, 3), requires_grad=True)
        target = torch.ones_like(velocity)
        loss = compute_velocity_tracking_loss(velocity, target, window=30)
        loss.backward()

        self.assertGreater(float(loss.detach()), 0.0)
        self.assertIsNotNone(velocity.grad)
        self.assertTrue(torch.isfinite(velocity.grad).all())


if __name__ == '__main__':
    unittest.main()
