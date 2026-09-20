import math
import unittest

import torch

from utils.tensor_utils import select_min_clearance_obstacle_sample


class LookaheadCollisionReductionTest(unittest.TestCase):
    def test_selects_minimum_clearance_instead_of_averaging_vectors(self):
        # Opposing vectors would average to zero and falsely look like a collision.
        samples = torch.tensor([
            [[1.0, 0.0, 0.0]],
            [[-1.0, 0.0, 0.0]],
        ])  # [samples=2, batch=1, xyz=3]

        selected, clearance = select_min_clearance_obstacle_sample(
            samples, margin=torch.tensor([0.07]), sample_dim=0,
        )

        self.assertTrue(torch.equal(selected, samples[0]))
        self.assertAlmostEqual(clearance.item(), 0.93, places=5)
        self.assertGreater(clearance.item(), 0.0)

    def test_any_colliding_sample_makes_step_clearance_negative(self):
        samples = torch.tensor([
            [[0.40, 0.0, 0.0]],
            [[0.02, 0.0, 0.0]],
            [[0.30, 0.0, 0.0]],
        ])

        selected, clearance = select_min_clearance_obstacle_sample(
            samples, margin=torch.tensor([0.07]), sample_dim=0,
        )

        self.assertTrue(torch.equal(selected, samples[1]))
        self.assertAlmostEqual(clearance.item(), math.sqrt(0.02 ** 2 + 1e-6) - 0.07, places=6)
        self.assertLess(clearance.item(), 0.0)

    def test_current_position_can_be_clear_while_lookahead_collides(self):
        samples = torch.tensor([
            [[0.40, 0.0, 0.0]],  # tau=0: actual position is collision-free
            [[0.02, 0.0, 0.0]],  # future sample crosses the collision margin
        ])
        margin = torch.tensor([0.07])

        actual_clearance = torch.sqrt((samples[0] * samples[0]).sum(-1) + 1e-6) - margin
        _, lookahead_clearance = select_min_clearance_obstacle_sample(
            samples, margin=margin, sample_dim=0,
        )

        self.assertGreater(actual_clearance.item(), 0.0)
        self.assertLess(lookahead_clearance.item(), 0.0)


if __name__ == '__main__':
    unittest.main()
