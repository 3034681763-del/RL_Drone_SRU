import unittest

import torch

from utils.logging_utils import compute_gradient_alignment_loss


class GradientAlignmentLossTest(unittest.TestCase):
    def test_parallel_gradients_have_zero_loss(self):
        scale = torch.tensor(2.0, requires_grad=True)
        left = (scale * torch.tensor([1.0, 2.0]),)
        right = (torch.tensor([1.0, 2.0], requires_grad=True),)

        loss, cosine, usable = compute_gradient_alignment_loss(left, right, scale)

        self.assertTrue(usable)
        self.assertAlmostEqual(float(cosine.detach()), 1.0, places=6)
        self.assertAlmostEqual(float(loss.detach()), 0.0, places=6)

    def test_opposite_gradients_have_maximum_loss(self):
        scale = torch.tensor(1.0, requires_grad=True)
        left = (scale * torch.tensor([1.0, 0.0]),)
        right = (torch.tensor([-1.0, 0.0]),)

        loss, cosine, usable = compute_gradient_alignment_loss(left, right, scale)

        self.assertTrue(usable)
        self.assertAlmostEqual(float(cosine.detach()), -1.0, places=6)
        self.assertAlmostEqual(float(loss.detach()), 2.0, places=6)

    def test_meta_target_is_detached_but_proxy_side_is_trainable(self):
        proxy_angle = torch.tensor(0.4, requires_grad=True)
        meta_angle = torch.tensor(-0.2, requires_grad=True)
        left = (torch.stack([torch.cos(proxy_angle), torch.sin(proxy_angle)]),)
        right = (torch.stack([torch.cos(meta_angle), torch.sin(meta_angle)]),)

        loss, _, usable = compute_gradient_alignment_loss(left, right, proxy_angle)
        proxy_grad, meta_grad = torch.autograd.grad(
            loss,
            (proxy_angle, meta_angle),
            allow_unused=True,
        )

        self.assertTrue(usable)
        self.assertIsNotNone(proxy_grad)
        self.assertGreater(abs(float(proxy_grad)), 1e-6)
        self.assertIsNone(meta_grad)

    def test_unusable_gradients_return_graph_connected_zero(self):
        reference = torch.tensor(3.0, requires_grad=True)
        loss, cosine, usable = compute_gradient_alignment_loss(
            (None,),
            (None,),
            reference,
        )

        self.assertFalse(usable)
        self.assertEqual(float(loss.detach()), 0.0)
        self.assertEqual(float(cosine.detach()), 0.0)
        self.assertEqual(float(torch.autograd.grad(loss, reference)[0]), 0.0)


if __name__ == '__main__':
    unittest.main()
