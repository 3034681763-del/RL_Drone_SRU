import unittest

import torch

from utils.logging_utils import (
    compute_gradient_alignment_loss,
    compute_meta_hypergrads_from_fast_grads,
)


class MetaGradientReuseTest(unittest.TestCase):
    def test_reused_fast_param_vjp_matches_direct_meta_gradient(self):
        phi = torch.tensor([0.4, -0.2], dtype=torch.float64, requires_grad=True)
        theta = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
        inner_lr = 0.15

        proxy_loss = phi[0] * theta.pow(2) + phi[1] * torch.sin(theta)
        inner_grad = torch.autograd.grad(proxy_loss, theta, create_graph=True)[0]
        fast_theta = theta - inner_lr * inner_grad
        meta_loss = (fast_theta - 1.3).pow(2) + 0.1 * fast_theta.pow(4)

        direct = torch.autograd.grad(meta_loss, phi, retain_graph=True)[0]
        outer_grad = torch.autograd.grad(meta_loss, fast_theta, retain_graph=True)[0]
        reused = compute_meta_hypergrads_from_fast_grads(
            (fast_theta,),
            (outer_grad,),
            (phi,),
        )[0]

        self.assertIsNotNone(reused)
        self.assertTrue(torch.allclose(reused, direct, atol=1e-10, rtol=1e-10))

    def test_reused_meta_plus_alignment_matches_total_loss_backward(self):
        phi = torch.tensor([0.3, -0.4], dtype=torch.float64, requires_grad=True)
        theta = torch.tensor([0.8, -0.1], dtype=torch.float64, requires_grad=True)
        inner_lr = 0.2
        alignment_weight = 0.1

        proxy_loss = phi[0].exp() * theta.pow(2).sum() + phi[1] * theta.prod()
        proxy_grads = torch.autograd.grad(
            proxy_loss,
            (theta,),
            create_graph=True,
            retain_graph=True,
        )
        fast_theta = theta - inner_lr * proxy_grads[0]
        meta_loss = (fast_theta - torch.tensor([1.2, -0.7], dtype=torch.float64)).pow(2).sum()
        outer_grads = torch.autograd.grad(
            meta_loss,
            (fast_theta,),
            retain_graph=True,
        )
        alignment_loss, _, usable = compute_gradient_alignment_loss(
            proxy_grads,
            outer_grads,
            reference=proxy_loss,
        )
        self.assertTrue(usable)

        direct_total = torch.autograd.grad(
            meta_loss + alignment_weight * alignment_loss,
            (phi,),
            retain_graph=True,
        )[0]
        reused_meta = compute_meta_hypergrads_from_fast_grads(
            (fast_theta,),
            outer_grads,
            (phi,),
            retain_graph=True,
        )[0]
        reused_alignment = torch.autograd.grad(
            alignment_weight * alignment_loss,
            (phi,),
        )[0]

        self.assertTrue(
            torch.allclose(
                reused_meta + reused_alignment,
                direct_total,
                atol=1e-10,
                rtol=1e-10,
            )
        )

    def test_missing_outer_grad_returns_unused_meta_grads(self):
        phi = torch.tensor(0.5, requires_grad=True)
        fast_theta = phi * 2.0

        reused = compute_meta_hypergrads_from_fast_grads(
            (fast_theta,),
            (None,),
            (phi,),
        )

        self.assertEqual(reused, (None,))


if __name__ == '__main__':
    unittest.main()
