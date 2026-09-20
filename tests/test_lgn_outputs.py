import unittest

import torch

from LossGenNet_transformer import LossGenNet


class LGNOutputsTest(unittest.TestCase):
    def test_lgn_forward_outputs_six_dynamic_loss_weights(self):
        model = LossGenNet(
            state_dim=13,
            geom_dim=19,
            progress_dim=32,
            output_temperature=0.75,
            weight_floor=0.2,
            preference_weight_floor=0.1,
            weight_ceiling=3.0,
        )
        model.eval()

        batch_size = 4
        depth = torch.randn(batch_size, 1, 12, 16, dtype=torch.float32)
        state = torch.randn(batch_size, 13, dtype=torch.float32)
        geom = torch.randn(batch_size, 19, dtype=torch.float32)
        progress = torch.randn(batch_size, 32, dtype=torch.float32)

        outputs, history = model(depth, state, geom, progress, hx=None)

        self.assertEqual(outputs.shape, torch.Size([batch_size, 6]))
        self.assertEqual(history.shape, torch.Size([batch_size, 1, 128]))
        self.assertTrue(torch.isfinite(outputs).all())
        self.assertTrue(torch.isfinite(history).all())
        self.assertTrue((outputs[:, :3] >= 0.2).all())
        self.assertTrue((outputs[:, 3:] >= 0.1).all())
        self.assertTrue((outputs <= 3.0).all())

    def test_history_stores_raw_fused_transformer_tokens(self):
        model = LossGenNet(
            state_dim=13,
            geom_dim=19,
            progress_dim=32,
        ).eval()
        depth = torch.randn(2, 1, 12, 16)
        state = torch.randn(2, 13)
        geom = torch.randn(2, 19)
        progress = torch.randn(2, 32)

        with torch.no_grad():
            v_emb = model.visual_proj(model.visual_net(depth))
            s_emb = model.state_proj(state)
            g_emb = model.geom_proj(geom)
            p_emb = model.progress_proj(progress)
            expected_token = model.pre_norm(
                model.fusion_proj(torch.cat([v_emb, s_emb, g_emb, p_emb], dim=-1))
            )
            _, history = model(depth, state, geom, progress)

        self.assertTrue(torch.allclose(history, expected_token.unsqueeze(1), atol=1e-6))

    def test_history_is_capped_at_max_sequence_length(self):
        model = LossGenNet(
            state_dim=13,
            geom_dim=19,
            progress_dim=32,
            max_seq_len=2,
        ).eval()
        depth = torch.randn(2, 1, 12, 16)
        state = torch.randn(2, 13)
        geom = torch.randn(2, 19)
        progress = torch.randn(2, 32)

        history = None
        with torch.no_grad():
            for _ in range(3):
                _, history = model(depth, state, geom, progress, history)

        self.assertEqual(history.shape, torch.Size([2, 2, 128]))

    def test_extreme_logits_still_respect_weight_bounds(self):
        model = LossGenNet(
            state_dim=13,
            geom_dim=19,
            progress_dim=32,
            weight_floor=0.25,
            preference_weight_floor=0.05,
            weight_ceiling=2.5,
        ).eval()
        with torch.no_grad():
            model.head[0].weight.zero_()
            model.head[0].bias.fill_(1.0)
            model.head[2].weight.zero_()
            model.head[2].bias.zero_()
            row_signs = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
            model.head[2].weight.copy_(row_signs[:, None].expand(-1, 128) * 1e4)
            outputs, _ = model(
                torch.zeros(1, 1, 12, 16),
                torch.zeros(1, 13),
                torch.zeros(1, 19),
                torch.zeros(1, 32),
            )

        self.assertTrue(torch.isfinite(outputs).all())
        self.assertTrue((outputs[:, :3] >= 0.25).all())
        self.assertTrue((outputs[:, 3:] >= 0.05).all())
        self.assertTrue((outputs <= 2.5).all())

    def test_new_preference_weights_start_close_to_off(self):
        torch.manual_seed(0)
        model = LossGenNet(
            state_dim=13,
            geom_dim=19,
            progress_dim=32,
        ).eval()
        with torch.no_grad():
            outputs, _ = model(
                torch.zeros(2, 1, 12, 16),
                torch.zeros(2, 13),
                torch.zeros(2, 19),
                torch.zeros(2, 32),
            )

        self.assertTrue((outputs[:, 3:] >= 0.01).all())
        self.assertTrue((outputs[:, 3:] < 0.06).all())


if __name__ == '__main__':
    unittest.main()
