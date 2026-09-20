import unittest

import torch
from torch import nn

from LossGenNet_transformer import LossGenNet
from WorkNet_transformer import WorkNet


class TransformerModelsTest(unittest.TestCase):
    def test_worker_uses_snapshot_transformer_and_token_memory(self):
        model = WorkNet(
            dim_obs=64,
            dim_action=4,
            max_seq_len=2,
        ).eval()
        self.assertIsInstance(model.transformer, nn.TransformerEncoder)
        self.assertEqual(len(model.transformer.layers), 2)
        self.assertEqual(model.transformer.layers[0].self_attn.num_heads, 6)

        depth = torch.randn(2, 1, 12, 16)
        state = torch.randn(2, 64)
        history = None
        with torch.no_grad():
            for _ in range(3):
                action, _, history = model(depth, state, history)

        self.assertEqual(action.shape, torch.Size([2, 4]))
        self.assertEqual(history.shape, torch.Size([2, 2, 192]))

    def test_lgn_uses_snapshot_transformer_with_six_output_head(self):
        model = LossGenNet(
            state_dim=13,
            geom_dim=19,
            progress_dim=32,
            max_seq_len=2,
        ).eval()
        self.assertIsInstance(model.transformer, nn.TransformerEncoder)
        self.assertEqual(len(model.transformer.layers), 2)
        self.assertEqual(model.transformer.layers[0].self_attn.num_heads, 4)
        self.assertEqual(model.head[2].out_features, 6)


if __name__ == '__main__':
    unittest.main()
