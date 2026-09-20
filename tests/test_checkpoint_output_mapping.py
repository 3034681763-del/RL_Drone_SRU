import os
import tempfile
import unittest

import torch
from torch import nn

from utils.io_utils import load_compatible_checkpoint


class _FiveOutputHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(2, 3),
            nn.ReLU(),
            nn.Linear(3, 5),
        )


class _SixOutputHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(2, 3),
            nn.ReLU(),
            nn.Linear(3, 6),
        )


class _ThreeOutputHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(2, 3),
            nn.ReLU(),
            nn.Linear(3, 3),
        )


class _SevenOutputHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(2, 3),
            nn.ReLU(),
            nn.Linear(3, 7),
        )


class CheckpointOutputMappingTest(unittest.TestCase):
    def test_direct_acceleration_checkpoint_expands_to_official_worker_head(self):
        model = _SevenOutputHead()
        old_weight = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        old_bias = torch.arange(4, dtype=torch.float32)

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = os.path.join(tmp_dir, 'four_output_worker.pth')
            torch.save({'head.2.weight': old_weight, 'head.2.bias': old_bias}, checkpoint_path)
            load_compatible_checkpoint(
                model,
                checkpoint_path,
                'TestWorker',
                torch.device('cpu'),
                zero_expanded=True,
                output_row_mappings={4: [0, None, 1, None, 2, None, 3]},
            )

        self.assertTrue(torch.equal(model.head[2].weight[[0, 2, 4, 6]], old_weight))
        self.assertTrue(torch.equal(model.head[2].bias[[0, 2, 4, 6]], old_bias))
        self.assertTrue(torch.equal(model.head[2].weight[[1, 3, 5]], torch.zeros((3, 3))))
        self.assertTrue(torch.equal(model.head[2].bias[[1, 3, 5]], torch.zeros(3)))

    def test_six_weight_head_expands_three_weights_and_preserves_new_rows(self):
        model = _SixOutputHead()
        initial_new_weight = model.head[2].weight[3:].detach().clone()
        initial_new_bias = model.head[2].bias[3:].detach().clone()
        old_weight = torch.arange(9, dtype=torch.float32).reshape(3, 3)
        old_bias = torch.arange(3, dtype=torch.float32)

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = os.path.join(tmp_dir, 'three_output_lgn.pth')
            torch.save({'head.2.weight': old_weight, 'head.2.bias': old_bias}, checkpoint_path)
            load_compatible_checkpoint(
                model,
                checkpoint_path,
                'TestLGN',
                torch.device('cpu'),
                zero_expanded=False,
                output_row_mappings={3: [0, 1, 2, None, None, None]},
            )

        self.assertTrue(torch.equal(model.head[2].weight[:3], old_weight))
        self.assertTrue(torch.equal(model.head[2].bias[:3], old_bias))
        self.assertTrue(torch.equal(model.head[2].weight[3:], initial_new_weight))
        self.assertTrue(torch.equal(model.head[2].bias[3:], initial_new_bias))

    def test_forced_legacy_six_row_mapping_preserves_new_rows(self):
        model = _SixOutputHead()
        initial_new_weight = model.head[2].weight[3:].detach().clone()
        initial_new_bias = model.head[2].bias[3:].detach().clone()
        old_weight = torch.arange(18, dtype=torch.float32).reshape(6, 3)
        old_bias = torch.arange(6, dtype=torch.float32)

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = os.path.join(tmp_dir, 'legacy_six_output_lgn.pth')
            torch.save({'head.2.weight': old_weight, 'head.2.bias': old_bias}, checkpoint_path)
            load_compatible_checkpoint(
                model,
                checkpoint_path,
                'TestLGN',
                torch.device('cpu'),
                zero_expanded=False,
                output_row_mappings={6: [3, 4, 5, None, None, None]},
                force_output_row_mapping=True,
            )

        self.assertTrue(torch.equal(model.head[2].weight[:3], old_weight[3:6]))
        self.assertTrue(torch.equal(model.head[2].bias[:3], old_bias[3:6]))
        self.assertTrue(torch.equal(model.head[2].weight[3:], initial_new_weight))
        self.assertTrue(torch.equal(model.head[2].bias[3:], initial_new_bias))

    def test_three_weight_head_loads_aux_rows_from_nine_output_checkpoint(self):
        model = _ThreeOutputHead()
        old_weight = torch.arange(27, dtype=torch.float32).reshape(9, 3)
        old_bias = torch.arange(9, dtype=torch.float32)

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = os.path.join(tmp_dir, 'nine_output_lgn.pth')
            torch.save({'head.2.weight': old_weight, 'head.2.bias': old_bias}, checkpoint_path)
            load_compatible_checkpoint(
                model,
                checkpoint_path,
                'TestLGN',
                torch.device('cpu'),
                zero_expanded=False,
                output_row_mappings={9: [6, 7, 8]},
            )

        self.assertTrue(torch.equal(model.head[2].weight, old_weight[6:9]))
        self.assertTrue(torch.equal(model.head[2].bias, old_bias[6:9]))

    def test_three_weight_head_loads_aux_rows_from_six_output_checkpoint(self):
        model = _ThreeOutputHead()
        old_weight = torch.arange(18, dtype=torch.float32).reshape(6, 3)
        old_bias = torch.arange(6, dtype=torch.float32)

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = os.path.join(tmp_dir, 'six_output_lgn.pth')
            torch.save({'head.2.weight': old_weight, 'head.2.bias': old_bias}, checkpoint_path)
            load_compatible_checkpoint(
                model,
                checkpoint_path,
                'TestLGN',
                torch.device('cpu'),
                zero_expanded=False,
                output_row_mappings={6: [3, 4, 5]},
            )

        self.assertTrue(torch.equal(model.head[2].weight, old_weight[3:6]))
        self.assertTrue(torch.equal(model.head[2].bias, old_bias[3:6]))

    def test_three_weight_head_loads_aux_rows_from_five_output_checkpoint(self):
        model = _ThreeOutputHead()
        old_weight = torch.arange(15, dtype=torch.float32).reshape(5, 3)
        old_bias = torch.arange(5, dtype=torch.float32)

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = os.path.join(tmp_dir, 'five_output_lgn.pth')
            torch.save({'head.2.weight': old_weight, 'head.2.bias': old_bias}, checkpoint_path)
            load_compatible_checkpoint(
                model,
                checkpoint_path,
                'TestLGN',
                torch.device('cpu'),
                zero_expanded=False,
                output_row_mappings={5: [2, 3, 4]},
            )

        self.assertTrue(torch.equal(model.head[2].weight, old_weight[2:5]))
        self.assertTrue(torch.equal(model.head[2].bias, old_bias[2:5]))

    def test_removed_direction_row_is_skipped_for_weight_and_bias(self):
        model = _FiveOutputHead()
        old_state = model.state_dict()
        old_weight = torch.arange(18, dtype=torch.float32).reshape(6, 3)
        old_bias = torch.arange(6, dtype=torch.float32)
        old_state['head.2.weight'] = old_weight
        old_state['head.2.bias'] = old_bias

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = os.path.join(tmp_dir, 'old_lgn.pth')
            torch.save(old_state, checkpoint_path)
            load_compatible_checkpoint(
                model,
                checkpoint_path,
                'TestLGN',
                torch.device('cpu'),
                output_row_indices=[0, 1, 3, 4, 5],
            )

        expected_rows = torch.tensor([0, 1, 3, 4, 5])
        self.assertTrue(torch.equal(model.head[2].weight, old_weight[expected_rows]))
        self.assertTrue(torch.equal(model.head[2].bias, old_bias[expected_rows]))

    def test_direction_row_is_inserted_when_loading_five_output_checkpoint(self):
        model = _SixOutputHead()
        old_state = model.state_dict()
        old_weight = torch.arange(15, dtype=torch.float32).reshape(5, 3)
        old_bias = torch.arange(5, dtype=torch.float32)
        old_state['head.2.weight'] = old_weight
        old_state['head.2.bias'] = old_bias

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = os.path.join(tmp_dir, 'five_output_lgn.pth')
            torch.save(old_state, checkpoint_path)
            load_compatible_checkpoint(
                model,
                checkpoint_path,
                'TestLGN',
                torch.device('cpu'),
                output_row_mappings={5: [0, 1, None, 2, 3, 4]},
            )

        self.assertTrue(torch.equal(model.head[2].weight[:2], old_weight[:2]))
        self.assertTrue(torch.equal(model.head[2].bias[:2], old_bias[:2]))
        self.assertTrue(torch.equal(model.head[2].weight[2], torch.zeros(3)))
        self.assertEqual(model.head[2].bias[2].item(), 0.0)
        self.assertTrue(torch.equal(model.head[2].weight[3:], old_weight[2:]))
        self.assertTrue(torch.equal(model.head[2].bias[3:], old_bias[2:]))


if __name__ == '__main__':
    unittest.main()
