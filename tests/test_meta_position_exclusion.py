import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _assignment_value(tree, target_name):
    matches = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == target_name:
                matches.append(node.value)
    if len(matches) != 1:
        raise AssertionError(
            f"Expected one assignment to {target_name!r}, found {len(matches)}"
        )
    return matches[0]


class MetaPositionExclusionTest(unittest.TestCase):
    def test_main_meta_loss_excludes_final_position(self):
        source = (REPO_ROOT / "mmgj_transformer.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        meta_loss_expr = _assignment_value(tree, "meta_loss")
        referenced_names = {
            node.id for node in ast.walk(meta_loss_expr) if isinstance(node, ast.Name)
        }

        self.assertNotIn("loss_meta_pos", referenced_names)
        self.assertIn("loss_meta_coll", referenced_names)
        self.assertIn("loss_meta_arrival_reward", referenced_names)

        # Removing LGN's M_pos must not remove Worker's direct goal supervision.
        worker_arrival_expr = _assignment_value(tree, "worker_arrival_task_loss")
        self.assertIsInstance(worker_arrival_expr, ast.BinOp)
        self.assertIsInstance(worker_arrival_expr.op, ast.Sub)
        worker_arrival_names = {
            node.id
            for node in ast.walk(worker_arrival_expr)
            if isinstance(node, ast.Name)
        }
        self.assertIn("worker_terminal_loss", worker_arrival_names)
        self.assertIn("worker_arrival_reward", worker_arrival_names)

    def test_unrolled_meta_loss_excludes_final_position(self):
        source = (REPO_ROOT / "utils" / "rollout_utils.py").read_text(encoding="utf-8")
        meta_terms_expr = _assignment_value(ast.parse(source), "meta_terms")
        self.assertIsInstance(meta_terms_expr, ast.Dict)
        keys = {
            key.value
            for key in meta_terms_expr.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }

        self.assertEqual(
            keys,
            {
                "collision",
                "height_bounds",
                "guidance_weighted",
                "smooth_jerk_weighted",
                "smooth_snap_weighted",
                "progress_weighted",
                "energy_weighted",
                "velocity_track_weighted",
                "stuck_weighted",
                "first_arrival_log_time_reward_weighted",
            },
        )


if __name__ == "__main__":
    unittest.main()
