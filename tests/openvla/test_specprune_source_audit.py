import ast
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("source_audit", ROOT / "scripts/audit_specprune_source.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SourceTests(unittest.TestCase):
    def test_pinned_source_identity(self):
        MODULE.verify_source_identity(MODULE.REFERENCE_SHA256, dict(MODULE.UPSTREAM_BLOBS))

    def test_source_drift_rejected(self):
        with self.assertRaisesRegex(ValueError, "source identity"):
            MODULE.verify_source_identity("wrong", MODULE.UPSTREAM_BLOBS)
        changed = dict(MODULE.UPSTREAM_BLOBS, **{"modeling_prismatic.py": "wrong"})
        with self.assertRaisesRegex(ValueError, "source identity"):
            MODULE.verify_source_identity(MODULE.REFERENCE_SHA256, changed)

    def test_negative_tail_expression(self):
        node = ast.parse("-ACTION_DIM * NUM_ACTIONS_CHUNK - 1", mode="eval").body
        self.assertEqual(MODULE.integer_expression(node, dict(ACTION_DIM=7, NUM_ACTIONS_CHUNK=8)), -57)

    def test_absolute_reference_expression(self):
        node = ast.parse("NUM_PATCHES + NUM_PROMPT_TOKENS + ACTION_DIM * NUM_ACTIONS_CHUNK", mode="eval").body
        self.assertEqual(MODULE.integer_expression(node, dict(NUM_PATCHES=513, NUM_PROMPT_TOKENS=34, ACTION_DIM=7, NUM_ACTIONS_CHUNK=8)), 603)

    def test_calls_are_not_executed(self):
        with self.assertRaises(ValueError):
            MODULE.integer_expression(ast.parse("danger()", mode="eval").body, {})

    def test_reads_actual_upstream_slice(self):
        tree = ast.parse((ROOT / "reports/specprune_source_audit_v1/upstream/modeling_prismatic.py").read_text())
        fn = MODULE.method(tree, "OpenVLAForActionPrediction", "_regression_or_discrete_prediction")
        span = MODULE.head_slice(fn)
        self.assertEqual(MODULE.integer_expression(span.lower, dict(ACTION_DIM=7, NUM_ACTIONS_CHUNK=8)), -57)
        self.assertEqual(MODULE.integer_expression(span.upper, {}), -1)


if __name__ == "__main__":
    unittest.main()
