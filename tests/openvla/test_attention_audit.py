"""Regression tests for the independent-runtime attention finding."""

import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "attention_audit", ROOT / "scripts/verify_openvla_attention_audit.py"
)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def fixture(name):
    return json.loads((ROOT / AUDIT.AUDIT_RELATIVE / name).read_text())["result"]


class AttentionAuditTests(unittest.TestCase):
    def setUp(self):
        self.original = fixture("original_full_v2.json")
        self.components = fixture("original_components_v2.json")
        self.compat = fixture("compatibility_components_v2.json")

    def test_real_evidence_is_authenticated(self):
        result = AUDIT.verify_files(ROOT)
        self.assertTrue(result["attention_semantics_mismatch_confirmed"])
        self.assertFalse(result["closed_loop_effect_measured"])

    def test_original_and_compatibility_have_different_semantics(self):
        self.assertEqual(AUDIT.classify(self.original), "bidirectional")
        self.assertEqual(AUDIT.classify(self.compat), "causal")

    def test_missing_call_rejected(self):
        self.original["calls"].pop()
        with self.assertRaises(ValueError):
            AUDIT.classify(self.original)

    def test_duplicate_case_rejected(self):
        self.original["calls"][1] = copy.deepcopy(self.original["calls"][0])
        with self.assertRaises(ValueError):
            AUDIT.classify(self.original)

    def test_padding_leak_rejected(self):
        self.original["calls"][2]["sdpa_layers"][0]["allowed_keys_by_query"][0][-1] = 1
        with self.assertRaises(ValueError):
            AUDIT.classify(self.original)

    def test_flag_alone_does_not_define_attention(self):
        # Padded causal attention has is_causal=False but a triangular mask.
        self.assertFalse(self.compat["calls"][2]["sdpa_layers"][0]["is_causal"])
        self.assertEqual(AUDIT.classify(self.compat), "causal")

    def test_wrong_flag_rejected(self):
        self.original["calls"][0]["sdpa_layers"][0]["is_causal"] = True
        with self.assertRaises(ValueError):
            AUDIT.classify(self.original)

    def test_nonfinite_intervention_rejected(self):
        self.original["sensitivities"][0]["earlier_tokens_change_when_valid_future_token_changes"] = float("nan")
        with self.assertRaises(ValueError):
            AUDIT.classify(self.original)

    def test_cache_production_difference_rejected(self):
        self.original["cache_production_hidden_difference"] = 0.1
        with self.assertRaises(ValueError):
            AUDIT.classify(self.original)

    def test_nonrestored_observer_rejected(self):
        self.original["sdpa_observer_restored"] = False
        with self.assertRaises(ValueError):
            AUDIT.classify(self.original)

    def test_different_source_rejected(self):
        self.compat["attention_source_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            AUDIT.reconcile(self.original, self.components, self.compat)

    def test_component_tensor_disagreement_rejected(self):
        self.components["hidden_state_sha256"]["cache_False_all_valid"] = "0" * 64
        with self.assertRaises(ValueError):
            AUDIT.reconcile(self.original, self.components, self.compat)


if __name__ == "__main__":
    unittest.main()
