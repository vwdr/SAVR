from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "analyze_acr_a3", ROOT / "scripts/analyze_acr_a3.py"
)
assert SPEC is not None and SPEC.loader is not None
ANALYZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZER)


def test_control_flow_attestation_requires_all_gates_before_checkpoint():
    markers = [
        'proof["proofs"]["factorized_token_parity"]',
        'proof["proofs"]["factorized_action_parity"]',
        'proof["proofs"]["scene_isolation"]',
        'proof["proofs"]["wrist_isolation"]',
        'proof["proofs"]["reuse_visual_tokens"]',
        'proof["proofs"]["reuse_component_truth"]',
        'proof["proofs"]["reuse_upstream_token_parity"]',
        'proof["proofs"]["reuse_current_state_action_parity"]',
        'proof["proofs"]["metadata_fail_closed"]',
        'proof["proofs"]["context_fail_closed"]',
        "checkpoint_after = validate_checkpoint",
    ]
    source = "\n".join(
        [
            "bitwise parity failed",
            "exact parity failed",
            "isolation failed",
            "did not fail closed",
            "did not reuse",
            *markers,
        ]
    )
    positions = ANALYZER.ordered_control_flow(source)
    assert list(positions.values()) == sorted(positions.values())


def test_control_flow_attestation_rejects_checkpoint_before_proofs():
    source = (ROOT / "scripts/run_acr_correctness.py").read_text()
    checkpoint = "checkpoint_after = validate_checkpoint"
    changed = source.replace(checkpoint, "")
    changed = checkpoint + changed
    with pytest.raises(RuntimeError, match="not sequential"):
        ANALYZER.ordered_control_flow(changed)
