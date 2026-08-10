from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"


def read(name: str) -> str:
    return (DOCS / name).read_text(encoding="utf-8")


def test_formal_spec_matches_frozen_method_identity_and_invariants() -> None:
    text = read("ACR_V5_FORMAL_METHOD_SPECIFICATION.md")
    required = (
        "acr-isolated-controller-v1",
        "IR-SA-ACR",
        "R^*(UR^+)^*(\\epsilon\\mid U)",
        "post-reuse-refresh",
        "isolation-state-mismatch",
        "maximum completed reuse streak is one",
        "NOT EXPERIMENTALLY VALIDATED",
    )
    assert all(item in text for item in required)
    assert "Unverified hypotheses" in text
    assert "task-success, reuse-rate, latency" not in text


def test_provenance_ledger_records_versions_negatives_and_resource_boundary() -> None:
    text = read("ACR_V5_IMPLEMENTATION_AND_PROVENANCE_LEDGER.md")
    normalized = " ".join(text.split())
    required = (
        "6007d6f",
        "cf8b731",
        "V3-D",
        "V4-A",
        "zero GPUs",
        "/home/ved/SAVR",
        "not a positive-results paper result",
        "/tmp/savr-v5-doc-sync-verify.json",
        "removed and its absence verified",
    )
    assert all(item in normalized for item in required)


def test_manuscript_guide_separates_supported_and_unsupported_claims() -> None:
    text = read("ACR_V5_MANUSCRIPT_TRANSLATION_GUIDE.md")
    assert "MANUSCRIPT NOT MODIFIED" in text
    assert "These are not yet defensible empirical contributions" in text
    assert "Do not call CPU tests experimental validation" in text
    assert "Success is preserved" in text
    assert "Predeclared paired non-inferiority result" in text


def test_roadmap_keeps_freezes_protected_data_and_gpu_coordination() -> None:
    text = read("ACR_V5_GATED_EVALUATION_ROADMAP.md")
    for phase in ("V5-B", "V5-C", "V5-D", "V5-E", "V5-F", "V5-G", "V5-H"):
        assert f"Phase {phase}" in text
    required = (
        "Freeze before output",
        "protected Goal/final data",
        "Before choosing a GPU, stop for user coordination",
        "No composite score can compensate for a failed primary gate",
        "one active phase",
    )
    assert all(item in text for item in required)
